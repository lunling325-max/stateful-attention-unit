"""
长明 — 三层网络 v2
感知（快）→ 工作记忆（中）→ 核心状态（慢）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sau import StatefulAttentionUnit


class SAUStack(nn.Module):
    """串联 N 个 SAU 块。每块有自己的持久状态。"""

    def __init__(self, n_blocks: int, d_model: int, d_state: int,
                 n_heads: int, alpha_init: float, beta_init: float,
                 step_scale: float = 0.3):
        super().__init__()
        self.blocks = nn.ModuleList([
            StatefulAttentionUnit(d_model, d_state, n_heads, alpha_init, beta_init, step_scale)
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, d_model]
        Returns:
            output: [B, d_model]
        """
        for block in self.blocks:
            if x.dim() == 2:
                x = x.unsqueeze(1)  # [B, d] → [B, 1, d]
            out = block(x)  # [B, d_model]
            x = out
        return x  # [B, d_model]

    def get_states(self) -> list:
        return [b.S.clone() for b in self.blocks]

    def get_effective_alpha_beta(self) -> list[tuple[float, float]]:
        """返回每个块当前生效的 α, β（sigmoid 后）。"""
        return [(torch.sigmoid(b.alpha).item(), torch.sigmoid(b.beta).item())
                for b in self.blocks]


class BiasProjector(nn.Module):
    """将核心状态投影到工作记忆层空间，作为偏置注入。"""

    def __init__(self, d_core: int, d_wm_state: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_core, d_core // 2),
            nn.GELU(),
            nn.Linear(d_core // 2, d_wm_state),
            nn.Tanh(),
        )

    def forward(self, core_state: torch.Tensor) -> torch.Tensor:
        return self.proj(core_state)


class ThreeLayerNetwork(nn.Module):
    """三层持续性网络 v2。

    改进:
    - α/β 使用 log-space 初始化，sigmoid 后可覆盖 (0.05, 0.95)
    - 慢通道累积制 + 冷却期
    - h_s 和 compressed_x 已归一化（SAU 内部 LayerNorm）
    - 核心状态 2 块，总维度与工作记忆对齐
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        # 感知层: 近乎纯前馈
        p_blocks: int = 2,    p_d_state: int = 24,
        p_alpha: float = -3.0, p_beta: float = 3.0,     # sigmoid: α≈0.047, β≈0.953
        # 工作记忆层: 平衡
        wm_blocks: int = 2,   wm_d_state: int = 64,
        wm_alpha: float = 0.0, wm_beta: float = 0.0,     # sigmoid: α=0.5, β=0.5
        # 核心状态层: 极稳定
        c_blocks: int = 2,    c_d_state: int = 64,
        c_alpha: float = 2.5, c_beta: float = -2.5,       # sigmoid: α≈0.924, β≈0.076
        # 慢通道
        surprise_threshold: float = 0.001,
        cooldown_steps: int = 10,
        inject_max: float = 0.15,
    ):
        super().__init__()

        self.surprise_threshold = surprise_threshold
        self.cooldown_steps = cooldown_steps
        self.inject_max = inject_max

        # 三层堆叠
        self.perception = SAUStack(p_blocks, d_model, p_d_state, n_heads,
                                    p_alpha, p_beta, step_scale=0.8)
        self.working_memory = SAUStack(wm_blocks, d_model, wm_d_state, n_heads,
                                        wm_alpha, wm_beta, step_scale=0.3)
        self.core_state = SAUStack(c_blocks, d_model, c_d_state, n_heads,
                                    c_alpha, c_beta, step_scale=0.05)

        # 偏置注入
        self.bias_proj = BiasProjector(c_d_state, wm_d_state)

        # 慢通道压缩
        self.slow_compress = nn.Linear(wm_d_state, c_d_state)

        # 慢通道状态
        self._prev_wm_states = None
        self._cumulative_surprise = 0.0
        self._cooldown_counter = 0
        self._step_count = 0

        # 方向级 surprise：跟踪工作记忆 delta 方向 EMA
        self._wm_delta_ema = None
        self._direction_decay = 0.9  # delta 方向 EMA 衰减
        self._direction_weight = 0.5  # 混合权重：方向 vs 量级

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [B, seq_len, d_model]
        Returns:
            dict with keys: output, p_change, wm_change, c_change, surprise, slow_triggered
        """
        B = x.shape[0]
        self._step_count += 1

        # ── 记录变换前状态 ──
        p_before = self.perception.get_states()
        wm_before = self.working_memory.get_states()
        c_before = self.core_state.get_states()

        # ── 偏置注入（轻量） ──
        wm_bias = self.bias_proj(self._get_core_state().expand(B, -1))
        self._inject_bias(wm_bias)

        # ── 前向传播 ──
        p_out = self.perception(x)
        wm_out = self.working_memory(p_out.unsqueeze(1))
        c_out = self.core_state(wm_out.unsqueeze(1))

        # ── 计算变化量 ──
        p_after = self.perception.get_states()
        wm_after = self.working_memory.get_states()
        c_after = self.core_state.get_states()

        p_change = self._state_change(p_before, p_after)
        wm_change = self._state_change(wm_before, wm_after)
        c_change = self._state_change(c_before, c_after)

        # ── 慢通道：累积制（方向 + 量级混合） ──
        # 量级 surprise (原有的)
        mag_surprise = wm_change

        # 方向 surprise: delta 方向偏离 EMA 的程度
        dir_surprise = 0.0
        wm_delta = (wm_after[-1] - wm_before[-1]).squeeze(0)  # 最后一块的 delta
        wm_delta_norm = wm_delta / (wm_delta.norm() + 1e-8)
        if self._wm_delta_ema is None:
            self._wm_delta_ema = wm_delta_norm
        else:
            dir_surprise = max(0.0, 1.0 - F.cosine_similarity(
                self._wm_delta_ema, wm_delta_norm, dim=0
            ).item())
            self._wm_delta_ema = (
                self._direction_decay * self._wm_delta_ema +
                (1 - self._direction_decay) * wm_delta_norm
            )
            self._wm_delta_ema = self._wm_delta_ema / (self._wm_delta_ema.norm() + 1e-8)

        surprise = mag_surprise * (1 - self._direction_weight) + dir_surprise * self._direction_weight
        slow_triggered = False

        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
        else:
            self._cumulative_surprise += surprise
            if self._cumulative_surprise > self.surprise_threshold:
                self._slow_inject(self._cumulative_surprise)
                self._cumulative_surprise = 0.0
                self._cooldown_counter = self.cooldown_steps
                slow_triggered = True

        return {
            'output': c_out,
            'p_change': p_change,
            'wm_change': wm_change,
            'c_change': c_change,
            'surprise': surprise,
            'cumulative_surprise': self._cumulative_surprise,
            'slow_triggered': slow_triggered,
        }

    def _get_core_state(self) -> torch.Tensor:
        return self.core_state.blocks[-1].S  # [1, d_core]

    def _inject_bias(self, bias: torch.Tensor):
        """轻量偏置注入：old * 0.95 + bias * 0.05"""
        first_block = self.working_memory.blocks[0]
        avg_bias = bias.mean(dim=0, keepdim=True)
        first_block.S = first_block.S * 0.95 + avg_bias * 0.05

    def _state_change(self, before: list, after: list) -> float:
        """计算状态变化（每维平均 MSE）。"""
        total_mse = 0.0
        total_dims = 0
        for s0, s1 in zip(before, after):
            total_mse += F.mse_loss(s0, s1).item() * s0.numel()
            total_dims += s0.numel()
        return total_mse / max(total_dims, 1)

    def _slow_inject(self, cum_surprise: float):
        """累积注入：weight 正比于累积 surprise，但有上限且渐近。"""
        wm_state = self.working_memory.blocks[-1].S
        compressed = self.slow_compress(wm_state)

        # Tanh 上限，避免一次性注入过大
        inject_weight = torch.tanh(torch.tensor(cum_surprise * 2.0)).item()
        inject_weight = min(inject_weight, self.inject_max)

        core_block = self.core_state.blocks[-1]
        core_block.S = core_block.S * (1 - inject_weight) + compressed * inject_weight

    def get_effective_params(self) -> dict:
        """返回三层当前生效的 α/β。"""
        return {
            'perception': self.perception.get_effective_alpha_beta(),
            'working_memory': self.working_memory.get_effective_alpha_beta(),
            'core_state': self.core_state.get_effective_alpha_beta(),
        }

    def reset(self):
        for stack in [self.perception, self.working_memory, self.core_state]:
            for block in stack.blocks:
                block.S = torch.zeros_like(block.S)
        self._prev_wm_states = None
        self._cumulative_surprise = 0.0
        self._cooldown_counter = 0
        self._step_count = 0
        self._wm_delta_ema = None
