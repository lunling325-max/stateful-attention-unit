"""
长明 — 空闲态 (Idle Loop)
无外部输入时，核心状态驱动自主演化。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeedGenerator(nn.Module):
    """从核心状态生成伪输入种子。

    机制：核心状态投影 + 可控噪声 → 伪外部输入。
    噪声幅度随空闲步数增加（避免原地打转）。
    """

    def __init__(self, d_core: int, d_model: int, noise_base: float = 0.1):
        super().__init__()
        self.noise_base = noise_base
        self.project = nn.Sequential(
            nn.Linear(d_core, d_core),
            nn.GELU(),
            nn.Linear(d_core, d_model),
        )

    def forward(self, core_state: torch.Tensor, idle_step: int) -> torch.Tensor:
        """
        Args:
            core_state: [1, d_core]  当前核心状态
            idle_step: 当前空闲步数（控噪声幅度）
        Returns:
            pseudo_input: [1, 1, d_model]  伪外部输入
        """
        seed = self.project(core_state)  # [1, d_model]

        # 噪声幅度先升后稳：防止正反馈导致激活发散
        noise_scale = self.noise_base * min(1.0 + 0.02 * idle_step, 3.0)
        noise = torch.randn_like(seed) * noise_scale

        pseudo = seed + noise
        return pseudo.unsqueeze(1)  # [1, 1, d_model]


class IdleController:
    """空闲态控制器。

    用法:
        controller = IdleController(network, d_model, decoder=decoder)
        for _ in range(idle_steps):
            result = controller.step()
            if result['triggered']:
                print(f"长明: {result['text']}")
    """

    def __init__(
        self,
        network,           # ThreeLayerNetwork
        d_model: int,
        output_threshold: float = 0.5,
        seed_refresh_every: int = 8,
        decoder=None,      # TextDecoder (optional)
    ):
        self.network = network
        self.d_model = d_model
        self.decoder = decoder

        d_core = network.core_state.blocks[0].d_state
        self.seed_gen = SeedGenerator(d_core, d_model)
        self.output_threshold = output_threshold
        self.seed_refresh_every = seed_refresh_every

        # 状态追踪
        self.idle_step = 0
        self.cached_seed = None
        self.trajectory = []  # [(activation, core_state_snapshot), ...]
        self.spontaneous_outputs = []  # [activation, ...]

    def step(self) -> dict:
        """
        执行一步空闲演化。

        Returns:
            dict with:
                activation: float  当前核心激活水平
                triggered: bool   是否触发自发输出
                p_change, wm_change, c_change: 各层变化
        """
        self.idle_step += 1

        # ── 种子选择 ──
        if self.idle_step % self.seed_refresh_every == 1 or self.cached_seed is None:
            core = self.network._get_core_state()
            self.cached_seed = self.seed_gen(core, self.idle_step)

        # ── 伪输入前向 ──
        # 用伪输入跑正常 forward，所有层状态都会更新
        result = self.network(self.cached_seed)

        # ── 激活检测 ──
        output = result['output']  # [1, d_model]
        activation = output.norm(dim=-1).item()

        triggered = activation > self.output_threshold

        # ── 记录 ──
        self.trajectory.append({
            'step': self.idle_step,
            'activation': activation,
            'p_change': result['p_change'],
            'wm_change': result['wm_change'],
            'c_change': result['c_change'],
            'slow_triggered': result['slow_triggered'],
        })

        if triggered:
            text = None
            if self.decoder is not None:
                text = self.decoder.decode(output, temperature=0.7)
            self.spontaneous_outputs.append({
                'step': self.idle_step,
                'activation': activation,
                'text': text,
            })

        return {
            'activation': activation,
            'triggered': triggered,
            'text': self.spontaneous_outputs[-1]['text'] if triggered and self.spontaneous_outputs else None,
            'p_change': result['p_change'],
            'wm_change': result['wm_change'],
            'c_change': result['c_change'],
        }

    def get_trajectory(self) -> list[dict]:
        """返回空闲轨迹。"""
        return self.trajectory

    def get_summary(self) -> dict:
        """空闲态运行摘要。"""
        if not self.trajectory:
            return {}
        acts = [t['activation'] for t in self.trajectory]
        return {
            'total_steps': self.idle_step,
            'spontaneous_count': len(self.spontaneous_outputs),
            'avg_activation': sum(acts) / len(acts),
            'max_activation': max(acts),
            'min_activation': min(acts),
            'activation_trend': self._trend(acts),
        }

    def _trend(self, values: list[float]) -> str:
        """判断激活趋势：上升/下降/平稳。"""
        if len(values) < 10:
            return 'warming'
        first = sum(values[:5]) / 5
        last = sum(values[-5:]) / 5
        if last > first * 1.1:
            return 'rising'
        elif last < first * 0.9:
            return 'falling'
        return 'stable'

    def reset(self):
        self.idle_step = 0
        self.cached_seed = None
        self.trajectory = []
        self.spontaneous_outputs = []
