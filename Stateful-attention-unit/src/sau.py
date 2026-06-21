"""
SAU forward 批处理 — 支持 return_sequence=True 产出 [B, L, d_model]

改动：不在 Python 层逐 token 循环，全序列一次传入 SAU。
保持 backward compatible：默认 return_sequence=False 行为不变。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StatefulAttentionUnit(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 256,
        n_heads: int = 8,
        alpha_init: float = 0.5,
        beta_init: float = 0.5,
        step_scale: float = 0.3,
    ):
        super().__init__()
        assert d_state < d_model, "d_state must be < d_model for information bottleneck"

        self.d_model = d_model
        self.d_state = d_state
        self.n_heads = n_heads

        self.s_q_proj = nn.Linear(d_state, d_model)
        self.x_kv_proj = nn.Linear(d_model, d_model * 2)
        self.x_q_proj = nn.Linear(d_model, d_state)
        self.s_kv_proj = nn.Linear(d_state, d_state * 2)
        self.out_proj = nn.Linear(d_model, d_model)
        self.compress = nn.Linear(d_model, d_state)

        self.state_ffn = nn.Sequential(
            nn.Linear(d_state, d_state * 4),
            nn.GELU(),
            nn.Linear(d_state * 4, d_state),
        )
        self.output_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        self.norm_x = nn.LayerNorm(d_model)
        self.norm_s = nn.LayerNorm(d_state)
        self.norm_hs = nn.LayerNorm(d_state)
        self.norm_cx = nn.LayerNorm(d_state)

        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta = nn.Parameter(torch.tensor(beta_init))
        self.step_scale = step_scale

        self.register_buffer("S", torch.zeros(1, d_state), persistent=False)

    def _cross_attention(self, q, kv, d_head, output_dim):
        B = q.shape[0]
        k, v = kv.chunk(2, dim=-1)

        q = q.view(B, -1, self.n_heads, d_head).transpose(1, 2)
        k = k.view(B, -1, self.n_heads, d_head).transpose(1, 2)
        v = v.view(B, -1, self.n_heads, d_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / (d_head ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, -1, output_dim)
        return out

    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, d_model]
            return_sequence: if True, returns [B, seq_len, d_model] (per-position)
                             if False, returns [B, d_model] (pooled, original behavior)

        S queries X → always cross-attention over full sequence (parallel)
        X queries S → if seq mode, each position independently queries S
                       if pooled, mean-pool first then query S
        """
        B, seq_len, _ = x.shape
        S = self.S.expand(B, self.d_state)

        # ── Step 1: S queries X (same for both modes) ──
        q_s = self.s_q_proj(S).unsqueeze(1)  # [B, 1, d_model]
        kv_x = self.x_kv_proj(x)              # [B, seq_len, d_model*2]
        x_filtered = self._cross_attention(
            q_s, kv_x, d_head=self.d_model // self.n_heads, output_dim=self.d_model
        )  # [B, 1, d_model] — S's view of the input

        if return_sequence:
            # ── Sequence mode: per-position output ──

            # X queries S: each position independently
            q_x = self.x_q_proj(x)  # [B, seq_len, d_state]
            kv_s = self.s_kv_proj(S).unsqueeze(1)  # [B, 1, d_state*2]
            # Cross-attention: each position queries the single state key-value
            # Reshape: treat each position as a separate query over the same KV
            q_x_flat = q_x.view(B * seq_len, 1, self.d_state)
            kv_s_flat = kv_s.expand(-1, seq_len, -1).reshape(B * seq_len, 1, self.d_state * 2)
            s_trigger = self._cross_attention(
                q_x_flat, kv_s_flat, d_head=self.d_state // self.n_heads, output_dim=self.d_state
            )  # [B*seq_len, 1, d_state]
            s_trigger = s_trigger.view(B, seq_len, self.d_state)

            # Per-position output: combine global context + local input
            x_context = x_filtered.expand(-1, seq_len, -1)  # [B, seq_len, d_model]
            h_x = self.output_ffn(self.norm_x(x_context + x))  # [B, seq_len, d_model]
            output = self.out_proj(h_x) + x  # [B, seq_len, d_model]

            # Aggregate state trigger for update
            s_trigger_pooled = s_trigger.mean(dim=1)  # [B, d_state]
            x_filtered_pooled = x_filtered.squeeze(1)  # [B, d_model]

        else:
            # ── Pooled mode (original behavior) ──
            q_x = self.x_q_proj(x.mean(dim=1))  # [B, d_state]
            kv_s = self.s_kv_proj(S)             # [B, d_state*2]
            s_trigger = self._cross_attention(
                q_x.unsqueeze(1), kv_s.unsqueeze(1),
                d_head=self.d_state // self.n_heads, output_dim=self.d_state
            ).squeeze(1)  # [B, d_state]

            s_trigger_pooled = s_trigger
            x_filtered_pooled = x_filtered.squeeze(1)

            h_x = self.output_ffn(self.norm_x(x_filtered_pooled))
            output = self.out_proj(h_x) + x.mean(dim=1)

        # ── State update (identical logic, pooled inputs) ──
        h_s = self.state_ffn(self.norm_s(s_trigger_pooled))
        compressed_x = self.compress(x_filtered_pooled)
        h_s_norm = self.norm_hs(h_s)
        cx_norm = self.norm_cx(compressed_x)
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        delta = alpha * h_s_norm + beta * cx_norm
        delta = delta / (delta.norm(dim=-1, keepdim=True) + 1e-8)
        step = self.step_scale / (self.d_state ** 0.5)
        # 主动内稳态 — 检测 |S| 偏离目标 2.0 时自动拉回
        S_norm = S.norm(dim=-1, keepdim=True)
        target = 2.0
        deviation = torch.clamp(S_norm - target, min=0.0)  # 只在上超时介入
        homeo = 1.0 * deviation * (S / (S_norm + 1e-8))    # 方向指向零
        S_new = S + step * (delta - homeo)                   # delta 减内稳态拉力

        self.S = S_new.mean(dim=0, keepdim=True).detach()

        return output
