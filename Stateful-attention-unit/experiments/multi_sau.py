"""
多 SAU 并行网络 — 最小实验
4 个 SAU，环形连接，轻耦合。观察自发分化 vs 趋同。
"""
import sys
sys.path.insert(0, 'C:/Users/shannon/Desktop/Changming/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
from sau import StatefulAttentionUnit

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class MultiSAURing(nn.Module):
    """N 个 SAU 并行，环形邻居连接，每步后混合邻居状态。"""

    def __init__(self, n_sau=4, d_model=128, d_state=64, n_heads=4,
                 gamma=0.1, alpha_inits=None, beta_inits=None):
        super().__init__()
        self.n_sau = n_sau
        self.gamma = gamma

        if alpha_inits is None:
            alpha_inits = [0.5] * n_sau
        if beta_inits is None:
            beta_inits = [0.5] * n_sau

        self.saus = nn.ModuleList([
            StatefulAttentionUnit(
                d_model=d_model, d_state=d_state, n_heads=n_heads,
                alpha_init=alpha_inits[i], beta_init=beta_inits[i], step_scale=0.3
            ) for i in range(n_sau)
        ])

        # 环形拓扑：每个 SAU 连前后两个邻居
        self.neighbors = [
            [(i - 1) % n_sau, (i + 1) % n_sau] for i in range(n_sau)
        ]

        self.step_count = 0

    def forward(self, x):
        """
        x: [B, seq_len, d_model]
        Returns: list of outputs [B, d_model] x N, + metrics dict
        """
        self.step_count += 1
        B = x.shape[0]

        # 保存混合前状态（用于指标）
        pre_states = [sau.S.clone() for sau in self.saus]

        # Step 1: 所有 SAU 独立处理输入
        outputs = []
        for sau in self.saus:
            out = sau(x)  # [B, d_model] + 更新 sau.S
            outputs.append(out)

        # Step 2: 混合邻居状态
        post_own = [sau.S.clone() for sau in self.saus]  # 自己的新状态

        for i in range(self.n_sau):
            neighbor_states = torch.cat([post_own[j] for j in self.neighbors[i]], dim=0)
            neighbor_mean = neighbor_states.mean(dim=0, keepdim=True)  # [1, d_state]
            # S_i = (1-gamma) * S_i + gamma * mean(neighbor_states)
            self.saus[i].S = (1 - self.gamma) * post_own[i] + self.gamma * neighbor_mean

        # 指标
        metrics = self._compute_metrics(pre_states)

        return outputs, metrics

    def _compute_metrics(self, pre_states):
        """计算分化指标。"""
        states = [sau.S.squeeze(0) for sau in self.saus]  # [d_state] each

        # 成对余弦相似度
        cos_sims = []
        for i in range(self.n_sau):
            for j in range(i + 1, self.n_sau):
                cos = F.cosine_similarity(states[i], states[j], dim=0).item()
                cos_sims.append(cos)

        # 每个 SAU 的有效 α/β
        effective_alphas = [torch.sigmoid(sau.alpha).item() for sau in self.saus]
        effective_betas = [torch.sigmoid(sau.beta).item() for sau in self.saus]

        # 状态范数
        state_norms = [s.norm().item() for s in states]

        # S 变化量（与混合前比较）
        state_deltas = []
        for i in range(self.n_sau):
            delta = (self.saus[i].S.squeeze(0) - pre_states[i].squeeze(0)).norm().item()
            state_deltas.append(delta)

        return {
            'cos_sim_mean': sum(cos_sims) / len(cos_sims) if cos_sims else 0,
            'cos_sim_range': max(cos_sims) - min(cos_sims) if cos_sims else 0,
            'alphas': effective_alphas,
            'betas': effective_betas,
            'state_norms': state_norms,
            'state_deltas': state_deltas,
        }

    def reset(self):
        for sau in self.saus:
            sau.S = torch.zeros_like(sau.S)
        self.step_count = 0


def run_experiment(name, alpha_inits, beta_inits, n_steps=200, gamma=0.1):
    """跑一个实验配置，返回轨迹。"""
    torch.manual_seed(42)

    net = MultiSAURing(
        n_sau=4, d_model=128, d_state=64, n_heads=4,
        gamma=gamma, alpha_inits=alpha_inits, beta_inits=beta_inits
    ).to(DEVICE)
    net.eval()

    # 随机文本序列（用随机嵌入模拟）
    with torch.no_grad():
        embedding = torch.randn(1, n_steps, 128, device=DEVICE) * 0.5

    trajectory = []
    for step in range(n_steps):
        x = embedding[:, step:step+1, :]  # [1, 1, 128]
        outputs, metrics = net(x)
        metrics['step'] = step
        trajectory.append(metrics)

    return trajectory


def analyze_trajectory(traj, name):
    """分析轨迹：前 20% vs 后 20% 的 metrics 对比。"""
    n = len(traj)
    early = slice(0, n // 5)
    late = slice(n - n // 5, n)

    def avg(key): return sum(t[key] for t in traj[late]) / (n // 5)

    early_alpha_range = max(traj[0]['alphas']) - min(traj[0]['alphas'])
    late_alpha_range = max(traj[-1]['alphas']) - min(traj[-1]['alphas'])

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  步骤数: {n}")
    print(f"  初始 α 范围: {early_alpha_range:.4f}  →  最终 α 范围: {late_alpha_range:.4f}")
    print(f"  最终 cos_sim_mean: {avg('cos_sim_mean'):.4f}  (1=完全趋同, 低=分化)")
    print(f"  最终 cos_sim_range: {avg('cos_sim_range'):.4f}")
    print(f"  最终 state_norms: {[f'{x:.2f}' for x in traj[-1]['state_norms']]}")
    print(f"  最终 alphas: {[f'{x:.4f}' for x in traj[-1]['alphas']]}")
    print(f"  最终 betas:  {[f'{x:.4f}' for x in traj[-1]['betas']]}")

    # 趋势：cos_sim 的斜率
    first10_mean = sum(t['cos_sim_mean'] for t in traj[:10]) / 10
    last10_mean = sum(t['cos_sim_mean'] for t in traj[-10:]) / 10
    trending = "趋同 ↑" if last10_mean > first10_mean + 0.05 else ("分化 ↓" if first10_mean > last10_mean + 0.05 else "稳态 →")
    print(f"  趋势 (cos_sim): {first10_mean:.4f} → {last10_mean:.4f}  [{trending}]")

    return late_alpha_range, avg('cos_sim_mean')


if __name__ == '__main__':

    results = {}

    # 实验 1: 完全相同初始参数 — 对称性能否被打破？
    print("\n▶ 实验 1: 4 个 SAU 完全相同参数")
    t1 = run_experiment("相同参数", [0.5]*4, [0.5]*4, n_steps=300)
    r1 = analyze_trajectory(t1, "实验 1: 完全相同参数")

    # 实验 2: 略微不同的 α 初始化 — 差异能否保持？
    print("\n▶ 实验 2: α 有小差异 (0.4, 0.45, 0.55, 0.6)")
    t2 = run_experiment("α 轻微分化", [0.4, 0.45, 0.55, 0.6], [0.5]*4, n_steps=300)
    r2 = analyze_trajectory(t2, "实验 2: α 轻微分化")

    # 实验 3: 较大 α 差异 — 环形拓扑能否阻止趋同？
    print("\n▶ 实验 3: α 有明显差异 (0.1, 0.3, 0.7, 0.9)")
    t3 = run_experiment("α 明显分化", [0.1, 0.3, 0.7, 0.9], [0.5]*4, n_steps=300)
    r3 = analyze_trajectory(t3, "实验 3: α 明显分化")

    # 实验 4: 无耦合 (gamma=0) 对照组 — 验证耦合是趋同的原因
    print("\n▶ 实验 4: 无耦合 (gamma=0) — α 明显差异")
    t4 = run_experiment("无耦合", [0.1, 0.3, 0.7, 0.9], [0.5]*4, n_steps=300, gamma=0.0)
    r4 = analyze_trajectory(t4, "实验 4: 无耦合对照组")

    # 实验 5: 强耦合 (gamma=0.3)
    print("\n▶ 实验 5: 强耦合 (gamma=0.3) — α 明显差异")
    t5 = run_experiment("强耦合", [0.1, 0.3, 0.7, 0.9], [0.5]*4, n_steps=300, gamma=0.3)
    r5 = analyze_trajectory(t5, "实验 5: 强耦合")

    print(f"\n{'='*60}")
    print("  总结")
    print(f"{'='*60}")
    print("  实验1 (相同参数):        能否自发打破对称？")
    print("  实验2 (α轻微分化):       差异能否保持？")
    print("  实验3 (α明显分化):       差异能否保持？")
    print("  实验4 (无耦合对照):      γ=0 应保持任意差异")
    print("  实验5 (强耦合):          γ=0.3 趋同是否更快？")
