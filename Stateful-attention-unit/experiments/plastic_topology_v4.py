"""
塑性拓扑多 SAU 网络 v4 — 连接图本身参与塑料性
全连接初始化 → 赫布强化/削弱 → 弱连接删剪 → 高相关重连
"""
import sys
sys.path.insert(0, 'C:/Users/shannon/Desktop/Changming/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
from sau import StatefulAttentionUnit

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class PlasticTopologySAU(nn.Module):
    """
    全连接初始化。边权重赫布更新。
    每 N 步：|w| < prune_threshold → 断开
    每 N 步：断开对中活动相关性 > reconnect_threshold → 重连（权重 0 初始化）
    """

    def __init__(self, n_sau=4, d_model=128, d_state=64, n_heads=4,
                 gamma=0.1, hebb_lr=0.003, ema_decay=0.95,
                 weight_decay=0.0005,
                 prune_threshold=0.08, reconnect_threshold=0.6,
                 topology_update_every=200):
        super().__init__()
        self.n_sau = n_sau
        self.gamma = gamma
        self.hebb_lr = hebb_lr
        self.ema_decay = ema_decay
        self.weight_decay = weight_decay
        self.prune_threshold = prune_threshold
        self.reconnect_threshold = reconnect_threshold
        self.topology_update_every = topology_update_every

        # SAU 实例 + 各自输入投影
        self.saus = nn.ModuleList([
            StatefulAttentionUnit(d_model, d_state, n_heads,
                                  alpha_init=0.5, beta_init=0.5, step_scale=0.3)
            for _ in range(n_sau)
        ])
        self.input_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
            for _ in range(n_sau)
        ])

        # 全连接初始化（除了自身）
        self.coupling_logits = nn.Parameter(torch.randn(n_sau, n_sau) * 0.1)
        self.register_buffer('self_mask', torch.eye(n_sau, dtype=torch.bool))
        # 连接状态：1=有边, 0=无边
        self.register_buffer('active_edges', torch.ones(n_sau, n_sau, dtype=torch.bool))
        self.active_edges[self.self_mask] = False

        # 活动 EMA
        self.register_buffer('dir_ema', torch.zeros(n_sau, d_state))
        self._ema_init = False
        self.step_count = 0

        # 拓扑变化记录
        self.topology_history = []

    def get_coupling_weights(self):
        """只有 active_edges=True 的位置有权重。"""
        w = 2.0 * torch.tanh(self.coupling_logits * 0.5)
        w = w * self.active_edges.float()
        return w

    def _update_topology(self):
        """删剪弱连接，重连高相关断开对。"""
        weights = self.get_coupling_weights()
        # 用 EMA 方向的余弦相似度
        with torch.no_grad():
            corr_mat = F.cosine_similarity(
                self.dir_ema.unsqueeze(0), self.dir_ema.unsqueeze(1), dim=-1
            )

            changes_made = {'pruned': 0, 'reconnected': 0}
            n_edges_before = self.active_edges.sum().item()

            # 删剪：|w| < threshold 且存在边
            w_abs = weights.abs()
            prune_mask = (w_abs < self.prune_threshold) & self.active_edges
            if prune_mask.any():
                self.active_edges[prune_mask] = False
                changes_made['pruned'] = prune_mask.sum().item()

            # 重连：corr > threshold 且无边存在（非自身）
            reconnect_mask = (corr_mat > self.reconnect_threshold) & ~self.active_edges & ~self.self_mask
            if reconnect_mask.any():
                self.active_edges[reconnect_mask] = True
                # 新连接权重初始化接近 0
                self.coupling_logits.data[reconnect_mask] = torch.randn(
                    reconnect_mask.sum(), device=self.coupling_logits.device
                ) * 0.05
                changes_made['reconnected'] = reconnect_mask.sum().item()

            n_edges_after = self.active_edges.sum().item()

            # 集群检测
            clusters = self._detect_clusters()

            self.topology_history.append({
                'step': self.step_count,
                'n_edges_before': n_edges_before,
                'n_edges_after': n_edges_after,
                'pruned': changes_made['pruned'],
                'reconnected': changes_made['reconnected'],
                'clusters': clusters,
                'w_mean_abs': w_abs[self.active_edges].mean().item() if self.active_edges.any() else 0,
            })

    def _detect_clusters(self):
        """简单连通分量检测。"""
        visited = set()
        clusters = []
        for i in range(self.n_sau):
            if i in visited:
                continue
            cluster = [i]
            visited.add(i)
            queue = [i]
            while queue:
                cur = queue.pop(0)
                for j in range(self.n_sau):
                    if j not in visited and self.active_edges[cur, j]:
                        visited.add(j)
                        cluster.append(j)
                        queue.append(j)
            clusters.append(cluster)
        return clusters

    def forward(self, inputs):
        self.step_count += 1

        # ── 1. 各 SAU 处理自己的输入 ──
        outputs = []
        for i, sau in enumerate(self.saus):
            x_i = self.input_proj[i](inputs[i])
            outputs.append(sau(x_i))

        post_own = torch.stack([sau.S.squeeze(0) for sau in self.saus])

        # ── 2. 扩散耦合（只经存在的边） ──
        weights = self.get_coupling_weights()
        d_state = self.saus[0].d_state
        step = self.saus[0].step_scale / (d_state ** 0.5)

        S_diff = post_own.unsqueeze(0) - post_own.unsqueeze(1)
        coupling = (weights.unsqueeze(-1) * S_diff).sum(dim=1)
        post_own = post_own + step * self.gamma * coupling

        for i in range(self.n_sau):
            self.saus[i].S = post_own[i].unsqueeze(0)

        # ── 3. EMA ──
        S_dir = post_own / (post_own.norm(dim=-1, keepdim=True) + 1e-8)
        if not self._ema_init:
            self.dir_ema = S_dir.clone()
            self._ema_init = True
        else:
            self.dir_ema = self.ema_decay * self.dir_ema + (1 - self.ema_decay) * S_dir
            self.dir_ema = self.dir_ema / (self.dir_ema.norm(dim=-1, keepdim=True) + 1e-8)

        # ── 4. 赫布更新 ──
        if self._ema_init:
            corr = F.cosine_similarity(
                self.dir_ema.unsqueeze(0), self.dir_ema.unsqueeze(1), dim=-1
            )
            active_mask = self.active_edges
            n_conn = active_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
            corr_masked = corr.masked_fill(~active_mask, 0.0)
            mean_corr = corr_masked.sum(dim=1, keepdim=True) / n_conn
            delta_logits = (corr - mean_corr) * active_mask.float()

            with torch.no_grad():
                self.coupling_logits.data += (
                    self.hebb_lr * delta_logits
                    - self.weight_decay * self.coupling_logits.data * active_mask.float()
                )

        # ── 5. 周期性拓扑更新 ──
        if self.step_count % self.topology_update_every == 0:
            self._update_topology()

        # ── 指标 ──
        states = [sau.S.squeeze(0) for sau in self.saus]
        cos_sims = []
        for i in range(self.n_sau):
            for j in range(i + 1, self.n_sau):
                cos = F.cosine_similarity(states[i], states[j], dim=0).item()
                cos_sims.append(cos)

        n_edges = self.active_edges.sum().item()
        max_edges = self.n_sau * (self.n_sau - 1)

        return outputs, {
            'cos_sim_mean': sum(cos_sims) / len(cos_sims),
            'n_edges': n_edges,
            'edge_density': n_edges / max_edges,
            'clusters': self._detect_clusters(),
        }


def make_mixed_inputs(n_steps, d_model=128):
    """三流混合输入（与 v3 相同）"""
    torch.manual_seed(42)
    t = torch.arange(n_steps, dtype=torch.float32)

    def gen_stream(pattern_fn):
        emb = torch.zeros(n_steps, d_model)
        for i in range(d_model):
            emb[:, i] = pattern_fn(t, i)
        emb += torch.randn(n_steps, d_model) * 0.15
        return emb.unsqueeze(0).to(DEVICE)

    def fast_pat(t, i):
        return (torch.sin(t * (0.5 + i*0.08)) * 0.4 +
                torch.sin(t * (1.2 + i*0.05)) * 0.3)

    def slow_pat(t, i):
        return torch.sin(t * 0.02 + float(i)*0.5) * 0.6

    def event_pat(t, i):
        base = torch.zeros_like(t)
        for et in [150, 400, 700, 1100, 1600]:
            window = (t >= et) & (t < et + 30)
            base[window] = torch.sin(t[window] * 0.3 + float(i)) * 1.5
        return base

    return {
        0: gen_stream(fast_pat),   # SAU0: fast
        1: gen_stream(fast_pat),   # SAU1: fast  
        2: gen_stream(slow_pat),   # SAU2: slow
        3: gen_stream(event_pat),  # SAU3: event
    }


def run_experiment(gamma, hebb_lr, wd, prune_thresh, reconnect_thresh, n_steps=3000):
    torch.manual_seed(42)
    net = PlasticTopologySAU(
        n_sau=4, gamma=gamma, hebb_lr=hebb_lr, weight_decay=wd,
        prune_threshold=prune_thresh, reconnect_threshold=reconnect_thresh
    ).to(DEVICE)
    net.eval()

    input_map = make_mixed_inputs(n_steps)
    trajectory = []

    with torch.no_grad():
        for step in range(n_steps):
            inputs = [input_map[i][:, step:step+1, :] for i in range(4)]
            _, metrics = net(inputs)
            metrics['step'] = step
            trajectory.append(metrics)

    return trajectory, net.topology_history, net.get_coupling_weights()


def analyze(name, traj, topo_hist, final_weights):
    n = len(traj)
    late = slice(n - n // 10, n)
    avg = lambda k: sum(t[k] for t in traj[late]) / max(n // 10, 1)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  步骤数: {n}")
    print(f"  最终边密度: {avg('edge_density'):.2%}")
    print(f"  最终 cos_sim_mean: {avg('cos_sim_mean'):.4f}")

    print(f"\n  拓扑演化:")
    for h in topo_hist:
        clusters_str = ' | '.join(
            f"{{{','.join(str(s) for s in c)}}}" for c in h['clusters']
        )
        print(f"    t={h['step']:5d}: 边={h['n_edges_after']}/{12} "
              f"删={h['pruned']} 连={h['reconnected']} "
              f"|w|={h['w_mean_abs']:.4f}  "
              f"集群={clusters_str}")

    print(f"\n  最终权重矩阵 (· = 无边):")
    w = final_weights.detach().cpu().numpy()
    labels = ['F0','F1','S2','E3']
    header = ' '.join(f'{l:>6}' for l in labels)
    print(f"         {header}")
    for i in range(4):
        row = ' '.join(
            f'{w[i,j]:+6.3f}' if abs(w[i,j]) > 1e-6 else '  ·   '
            for j in range(4)
        )
        print(f"    {labels[i]}: {row}")


if __name__ == '__main__':
    configs = [
        # (gamma, hebb_lr, wd, prune_thresh, reconnect_thresh, name)
        (0.1, 0.003, 0.0005, 0.08, 0.6, "基准"),
        (0.1, 0.003, 0.0005, 0.15, 0.7, "高删剪阈值"),
        (0.1, 0.003, 0.001,  0.08, 0.6, "强衰减"),
    ]

    for gamma, hebb_lr, wd, pt, rt, name in configs:
        print(f"\n▶ {name} (γ={gamma}, prune<{pt}, reconnect>{rt})")
        traj, topo, final_w = run_experiment(gamma, hebb_lr, wd, pt, rt, n_steps=3000)
        analyze(name, traj, topo, final_w)
