"""
塑性多 SAU 网络 v2 — 稀疏连接 + 感受野分化 + 文本输入
模拟生物神经系统：每个 SAU 看到输入的不同方面，通过赫布塑料性自发组织。
"""
import sys
sys.path.insert(0, 'C:/Users/shannon/Desktop/Changming/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
from sau import StatefulAttentionUnit

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class SparsePlasticSAUNet(nn.Module):
    """
    N 个 SAU，稀疏连接（环形，每个连 k 个邻居），
    每个 SAU 有自己的输入投影（感受野分化），
    耦合权重通过对比赫布规则在线更新，带权重衰减防饱和。
    """

    def __init__(self, n_sau=8, d_model=128, d_state=64, n_heads=4,
                 gamma=0.1, hebb_lr=0.005, ema_decay=0.95,
                 weight_decay=0.001, k_neighbors=2):
        super().__init__()
        self.n_sau = n_sau
        self.gamma = gamma
        self.hebb_lr = hebb_lr
        self.ema_decay = ema_decay
        self.weight_decay = weight_decay
        self.k_neighbors = k_neighbors

        # SAU 实例 + 各自独立的输入投影（感受野）
        self.saus = nn.ModuleList([
            StatefulAttentionUnit(d_model, d_state, n_heads,
                                  alpha_init=0.5, beta_init=0.5, step_scale=0.3)
            for _ in range(n_sau)
        ])

        # 每个 SAU 的输入投影：[d_model → d_model]，不同随机初始化
        self.input_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
            for _ in range(n_sau)
        ])

        # 稀疏拓扑：环形，每个 SAU 连前后 k 个邻居
        adj = torch.zeros(n_sau, n_sau, dtype=torch.bool)
        for i in range(n_sau):
            for offset in range(1, k_neighbors + 1):
                adj[i, (i + offset) % n_sau] = True
                adj[i, (i - offset) % n_sau] = True
        self.register_buffer('adjacency', adj)

        # 塑料性耦合 logits（只存在于有连接的位置）
        self.coupling_logits = nn.Parameter(
            torch.zeros(n_sau, n_sau) * self.adjacency.float()
        )

        # 活动 EMA
        self.register_buffer('dir_ema', torch.zeros(n_sau, d_state))
        self._ema_init = False
        self.step_count = 0

    def get_coupling_weights(self):
        """当前有效耦合权重，自身=0，无连接=0。"""
        # 使用 soft sign 替代 tanh：更缓和的饱和，范围 [-2, 2]
        w = 2.0 * torch.tanh(self.coupling_logits * 0.5)
        w = w * self.adjacency.float()
        return w

    def forward(self, x):
        """
        x: [B, seq_len, d_model]  (原始输入，每个 SAU 用自己的投影)
        Returns: outputs list, metrics dict
        """
        self.step_count += 1
        B = x.shape[0]

        # ── Step 1: 每个 SAU 用自己的感受野处理输入 ──
        outputs = []
        pre_states = [sau.S.clone() for sau in self.saus]
        for i, sau in enumerate(self.saus):
            x_i = self.input_proj[i](x)  # 感受野投影
            outputs.append(sau(x_i))

        post_own = torch.stack([sau.S.squeeze(0) for sau in self.saus])

        # ── Step 2: 扩散耦合（仅限有连接的邻居） ──
        weights = self.get_coupling_weights()
        d_state = self.saus[0].d_state
        step = self.saus[0].step_scale / (d_state ** 0.5)

        S_diff = post_own.unsqueeze(0) - post_own.unsqueeze(1)
        coupling = (weights.unsqueeze(-1) * S_diff).sum(dim=1)
        post_own = post_own + step * self.gamma * coupling

        # 写回
        for i in range(self.n_sau):
            self.saus[i].S = post_own[i].unsqueeze(0)

        # ── Step 3: 更新活动 EMA ──
        S_dir = post_own / (post_own.norm(dim=-1, keepdim=True) + 1e-8)
        if not self._ema_init:
            self.dir_ema = S_dir.clone()
            self._ema_init = True
        else:
            self.dir_ema = self.ema_decay * self.dir_ema + (1 - self.ema_decay) * S_dir
            self.dir_ema = self.dir_ema / (self.dir_ema.norm(dim=-1, keepdim=True) + 1e-8)

        # ── Step 4: 赫布更新 + 权重衰减 ──
        if self._ema_init:
            corr = F.cosine_similarity(
                self.dir_ema.unsqueeze(0), self.dir_ema.unsqueeze(1), dim=-1
            )

            # 对比信号
            diag_mask = torch.eye(self.n_sau, dtype=torch.bool, device=corr.device)
            active_mask = self.adjacency & ~diag_mask
            n_connections = active_mask.sum(dim=1, keepdim=True).float().clamp(min=1)

            corr_masked = corr.masked_fill(~active_mask, 0.0)
            mean_corr = corr_masked.sum(dim=1, keepdim=True) / n_connections

            delta_logits = (corr - mean_corr) * active_mask.float()

            with torch.no_grad():
                # 赫布更新 + 权重衰减（向零拉回，防止饱和）
                self.coupling_logits.data += (
                    self.hebb_lr * delta_logits
                    - self.weight_decay * self.coupling_logits.data * active_mask.float()
                )

        # ── 指标 ──
        metrics = self._compute_metrics(weights)

        return outputs, metrics

    def _compute_metrics(self, weights):
        states = [sau.S.squeeze(0) for sau in self.saus]

        cos_sims = []
        for i in range(self.n_sau):
            for j in range(i + 1, self.n_sau):
                cos = F.cosine_similarity(states[i], states[j], dim=0).item()
                cos_sims.append(cos)

        # 聚类指标：权重是否出现集团
        w_active = weights[self.adjacency]
        pos_frac = (w_active > 0.1).float().mean().item()
        neg_frac = (w_active < -0.1).float().mean().item()

        return {
            'cos_sim_mean': sum(cos_sims) / len(cos_sims),
            'state_norms': [s.norm().item() for s in states],
            'w_pos_frac': pos_frac,
            'w_neg_frac': neg_frac,
            'w_mean_abs': w_active.abs().mean().item(),
            'w_max_abs': w_active.abs().max().item() if w_active.numel() > 0 else 0,
        }

    def reset(self):
        for sau in self.saus:
            sau.S = torch.zeros_like(sau.S)
        torch.nn.init.zeros_(self.coupling_logits)
        self.dir_ema = torch.zeros_like(self.dir_ema)
        self._ema_init = False
        self.step_count = 0

    def get_clusters(self):
        """基于耦合权重的简单聚类：正权 = 同组，负权 = 异组。"""
        weights = self.get_coupling_weights().detach().cpu().numpy()
        n = self.n_sau

        # 构建亲和矩阵：正权→亲和，负权→排斥
        affinity = (weights + weights.T) / 2  # 对称化

        # 贪心聚类：把有强正连接的分在一组
        visited = set()
        clusters = []
        threshold = 0.2

        for i in range(n):
            if i in visited:
                continue
            cluster = [i]
            visited.add(i)
            # BFS：找所有与当前聚类成员有正连接的 SAU
            queue = [i]
            while queue:
                cur = queue.pop(0)
                for j in range(n):
                    if j not in visited and affinity[cur, j] > threshold:
                        visited.add(j)
                        cluster.append(j)
                        queue.append(j)
            clusters.append(cluster)

        return clusters


def load_text_embeddings(n_steps, d_model=128):
    """用随机投影 + 正弦位置编码模拟文本嵌入。"""
    torch.manual_seed(42)
    # 多频率合成模拟文本的韵律结构
    t = torch.arange(n_steps, dtype=torch.float32)
    embeddings = torch.zeros(n_steps, d_model)
    for i in range(d_model):
        freq = 1.0 + i * 0.3
        phase = torch.randn(1).item() * 3.14
        embeddings[:, i] = torch.sin(t * freq * 0.1 + phase) * 0.5
        if i % 3 == 0:
            embeddings[:, i] += torch.sin(t * freq * 0.05) * 0.3

    # 加一些"内容"跳变（模拟句子边界）
    for boundary in [200, 500, 800, 1100, 1500, 1800]:
        embeddings[boundary:boundary+20] += torch.randn(20, d_model) * 0.8

    return embeddings.unsqueeze(0)  # [1, n_steps, d_model]


def run_experiment(name, gamma, hebb_lr, wd, k_neighbors, n_steps=2000):
    torch.manual_seed(42)

    net = SparsePlasticSAUNet(
        n_sau=8, d_model=128, d_state=64, n_heads=4,
        gamma=gamma, hebb_lr=hebb_lr, weight_decay=wd, k_neighbors=k_neighbors
    ).to(DEVICE)
    net.eval()

    embedding = load_text_embeddings(n_steps).to(DEVICE)

    trajectory = []
    snapshots = {}  # 每隔 500 步保存权重快照

    with torch.no_grad():
        for step in range(n_steps):
            x = embedding[:, step:step+1, :]
            outputs, metrics = net(x)
            metrics['step'] = step
            trajectory.append(metrics)

            if step % 500 == 499 or step == 0:
                snapshots[step] = {
                    'weights': net.get_coupling_weights().detach().cpu().clone(),
                    'clusters': net.get_clusters(),
                    'step': step + 1,
                }

    return trajectory, snapshots, net.get_coupling_weights(), net.get_clusters()


def analyze(name, traj, snapshots, final_weights, final_clusters):
    n = len(traj)
    late = slice(n - n // 10, n)

    def avg(key): return sum(t[key] for t in traj[late]) / max((n // 10), 1)

    first100 = sum(t['cos_sim_mean'] for t in traj[:100]) / 100
    last100 = sum(t['cos_sim_mean'] for t in traj[-100:]) / 100

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  步骤数: {n}")
    print(f"  cos_sim 趋势: {first100:.4f} → {last100:.4f}")
    print(f"  最终 w_mean_abs: {avg('w_mean_abs'):.4f}  w_max_abs: {avg('w_max_abs'):.4f}")
    print(f"  最终 w_pos: {avg('w_pos_frac'):.1%}  w_neg: {avg('w_neg_frac'):.1%}")

    # 聚类演化
    print(f"\n  聚类演化:")
    for step, snap in snapshots.items():
        clusters = snap['clusters']
        cluster_str = ' | '.join(
            f"{{{','.join(str(s) for s in c)}}}" for c in clusters
        )
        print(f"    t={snap['step']:5d}: {cluster_str}")

    # 最终权重矩阵
    print(f"\n  最终耦合权重矩阵:")
    w = final_weights.detach().cpu().numpy()
    for i in range(w.shape[0]):
        row = ' '.join(f'{w[i,j]:+6.3f}' if w[i,j] != 0 else '  ·   ' for j in range(w.shape[1]))
        print(f"    SAU{i}: {row}")


if __name__ == '__main__':
    configs = [
        # (gamma, hebb_lr, weight_decay, k_neighbors, name)
        (0.1, 0.003, 0.0005, 2, "基准: 环形k=2, γ=0.1, 慢塑料+轻衰减"),
        (0.1, 0.003, 0.002,  2, "强衰减: 更积极的权重衰减"),
        (0.2, 0.003, 0.0005, 2, "强耦合: γ=0.2"),
        (0.1, 0.003, 0.0005, 3, "宽邻域: 环形k=3"),
    ]

    for gamma, hebb_lr, wd, k, name in configs:
        print(f"\n▶ {name}")
        traj, snaps, final_w, final_c = run_experiment(
            name, gamma, hebb_lr, wd, k, n_steps=2000
        )
        analyze(name, traj, snaps, final_w, final_c)
