"""
LIF-PS (Leaky Integrate-and-Fire with Persistent State)
SAU 的持久性基因 + 神经元的离散通信。
膜电位累积 → 阈值发放 → 部分重置（不是清零）
静默期 S 持续演化。通信用 spike + STDP。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class LIFPlusState(nn.Module):
    """
    单神经元：膜电位 V（标量） + 持久状态 S（向量）。

    V 动态:
      V_new = V * leak + input_gain * I
      if V_new > threshold: spike! V ← V_rest, S ← S + ΔS_spike
      else: V ← V_new

    S 动态 (SAU 基因):
      S_new = S + step * (FFN(S) - homeo)
      homeo = clamp(|S| - target, 0) * S/|S|

    spike 携带 S 的当前方向。
    """

    def __init__(self, input_dim=128, d_state=64,
                 threshold=1.0, leak=0.9, input_gain=0.3,
                 v_rest=0.1, step_scale=0.1, target_norm=2.0):
        super().__init__()
        self.d_state = d_state
        self.threshold = threshold
        self.leak = leak
        self.input_gain = input_gain
        self.v_rest = v_rest
        self.step_scale = step_scale
        self.target_norm = target_norm

        # 输入投影
        self.input_proj = nn.Linear(input_dim, 1)  # → scalar V contribution

        # S 演化网络 (SAU 的 state_ffn)
        self.state_ffn = nn.Sequential(
            nn.Linear(d_state, d_state * 4),
            nn.GELU(),
            nn.Linear(d_state * 4, d_state),
        )
        self.norm_s = nn.LayerNorm(d_state)

        # spike 时 S 收到的更新投影
        self.spike_proj = nn.Sequential(
            nn.Linear(d_state + 1, d_state),  # S direction + spike strength
            nn.GELU(),
            nn.Linear(d_state, d_state),
        )

        # 持久状态 S
        self.register_buffer('S', torch.zeros(1, d_state))
        # 膜电位
        self.register_buffer('V', torch.tensor(v_rest))
        # 上次 spike 的方向
        self.register_buffer('last_spike_dir', torch.zeros(d_state))
        self.register_buffer('last_spike_time', torch.tensor(-1000))
        self.register_buffer('_spike_count', torch.tensor(0))

        self.step_count = 0

    def forward(self, x):
        """
        x: [B, 1, input_dim]  外部输入
        Returns: (output_vector, did_spike, spike_info)
        output_vector = S 的投影，给下游用
        """
        self.step_count += 1
        B = x.shape[0]

        # ── V 动态：leak + 输入整合 ──
        I = self.input_proj(x.squeeze(1)).squeeze(-1).mean()  # scalar
        V_new = self.V * self.leak + self.input_gain * I

        # ── S 持续演化 (静默期也在跑) ──
        h_s = self.state_ffn(self.norm_s(self.S))
        S_norm = self.S.norm(dim=-1, keepdim=True)
        homeo = torch.clamp(S_norm - self.target_norm, min=0.0) * (
            self.S / (S_norm + 1e-8)
        )
        step = self.step_scale / (self.d_state ** 0.5)
        S_new = self.S + step * (h_s - homeo)

        # ── 阈值判定 ──
        if V_new > self.threshold:
            # spike!
            self._spike_count += 1

            # spike 方向 = 当前 S 的方向
            S_dir = S_new.squeeze(0) / (S_new.norm() + 1e-8)

            # spike 强度（超出阈值的幅度）
            spike_strength = (V_new - self.threshold).item()

            # 用 spike 信号更新 S
            spike_input = torch.cat([
                S_dir, torch.tensor([spike_strength], device=S_dir.device)
            ]).unsqueeze(0)
            delta_S_spike = self.spike_proj(spike_input)
            S_new = S_new + 0.1 * delta_S_spike

            # 重置 V
            self.V = torch.tensor(self.v_rest, device=self.V.device)

            # 记录
            self.last_spike_dir = S_dir.clone()
            self.last_spike_time = torch.tensor(self.step_count, device=self.V.device)

            spike_info = {
                'direction': S_dir,
                'strength': spike_strength,
                'time': self.step_count,
            }
            did_spike = True
        else:
            self.V = V_new
            spike_info = None
            did_spike = False

        self.S = S_new.detach()

        # 输出 = S 的投影（给下游用，比如输入到其他神经元）
        output = self.S.clone()  # [1, d_state]

        return output, did_spike, spike_info

    def reset(self):
        self.S = torch.zeros_like(self.S)
        self.V = torch.tensor(self.v_rest)
        self.last_spike_dir = torch.zeros_like(self.last_spike_dir)
        self.last_spike_time = torch.tensor(-1000)
        self._spike_count = torch.tensor(0)
        self.step_count = 0


class STDPNetwork(nn.Module):
    """N 个 LIF-PS 神经元。全连接 + STDP 学习。"""

    def __init__(self, n_neurons=8, input_dim=128, d_state=64,
                 threshold=1.0, stdp_lr=0.005, stdp_window=30,
                 stdp_tau=15):
        super().__init__()
        self.n = n_neurons

        # 神经元
        self.neurons = nn.ModuleList([
            LIFPlusState(input_dim, d_state, threshold=threshold)
            for _ in range(n_neurons)
        ])

        # STDP 权重：w_ij = neuron j 对 neuron i 的影响
        # 初始随机
        self.weight_logits = nn.Parameter(torch.randn(n_neurons, n_neurons) * 0.1)
        self.register_buffer('self_mask', torch.eye(n_neurons, dtype=torch.bool))

        self.stdp_lr = stdp_lr
        self.stdp_window = stdp_window
        self.stdp_tau = stdp_tau

        # 全局 spike 历史：[(step, neuron_idx, direction), ...]
        self.spike_history = []
        self.global_step = 0

    def get_weights(self):
        w = torch.tanh(self.weight_logits * 0.3)
        w = w.masked_fill(self.self_mask, 0.0)
        return w

    def forward(self, external_inputs):
        """
        external_inputs: list of [B, 1, input_dim] × n
        Returns: outputs list, metrics
        """
        self.global_step += 1
        B = external_inputs[0].shape[0]
        weights = self.get_weights()

        # ── 1. 构建循环输入 ──
        prev_states = torch.stack(
            [n.S.squeeze(0) for n in self.neurons]
        )  # [N, d_state]

        recurrent_inputs = weights @ prev_states  # [N, d_state]
        recurrent_I = recurrent_inputs.norm(dim=-1) * 0.3  # [N]

        # ── 2. 神经元前向 ──
        outputs = []
        spikes_this_step = []  # [(idx, info), ...]

        for i, neuron in enumerate(self.neurons):
            x = external_inputs[i]
            x_augmented = x + recurrent_I[i] * 0.15
            out, spiked, info = neuron(x_augmented)
            outputs.append(out)
            if spiked:
                spikes_this_step.append((i, info))

        # ── 3. STDP 更新：用 spike 历史回溯 ──
        if spikes_this_step:
            for i, info in spikes_this_step:
                # 记录
                self.spike_history.append({
                    'step': self.global_step,
                    'neuron': i,
                    'direction': info['direction'].clone(),
                })
            # 用回溯做 STDP
            self._stdp_update(spikes_this_step)

        # ── 4. 清理旧历史 ──
        cutoff = self.global_step - self.stdp_window * 3
        self.spike_history = [h for h in self.spike_history if h['step'] > cutoff]

        # ── 5. 指标 ──
        states = torch.stack([n.S.squeeze(0) for n in self.neurons])
        cos_sims = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                cos = F.cosine_similarity(states[i], states[j], dim=0).item()
                cos_sims.append(cos)

        total_spikes = sum(n._spike_count.item() for n in self.neurons)
        w_active = weights[~self.self_mask]
        v_vals = [n.V.item() for n in self.neurons]

        return outputs, {
            'cos_sim_mean': sum(cos_sims) / len(cos_sims),
            'spikes_this_step': len(spikes_this_step),
            'total_spikes': total_spikes,
            'w_mean_abs': w_active.abs().mean().item(),
            'w_max_abs': w_active.abs().max().item(),
            'V_mean': sum(v_vals) / len(v_vals),
        }

    def _stdp_update(self, spikes_this_step):
        """
        STDP: 对每个刚发放的神经元，扫描历史 buffer 找前序 spike。
        如果 neuron j 的 spike 在 neuron i 之前（0 < Δt ≤ window），w_ij 加强。
        """
        for i, info_i in spikes_this_step:
            t_i = self.global_step

            for hist in self.spike_history:
                j = hist['neuron']
                if j == i:
                    continue
                t_j = hist['step']
                dt = t_i - t_j

                if dt <= 0 or dt > self.stdp_window:
                    continue

                # j 先发，i 后发 → 可能的因果 → strengthen w_ij
                delta = self.stdp_lr * torch.exp(
                    torch.tensor(-dt / self.stdp_tau, device=self.weight_logits.device)
                )

                # 方向一致性加成：同向更强
                cos_dir = F.cosine_similarity(
                    hist['direction'], info_i['direction'], dim=0
                ).item()
                delta = delta * max(0.1, cos_dir)

                with torch.no_grad():
                    self.weight_logits.data[i, j] += delta

    def reset(self):
        for n in self.neurons:
            n.reset()
        torch.nn.init.normal_(self.weight_logits, 0, 0.1)

    def get_spike_counts(self):
        return [n._spike_count.item() for n in self.neurons]


def make_inputs(n_steps, input_dim=128):
    """三流输入。"""
    torch.manual_seed(42)
    t = torch.arange(n_steps, dtype=torch.float32)

    def gen(pat):
        emb = torch.zeros(n_steps, input_dim)
        for i in range(input_dim):
            emb[:, i] = pat(t, i)
        emb += torch.randn(n_steps, input_dim) * 0.2
        return emb.unsqueeze(0).to(DEVICE)

    def fast(t, i): return torch.sin(t*(0.5+i*0.08))*0.5 + torch.sin(t*(1.5+i*0.05))*0.3
    def slow(t, i): return torch.sin(t*0.03+float(i)*0.5)*0.5
    def event(t, i):
        b = torch.zeros_like(t)
        for et in [200, 500, 900, 1400]:
            b[(t>=et)&(t<et+40)] = torch.sin(t[(t>=et)&(t<et+40)]*0.3+float(i))*2.0
        return b

    f = gen(fast); s = gen(slow); e = gen(event)
    # 8 neurons: 0,1,2=fast; 3,4,5=slow; 6,7=event
    return {i: f if i<3 else (s if i<6 else e) for i in range(8)}


def run_experiment(name, threshold, n_steps=2000):
    torch.manual_seed(42)
    net = STDPNetwork(
        n_neurons=8, input_dim=128, d_state=64,
        threshold=threshold, stdp_lr=0.05, stdp_window=100
    ).to(DEVICE)
    net.eval()

    input_map = make_inputs(n_steps)
    traj = []

    with torch.no_grad():
        for step in range(n_steps):
            inputs = [input_map[i][:, step:step+1, :] for i in range(8)]
            _, m = net(inputs)
            m['step'] = step
            traj.append(m)

    return traj, net


def analyze(name, traj, net):
    n = len(traj)
    late = slice(n - n // 10, n)
    avg = lambda k: sum(t[k] for t in traj[late]) / max(n // 10, 1)

    total = traj[-1]['total_spikes']
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  步骤: {n}  总 spike: {total:.0f}  ({total/n/8:.1%} 发放率/神经元)")
    print(f"  cos_sim: {avg('cos_sim_mean'):.4f}  V_mean: {avg('V_mean'):.4f}")
    print(f"  w_mean_abs: {avg('w_mean_abs'):.4f}  w_max: {avg('w_max_abs'):.4f}")

    sc = net.get_spike_counts()
    print(f"  各神经元 spike: {[f'{s:.0f}' for s in sc]}")

    w = net.get_weights().detach().cpu().numpy()
    ratio = (w > 0.1).sum() / max((w != 0).sum(), 1)
    print(f"  正权比例: {ratio:.1%}")


if __name__ == '__main__':
    configs = [
        ("阈值=0.5 (易发放)", 0.5),
        ("阈值=1.0 (基准)", 1.0),
        ("阈值=1.5 (难发放)", 1.5),
    ]

    for name, thresh in configs:
        traj, net = run_experiment(name, thresh, n_steps=2000)
        analyze(name, traj, net)
