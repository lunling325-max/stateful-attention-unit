"""
STDP 有效性验证 —— 置换检验
H0: STDP 产生的权重结构不优于随机打乱的 spike 时序
H1: 真实 spike 时序中的因果结构被 STDP 捕获，产生显著更强的权重方差
"""
import sys
sys.path.insert(0, 'C:/Users/shannon/Desktop/Changming/experiments')

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from lif_ps_v1 import LIFPlusState, STDPNetwork, make_inputs

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def structure_score(weight_matrix):
    """权重矩阵的结构化程度：去掉自身后所有权重的标准差。"""
    n = weight_matrix.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=weight_matrix.device)
    w = weight_matrix[mask]
    return w.std().item()


def run_with_spike_log(n_steps=2000, threshold=0.5):
    """跑网络，记录所有 spike 事件。"""
    torch.manual_seed(42)
    net = STDPNetwork(
        n_neurons=8, input_dim=128, d_state=64,
        threshold=threshold, stdp_lr=0.05, stdp_window=100
    ).to(DEVICE)
    net.eval()
    imap = make_inputs(n_steps)

    spike_log = []  # [(step, neuron_idx, direction), ...]

    with torch.no_grad():
        for step in range(n_steps):
            net.global_step += 1
            inputs = [imap[i][:, step:step+1, :] for i in range(8)]
            weights = net.get_weights()
            prev_states = torch.stack([n.S.squeeze(0) for n in net.neurons])
            recurrent_inputs = weights @ prev_states
            recurrent_I = recurrent_inputs.norm(dim=-1) * 0.5

            for i, neuron in enumerate(net.neurons):
                x = inputs[i]
                x_aug = x + recurrent_I[i] * 0.5
                out, spiked, info = neuron(x_aug)
                if spiked:
                    spike_log.append({
                        'step': step,
                        'neuron': i,
                        'direction': info['direction'].clone(),
                    })
                    net.spike_history.append({
                        'step': net.global_step,
                        'neuron': i,
                        'direction': info['direction'].clone(),
                    })

            # STDP
            spikes_now = [(s['neuron'], {'time': s['step'], 'direction': s['direction']})
                          for s in spike_log if s['step'] == step]
            if spikes_now:
                net._stdp_update(spikes_now)

            # 清理
            cutoff = net.global_step - net.stdp_window * 3
            net.spike_history = [h for h in net.spike_history if h['step'] > cutoff]

    return net.get_weights().detach().cpu(), spike_log


def shuffle_spike_log(log):
    """保持每个神经元的发放次数和方向不变，打乱时间顺序。"""
    # 按神经元分组
    by_neuron = {}
    for entry in log:
        n = entry['neuron']
        if n not in by_neuron:
            by_neuron[n] = []
        by_neuron[n].append(entry.copy())

    # 收集所有时间戳并打乱
    all_steps = sorted(list(set(entry['step'] for entry in log)))
    shuffled_steps = all_steps.copy()
    random.shuffle(shuffled_steps)

    # 收集所有 spike（保持 neuron 标签和方向），重新分配时间
    all_spikes = []
    for entries in by_neuron.values():
        all_spikes.extend(entries)

    # 打乱 spike 顺序后重新分配时间戳
    random.shuffle(all_spikes)
    for i, entry in enumerate(all_spikes):
        entry['step'] = shuffled_steps[i % len(shuffled_steps)]

    return sorted(all_spikes, key=lambda x: x['step'])


def run_with_given_spikes(spike_log, n_steps=2000):
    """用给定的 spike 序列重跑网络（只做 STDP，不做循环耦合影响）。"""
    torch.manual_seed(999)  # 不同 seed，只测 STDP
    net = STDPNetwork(
        n_neurons=8, input_dim=128, d_state=64,
        threshold=0.5, stdp_lr=0.05, stdp_window=100
    ).to(DEVICE)
    net.eval()

    # 按步骤索引 spike
    spikes_by_step = {}
    for s in spike_log:
        step = s['step']
        if step not in spikes_by_step:
            spikes_by_step[step] = []
        spikes_by_step[step].append(s)

    net.global_step = 0
    for step in range(n_steps):
        net.global_step += 1
        if step in spikes_by_step:
            for s in spikes_by_step[step]:
                net.spike_history.append({
                    'step': net.global_step,
                    'neuron': s['neuron'],
                    'direction': s['direction'].clone(),
                })
                info = {'time': step, 'direction': s['direction']}
                net._stdp_update([(s['neuron'], info)])

        cutoff = net.global_step - net.stdp_window * 3
        net.spike_history = [h for h in net.spike_history if h['step'] > cutoff]

    return net.get_weights().detach().cpu()


if __name__ == '__main__':
    print("=" * 60)
    print("  STDP 置换检验")
    print("=" * 60)

    # Step 1: 跑真实网络，记录 spike log + 真实权重
    print("\n▶ 跑真实网络 (threshold=0.5)...")
    real_weights, spike_log = run_with_spike_log(n_steps=2000, threshold=0.5)
    real_score = structure_score(real_weights)
    n_spikes = len(spike_log)
    print(f"  总 spike: {n_spikes}")
    print(f"  真实结构分数: {real_score:.6f}")

    # Step 2: 置换检验
    n_permutations = 100
    null_scores = []

    print(f"\n▶ 跑 {n_permutations} 次置换检验...")
    for perm_idx in range(n_permutations):
        shuffled = shuffle_spike_log(spike_log)
        w_null = run_with_given_spikes(shuffled, n_steps=2000)
        score = structure_score(w_null)
        null_scores.append(score)
        if (perm_idx + 1) % 20 == 0:
            print(f"    {perm_idx+1}/{n_permutations}...")

    # Step 3: 统计分析
    null_scores_sorted = sorted(null_scores)
    p_value = sum(1 for s in null_scores if s >= real_score) / n_permutations
    percentile_95 = null_scores_sorted[int(n_permutations * 0.95)]

    print(f"\n{'='*60}")
    print(f"  结果")
    print(f"{'='*60}")
    print(f"  真实结构分数:  {real_score:.6f}")
    print(f"  零分布均值:    {sum(null_scores)/n_permutations:.6f}")
    print(f"  零分布标准差:  {torch.tensor(null_scores).std().item():.6f}")
    print(f"  零分布 95% 分位: {percentile_95:.6f}")
    print(f"  p 值:          {p_value:.4f}")

    if p_value < 0.01:
        print(f"\n  ✅ 拒绝 H0: STDP 产生的结构显著强于随机 (p={p_value:.4f})")
    elif p_value < 0.05:
        print(f"\n  ⚠️  边缘显著 (p={p_value:.4f})")
    else:
        print(f"\n  ❌ 不能拒绝 H0: STDP 未产生随机之上的结构 (p={p_value:.4f})")

    # Step 4: 展示真实权重矩阵
    print(f"\n  真实权重矩阵:")
    w = real_weights.numpy()
    labels = ['F0','F1','S2','S3','E4','E5','S6','E7']
    header = ' '.join(f'{l:>6}' for l in labels)
    print(f"         {header}")
    for i in range(8):
        row = ' '.join(f'{w[i,j]:+6.4f}' if abs(w[i,j]) > 0.02 else '  ·   ' for j in range(8))
        print(f"    {labels[i]}: {row}")
