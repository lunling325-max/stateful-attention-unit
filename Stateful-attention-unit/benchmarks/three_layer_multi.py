"""
三层网络熟悉度 — 多 trial + 慢通道观测

验证：
1. 三层分离是否跨种子稳定
2. 慢通道累积 surprise → 触发 → 核心状态最终响应
"""
import torch, torch.nn.functional as F, sys, time
sys.path.insert(0, 'C:/Users/shannon/Desktop/Changming/src')
from network import ThreeLayerNetwork

DEVICE = torch.device('cuda')
D_MODEL, CHUNK = 128, 8
N_FAMILIAR = 10
N_NOVEL = 30        # 延长新颖期，给慢通道积累时间
TRIALS = 5

print(f"设备: {DEVICE} | {torch.cuda.get_device_name(0)}", flush=True)
print(f"d_model={D_MODEL}, chunk={CHUNK}, trials={TRIALS}, 新颖期={N_NOVEL}步", flush=True)

def get_last_states(net):
    return {
        'p': net.perception.blocks[-1].S.clone(),
        'wm': net.working_memory.blocks[-1].S.clone(),
        'c': net.core_state.blocks[-1].S.clone(),
    }

def compute_dirs(before, after):
    dirs = {}
    for k in before:
        d = (after[k] - before[k]).squeeze(0)
        dirs[k] = d / (d.norm() + 1e-8)
    return dirs

def compute_scores(emas, dirs):
    return {k: F.cosine_similarity(emas[k], dirs[k], dim=0).item() for k in dirs}

def update_emas(emas, dirs, decay=0.9):
    for k in dirs:
        if emas[k] is None:
            emas[k] = dirs[k]
        else:
            emas[k] = decay * emas[k] + (1 - decay) * dirs[k]
            emas[k] = emas[k] / (emas[k].norm() + 1e-8)

all_trials = []

for trial in range(TRIALS):
    torch.manual_seed(trial * 123)
    net = ThreeLayerNetwork(d_model=D_MODEL, n_heads=4).to(DEVICE)
    net.reset()

    familiar = torch.randn(1, CHUNK, D_MODEL, device=DEVICE)
    novel = torch.randn(1, CHUNK, D_MODEL, device=DEVICE)

    emas = {'p': None, 'wm': None, 'c': None}

    # 熟悉期
    for _ in range(N_FAMILIAR):
        before = get_last_states(net)
        _ = net(familiar)
        after = get_last_states(net)
        dirs = compute_dirs(before, after)
        update_emas(emas, dirs)

    # 新颖期 — 记录每层 + 慢通道
    trial_data = {'p': [], 'wm': [], 'c': [], 'surprise': [], 'triggered': []}
    for step in range(N_NOVEL):
        before = get_last_states(net)
        result = net(novel)
        after = get_last_states(net)

        dirs = compute_dirs(before, after)
        scores = compute_scores(emas, dirs)
        for k in scores:
            trial_data[k].append(scores[k])

        trial_data['surprise'].append(result['cumulative_surprise'])
        trial_data['triggered'].append(result['slow_triggered'])

        update_emas(emas, dirs)

    all_trials.append(trial_data)

    # 单 trial 摘要
    p_first = trial_data['p'][0]
    wm_first = trial_data['wm'][0]
    c_first = trial_data['c'][0]
    c_final = trial_data['c'][-1]
    triggers = sum(trial_data['triggered'])
    print(f"\ntrial {trial+1}:", flush=True)
    print(f"  首步: 感知={p_first:.4f}  工作记忆={wm_first:.4f}  核心={c_first:.4f}", flush=True)
    print(f"  核心末步: {c_final:.4f}  (Δ={c_first-c_final:+.4f})", flush=True)
    print(f"  慢通道触发: {triggers}次  (最终累积surprise={trial_data['surprise'][-1]:.4f})", flush=True)

# ─── 跨 trial 统计 ───
print(f"\n═══ 跨 {TRIALS} trial 统计 ═══", flush=True)

# 首步响应
p_firsts = [t['p'][0] for t in all_trials]
wm_firsts = [t['wm'][0] for t in all_trials]
c_firsts = [t['c'][0] for t in all_trials]

import numpy as np
print(f"  感知首步: {np.mean(p_firsts):.4f} ± {np.std(p_firsts):.4f}", flush=True)
print(f"  工作记忆首步: {np.mean(wm_firsts):.4f} ± {np.std(wm_firsts):.4f}", flush=True)
print(f"  核心首步: {np.mean(c_firsts):.4f} ± {np.std(c_firsts):.4f}", flush=True)

sep_p_wm = np.mean([p - w for p, w in zip(p_firsts, wm_firsts)])
sep_wm_c = np.mean([w - c for w, c in zip(wm_firsts, c_firsts)])
print(f"  感知↔工作记忆分离: {abs(sep_p_wm):.4f}", flush=True)
print(f"  工作记忆↔核心分离: {abs(sep_wm_c):.4f}", flush=True)

# 核心状态在 30 步新颖期后的变化
c_drops = [t['c'][0] - t['c'][-1] for t in all_trials]
print(f"  核心30步后下降: {np.mean(c_drops):.4f} ± {np.std(c_drops):.4f}", flush=True)

# 慢通道
total_triggers = [sum(t['triggered']) for t in all_trials]
print(f"  慢通道触发次数: {np.mean(total_triggers):.1f} ± {np.std(total_triggers):.1f}", flush=True)

# 结论
if abs(sep_p_wm) > 0.1 and abs(sep_wm_c) > 0.3:
    print(f"\n  ✅ 三层时间分离跨种子稳定", flush=True)
    print(f"     感知: 对外界变化高度敏感", flush=True)
    print(f"     工作记忆: 中等响应", flush=True)
    print(f"     核心: 几乎不受影响（α=0.92 + step=0.05）", flush=True)
else:
    print(f"\n  ⚠️ 分离不稳定，需进一步分析", flush=True)
