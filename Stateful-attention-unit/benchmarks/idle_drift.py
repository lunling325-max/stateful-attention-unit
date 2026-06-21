"""
空闲态三层演化 — 零输入下各层的自驱漂移

核心问题：没人说话时，三层各自在干什么？
预期：感知(step=0.8)快漂，工作记忆(step=0.3)中速，核心(step=0.05)几乎不动
"""
import torch, torch.nn.functional as F, sys, time, numpy as np
sys.path.insert(0, 'C:/Users/shannon/Desktop/Changming/src')
from network import ThreeLayerNetwork

DEVICE = torch.device('cuda')
D_MODEL, CHUNK = 128, 8
IDLE_STEPS = 100
SAMPLE_INTERVAL = 5  # 每 5 步采样一次
TRIALS = 3

print(f"设备: {DEVICE} | {torch.cuda.get_device_name(0)}", flush=True)
print(f"空闲步数: {IDLE_STEPS}, 采样间隔: {SAMPLE_INTERVAL}\n", flush=True)

def get_last_states(net):
    return {
        'p': net.perception.blocks[-1].S.clone().squeeze(0),
        'wm': net.working_memory.blocks[-1].S.clone().squeeze(0),
        'c': net.core_state.blocks[-1].S.clone().squeeze(0),
    }

def compute_autocorr(trajectory):
    """state 自相关: cos_sim(S_t, S_0) 随时间衰减"""
    if len(trajectory) < 2:
        return []
    s0 = trajectory[0]
    return [F.cosine_similarity(s0, s, dim=0).item() for s in trajectory]

all_results = []

for trial in range(TRIALS):
    torch.manual_seed(trial * 137)
    net = ThreeLayerNetwork(d_model=D_MODEL, n_heads=4).to(DEVICE)
    net.eval()
    net.reset()

    # 先用一些输入"热身"（给 state 一个非零起点）
    warmup = torch.randn(1, CHUNK, D_MODEL, device=DEVICE)
    for _ in range(5):
        _ = net(warmup)

    # 记录初始态
    trajectories = {'p': [], 'wm': [], 'c': []}
    s0 = get_last_states(net)
    for k in s0:
        trajectories[k].append(s0[k])

    # 空闲期 — 喂零向量
    zero_input = torch.zeros(1, CHUNK, D_MODEL, device=DEVICE)
    for step in range(IDLE_STEPS):
        _ = net(zero_input)
        if (step + 1) % SAMPLE_INTERVAL == 0:
            s = get_last_states(net)
            for k in s:
                trajectories[k].append(s[k])

    # 计算自相关
    results = {}
    for k, name in [('p', '感知'), ('wm', '工作记忆'), ('c', '核心')]:
        ac = compute_autocorr(trajectories[k])
        final_ac = ac[-1] if ac else 1.0
        half_life = next((i * SAMPLE_INTERVAL for i, v in enumerate(ac) if v < 0.5), IDLE_STEPS)
        results[k] = {'ac': ac, 'final': final_ac, 'half_life': half_life, 'name': name}

    all_results.append(results)

    print(f"trial {trial+1}:", flush=True)
    for k in ['p', 'wm', 'c']:
        r = results[k]
        print(f"  {r['name']:>6}: 最终自相关={r['final']:.4f}  半衰期≈{r['half_life']}步", flush=True)
    print(flush=True)

# ─── 跨 trial ───
print(f"═══ 跨 {TRIALS} trial ═══", flush=True)
print(f"  {'层':>6} | {'最终自相关':>10} | {'半衰期':>8}", flush=True)
print("  " + "-" * 35, flush=True)
for k in ['p', 'wm', 'c']:
    finals = [r[k]['final'] for r in all_results]
    halves = [r[k]['half_life'] for r in all_results]
    name = all_results[0][k]['name']
    print(f"  {name:>6} | {np.mean(finals):>8.4f}±{np.std(finals):.4f} | {np.mean(halves):>6.0f}±{np.std(halves):.0f}", flush=True)

# 验证
p_fast = np.mean([r['p']['final'] for r in all_results])
c_slow = np.mean([r['c']['final'] for r in all_results])
print(f"\n  感知漂移最快: {'✅' if p_fast < 0.5 else '⚠️ 自相关仍高'}", flush=True)
print(f"  核心最稳定:   {'✅' if c_slow > 0.9 else '⚠️ 也在漂移'}", flush=True)
print(f"  空闲态分层:   {'✅ 三层不同的自驱节奏' if p_fast < 0.5 and c_slow > 0.9 else '⚠️'}", flush=True)
