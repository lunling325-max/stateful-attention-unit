"""
三层网络熟悉度 — 验证 α/β/step_scale 是否产生不同时间尺度

预期：
  感知层 (β=0.95, step=0.8) → 换输入立刻偏离
  工作记忆 (α=β=0.5, step=0.3) → 逐渐偏离
  核心状态 (α=0.92, step=0.05) → 几乎不动

如果三层响应差不多 → α/β 没生效，架构设计需要重新审视。
"""
import torch, torch.nn.functional as F, sys, time
sys.path.insert(0, 'C:/Users/shannon/Desktop/Changming/src')
from network import ThreeLayerNetwork

DEVICE = torch.device('cuda')
D_MODEL, CHUNK = 128, 8
N_FAMILIAR = 8   # 熟悉期轮数
N_SWITCH = 8      # 切换后轮数

print(f"设备: {DEVICE} | {torch.cuda.get_device_name(0)}", flush=True)
print(f"d_model={D_MODEL}, chunk={CHUNK}", flush=True)

# 初始化三层网络（默认参数：log-space α/β）
net = ThreeLayerNetwork(d_model=D_MODEL, n_heads=4).to(DEVICE)
net.reset()

# 两个不同的 chunk
familiar = torch.randn(1, CHUNK, D_MODEL, device=DEVICE)
novel = torch.randn(1, CHUNK, D_MODEL, device=DEVICE)

# ─── 工具函数 ───
def get_last_states():
    """提取每层最后一个 block 的 state"""
    return {
        'p': net.perception.blocks[-1].S.clone(),
        'wm': net.working_memory.blocks[-1].S.clone(),
        'c': net.core_state.blocks[-1].S.clone(),
    }

def compute_delta_dirs(before, after):
    """计算每层 delta 方向（归一化）"""
    dirs = {}
    for k in before:
        d = (after[k] - before[k]).squeeze(0)
        dirs[k] = d / (d.norm() + 1e-8)
    return dirs

# ─── Phase 1: 熟悉期 ───
print("\n═══ Phase 1: 熟悉期 (重复同一 chunk) ═══", flush=True)
emas = {'p': None, 'wm': None, 'c': None}
decay = 0.9

for step in range(N_FAMILIAR):
    before = get_last_states()
    _ = net(familiar)
    after = get_last_states()

    dirs = compute_delta_dirs(before, after)

    # 更新 EMA
    for k in dirs:
        if emas[k] is None:
            emas[k] = dirs[k]
        else:
            emas[k] = decay * emas[k] + (1 - decay) * dirs[k]
            emas[k] = emas[k] / (emas[k].norm() + 1e-8)

    if step == 0 or step == N_FAMILIAR - 1:
        scores = {k: F.cosine_similarity(emas[k], dirs[k], dim=0).item() for k in dirs}
        print(f"  step {step:>2}: 感知={scores['p']:.4f}  工作记忆={scores['wm']:.4f}  核心={scores['c']:.4f}", flush=True)

# ─── Phase 2: 切换到陌生 ───
print("\n═══ Phase 2: 切换到陌生 chunk ═══", flush=True)
print(f"  {'step':>5} | {'感知(β=0.95)':>14} | {'工作记忆(α=β=0.5)':>18} | {'核心(α=0.92)':>14}", flush=True)
print("  " + "-" * 62, flush=True)

history = {'p': [], 'wm': [], 'c': []}
for step in range(N_SWITCH):
    before = get_last_states()
    _ = net(novel)
    after = get_last_states()

    dirs = compute_delta_dirs(before, after)
    scores = {k: F.cosine_similarity(emas[k], dirs[k], dim=0).item() for k in dirs}

    for k in scores:
        history[k].append(scores[k])

    # 同时更新 EMA（模拟网络适应新输入）
    for k in dirs:
        emas[k] = decay * emas[k] + (1 - decay) * dirs[k]
        emas[k] = emas[k] / (emas[k].norm() + 1e-8)

    markers = []
    for k, name in [('p', '感知'), ('wm', '工作记忆'), ('c', '核心')]:
        if scores[k] < 0.9:
            markers.append(f"{name}↓")
    marker_str = '  ← ' + ','.join(markers) if markers else ''

    print(f"  step {step:>2} | {scores['p']:>14.4f} | {scores['wm']:>18.4f} | {scores['c']:>14.4f}{marker_str}", flush=True)

# ─── 分析 ───
print(f"\n═══ 分析 ═══", flush=True)

# 检测速度：cos_sim 首次跌破阈值所需的步数
thresholds = [0.95, 0.90, 0.85, 0.80]
print(f"  {'阈值':>6} | {'感知层':>8} | {'工作记忆':>10} | {'核心状态':>10}", flush=True)
print("  " + "-" * 42, flush=True)
for thresh in thresholds:
    row = f"  {thresh:>6.2f} |"
    for k in ['p', 'wm', 'c']:
        steps_to_detect = next((i for i, s in enumerate(history[k]) if s < thresh), '—')
        row += f" {str(steps_to_detect):>8}"
    print(row, flush=True)

# 平均下降幅度
for k, name in [('p', '感知层'), ('wm', '工作记忆层'), ('c', '核心状态层')]:
    drop = history[k][0] - history[k][-1]
    print(f"  {name}: 首步={history[k][0]:.4f} → 末步={history[k][-1]:.4f} (Δ={drop:+.4f})", flush=True)

# 时间分离验证
p_fast = history['p'][0] < history['wm'][0]  # 感知层应该更快响应
c_slow = history['c'][0] > history['p'][0]    # 核心层应该更慢

print(f"\n  感知最快响应: {'✅' if p_fast else '❌'}", flush=True)
print(f"  核心最慢响应: {'✅' if c_slow else '❌'}", flush=True)
print(f"  三层分离生效: {'✅ 架构验证通过' if p_fast and c_slow else '❌ α/β 未产生有效分离'}", flush=True)
