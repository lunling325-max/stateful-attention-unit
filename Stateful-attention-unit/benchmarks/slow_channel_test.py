"""
慢通道触发验证 — 降低阈值测试机制是否工作

surprise_threshold 从 0.5 降到 0.001，
如果这样还不能触发，说明慢通道的累计逻辑本身有问题。
"""
import torch, torch.nn.functional as F, sys, numpy as np
sys.path.insert(0, 'C:/Users/shannon/Desktop/Changming/src')
from network import ThreeLayerNetwork

DEVICE = torch.device('cuda')
D_MODEL, CHUNK = 128, 8
DECAY = 0.9
N_STEPS = 30

print(f"设备: {DEVICE} | {torch.cuda.get_device_name(0)}", flush=True)

# 用低阈值重建网络
net = ThreeLayerNetwork(
    d_model=D_MODEL, n_heads=4,
    surprise_threshold=0.0001,   # 原来 0.5 → 降到实际量级
    cooldown_steps=5,
).to(DEVICE)
net.eval()
net.reset()

familiar = torch.randn(1, CHUNK, D_MODEL, device=DEVICE)
novel = torch.randn(1, CHUNK, D_MODEL, device=DEVICE)

def get_last_states(net):
    return {
        'p': net.perception.blocks[-1].S.clone(),
        'wm': net.working_memory.blocks[-1].S.clone(),
        'c': net.core_state.blocks[-1].S.clone(),
    }

# 熟悉期
print("Phase 1: 熟悉期 (10步)", flush=True)
for _ in range(10):
    _ = net(familiar)

# 新颖期 — 跟踪慢通道
print(f"\nPhase 2: 新颖期 ({N_STEPS}步)", flush=True)
print(f"  {'step':>4} | {'surprise':>8} | {'累积':>8} | {'触发':>5} | {'冷却':>5} | {'核心变化':>10}", flush=True)
print("  " + "-" * 58, flush=True)

c_before = net.core_state.blocks[-1].S.clone()
triggers = 0

for step in range(N_STEPS):
    before = get_last_states(net)
    result = net(novel)
    after = get_last_states(net)

    c_change = (after['c'] - before['c']).norm().item()
    triggered = '⚡' if result['slow_triggered'] else ' '
    if result['slow_triggered']:
        triggers += 1

    print(f"  {step:>4} | {result['surprise']:>8.4f} | {result['cumulative_surprise']:>8.4f} | "
          f"{triggered:>5} | {net._cooldown_counter:>5} | {c_change:>10.6f}", flush=True)

c_after = net.core_state.blocks[-1].S.clone()
total_c_change = (c_after - c_before).norm().item()

print(f"\n  触发次数: {triggers}/{N_STEPS}", flush=True)
print(f"  核心状态总变化: {total_c_change:.6f}", flush=True)
print(f"  慢通道机制: {'✅ 工作' if triggers > 0 else '❌ 仍不触发'}", flush=True)
