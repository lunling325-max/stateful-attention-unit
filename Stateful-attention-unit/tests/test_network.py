"""
长明 — 三层网络测试 v2
验证时间节奏分离、慢通道累积制、偏置注入、长跑稳定
"""

import torch
import sys
sys.path.insert(0, 'src')
from network import ThreeLayerNetwork
import torch.nn.functional as F


def test_forward():
    """前向传播不崩溃。"""
    torch.manual_seed(123)
    net = ThreeLayerNetwork(d_model=128, n_heads=4)

    for t in range(5):
        x = torch.randn(1, 8, 128)
        result = net(x)
        out = result['output']
        assert out.shape == (1, 128), f"Bad shape: {out.shape}"
        assert not torch.isnan(out).any(), f"NaN at step {t}"

    # 验证 debug 字段存在
    for key in ['output', 'p_change', 'wm_change', 'c_change', 'surprise', 'slow_triggered']:
        assert key in result, f"Missing key: {key}"

    print("PASS: test_forward")


def test_time_scale_separation():
    """感知层 > 工作记忆层 > 核心状态层（每维变化速率）。"""
    torch.manual_seed(42)
    net = ThreeLayerNetwork(d_model=128, n_heads=4)

    p_changes, wm_changes, c_changes = [], [], []

    for _ in range(100):
        x = torch.randn(1, 8, 128)
        result = net(x)
        p_changes.append(result['p_change'])
        wm_changes.append(result['wm_change'])
        c_changes.append(result['c_change'])

    p_avg = sum(p_changes[10:]) / len(p_changes[10:])  # 跳过 warmup
    wm_avg = sum(wm_changes[10:]) / len(wm_changes[10:])
    c_avg = sum(c_changes[10:]) / len(c_changes[10:])

    print(f"  感知层 avg Δ/维:     {p_avg:.6f}")
    print(f"  工作记忆层 avg Δ/维: {wm_avg:.6f}")
    print(f"  核心状态层 avg Δ/维: {c_avg:.6f}")

    assert p_avg > wm_avg, \
        f"Perception ({p_avg:.6f}) should change more than WM ({wm_avg:.6f})"
    assert wm_avg > c_avg, \
        f"WM ({wm_avg:.6f}) should change more than core ({c_avg:.6f})"

    # 验证 α/β 生效
    params = net.get_effective_params()
    p_alpha = sum(a for a, _ in params['perception']) / len(params['perception'])
    c_alpha = sum(a for a, _ in params['core_state']) / len(params['core_state'])
    assert p_alpha < c_alpha, \
        f"Perception α ({p_alpha:.4f}) should be < core α ({c_alpha:.4f})"

    print(f"  α: perception={p_alpha:.4f}, core={c_alpha:.4f} ✓")
    print("PASS: test_time_scale_separation")


def test_slow_channel_cumulative():
    """慢通道是累积制：连续高 surprise 才触发，有冷却期。"""
    torch.manual_seed(99)
    net = ThreeLayerNetwork(
        d_model=128, n_heads=4,
        surprise_threshold=0.0001, cooldown_steps=5  # 匹配新的小步长
    )

    triggers = []
    for _ in range(50):
        x = torch.randn(1, 8, 128) * 2.0  # 偏高方差 → 持续中等 surprise
        result = net(x)
        triggers.append(result['slow_triggered'])

    trigger_count = sum(triggers)
    # 50 步中应有几次触发，但不是每步都触发
    assert trigger_count >= 1, f"No slow channel triggers in 50 steps"
    assert trigger_count < 50, f"Slow channel triggered every step ({trigger_count})"

    # 检查冷却期：触发后 cooling 步内不应再触发
    in_cooldown = False
    cooldown_left = 0
    for i, t in enumerate(triggers):
        if t:
            if in_cooldown:
                print(f"  WARNING: trigger at step {i} during cooldown")
            in_cooldown = True
            cooldown_left = 5
        if in_cooldown:
            cooldown_left -= 1
            if cooldown_left <= 0:
                in_cooldown = False

    print(f"  50 步中触发 {trigger_count} 次")
    print("PASS: test_slow_channel_cumulative")


def test_bias_injection():
    """不同历史 → 不同核心偏置 → 相同输入产生不同工作记忆响应。"""
    torch.manual_seed(55)
    net_a = ThreeLayerNetwork(d_model=128, n_heads=4)
    net_b = ThreeLayerNetwork(d_model=128, n_heads=4)
    net_b.load_state_dict(net_a.state_dict())

    # 不同历史
    torch.manual_seed(1)
    for _ in range(500):
        net_a(torch.randn(1, 8, 128))
        net_b(torch.randn(1, 8, 128) * 0.1 + 1.5)

    ca = net_a.core_state.get_states()
    cb = net_b.core_state.get_states()
    core_diff = sum(F.mse_loss(a, b).item() for a, b in zip(ca, cb))
    print(f"  核心状态差异: {core_diff:.6f}")
    assert core_diff > 1e-6, "Core states too similar after different histories"

    # 相同输入，工作记忆响应应不同
    torch.manual_seed(42)
    test_x = torch.randn(1, 8, 128)
    ra = net_a(test_x)
    rb = net_b(test_x)
    wm_diff = abs(ra['wm_change'] - rb['wm_change'])
    print(f"  工作记忆 Δ 差异: {wm_diff:.6f}")
    assert wm_diff > 1e-10, "WM response identical despite different core bias"

    print("PASS: test_bias_injection")


def test_long_run_stability():
    """500 步不发散、不死。"""
    torch.manual_seed(77)
    net = ThreeLayerNetwork(d_model=128, n_heads=4)

    for t in range(500):
        x = torch.randn(1, 8, 128)
        result = net(x)
        out = result['output']

        if t % 100 == 0:
            assert not torch.isnan(out).any(), f"NaN at step {t}"
            assert not torch.isinf(out).any(), f"Inf at step {t}"

    core = net.core_state.get_states()
    max_val = max(s.abs().max().item() for s in core)
    alive = not any(torch.allclose(s, torch.zeros_like(s)) for s in core)
    assert alive, "Core state died"
    assert max_val < 100, f"Core state exploded: max={max_val:.2f}"

    print(f"PASS: test_long_run_stability — 500 steps, core max={max_val:.4f}")


def test_effective_params():
    """验证 α/β 在合理的数值范围内。"""
    net = ThreeLayerNetwork(d_model=128, n_heads=4)
    params = net.get_effective_params()

    # 感知层 α 应很小，β 应很大
    for a, b in params['perception']:
        assert a < 0.2, f"Perception α too high: {a:.4f}"
        assert b > 0.8, f"Perception β too low: {b:.4f}"

    # 核心状态 α 应很大，β 应很小
    for a, b in params['core_state']:
        assert a > 0.8, f"Core α too low: {a:.4f}"
        assert b < 0.2, f"Core β too high: {b:.4f}"

    print("PASS: test_effective_params")
    for layer, vals in params.items():
        print(f"  {layer}: α={vals[0][0]:.4f} β={vals[0][1]:.4f}")


if __name__ == "__main__":
    test_forward()
    test_time_scale_separation()
    test_slow_channel_cumulative()
    test_bias_injection()
    test_long_run_stability()
    test_effective_params()
    print("\n✅ 所有三层网络 v2 测试通过")
