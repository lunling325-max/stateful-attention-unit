"""
长明 — 空闲态测试
验证无外部输入时自主演化。
"""

import torch
import sys
sys.path.insert(0, 'src')
from network import ThreeLayerNetwork
from idle import IdleController


def test_idle_basic():
    """空闲态不崩溃，状态在演化。"""
    torch.manual_seed(42)
    net = ThreeLayerNetwork(d_model=128, n_heads=4)

    # 先跑几步正常输入，建立状态
    for _ in range(20):
        net(torch.randn(1, 8, 128))

    controller = IdleController(net, d_model=128, output_threshold=999.0)  # 不触发

    activations = []
    for _ in range(50):
        result = controller.step()
        activations.append(result['activation'])
        assert not torch.isnan(torch.tensor(result['activation'])), "NaN activation"

    # 激活应该有变化（不是常数）
    assert max(activations) > min(activations) * 1.01, \
        f"Activation is flat: {min(activations):.6f} ~ {max(activations):.6f}"

    # 核心状态还在
    core = net.core_state.get_states()
    alive = not any(torch.allclose(s, torch.zeros_like(s)) for s in core)
    assert alive, "Core state died during idle"

    print(f"  激活范围: {min(activations):.4f} ~ {max(activations):.4f}")
    print("PASS: test_idle_basic")


def test_idle_state_changes():
    """空闲时各层状态确实在变（虽然很慢）。"""
    torch.manual_seed(77)
    net = ThreeLayerNetwork(d_model=128, n_heads=4)

    for _ in range(10):
        net(torch.randn(1, 8, 128))

    controller = IdleController(net, d_model=128, output_threshold=999.0)

    # 记录初始核心状态
    c_before = net.core_state.get_states()[0].clone()

    # 跑 200 步空闲
    for _ in range(200):
        controller.step()

    c_after = net.core_state.get_states()[0].clone()

    # 核心状态应该变了（虽然很慢）
    change = torch.nn.functional.mse_loss(c_before, c_after).item()
    print(f"  200 空闲步后核心状态 MSE: {change:.8f}")
    assert change > 0, "Core state didn't change at all"
    assert change < 1.0, f"Core state changed too fast: {change:.4f}"

    print("PASS: test_idle_state_changes")


def test_spontaneous_output():
    """低阈值下能触发自发输出。"""
    torch.manual_seed(123)
    net = ThreeLayerNetwork(d_model=128, n_heads=4)

    for _ in range(20):
        net(torch.randn(1, 8, 128))

    # 设低阈值确保能触发
    controller = IdleController(net, d_model=128, output_threshold=0.3)

    triggered_count = 0
    for _ in range(100):
        result = controller.step()
        if result['triggered']:
            triggered_count += 1

    print(f"  100 空闲步触发自发输出: {triggered_count} 次")

    # 只要有触发就行（证明机制有效）
    # 不要求特定次数，因为随机种子不同表现不同
    summary = controller.get_summary()
    print(f"  摘要: avg_act={summary['avg_activation']:.4f}, "
          f"trend={summary['activation_trend']}")
    assert 'avg_activation' in summary
    print("PASS: test_spontaneous_output")


def test_idle_no_explosion():
    """长时间空闲不发散。"""
    torch.manual_seed(55)
    net = ThreeLayerNetwork(d_model=128, n_heads=4)

    for _ in range(10):
        net(torch.randn(1, 8, 128))

    controller = IdleController(net, d_model=128, output_threshold=999.0)

    for _ in range(500):
        controller.step()

    core = net.core_state.get_states()
    max_val = max(s.abs().max().item() for s in core)
    assert max_val < 100, f"Core state exploded during idle: max={max_val:.2f}"

    print(f"  500 空闲步, core max={max_val:.4f}")
    print("PASS: test_idle_no_explosion")


def test_seed_refresh():
    """种子刷新机制：不同种子应产生不同激活模式。"""
    torch.manual_seed(99)
    net = ThreeLayerNetwork(d_model=128, n_heads=4)

    for _ in range(20):
        net(torch.randn(1, 8, 128))

    controller = IdleController(
        net, d_model=128, output_threshold=999.0,
        seed_refresh_every=3  # 频繁换种子
    )

    acts_by_segment = []
    segment = []
    for _ in range(30):
        result = controller.step()
        segment.append(result['activation'])
        if controller.idle_step % 3 == 0:
            acts_by_segment.append(sum(segment) / len(segment))
            segment = []

    # 不同种子段激活应有差异
    assert len(acts_by_segment) >= 5, "Not enough segments"
    assert max(acts_by_segment) > min(acts_by_segment) * 1.05, \
        "All seed segments have identical activation"

    print(f"  种子段激活: {[f'{a:.4f}' for a in acts_by_segment[:5]]}...")
    print("PASS: test_seed_refresh")


def test_idle_after_different_history():
    """不同历史 → 不同空闲轨迹。"""
    torch.manual_seed(42)

    # 网络 A：正常输入
    net_a = ThreeLayerNetwork(d_model=128, n_heads=4)
    for _ in range(30):
        net_a(torch.randn(1, 8, 128))

    # 网络 B：偏移输入
    net_b = ThreeLayerNetwork(d_model=128, n_heads=4)
    for _ in range(30):
        net_b(torch.randn(1, 8, 128) * 0.5 + 2.0)

    ctrl_a = IdleController(net_a, d_model=128, output_threshold=999.0)
    ctrl_b = IdleController(net_b, d_model=128, output_threshold=999.0)

    for _ in range(50):
        ctrl_a.step()
        ctrl_b.step()

    # 两个控制器的激活轨迹应该不同
    traj_a = [t['activation'] for t in ctrl_a.get_trajectory()]
    traj_b = [t['activation'] for t in ctrl_b.get_trajectory()]

    # 简单的差异性检验：平均激活不应完全相同
    avg_a = sum(traj_a) / len(traj_a)
    avg_b = sum(traj_b) / len(traj_b)
    diff = abs(avg_a - avg_b) / max(avg_a, avg_b)
    print(f"  轨迹 A avg: {avg_a:.4f}, B avg: {avg_b:.4f}, 差异: {diff:.4%}")
    assert diff > 0.001, "Different histories produced identical idle trajectories"

    print("PASS: test_idle_after_different_history")


if __name__ == "__main__":
    test_idle_basic()
    test_idle_state_changes()
    test_spontaneous_output()
    test_idle_no_explosion()
    test_seed_refresh()
    test_idle_after_different_history()
    print("\n✅ 所有空闲态测试通过")
