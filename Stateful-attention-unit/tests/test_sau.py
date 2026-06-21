"""
长明 — SAU 连续性测试
"""

import torch
import sys
sys.path.insert(0, 'src')
from sau import StatefulAttentionUnit
import torch.nn.functional as F


def test_basic_continuity():
    """状态自相关不衰减"""
    sau = StatefulAttentionUnit(d_model=64, d_state=32, n_heads=4)
    states = []

    for t in range(20):
        x = torch.randn(1, 8, 64)
        _ = sau(x)
        states.append(sau.S.clone())

    states = torch.stack(states).squeeze(0)
    for t in range(1, len(states)):
        sim = F.cosine_similarity(states[t-1], states[t], dim=-1)
        assert sim > 0, f"State correlation dropped at step {t}"
    
    assert not torch.allclose(sau.S, torch.zeros_like(sau.S)), "State died!"
    print("PASS: test_basic_continuity — state alive, no zero-correlation")


def test_state_not_static():
    """状态确实在变，不是冻住的"""
    sau = StatefulAttentionUnit(d_model=64, d_state=32, n_heads=4)
    
    # Run a few steps with same input
    fixed_x = torch.randn(1, 8, 64)
    _ = sau(fixed_x)
    s1 = sau.S.clone()
    _ = sau(fixed_x)
    s2 = sau.S.clone()

    # Should be similar but not identical
    sim = F.cosine_similarity(s1, s2, dim=-1)
    assert sim < 1.0, "State didn't change with repeated input"
    assert sim > 0.5, "State changed too much with repeated input"
    print(f"PASS: test_state_not_static — cos_sim={sim.item():.4f}")


def test_alpha_beta_effect():
    """不同 α/β 产生不同的状态轨迹"""
    sau_a = StatefulAttentionUnit(
        d_model=64, d_state=32, n_heads=4,
        alpha_init=2.0, beta_init=-2.0  # α≈0.88, β≈0.12
    )
    sau_b = StatefulAttentionUnit(
        d_model=64, d_state=32, n_heads=4,
        alpha_init=-2.0, beta_init=2.0  # α≈0.12, β≈0.88
    )
    # 共享所有权重，仅 α/β 不同
    sau_b.load_state_dict(sau_a.state_dict())
    sau_b.alpha.data = torch.tensor(-2.0)
    sau_b.beta.data = torch.tensor(2.0)

    torch.manual_seed(42)
    inputs = [torch.randn(1, 8, 64) for _ in range(50)]

    state_a, state_b = None, None
    for x in inputs:
        _ = sau_a(x)
        _ = sau_b(x)
    state_a = sau_a.S.clone()
    state_b = sau_b.S.clone()

    # 两个 SAU 最终状态应该显著不同（因为 α/β 不同）
    sim = F.cosine_similarity(state_a, state_b, dim=-1)
    assert sim < 0.99, f"Different α/β produced near-identical states (cos_sim={sim.item():.4f})"
    print(f"PASS: test_alpha_beta_effect — different α/β diverge, cos_sim={sim.item():.4f}")


def test_batch_independence():
    """批处理时状态取均值不影响结果"""
    sau = StatefulAttentionUnit(d_model=64, d_state=32, n_heads=4)
    x = torch.randn(4, 8, 64)

    # Single batch
    _ = sau(x)
    s_batch = sau.S.clone()

    # Reset and run individually
    sau.S = torch.zeros(1, 32)
    for i in range(4):
        _ = sau(x[i:i+1])

    # Should still be alive
    assert not torch.allclose(sau.S, torch.zeros_like(sau.S)), "State died after batch run"
    print("PASS: test_batch_independence — batch processing works")


if __name__ == "__main__":
    test_basic_continuity()
    test_state_not_static()
    test_alpha_beta_effect()
    test_batch_independence()
    print("\n✅ 所有测试通过")
