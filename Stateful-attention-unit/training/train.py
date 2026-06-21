"""
长明 — 训练脚本 v1
字符预测任务：喂字符序列，SAU 状态累积，预测下一字符
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '../src')
from network import ThreeLayerNetwork
from decoder import TextDecoder, CHAR_TO_IDX, IDX_TO_CHAR, VOCAB_SIZE
from idle import IdleController


# ─── 训练数据 ───
TEXT = "the cat sat on the mat. " * 3  # ~72 字符，小但够


class CharEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, char_idx: int) -> torch.Tensor:
        """char_idx: int → [1, 1, d_model]"""
        idx = torch.tensor([char_idx], device=self.embed.weight.device)
        return self.embed(idx).unsqueeze(1)  # [1, 1, d_model]


def train():
    print("=" * 60)
    print("  长明 — 训练演示")
    print("=" * 60)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    d_model = 128
    epochs = 200  # GPU + 梯度累积，CPU 压力轻

    # ── 初始化 ──
    print(f"\n训练数据 ({len(TEXT)} 字符): \"{TEXT[:60]}...\"")
    net = ThreeLayerNetwork(d_model=d_model, n_heads=4).to(device)
    decoder = TextDecoder(d_model=d_model, hidden_dim=256).to(device)
    embedder = CharEmbedding(VOCAB_SIZE, d_model).to(device)

    # 切换到训练模式（SAU 不 detach 状态）
    net.train()
    decoder.train()

    optimizer = torch.optim.Adam(
        list(net.parameters()) + list(decoder.parameters()) + list(embedder.parameters()),
        lr=0.001,
    )

    # ── 训练循环 ──
    print(f"\n训练 {epochs} 轮...")
    loss_history = []

    for epoch in range(epochs):
        net.reset()  # 每 epoch 重置状态
        total_loss = 0.0
        optimizer.zero_grad()  # 累积梯度，epoch 末统一 step

        steps = len(TEXT) - 1
        for i in range(steps):
            current_char = TEXT[i]
            target_char = TEXT[i + 1]

            if current_char not in CHAR_TO_IDX or target_char not in CHAR_TO_IDX:
                continue

            cur_idx = CHAR_TO_IDX[current_char]
            tgt_idx = CHAR_TO_IDX[target_char]

            # 前向
            x = embedder(cur_idx)          # [1, 1, d_model]
            result = net(x)
            output = result['output']       # [1, d_model]
            logits = decoder(output)         # [1, vocab_size]

            # 损失（缩放避免梯度累积过大）
            loss = nn.functional.cross_entropy(
                logits, torch.tensor([tgt_idx], device=device)
            ) / steps
            total_loss += loss.item() * steps  # 还原原始 loss 用于显示

            # 反向传播（累积梯度）
            loss.backward()

        # epoch 末统一更新
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer.step()

        avg_loss = total_loss / (len(TEXT) - 1)
        loss_history.append(avg_loss)

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:>3}: loss={avg_loss:.4f}")

    print(f"  epoch {epochs:>3}: loss={avg_loss:.4f}")

    # ── 训练后评估 ──
    net.eval()
    decoder.eval()
    net.reset()

    print("\n--- 训练后预测测试 ---")
    test_text = "the "
    print(f"  输入: \"{test_text}\"")

    pred_chars = []
    for ch in test_text:
        if ch in CHAR_TO_IDX:
            x = embedder(CHAR_TO_IDX[ch])
            result = net(x)
            logits = decoder(result['output'])
            pred_idx = logits.argmax(dim=-1).item()
            pred_chars.append(IDX_TO_CHAR.get(pred_idx, '?'))
    print(f"  预测: \"{''.join(pred_chars)}\"")

    # ── 空闲态输出 ──
    print("\n--- 训练后空闲态 ---")
    net.eval()
    controller = IdleController(
        net, d_model=d_model,
        output_threshold=5.2,
        seed_refresh_every=8,
        decoder=decoder,
    )
    controller.seed_gen = controller.seed_gen.to(device)

    output_count = 0
    for _ in range(100):
        result = controller.step()
        if result['triggered'] and result['text'] and result['text'].strip():
            output_count += 1
            if output_count <= 5:
                print(f"  [{controller.idle_step:>3}] \"{result['text']}\"")

    summary = controller.get_summary()
    print(f"\n  空闲输出: {output_count} 次")
    print(f"  平均激活: {summary['avg_activation']:.3f}")
    print(f"  激活趋势: {summary['activation_trend']}")

    # ── 保存模型 ──
    print("\n--- 保存模型 ---")
    torch.save({
        'network': net.state_dict(),
        'decoder': decoder.state_dict(),
        'embedder': embedder.state_dict(),
    }, 'checkpoint.pt')
    print("  已保存到 checkpoint.pt")

    print("\n" + "=" * 60)
    print("  训练完成")
    print("=" * 60)

    return loss_history


if __name__ == "__main__":
    train()
