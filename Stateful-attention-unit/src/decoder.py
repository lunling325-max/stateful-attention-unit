"""
长明 — 文本解码头
从核心状态输出解码为文字。无需预训练，初始为随机映射，
不同核心状态自然产生不同输出。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 可打印 ASCII（32-126）+ 换行符
CHARS = ''.join(chr(i) for i in range(32, 127)) + '\n'
VOCAB_SIZE = len(CHARS)
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHARS)}


class TextDecoder(nn.Module):
    """从核心状态输出解码为字符序列。

    输入: [B, d_model]  核心状态层输出向量
    输出: [B, vocab_size]  字符 logits
    """

    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, VOCAB_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, d_model] → logits: [B, vocab_size]"""
        return self.mlp(x)

    def decode(
        self, x: torch.Tensor,
        max_len: int = 40,
        temperature: float = 0.8,
        top_k: int = 20,
    ) -> str:
        """将输出向量解码为字符串。

        Args:
            x: [B, d_model] 或 [d_model]  输出向量
            max_len: 最大输出字符数
            temperature: 采样温度（越高越随机）
            top_k: 只从 top-k 候选中采样
        Returns:
            解码后的字符串
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        logits = self.forward(x)  # [1, vocab_size]

        # Temperature + top-k 采样
        logits = logits / temperature

        if top_k > 0:
            top_k = min(top_k, VOCAB_SIZE)
            top_values, _ = torch.topk(logits, top_k, dim=-1)
            min_top = top_values[:, -1:]
            logits = torch.where(logits < min_top, torch.tensor(-float('inf')), logits)

        probs = F.softmax(logits, dim=-1)

        # 采样多个字符
        chars = []
        for _ in range(max_len):
            idx = torch.multinomial(probs.squeeze(0), 1).item()
            c = IDX_TO_CHAR[idx]
            if c == '\n':
                break
            chars.append(c)

        return ''.join(chars)

    def decode_beam(
        self, x: torch.Tensor,
        max_len: int = 30,
        temperature: float = 0.7,
    ) -> str:
        """简单贪心 + 温度：每步取最高概率（加温度随机）。

        用于更可控的输出。
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        logits = self.forward(x) / temperature
        probs = F.softmax(logits, dim=-1)

        chars = []
        for _ in range(max_len):
            idx = probs.argmax(dim=-1).item()
            c = IDX_TO_CHAR[idx]
            if c == '\n':
                break
            chars.append(c)
            # 轻微扰动，防止重复输出同一字符
            probs = probs * 0.95 + torch.rand_like(probs) * 0.05

        return ''.join(chars)


def test_decoder():
    """快速验证解码器输出。"""
    torch.manual_seed(42)
    decoder = TextDecoder(d_model=128)

    # 不同输入产生不同输出
    x1 = torch.randn(1, 128)
    x2 = torch.randn(1, 128) * 2.0

    s1 = decoder.decode(x1, temperature=0.5)
    s2 = decoder.decode(x2, temperature=0.5)

    print(f"  输入1 → \"{s1}\"")
    print(f"  输入2 → \"{s2}\"")
    assert s1 != s2, "Different inputs produced identical output"
    assert len(s1) > 0, "Empty output from decoder"
    print("PASS: test_decoder")


if __name__ == "__main__":
    test_decoder()
