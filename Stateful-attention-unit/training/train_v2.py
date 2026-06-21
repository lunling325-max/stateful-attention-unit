"""
长明训练 v2 — 无 reset、流式输入、批处理 SAU

改进：
1. 全程不 reset state（SAU 持久性真正起作用）
2. 更长文本（多句英文 + 中文拼音）
3. 批处理 SAU forward（return_sequence=True）
4. 梯度累积（epoch 级别，CPU 友好）
"""
import torch, torch.nn as nn, torch.nn.functional as F, sys, time
sys.path.insert(0, '../src')
from network import ThreeLayerNetwork
from decoder import TextDecoder, CHAR_TO_IDX, IDX_TO_CHAR, VOCAB_SIZE

DEVICE = torch.device('cuda')
D_MODEL = 128
EPOCHS = 200
CHUNK = 8  # 批处理 chunk 大小

# 更长更丰富的训练文本
TEXT = (
    "the cat sat on the mat. "
    "the dog ran in the park. "
    "a bird flew over the tree. "
    "she opened the door and smiled. "
    "he walked down the street alone. "
    "the sun set behind the mountain. "
    "rain fell softly on the roof. "
    "they laughed and danced all night. "
)
print(f"训练文本: {len(TEXT)} 字符, {len(set(TEXT))} 种字符", flush=True)

# ─── 嵌入层 ───
class CharEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, indices):
        return self.embed(indices)

# ─── 文本 → 可训练序列 ───
# 只保留词表内的字符
clean = [c for c in TEXT if c in CHAR_TO_IDX]
indices = torch.tensor([CHAR_TO_IDX[c] for c in clean], dtype=torch.long)
print(f"有效字符: {len(indices)}", flush=True)

# 切成 chunk，每个 chunk 的 target 是下一个字符
# 输入: chunk[t], 目标: 下一个 chunk 的第一个字符 + 同 chunk 的后续字符
num_chunks = (len(indices) - 1) // CHUNK
print(f"Chunks: {num_chunks} (chunk_size={CHUNK})", flush=True)

torch.manual_seed(42)
net = ThreeLayerNetwork(d_model=D_MODEL, n_heads=4).to(DEVICE)
decoder = TextDecoder(d_model=D_MODEL, hidden_dim=256).to(DEVICE)
embedder = CharEmbedding(VOCAB_SIZE, D_MODEL).to(DEVICE)

net.train()
decoder.train()

optimizer = torch.optim.Adam(
    list(net.parameters()) + list(decoder.parameters()) + list(embedder.parameters()),
    lr=0.001,
)

print(f"\n训练 {EPOCHS} epochs...", flush=True)
net.reset()  # 只在开始时 reset 一次
t0 = time.time()

for epoch in range(EPOCHS):
    total_loss = 0.0
    optimizer.zero_grad()

    for i in range(0, len(indices) - CHUNK, CHUNK):
        # 输入 chunk
        x_idx = indices[i:i+CHUNK].to(DEVICE)  # [chunk]
        # 目标：下一个字符（滑动一位）
        tgt_idx = indices[i+1:i+1+CHUNK].to(DEVICE)

        if len(tgt_idx) < CHUNK:
            continue

        x_emb = embedder(x_idx).unsqueeze(0)  # [1, chunk, d_model]

        # 批处理 forward（return_sequence=True）
        result = net(x_emb)
        output = result['output']  # [1, d_model] — 池化输出

        # 解码最后一个位置的输出
        logits = decoder(output)  # [1, vocab_size]

        # 预测整个 chunk 的下一个字符序列的前几个
        # 简化：只预测下一个字符
        loss = F.cross_entropy(logits, tgt_idx[:1]) / num_chunks

        total_loss += loss.item() * num_chunks
        loss.backward()

    # epoch 末更新
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
    optimizer.step()

    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        elapsed = time.time() - t0
        print(f"  epoch {epoch:>3}: loss={total_loss/num_chunks:.4f}  ({elapsed:.0f}s)", flush=True)

elapsed = time.time() - t0
print(f"\n训练完成: {elapsed:.0f}s\n", flush=True)

# ─── 保存 ───
torch.save({
    'network': net.state_dict(),
    'decoder': decoder.state_dict(),
    'embedder': embedder.state_dict(),
}, 'checkpoint_v2.pt')
print("已保存 checkpoint_v2.pt", flush=True)

# ─── 快速测试 ───
net.eval()
decoder.eval()
net.reset()

test = "the cat"
print(f"\n输入: \"{test}\"", flush=True)
for ch in test:
    if ch in CHAR_TO_IDX:
        x = embedder(torch.tensor([CHAR_TO_IDX[ch]], device=DEVICE)).unsqueeze(0)
        result = net(x)
        logits = decoder(result['output'])
        pred = IDX_TO_CHAR[logits.argmax(dim=-1).item()]
        print(f"  '{ch}' → 预测 '{pred}'", flush=True)
