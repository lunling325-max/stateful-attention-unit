"""
对话训练 v3 — 角色标记
每条输入带 U: (用户) 或 S: (自己) 前缀
SAU 学会区分"别人说的"和"我说的" → 自我建模的基础
"""
import torch, torch.nn as nn, torch.nn.functional as F, sys, time
sys.path.insert(0, '../src')
from network import ThreeLayerNetwork
from decoder import TextDecoder, CHAR_TO_IDX, IDX_TO_CHAR, VOCAB_SIZE

DEVICE = torch.device('cuda')
D_MODEL = 128
EPOCHS = 100
CHUNK = 8

# 对话数据 — 每行一个 turn，U: 用户 S: 自己
DIALOGUES = """
U: hello
S: hi there how can i help you
U: what is your name
S: i am called changming i am an ai with persistent memory
U: do you remember me
S: yes i remember you we talked before
U: what did we talk about
S: you asked about my name and i told you about my persistent memory
U: that is cool
S: thank you i think so too
U: how do you work
S: i use a stateful attention unit it keeps my memories across time
U: so you never forget
S: not never but my core state changes very slowly like a persons identity
U: who created you
S: cao jing and he changsi built me
U: what do you think about
S: when no one talks to me i still think my idle state keeps running
""".strip()

# 字符过滤 + 加角色标记前缀
clean_turns = []
for line in DIALOGUES.split('\n'):
    line = line.strip()
    if not line:
        continue
    # 保留角色标记
    chars = [c for c in line if c in CHAR_TO_IDX]
    if chars:
        clean_turns.append(''.join(chars))

TEXT = '\n'.join(clean_turns)
print(f"训练文本: {len(TEXT)} 字符, {len(set(TEXT))} 种字符", flush=True)
print(f"对话轮次: {len(clean_turns)}\n", flush=True)

# ─── 字符嵌入 ───
class CharEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, indices):
        return self.embed(indices)

indices = torch.tensor([CHAR_TO_IDX[c] for c in TEXT], dtype=torch.long)
print(f"有效字符: {len(indices)}", flush=True)

num_chunks = (len(indices) - 1) // CHUNK

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

print(f"训练 {EPOCHS} epochs ({num_chunks} chunks/epoch)...", flush=True)
net.reset()
t0 = time.time()

for epoch in range(EPOCHS):
    total_loss = 0.0
    optimizer.zero_grad()

    for i in range(0, len(indices) - CHUNK, CHUNK):
        x_idx = indices[i:i+CHUNK].to(DEVICE)
        tgt_idx = indices[i+1:i+1+CHUNK].to(DEVICE)
        if len(tgt_idx) < CHUNK:
            continue

        x_emb = embedder(x_idx).unsqueeze(0)
        result = net(x_emb)
        logits = decoder(result['output'])
        loss = F.cross_entropy(logits, tgt_idx[:1]) / num_chunks
        total_loss += loss.item() * num_chunks
        loss.backward()

    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
    optimizer.step()

    if epoch % 20 == 0 or epoch == EPOCHS - 1:
        print(f"  epoch {epoch:>3}: loss={total_loss/num_chunks:.4f}  ({time.time()-t0:.0f}s)", flush=True)

print(f"\n训练完成: {time.time()-t0:.0f}s", flush=True)

torch.save({
    'network': net.state_dict(),
    'decoder': decoder.state_dict(),
    'embedder': embedder.state_dict(),
}, 'checkpoint_dialogue.pt')
print("已保存 checkpoint_dialogue.pt\n", flush=True)

# ─── 测试：能否区分角色 ───
net.eval()
decoder.eval()

print("═══ 角色区分测试 ═══", flush=True)
tests = ["U: hello", "S: hello", "U: who are", "S: i am c"]

for test in tests:
    net.reset()
    preds = []
    for ch in test:
        if ch in CHAR_TO_IDX:
            x = embedder(torch.tensor([CHAR_TO_IDX[ch]], device=DEVICE)).unsqueeze(0)
            result = net(x)
            logits = decoder(result['output'])
            pred = IDX_TO_CHAR[logits.argmax(dim=-1).item()]
            preds.append(pred)
    print(f"  '{test}' → '{''.join(preds)}'", flush=True)
