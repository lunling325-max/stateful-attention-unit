"""
单预测神经元 v3 — 预测下一步，而非当前步
S_t = decay*S_{t-1} + P*x_t     (带历史)
x̂_{t+1} = W @ S_t              (预测)
error = x_{t+1} - x̂_{t+1}       (下一时间步才比较)
delta rule online
"""
import torch, torch.nn as nn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class PredictorNeuron(nn.Module):
    def __init__(self, d_in=8, d_st=4, lr=0.01, decay=0.9, target_norm=2.0, momentum=0.9):
        super().__init__()
        self.d_in = d_in; self.d_st = d_st; self.lr = lr
        self.decay = decay; self.target_norm = target_norm; self.momentum = momentum

        self.W = nn.Parameter(torch.randn(d_st, d_in) * 0.01)
        self.P = nn.Parameter(torch.randn(d_in, d_st) * 0.01)

        self.register_buffer('S', torch.zeros(d_st))
        self.register_buffer('prediction', torch.zeros(d_in))
        self.register_buffer('prev_x', None)
        self.register_buffer('W_vel', torch.zeros(d_st, d_in))
        self.register_buffer('P_vel', torch.zeros(d_in, d_st))

    def forward(self, x):
        # ── 1. 预测误差（来自上一步的预测） ──
        if self.prev_x is None:
            error = torch.zeros(self.d_in, device=x.device)
            pe = 0.0
        else:
            error = x - self.prediction  # 实际 vs 上一步预测
            pe = error.norm().item()

        # ── 2. 更新状态: S = decay*S_prev + P*current_x ──
        Px = (x.unsqueeze(0) @ self.P).squeeze(0)
        self.S = self.decay * self.S + Px
        # 归一化（防爆炸）
        sn = self.S.norm()
        if sn > 0:
            self.S = self.S * (self.target_norm / (sn + 1e-8))

        # ── 3. delta rule + momentum 更新 W ──
        if self.prev_x is not None:
            S_old = self.S_prev if hasattr(self, 'S_prev') else self.S
            gW = torch.outer(S_old, error)
            self.W_vel = self.momentum * self.W_vel + gW
            with torch.no_grad():
                self.W.data += self.lr * self.W_vel

        # ── 4. delta rule + momentum 更新 P ──
        if self.prev_x is not None:
            gP = torch.outer(self.prev_x, self.W @ error)
            self.P_vel = self.momentum * self.P_vel + gP
            with torch.no_grad():
                self.P.data += self.lr * 0.1 * self.P_vel

        # ── 5. 生成下一步的预测 ──
        self.prediction = (self.S.unsqueeze(0) @ self.W).squeeze(0)
        self.prev_x = x
        self.S_prev = self.S.clone()

        return pe

    def reset(self):
        self.S.zero_(); self.prediction.zero_(); self.prev_x = None
        self.W_vel.zero_(); self.P_vel.zero_()
        torch.nn.init.normal_(self.W, 0, 0.01)
        torch.nn.init.normal_(self.P, 0, 0.01)


def run(d_in, d_st, n=5000):
    torch.manual_seed(42)
    t = torch.arange(n, dtype=torch.float32)
    sig = torch.zeros(n, d_in)
    for i in range(d_in):
        sig[:,i] = torch.sin(t*0.05*(1+i*0.3)+i)*0.7
    sig += torch.randn(n, d_in)*0.05

    m = PredictorNeuron(d_in, d_st, lr=0.005).to(DEVICE)
    errs = []
    for s in range(n):
        e = m(sig[s].to(DEVICE))
        if e > 0:
            errs.append(e)

    if len(errs) < 100:
        print(f"  d_in={d_in}, d_st={d_st}: 无足够误差数据")
        return

    early = sum(errs[50:len(errs)//10])/(len(errs)//10-50)
    late = sum(errs[-len(errs)//10:])/(len(errs)//10)
    r = late/early if early>0 else 1
    label = '✅' if r<0.5 else ('↓' if r<0.9 else '→')
    print(f"  d_in={d_in}, d_st={d_st}  |  {early:.4f} → {late:.4f}  ratio={r:.3f}  {label}")
    return errs


if __name__ == '__main__':
    print("单预测神经元 v3 — 预测下一步\n")
    run(8, 4, 5000)
    run(16, 8, 5000)
