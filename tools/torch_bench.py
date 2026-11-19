"""StonefishAI vs PyTorch - same architecture, same data, same GPU, same hardware.

Goal: MEASURE the difference between a "from-scratch Rust engine" and an
"industry-standard framework". To keep the comparison fair the PyTorch side
StonefishAI'nin v2 mimarisini birebir uygular:

  implements StonefishAI's v2 architecture exactly:
  RMSNorm -> SwiGLU (silu(xW1) * (xW3)) -> W2 -> residual
  final RMSNorm -> logits = x * tok_embT  (weight tying)

Two attention modes are reported:
  naive : scores are materialized (what StonefishAI does)
  sdpa  : torch SDPA (flash/memory-efficient kernels) - PyTorch at its best
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * inv * self.g


def rope(x, n_heads, d_head, base=10000.0):
    """x: (B, S, H, Dh) -> rotated."""
    S = x.size(1)
    half = d_head // 2
    j = torch.arange(half, device=x.device, dtype=torch.float32)
    freq = base ** (-2.0 * j / d_head)
    pos = torch.arange(S, device=x.device, dtype=torch.float32)
    ang = pos[:, None] * freq[None, :]  # (S, half)
    cos = ang.cos()[None, :, None, :]
    sin = ang.sin()[None, :, None, :]
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
    return out.flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, d, n_heads, hidden, naive):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.n2 = RMSNorm(d)
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.ff = SwiGLU(d, hidden)
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.naive = naive

    def forward(self, x):
        B, S, D = x.shape
        h = self.n1(x)
        q = self.wq(h).view(B, S, self.n_heads, self.d_head)
        k = self.wk(h).view(B, S, self.n_heads, self.d_head)
        v = self.wv(h).view(B, S, self.n_heads, self.d_head)
        q = rope(q, self.n_heads, self.d_head)
        k = rope(k, self.n_heads, self.d_head)
        if self.naive:
            # materialize the scores (the same work StonefishAI does)
            qq = q.transpose(1, 2)  # (B,H,S,Dh)
            kk = k.transpose(1, 2)
            vv = v.transpose(1, 2)
            scores = (qq @ kk.transpose(-1, -2)) / (self.d_head ** 0.5)
            mask = torch.triu(
                torch.full((S, S), float("-inf"), device=x.device), diagonal=1
            )
            att = torch.softmax(scores + mask, dim=-1)
            out = att @ vv
            out = out.transpose(1, 2).reshape(B, S, D)
        else:
            qq = q.transpose(1, 2)
            kk = k.transpose(1, 2)
            vv = v.transpose(1, 2)
            out = F.scaled_dot_product_attention(qq, kk, vv, is_causal=True)
            out = out.transpose(1, 2).reshape(B, S, D)
        x = x + self.wo(out)
        return x + self.ff(self.n2(x))


class GPT(nn.Module):
    def __init__(self, vocab, ctx, d, layers, heads, naive):
        super().__init__()
        # exactly the same rule as StonefishAI's swiglu_hidden():
        raw = -(-(8 * d) // 3)  # round up
        step = 256 if raw >= 1024 else (64 if raw >= 256 else 8)
        hidden = -(-raw // step) * step
        self.tok = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, heads, hidden, naive) for _ in range(layers)])
        self.nf = RMSNorm(d)

    def forward(self, ids):
        x = self.tok(ids)
        for b in self.blocks:
            x = b(x)
        return self.nf(x) @ self.tok.weight.t()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--accum", type=int, default=16)
    p.add_argument("--vocab", type=int, default=16384)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--attn", choices=["naive", "sdpa"], default="naive")
    a = p.parse_args()

    dev = "cuda"
    torch.manual_seed(1234)
    model = GPT(a.vocab, a.ctx, a.d, a.layers, a.heads, a.attn == "naive").to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-6, betas=(0.9, 0.999))
    lossf = nn.CrossEntropyLoss()

    tokens_per_step = a.batch * a.accum * a.ctx
    print(
        f"PyTorch {torch.__version__} | {torch.cuda.get_device_name(0)} | attn={a.attn}"
    )
    print(
        f"model: d={a.d} layers={a.layers} heads={a.heads} ctx={a.ctx} vocab={a.vocab}"
        f" -> {n_params/1e6:.1f}M parameters"
    )
    print(f"batch: {a.batch} x accum {a.accum} = {tokens_per_step} tokens/step")
    print(f"tf32: {torch.backends.cuda.matmul.allow_tf32}")

    x = torch.randint(0, a.vocab, (a.batch, a.ctx + 1), device=dev)

    # warm-up
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        for _m in range(a.accum):
            logits = model(x[:, :-1])
            loss = lossf(logits.reshape(-1, a.vocab), x[:, 1:].reshape(-1)) / a.accum
            loss.backward()
        opt.step()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(a.steps):
        opt.zero_grad(set_to_none=True)
        for _m in range(a.accum):
            logits = model(x[:, :-1])
            loss = lossf(logits.reshape(-1, a.vocab), x[:, 1:].reshape(-1)) / a.accum
            loss.backward()
        # StonefishAI's step() performs global L2 clipping (clamp 1.0).
        # For fairness we do the same work on the PyTorch side.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.cuda.synchronize()
        opt.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    tok_s = a.steps * tokens_per_step / dt
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nstep time: {dt/a.steps*1000:.0f} ms")
    print(f"speed      : {tok_s:,.0f} tokens/s")
    print(f"zirve VRAM : {peak:.1f} GB")
    print(f"son loss   : {loss.item()*a.accum:.4f}")


if __name__ == "__main__":
    main()
