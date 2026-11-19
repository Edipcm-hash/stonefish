#!/usr/bin/env python3
"""Cross-framework parity: does StonefishAI compute the same loss as PyTorch?

This is the check that matters for the "from scratch" claim. Anyone can write a
fast loop; the question is whether the numbers are right. So we take a trained
SFEN001 checkpoint, load it into an exact PyTorch implementation of the same
architecture, and evaluate the *same* batches that `stonefish_ai en-eval` uses.
If the two independent implementations agree, the engine is correct.

The batching is replicated exactly from src/en_train.rs::eval:
    batch = 8, seq = cfg.max_seq_len
    for b in 0..n_batches: for k in 0..8: start = (b*8 + k) * seq
    x = tokens[start .. start+seq], y = tokens[start+1 .. start+seq+1]

Usage:
    python tools/torch_parity.py --state stonefish_en.final.state \
        --valid data/en/tiny.valid.bin --batches 8

    # then compare with:
    stonefish_ai en-eval data/en/tiny stonefish_en.final.state 8
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch_bench import GPT  # noqa: E402  (same architecture as the Rust side)


def load_state(path: str):
    """Parse SFEN001 into (cfg, ordered list of (name, array))."""
    with open(path, "rb") as fh:
        blob = fh.read()

    off = 0

    def take(n):
        nonlocal off
        out = blob[off : off + n]
        off += n
        if len(out) != n:
            raise ValueError("truncated state file")
        return out

    magic = take(8)
    if magic != b"SFEN001\0":
        raise ValueError(f"wrong magic {magic!r} - not an SFEN001 weight file")
    (version,) = struct.unpack("<I", take(4))
    if version != 1:
        raise ValueError(f"unsupported version {version}")

    vocab, ctx, d, layers, heads = struct.unpack("<5I", take(20))
    (adam_t,) = struct.unpack("<Q", take(8))
    (n_tensors,) = struct.unpack("<I", take(4))

    cfg = dict(vocab=vocab, ctx=ctx, d=d, layers=layers, heads=heads, adam_t=adam_t)

    # The order comes from GpuGpt::params_mut():
    #   tok_emb, then per layer (norm1_g, wq, wk, wv, wo, norm2_g,
    #   ff_gate, ff_up, ff_down), then lnf_g.
    names = ["tok_emb"]
    for _ in range(layers):
        names += ["norm1_g", "wq", "wk", "wv", "wo", "norm2_g", "ff_gate", "ff_up", "ff_down"]
    names.append("lnf_g")
    if n_tensors != len(names):
        raise ValueError(f"tensor count {n_tensors} != expected {len(names)}")

    tensors = []
    for name in names:
        rows, cols = struct.unpack("<2I", take(8))
        data = np.frombuffer(take(rows * cols * 4), dtype="<f4").reshape(rows, cols)
        tensors.append((name, np.array(data, dtype=np.float32)))

    return cfg, tensors


def build_and_load(cfg, tensors, device):
    model = GPT(cfg["vocab"], cfg["ctx"], cfg["d"], cfg["layers"], cfg["heads"], naive=True)
    model = model.to(device).float()

    with torch.no_grad():
        model.tok.weight.copy_(torch.from_numpy(tensors[0][1]))
        model.nf.g.copy_(torch.from_numpy(tensors[-1][1].reshape(-1)))

        idx = 1
        for li in range(cfg["layers"]):
            blk = model.blocks[li]
            g1 = torch.from_numpy(tensors[idx + 0][1].reshape(-1))
            wq = torch.from_numpy(tensors[idx + 1][1])
            wk = torch.from_numpy(tensors[idx + 2][1])
            wv = torch.from_numpy(tensors[idx + 3][1])
            wo = torch.from_numpy(tensors[idx + 4][1])
            g2 = torch.from_numpy(tensors[idx + 5][1].reshape(-1))
            gate = torch.from_numpy(tensors[idx + 6][1])
            up = torch.from_numpy(tensors[idx + 7][1])
            down = torch.from_numpy(tensors[idx + 8][1])
            idx += 9

            # The Rust side stores W as (in, out) and computes y = x @ W;
            # nn.Linear stores (out, in) and computes x @ W.T, so transpose.
            blk.n1.g.copy_(g1)
            blk.wq.weight.copy_(wq.t())
            blk.wk.weight.copy_(wk.t())
            blk.wv.weight.copy_(wv.t())
            blk.wo.weight.copy_(wo.t())
            blk.n2.g.copy_(g2)
            blk.ff.gate.weight.copy_(gate.t())
            blk.ff.up.weight.copy_(up.t())
            blk.ff.down.weight.copy_(down.t())

    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="stonefish_en.final.state")
    ap.add_argument("--valid", default="data/en/tiny.valid.bin")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="load and check the weights, do not run batches")
    ap.add_argument("--expected", type=float, default=None,
                    help="loss printed by en-eval, to compute the delta")
    a = ap.parse_args()

    cfg, tensors = load_state(a.state)
    n_params = sum(int(np.prod(t.shape)) for _, t in tensors)
    print(f"state  : d={cfg['d']} layers={cfg['layers']} heads={cfg['heads']} "
          f"ctx={cfg['ctx']} vocab={cfg['vocab']} step={cfg['adam_t']}")
    print(f"weights: {n_params:,} parameters in {len(tensors)} tensors")

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    model = build_and_load(cfg, tensors, dev).eval()
    n_model = sum(p.numel() for p in model.parameters())
    print(f"model  : {n_model:,} parameters on {dev}")
    if n_model != n_params:
        sys.exit(f"parameter count mismatch: state {n_params:,} vs model {n_model:,}")
    if a.dry_run:
        print("dry run: weights loaded and shapes matched")
        return

    tokens = np.fromfile(a.valid, dtype="<u2").astype(np.int64)
    seq, batch = cfg["ctx"], 8
    per_batch = batch * seq
    n_batches = min((len(tokens) - 1) // per_batch, a.batches)
    print(f"valid  : {len(tokens):,} tokens, using {n_batches} x {per_batch} = "
          f"{n_batches * per_batch:,}")

    lossf = torch.nn.CrossEntropyLoss(reduction="sum")
    total = 0.0
    with torch.no_grad():
        for b in range(n_batches):
            xs, ys = [], []
            for k in range(batch):
                start = (b * batch + k) * seq
                xs.append(tokens[start : start + seq])
                ys.append(tokens[start + 1 : start + seq + 1])
            x = torch.from_numpy(np.stack(xs)).to(dev)
            y = torch.from_numpy(np.stack(ys)).to(dev)
            logits = model(x)
            total += lossf(logits.reshape(-1, cfg["vocab"]), y.reshape(-1)).item()
            print(f"  batch {b + 1}/{n_batches}  running mean {total / (b + 1) / per_batch:.6f}")

    loss = total / (n_batches * per_batch)
    print(f"\nPyTorch loss : {loss:.6f}")
    print(f"PyTorch ppl  : {np.exp(loss):.4f}")
    if a.expected is not None:
        delta = abs(loss - a.expected)
        print(f"StonefishAI  : {a.expected:.6f}")
        print(f"delta        : {delta:.6f}")
        verdict = "MATCH" if delta < 2e-3 else "MISMATCH - investigate"
        print(f"verdict      : {verdict}")


if __name__ == "__main__":
    main()
