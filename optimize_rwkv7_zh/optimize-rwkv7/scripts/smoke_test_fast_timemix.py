#!/usr/bin/env python
"""Smoke test for a project that exposes lit_gpt.rwkv7.RWKV7TimeMix."""

from __future__ import annotations

import argparse
import json
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--num-tested-layers", type=int, default=2)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the fused RWKV-7 kernels")

    from lit_gpt.rwkv7 import RWKV7TimeMix

    torch.manual_seed(1234)
    device = torch.device(args.device)
    dtype = torch.bfloat16

    layers = torch.nn.ModuleList(
        [
            RWKV7TimeMix(
                n_embd=args.n_embd,
                n_layer=args.n_layer,
                layer_idx=i,
                head_size=args.head_size,
            )
            for i in range(args.num_tested_layers)
        ]
    ).to(device=device, dtype=dtype)
    layers.train()

    x = torch.randn(args.batch_size, args.seq_len, args.n_embd, device=device, dtype=dtype, requires_grad=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    v_first = None
    y = x
    for layer in layers:
        y, v_first = layer(y, v_first)
    loss = y.float().square().mean()
    loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    finite = bool(torch.isfinite(loss.detach()).item())
    finite = finite and all(
        p.grad is None or bool(torch.isfinite(p.grad.detach()).all().item())
        for p in layers.parameters()
    )
    result = {
        "ok": finite,
        "loss": float(loss.detach().cpu()),
        "elapsed_s": elapsed,
        "device": str(device),
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "n_embd": args.n_embd,
        "head_size": args.head_size,
        "tested_layers": args.num_tested_layers,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not finite:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
