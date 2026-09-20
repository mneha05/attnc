#!/usr/bin/env python3
"""Reproducible GPU benchmark matrix for attnc, SDPA, FlexAttention, and adapters."""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
from pathlib import Path

from attnc import attention, compile


VARIANTS = {
    "causal": {"causal": True},
    "window": {"causal": True, "window": 4096},
    "gqa": {"causal": True, "gqa": True},
    "softcap": {"causal": True, "softcap": 50.0},
    "alibi": {"causal": True, "alibi": True},
    "combined": {"causal": True, "window": 4096, "gqa": True, "softcap": 50.0, "alibi": True},
}


def timed_cuda(torch, fn, warmup=5, iterations=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record(); fn(); end.record(); end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def make_spec(name, d, dtype, hq, hkv):
    opts = VARIANTS[name]
    spec = attention(head_dim=d, dtype=dtype)
    if opts.get("causal"): spec = spec.causal()
    if opts.get("window"): spec = spec.sliding_window(opts["window"])
    if opts.get("gqa"): spec = spec.gqa(hkv)
    if opts.get("softcap"): spec = spec.softcap(opts["softcap"])
    if opts.get("alibi"): spec = spec.alibi([2.0 ** (-8.0 * (i + 1) / hq) for i in range(hq)])
    return spec


def sdpa_callable(torch, q, k, v, name):
    opts = VARIANTS[name]
    if opts.get("window") or opts.get("softcap") or opts.get("alibi"):
        return None
    if q.shape[1] != k.shape[1]:
        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    nq, nk = q.shape[-2], k.shape[-2]
    qi = torch.arange(nq, device=q.device)[:, None] + max(nk - nq, 0)
    ki = torch.arange(nk, device=q.device)[None, :]
    mask = qi >= ki
    return lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def flex_callable(torch, q, k, v, name):
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    except (ImportError, AttributeError):
        return None
    opts = VARIANTS[name]
    nq, nk = q.shape[-2], k.shape[-2]
    offset = max(nk - nq, 0)
    slopes = torch.tensor([2.0 ** (-8.0 * (i + 1) / q.shape[1]) for i in range(q.shape[1])], device=q.device)

    def score_mod(score, b, h, qi, ki):
        qa = qi + offset
        if opts.get("alibi"): score = score - slopes[h] * (qa - ki)
        if opts.get("softcap"):
            cap = opts["softcap"]
            score = cap * torch.tanh(score / cap)
        return score

    def mask_mod(b, h, qi, ki):
        qa = qi + offset
        allowed = qa >= ki if opts.get("causal") else torch.ones_like(qa, dtype=torch.bool)
        if opts.get("window"): allowed = allowed & ((qa - ki) < opts["window"])
        return allowed

    block_mask = create_block_mask(mask_mod, q.shape[0], q.shape[1], nq, nk, device=q.device)
    compiled = torch.compile(flex_attention)
    return lambda: compiled(q, k, v, score_mod=score_mod, block_mask=block_mask, enable_gqa=bool(opts.get("gqa")))


def load_adapter(path):
    if not path: return None
    module, func = path.split(":", 1)
    return getattr(importlib.import_module(module), func)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="sm_80")
    p.add_argument("--device", default="cuda")
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--peak-gbps", type=float, help="Board peak bandwidth; enables percent-of-peak reporting")
    p.add_argument("--handwritten", help="Optional module:function adapter")
    p.add_argument("--output", default="benchmark-results.json")
    args = p.parse_args()
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Install the torch and CUDA extras before benchmarking") from exc
    if not torch.cuda.is_available(): raise SystemExit("CUDA device unavailable")

    shapes = [(1, 1024), (1, 8192), (1, 32768), (2048, 2048), (8192, 8192)]
    if args.quick: shapes = [(1, 1024), (256, 256)]
    variants = list(VARIANTS) if not args.quick else ["causal", "combined"]
    handwritten = load_adapter(args.handwritten)
    rows = []
    for nq, nk in shapes:
        for name in variants:
            opts = VARIANTS[name]
            hkv = args.kv_heads if opts.get("gqa") else args.heads
            q = torch.randn(1, args.heads, nq, args.head_dim, device=args.device, dtype=torch.float16)
            k = torch.randn(1, hkv, nk, args.head_dim, device=args.device, dtype=torch.float16)
            v = torch.randn_like(k)
            logical_bytes = (q.numel() + k.numel() + v.numel() + q.numel()) * q.element_size()

            runners = {}
            kernel = compile(make_spec(name, args.head_dim, "fp16", args.heads, hkv), arch=args.arch, backend="cuda")
            runners["attnc"] = lambda kernel=kernel: kernel(q, k, v)
            sdpa = sdpa_callable(torch, q, k, v, name)
            if sdpa: runners["pytorch_sdpa"] = sdpa
            flex = flex_callable(torch, q, k, v, name)
            if flex: runners["flex_attention"] = flex
            if handwritten:
                runners["handwritten"] = lambda: handwritten(q, k, v, variant=name)

            for implementation, fn in runners.items():
                try:
                    ms = timed_cuda(torch, fn, warmup=2 if args.quick else 5, iterations=5 if args.quick else 20)
                    gbps = logical_bytes / (ms / 1e3) / 1e9
                    rows.append({
                        "implementation": implementation, "variant": name, "q_len": nq, "kv_len": nk,
                        "head_dim": args.head_dim, "milliseconds": ms, "effective_gbps": gbps,
                        "effective_percent_peak": None if not args.peak_gbps else 100 * gbps / args.peak_gbps,
                        "status": "ok", "arch": args.arch,
                    })
                except Exception as exc:
                    rows.append({"implementation": implementation, "variant": name, "q_len": nq, "kv_len": nk,
                                 "head_dim": args.head_dim, "status": "error", "error": repr(exc), "arch": args.arch})
            del q, k, v
    payload = {
        "schema": 1,
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name()},
        "note": "Effective bandwidth uses logical tensor bytes, not DRAM counters.",
        "results": rows,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
