"""Command-line inspection and compilation utilities."""

from __future__ import annotations

import argparse
import json

from .frontend import attention
from .runtime import compile_attention


def _spec(args: argparse.Namespace):
    spec = attention(head_dim=args.head_dim, dtype=args.dtype)
    if args.causal: spec = spec.causal()
    if args.window: spec = spec.sliding_window(args.window)
    if args.kv_heads: spec = spec.gqa(args.kv_heads)
    if args.softcap: spec = spec.softcap(args.softcap)
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(prog="attnc", description="Compile and inspect fused attention variants")
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--arch", default="sm_80")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--window", type=int)
    parser.add_argument("--kv-heads", type=int)
    parser.add_argument("--softcap", type=float)
    parser.add_argument("--dump-cuda")
    parser.add_argument("--dump-ptx")
    parser.add_argument("--dump-ir", action="store_true")
    args = parser.parse_args()
    kernel = compile_attention(_spec(args), arch=args.arch, backend="numpy")
    if args.dump_ir: print(json.dumps(kernel.ir.to_dict(), indent=2, sort_keys=True))
    if args.dump_cuda: kernel.dump_cuda(args.dump_cuda)
    if args.dump_ptx: kernel.dump_ptx(args.dump_ptx)
    if not (args.dump_ir or args.dump_cuda or args.dump_ptx): print(kernel.source)


if __name__ == "__main__":
    main()

