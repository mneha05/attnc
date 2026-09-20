# attnc

**A Python-embedded DSL and JIT compiler for fused attention variants.**

`attnc` lowers causal masks, sliding windows, grouped-query attention, logit
softcaps, ALiBi, and traced user functions into a small IR, optimizes the mask,
and emits one fused CUDA kernel. A NumPy interpreter is the correctness oracle;
NVRTC/CuPy provides JIT execution and PyTorch interop through DLPack.

> Status: v0.1 is an end-to-end, correctness-first implementation. The CPU
> compiler/oracle suite passes 160 differential cases. CUDA execution and
> performance have not been measured in this development environment, so this
> repository intentionally makes no speedup or bandwidth claim yet.

## Quick start

```bash
pip install -e .                    # oracle/compiler path
pip install -e '.[cuda,torch]'      # NVIDIA execution + Torch interop
```

```python
from attnc import attention, compile

spec = (attention(head_dim=128, dtype="fp16")
        .causal()
        .sliding_window(4096)
        .gqa(kv_heads=8)
        .softcap(50.0))

kernel = compile(spec, arch="sm_121")
out = kernel(q, k, v)  # NumPy, CuPy, or CUDA Torch tensors
```

`backend="auto"` selects CUDA when CuPy and a device are available and the
NumPy oracle otherwise. CUDA tensors must be contiguous and shaped
`[batch, heads, sequence, head_dim]`.

## Traced frontend

The tracer executes tiny Python functions once on proxy expressions. Captured
tables use `attnc.table`, allowing natural symbolic indexing.

```python
from attnc import compile, table

slopes = table([0.25, 0.125, 0.0625, 0.03125], name="slopes")

def alibi(score, b, h, q_idx, kv_idx):
    return score - slopes[h] * (q_idx - kv_idx)

def window_mask(b, h, q_idx, kv_idx):
    return (q_idx >= kv_idx) & ((q_idx - kv_idx) < 4096)

kernel = compile(score_mod=alibi, mask_mod=window_mask,
                 head_dim=128, dtype="fp16", arch="sm_121")
```

Supported traced operations are `+ - * /`, comparisons, `& | ~`, table
lookups, `attnc.tanh`, and `attnc.where`. Python `and`/`or` cannot be overloaded;
use `&`/`|` with parenthesized comparisons.

## What is implemented

- Immutable combinator API and proxy-object tracer lowering to one serializable IR.
- Constant folding, boolean simplification, and conservative dead-branch removal.
- Closed-form causal/window analysis and masked/unmasked/partial KV-tile paths.
- Fused warp kernel with scaled QK, score mods, online softmax, and PV.
- GQA head mapping and correct right-aligned positions for decode queries.
- First-call CUDA autotuning over architecture-aware KV tile candidates.
- Content-addressed IR/CUDA/tuning cache.
- NumPy oracle, 160 randomized differential cases, and algebraic properties.
- Optional CuPy JIT, zero-copy Torch DLPack dispatch, CUDA/IR/PTX dumping.
- Reproducible T4/GB10 benchmark matrix plus hand-written adapter hook.

See [the design note](docs/DESIGN.md) for compiler and numerical details.

## Inspect generated code

```bash
attnc --head-dim 128 --causal --window 4096 --softcap 50 \
  --arch sm_121 --dump-ir --dump-cuda kernel.cu

# Requires an installed NVRTC library that supports the requested architecture.
attnc --head-dim 128 --causal --arch sm_121 --dump-ptx kernel.ptx
```

## Correctness

```bash
python -m unittest discover -s tests -v
```

The differential suite uses an independently written dense attention function
and covers 32 seeds across five compositions (160 cases), including GQA and
non-square decode shapes. Mask decisions are exact; values use fp32 tolerance
in CPU tests. GPU fp16 tolerance should be recorded with GPU results.

## Benchmarking

```bash
python benchmarks/bench.py --arch sm_75 --peak-gbps 320 --output results-t4.json
python benchmarks/bench.py --arch sm_121 --peak-gbps YOUR_BOARD_VALUE --output results-gb10.json
```

The harness covers decode 1K/8K/32K and prefill 2K/8K across six variants. It
reports medians, effective bandwidth, optional effective percent of a supplied
board peak, and errors rather than silently dropping failed cells. A literal
DRAM percent-of-peak claim still requires Nsight Compute counters. See
[the benchmark protocol](benchmarks/README.md).

## Honest next steps

The current backend proves the compiler stack and forward semantics. The next
performance milestone is replacing the warp-per-row skeleton with tensor-core
prefill and context-split decode skeletons from `gb10-attn`, then measuring
compiler-generated bodies against those same hand-written ceilings. Paged KV
and a hetero-serve dispatch adapter follow after that.

## Resume bullets—only after measuring

- Built `attnc`, a Python-embedded DSL and JIT compiler generating fused CUDA
  attention kernels for compositions of causal, sliding-window, GQA, softcap,
  and ALiBi variants, with proxy tracing, IR passes, NVRTC codegen, and caching.
- Implemented block-level mask analysis that skips fully masked key tiles and
  removes predicates from fully unmasked tiles; reached **X%** of the
  hand-written baseline and **Y%** effective peak bandwidth on NVIDIA GB10.
- Verified generated semantics with a NumPy oracle across 160 randomized
  variant/shape cases plus mask, MHA/GQA, and decode-offset property tests.

Do not replace `X` or `Y` until the committed benchmark JSON supports them.

## Non-goals

No backward pass, training, general tensor compiler, multi-GPU runtime, or
Windows support. v0.1 supports contiguous KV; paged KV is explicitly future
work.

## License

MIT
