"""NumPy reference interpreter for the attention IR."""

from __future__ import annotations

from typing import Any, Mapping
import math
import numpy as np

from .ir import AttentionIR, Expr


def evaluate(expr: Expr, env: Mapping[str, Any], tables: Mapping[str, tuple[float, ...]]) -> Any:
    if expr.op == "const": return expr.value
    if expr.op == "var": return env[expr.value]
    if expr.op == "lookup": return tables[expr.value][int(evaluate(expr.args[0], env, tables))]
    args = [evaluate(arg, env, tables) for arg in expr.args]
    binary = {
        "add": lambda: args[0] + args[1], "sub": lambda: args[0] - args[1],
        "mul": lambda: args[0] * args[1], "div": lambda: args[0] / args[1],
        "lt": lambda: args[0] < args[1], "le": lambda: args[0] <= args[1],
        "gt": lambda: args[0] > args[1], "ge": lambda: args[0] >= args[1],
        "eq": lambda: args[0] == args[1], "and": lambda: bool(args[0]) and bool(args[1]),
        "or": lambda: bool(args[0]) or bool(args[1]),
        "minimum": lambda: min(args), "maximum": lambda: max(args),
    }
    if expr.op in binary: return binary[expr.op]()
    if expr.op == "neg": return -args[0]
    if expr.op == "not": return not args[0]
    if expr.op == "tanh": return math.tanh(args[0])
    if expr.op == "exp": return math.exp(args[0])
    if expr.op == "abs": return abs(args[0])
    if expr.op == "select": return args[1] if args[0] else args[2]
    raise ValueError(f"unknown IR op: {expr.op}")


def _validate(q: np.ndarray, k: np.ndarray, v: np.ndarray, ir: AttentionIR) -> tuple[int, int, int, int, int]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [batch, heads, sequence, head_dim]")
    b, hq, nq, d = q.shape
    bk, hkv, nk, dk = k.shape
    if (bk, dk) != (b, d) or v.shape != (b, hkv, nk, d):
        raise ValueError("incompatible q/k/v shapes")
    if d != ir.layout.head_dim:
        raise ValueError(f"expected head_dim={ir.layout.head_dim}, got {d}")
    if hq % hkv:
        raise ValueError("query head count must be divisible by key/value head count")
    if ir.layout.kv_heads is not None and hkv != ir.layout.kv_heads:
        raise ValueError(f"expected {ir.layout.kv_heads} KV heads, got {hkv}")
    return b, hq, hkv, nq, nk


def run_reference(q: Any, k: Any, v: Any, ir: AttentionIR, *, scale: float | None = None, q_offset: int | None = None) -> np.ndarray:
    q = np.asarray(q)
    k = np.asarray(k)
    v = np.asarray(v)
    bsz, q_heads, kv_heads, q_len, kv_len = _validate(q, k, v, ir)
    scale = float(scale if scale is not None else 1.0 / math.sqrt(ir.layout.head_dim))
    q_offset = max(kv_len - q_len, 0) if q_offset is None else int(q_offset)
    out = np.zeros(q.shape, dtype=np.float32)

    for b in range(bsz):
        for h in range(q_heads):
            kh = h * kv_heads // q_heads
            raw = np.asarray(q[b, h], dtype=np.float32) @ np.asarray(k[b, kh], dtype=np.float32).T
            raw *= scale
            scores = np.full((q_len, kv_len), -np.inf, dtype=np.float32)
            for qi in range(q_len):
                q_abs = qi + q_offset
                for ki in range(kv_len):
                    env = {"score": float(raw[qi, ki]), "b": b, "h": h, "q_idx": q_abs, "kv_idx": ki}
                    if evaluate(ir.mask, env, ir.tables):
                        scores[qi, ki] = evaluate(ir.score, env, ir.tables)
            for qi in range(q_len):
                finite = np.isfinite(scores[qi])
                if not finite.any():
                    continue
                row = scores[qi, finite]
                weights = np.exp(row - row.max())
                weights /= weights.sum()
                out[b, h, qi] = weights @ np.asarray(v[b, kh, finite], dtype=np.float32)
    return out
