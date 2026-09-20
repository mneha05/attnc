"""Middle-end simplification and mask sparsity analysis."""

from __future__ import annotations

from dataclasses import replace
from enum import IntEnum
from typing import Any
import math

from .ir import AttentionIR, Expr, MaskBounds


class TileState(IntEnum):
    FULLY_MASKED = 0
    PARTIAL = 1
    FULLY_UNMASKED = 2


_BINARY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "eq": lambda a, b: a == b,
    "and": lambda a, b: bool(a) and bool(b),
    "or": lambda a, b: bool(a) or bool(b),
    "minimum": min,
    "maximum": max,
}


def simplify(expr: Expr) -> Expr:
    args = tuple(simplify(arg) for arg in expr.args)
    if expr.op in _BINARY and all(arg.op == "const" for arg in args):
        return Expr.const(_BINARY[expr.op](args[0].value, args[1].value))
    if expr.op == "neg" and args[0].op == "const":
        return Expr.const(-args[0].value)
    if expr.op == "not" and args[0].op == "const":
        return Expr.const(not args[0].value)
    if expr.op in {"tanh", "exp", "abs"} and args[0].op == "const":
        fn = {"tanh": math.tanh, "exp": math.exp, "abs": abs}[expr.op]
        return Expr.const(fn(args[0].value))
    if expr.op == "select" and args[0].op == "const":
        return args[1] if args[0].value else args[2]
    if expr.op == "and":
        if args[0].op == "const": return args[1] if args[0].value else args[0]
        if args[1].op == "const": return args[0] if args[1].value else args[1]
    if expr.op == "or":
        if args[0].op == "const": return args[0] if args[0].value else args[1]
        if args[1].op == "const": return args[1] if args[1].value else args[0]
    if expr.op in {"add", "sub"} and args[1].op == "const" and args[1].value == 0:
        return args[0]
    if expr.op == "mul" and args[1].op == "const":
        if args[1].value == 1: return args[0]
        if args[1].value == 0: return args[1]
    return Expr(expr.op, args, expr.value)


def optimize(ir: AttentionIR) -> AttentionIR:
    mask = simplify(ir.mask)
    score = simplify(ir.score)
    bounds = ir.mask_bounds or infer_mask_bounds(mask)
    return replace(ir, mask=mask, score=score, mask_bounds=bounds)


def _is_var(expr: Expr, name: str) -> bool:
    return expr.op == "var" and expr.value == name


def _match_sub(expr: Expr, left: str, right: str) -> bool:
    return expr.op == "sub" and _is_var(expr.args[0], left) and _is_var(expr.args[1], right)


def infer_mask_bounds(expr: Expr) -> MaskBounds | None:
    """Recognize causal and causal/sliding-window predicates safely."""
    causal = False
    window: int | None = None

    def visit(node: Expr) -> bool:
        nonlocal causal, window
        if node.op == "const" and bool(node.value):
            return True
        if node.op == "and":
            return visit(node.args[0]) and visit(node.args[1])
        if node.op in {"ge", "le"}:
            a, b = node.args
            if (node.op == "ge" and _is_var(a, "q_idx") and _is_var(b, "kv_idx")) or (
                node.op == "le" and _is_var(a, "kv_idx") and _is_var(b, "q_idx")
            ):
                causal = True
                return True
        if node.op in {"lt", "le"} and _match_sub(node.args[0], "q_idx", "kv_idx"):
            rhs = node.args[1]
            if rhs.op == "const" and isinstance(rhs.value, (int, float)):
                raw = int(rhs.value)
                window = raw if node.op == "lt" else raw + 1
                return window > 0
        return False

    return MaskBounds(causal, window) if visit(expr) else None


def classify_tile(bounds: MaskBounds | None, q_start: int, q_end: int, kv_start: int, kv_end: int) -> TileState:
    """Classify half-open [q_start,q_end) x [kv_start,kv_end) tiles."""
    if q_end <= q_start or kv_end <= kv_start:
        raise ValueError("tile ranges must be non-empty")
    if bounds is None:
        return TileState.PARTIAL

    # Allowed key interval for each q: [q-window+1, q] with absent bounds open.
    if bounds.causal and kv_start > q_end - 1:
        return TileState.FULLY_MASKED
    if bounds.window is not None and kv_end - 1 < q_start - bounds.window + 1:
        return TileState.FULLY_MASKED

    upper_ok = not bounds.causal or kv_end - 1 <= q_start
    lower_ok = bounds.window is None or kv_start >= (q_end - 1) - bounds.window + 1
    if upper_ok and lower_ok:
        return TileState.FULLY_UNMASKED
    return TileState.PARTIAL

