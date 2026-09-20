"""Proxy-object symbolic tracer for score_mod and mask_mod functions."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Callable, Mapping, Sequence

from .ir import AttentionIR, BATCH, Expr, HEAD, KV_IDX, Layout, Q_IDX, SCORE, as_expr, call, table_lookup
from .passes import infer_mask_bounds


@dataclass(frozen=True)
class SymbolicTable:
    name: str
    values: tuple[float, ...]

    def __getitem__(self, index: Any) -> Expr:
        return table_lookup(self.name, as_expr(index))


def table(values: Sequence[float], name: str = "table") -> SymbolicTable:
    return SymbolicTable(name, tuple(float(v) for v in values))


def tanh(value: Any) -> Expr:
    return call("tanh", value)


def where(predicate: Any, when_true: Any, when_false: Any) -> Expr:
    return call("select", predicate, when_true, when_false)


def trace(
    *,
    score_mod: Callable[..., Any] | None,
    mask_mod: Callable[..., Any] | None,
    head_dim: int,
    dtype: str = "fp16",
    kv_heads: int | None = None,
    tables: Mapping[str, SymbolicTable | Sequence[float]] | None = None,
) -> AttentionIR:
    """Execute tiny user functions once on proxy expressions to record their ops."""
    traced_tables: dict[str, tuple[float, ...]] = {}
    for name, value in (tables or {}).items():
        symbolic = value if isinstance(value, SymbolicTable) else table(value, name)
        traced_tables[symbolic.name] = symbolic.values

    # Captured SymbolicTable objects make the natural `slopes[h]` spelling work.
    for fn in (score_mod, mask_mod):
        if fn is None:
            continue
        closure = inspect.getclosurevars(fn)
        for value in (*closure.nonlocals.values(), *closure.globals.values()):
            if isinstance(value, SymbolicTable):
                traced_tables[value.name] = value.values

    score = SCORE if score_mod is None else as_expr(score_mod(SCORE, BATCH, HEAD, Q_IDX, KV_IDX))
    mask = Expr.const(True) if mask_mod is None else as_expr(mask_mod(BATCH, HEAD, Q_IDX, KV_IDX))
    bounds = infer_mask_bounds(mask)
    features = tuple(name for name, present in (
        ("custom_mask", mask_mod is not None),
        ("custom_score", score_mod is not None),
        ("gqa", kv_heads is not None),
    ) if present)
    return AttentionIR(
        layout=Layout(head_dim=head_dim, dtype=dtype, kv_heads=kv_heads),
        mask=mask,
        score=score,
        tables=traced_tables,
        mask_bounds=bounds,
        features=features,
    )
