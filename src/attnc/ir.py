"""Small, serializable intermediate representation for attention variants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import hashlib
import json


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple["Expr", ...] = ()
    value: Any = None

    @staticmethod
    def const(value: Any) -> "Expr":
        return Expr("const", value=value)

    @staticmethod
    def var(name: str) -> "Expr":
        return Expr("var", value=name)

    def _bin(self, op: str, other: Any) -> "Expr":
        return Expr(op, (self, as_expr(other)))

    def __add__(self, other: Any) -> "Expr": return self._bin("add", other)
    def __radd__(self, other: Any) -> "Expr": return as_expr(other)._bin("add", self)
    def __sub__(self, other: Any) -> "Expr": return self._bin("sub", other)
    def __rsub__(self, other: Any) -> "Expr": return as_expr(other)._bin("sub", self)
    def __mul__(self, other: Any) -> "Expr": return self._bin("mul", other)
    def __rmul__(self, other: Any) -> "Expr": return as_expr(other)._bin("mul", self)
    def __truediv__(self, other: Any) -> "Expr": return self._bin("div", other)
    def __rtruediv__(self, other: Any) -> "Expr": return as_expr(other)._bin("div", self)
    def __neg__(self) -> "Expr": return Expr("neg", (self,))
    def __lt__(self, other: Any) -> "Expr": return self._bin("lt", other)
    def __le__(self, other: Any) -> "Expr": return self._bin("le", other)
    def __gt__(self, other: Any) -> "Expr": return self._bin("gt", other)
    def __ge__(self, other: Any) -> "Expr": return self._bin("ge", other)
    def __eq__(self, other: Any) -> "Expr":  # type: ignore[override]
        return self._bin("eq", other)
    def __and__(self, other: Any) -> "Expr": return self._bin("and", other)
    def __rand__(self, other: Any) -> "Expr": return as_expr(other)._bin("and", self)
    def __or__(self, other: Any) -> "Expr": return self._bin("or", other)
    def __ror__(self, other: Any) -> "Expr": return as_expr(other)._bin("or", self)
    def __invert__(self) -> "Expr": return Expr("not", (self,))

    def __bool__(self) -> bool:
        raise TypeError("symbolic expressions cannot be converted to bool; use &, |, and ~")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"op": self.op}
        if self.args:
            result["args"] = [arg.to_dict() for arg in self.args]
        if self.value is not None:
            if hasattr(self.value, "item"):
                result["value"] = self.value.item()
            else:
                result["value"] = self.value
        return result


def as_expr(value: Any) -> Expr:
    return value if isinstance(value, Expr) else Expr.const(value)


@dataclass(frozen=True)
class Layout:
    head_dim: int
    dtype: str = "fp16"
    kv_heads: int | None = None
    kv_layout: str = "contiguous"

    def __post_init__(self) -> None:
        if self.head_dim <= 0 or self.head_dim > 512:
            raise ValueError("head_dim must be in [1, 512]")
        if self.dtype not in {"fp16", "fp32"}:
            raise ValueError("dtype must be 'fp16' or 'fp32'")
        if self.kv_heads is not None and self.kv_heads <= 0:
            raise ValueError("kv_heads must be positive")
        if self.kv_layout not in {"contiguous", "paged"}:
            raise ValueError("kv_layout must be contiguous or paged")


@dataclass(frozen=True)
class MaskBounds:
    """Closed-form bounds for masks representable as a key interval per query."""

    causal: bool = False
    window: int | None = None


@dataclass(frozen=True)
class AttentionIR:
    layout: Layout
    mask: Expr = field(default_factory=lambda: Expr.const(True))
    score: Expr = field(default_factory=lambda: Expr.var("score"))
    tables: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    mask_bounds: MaskBounds | None = None
    features: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout": {
                "head_dim": self.layout.head_dim,
                "dtype": self.layout.dtype,
                "kv_heads": self.layout.kv_heads,
                "kv_layout": self.layout.kv_layout,
            },
            "mask": self.mask.to_dict(),
            "score": self.score.to_dict(),
            "tables": {k: list(v) for k, v in sorted(self.tables.items())},
            "mask_bounds": None if self.mask_bounds is None else {
                "causal": self.mask_bounds.causal,
                "window": self.mask_bounds.window,
            },
            "features": list(self.features),
        }

    def stable_hash(self, *, arch: str, options: Mapping[str, Any] | None = None) -> str:
        payload = {"ir": self.to_dict(), "arch": arch, "options": dict(options or {})}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:20]


SCORE = Expr.var("score")
BATCH = Expr.var("b")
HEAD = Expr.var("h")
Q_IDX = Expr.var("q_idx")
KV_IDX = Expr.var("kv_idx")


def table_lookup(name: str, index: Expr) -> Expr:
    return Expr("lookup", (index,), value=name)


def call(name: str, *args: Any) -> Expr:
    if name not in {"tanh", "exp", "abs", "minimum", "maximum", "select"}:
        raise ValueError(f"unsupported intrinsic: {name}")
    return Expr(name, tuple(as_expr(arg) for arg in args))

