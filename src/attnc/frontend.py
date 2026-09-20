"""Fluent combinator frontend."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from .ir import AttentionIR, Expr, HEAD, KV_IDX, Layout, MaskBounds, Q_IDX, SCORE, call, table_lookup


@dataclass(frozen=True)
class AttentionSpec:
    layout: Layout
    use_causal: bool = False
    window_size: int | None = None
    softcap_value: float | None = None
    alibi_values: tuple[float, ...] | None = None

    def causal(self) -> "AttentionSpec":
        return replace(self, use_causal=True)

    def sliding_window(self, size: int) -> "AttentionSpec":
        if size <= 0:
            raise ValueError("sliding window size must be positive")
        return replace(self, window_size=int(size))

    def gqa(self, kv_heads: int) -> "AttentionSpec":
        return replace(self, layout=replace(self.layout, kv_heads=int(kv_heads)))

    def softcap(self, value: float) -> "AttentionSpec":
        if value <= 0:
            raise ValueError("softcap must be positive")
        return replace(self, softcap_value=float(value))

    def alibi(self, slopes: Sequence[float]) -> "AttentionSpec":
        values = tuple(float(x) for x in slopes)
        if not values:
            raise ValueError("ALiBi slopes cannot be empty")
        return replace(self, alibi_values=values)

    def lower(self) -> AttentionIR:
        mask = Expr.const(True)
        features: list[str] = []
        if self.use_causal:
            mask = mask & (Q_IDX >= KV_IDX)
            features.append("causal")
        if self.window_size is not None:
            mask = mask & ((Q_IDX - KV_IDX) < self.window_size)
            features.append("sliding_window")

        score = SCORE
        tables: dict[str, tuple[float, ...]] = {}
        if self.alibi_values is not None:
            tables["alibi_slopes"] = self.alibi_values
            score = score - table_lookup("alibi_slopes", HEAD) * (Q_IDX - KV_IDX)
            features.append("alibi")
        if self.softcap_value is not None:
            cap = self.softcap_value
            score = cap * call("tanh", score / cap)
            features.append("softcap")
        if self.layout.kv_heads is not None:
            features.append("gqa")

        bounds = MaskBounds(self.use_causal, self.window_size)
        return AttentionIR(
            layout=self.layout,
            mask=mask,
            score=score,
            tables=tables,
            mask_bounds=bounds,
            features=tuple(features),
        )


def attention(*, head_dim: int, dtype: str = "fp16", kv_layout: str = "contiguous") -> AttentionSpec:
    return AttentionSpec(Layout(head_dim=head_dim, dtype=dtype, kv_layout=kv_layout))

