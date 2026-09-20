"""Small architecture-aware search space for first-call CUDA autotuning."""

from __future__ import annotations


def candidate_kv_tiles(arch: str, head_dim: int) -> tuple[int, ...]:
    if arch in {"sm_75", "sm_70"}:  # T4 / Volta-class register pressure
        return (32, 64) if head_dim >= 128 else (32, 64, 128)
    if arch.startswith("sm_12") or arch in {"sm_90", "sm_100"}:
        return (64, 128, 256) if head_dim <= 128 else (32, 64, 128)
    return (32, 64, 128)


def default_kv_tile(arch: str, head_dim: int) -> int:
    candidates = candidate_kv_tiles(arch, head_dim)
    return 64 if 64 in candidates else candidates[0]

