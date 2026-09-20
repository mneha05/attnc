"""Compilation, caching, dispatch, and optional CUDA execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import json
import math
import os
import numpy as np

from .codegen import generate_cuda
from .autotune import candidate_kv_tiles, default_kv_tile
from .frontend import AttentionSpec
from .interpreter import run_reference
from .ir import AttentionIR
from .passes import optimize
from .tracer import SymbolicTable, trace


def _cache_root() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "attnc"


def cuda_available() -> bool:
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@dataclass
class CompiledAttention:
    ir: AttentionIR
    arch: str
    backend: str
    source: str
    cache_key: str
    kv_tile: int = 64
    autotune: bool = False
    _module: Any = None
    _kernel: Any = None

    def __post_init__(self) -> None:
        root = _cache_root() / self.cache_key
        root.mkdir(parents=True, exist_ok=True)
        (root / "kernel.cu").write_text(self.source)
        (root / "ir.json").write_text(json.dumps(self.ir.to_dict(), indent=2, sort_keys=True))

    @property
    def cache_dir(self) -> Path:
        return _cache_root() / self.cache_key

    def dump_cuda(self, path: str | os.PathLike[str]) -> Path:
        target = Path(path)
        target.write_text(self.source)
        return target

    def dump_ptx(self, path: str | os.PathLike[str]) -> Path:
        from .nvrtc import compile_ptx
        target = Path(path)
        target.write_bytes(compile_ptx(self.source, arch=self.arch))
        return target

    def _ensure_cuda(self) -> Any:
        if self._kernel is not None:
            return self._kernel
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError("CUDA execution requires `pip install 'attnc[cuda]'`") from exc
        self._module = cp.RawModule(code=self.source, options=("--std=c++17", "--use_fast_math"), name_expressions=("attnc_attention",))
        self._kernel = self._module.get_function("attnc_attention")
        return self._kernel

    def _autotune_cuda(self, cp: Any, args: list[Any], grid: tuple[int, ...], shape_key: str) -> None:
        record = self.cache_dir / f"tune-{shape_key}.json"
        candidates = candidate_kv_tiles(self.arch, self.ir.layout.head_dim)
        if record.exists():
            try:
                chosen = int(json.loads(record.read_text())["kv_tile"])
                if chosen in candidates:
                    self.kv_tile = chosen
                    self.source = generate_cuda(self.ir, kv_tile=chosen)
                    self._module = self._kernel = None
                    self.autotune = False
                    return
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass
        timings: dict[int, float] = {}
        best: tuple[float, int, Any, Any, str] | None = None
        for tile in candidates:
            source = generate_cuda(self.ir, kv_tile=tile)
            module = cp.RawModule(code=source, options=("--std=c++17", "--use_fast_math"), name_expressions=("attnc_attention",))
            kernel = module.get_function("attnc_attention")
            kernel(grid, (32,), tuple(args))
            cp.cuda.runtime.deviceSynchronize()
            start, end = cp.cuda.Event(), cp.cuda.Event()
            start.record()
            for _ in range(5): kernel(grid, (32,), tuple(args))
            end.record(); end.synchronize()
            elapsed = float(cp.cuda.get_elapsed_time(start, end)) / 5.0
            timings[tile] = elapsed
            if best is None or elapsed < best[0]: best = (elapsed, tile, module, kernel, source)
        assert best is not None
        _, self.kv_tile, self._module, self._kernel, self.source = best
        record.write_text(json.dumps({"kv_tile": self.kv_tile, "milliseconds": timings}, indent=2, sort_keys=True))
        (self.cache_dir / "kernel.cu").write_text(self.source)
        self.autotune = False

    def __call__(self, q: Any, k: Any, v: Any, *, scale: float | None = None, q_offset: int | None = None) -> Any:
        if self.backend == "numpy":
            return run_reference(q, k, v, self.ir, scale=scale, q_offset=q_offset)
        return self._call_cuda(q, k, v, scale=scale, q_offset=q_offset)

    def _call_cuda(self, q: Any, k: Any, v: Any, *, scale: float | None, q_offset: int | None) -> Any:
        import cupy as cp
        was_torch = q.__class__.__module__.startswith("torch")
        if was_torch:
            q_cp, k_cp, v_cp = (cp.from_dlpack(x) for x in (q, k, v))
        else:
            q_cp, k_cp, v_cp = (cp.asarray(x) for x in (q, k, v))
        if not (q_cp.flags.c_contiguous and k_cp.flags.c_contiguous and v_cp.flags.c_contiguous):
            raise ValueError("q, k, and v must be contiguous")
        b, hq, nq, d = q_cp.shape
        bk, hkv, nk, dk = k_cp.shape
        if (bk, dk) != (b, d) or v_cp.shape != (b, hkv, nk, d) or d != self.ir.layout.head_dim:
            raise ValueError("incompatible q/k/v shapes")
        if hq % hkv:
            raise ValueError("query head count must be divisible by key/value head count")
        if self.ir.layout.kv_heads is not None and hkv != self.ir.layout.kv_heads:
            raise ValueError(f"expected {self.ir.layout.kv_heads} KV heads, got {hkv}")
        expected = cp.float16 if self.ir.layout.dtype == "fp16" else cp.float32
        if q_cp.dtype != expected or k_cp.dtype != expected or v_cp.dtype != expected:
            raise TypeError(f"compiled kernel expects {self.ir.layout.dtype}")
        out = cp.empty_like(q_cp)
        args: list[Any] = [q_cp, k_cp, v_cp, out]
        for name in sorted(self.ir.tables):
            args.append(cp.asarray(self.ir.tables[name], dtype=cp.float32))
        q_offset = max(nk - nq, 0) if q_offset is None else int(q_offset)
        args.extend([np.int32(b), np.int32(hq), np.int32(hkv), np.int32(nq), np.int32(nk), np.int32(q_offset), np.float32(scale or 1.0 / math.sqrt(d))])
        if self.autotune:
            self._autotune_cuda(cp, args, (b * hq * nq,), f"{b}x{hq}x{hkv}x{nq}x{nk}x{d}")
        self._ensure_cuda()((b * hq * nq,), (32,), tuple(args))
        if was_torch:
            import torch
            return torch.from_dlpack(out)
        return out


def compile_attention(
    spec: AttentionSpec | AttentionIR | None = None,
    *,
    score_mod: Callable[..., Any] | None = None,
    mask_mod: Callable[..., Any] | None = None,
    head_dim: int | None = None,
    dtype: str = "fp16",
    kv_heads: int | None = None,
    tables: Mapping[str, SymbolicTable | Sequence[float]] | None = None,
    arch: str = "sm_80",
    backend: str = "auto",
    autotune: bool = True,
) -> CompiledAttention:
    if spec is not None and (score_mod is not None or mask_mod is not None):
        raise ValueError("pass either spec or score_mod/mask_mod, not both")
    if isinstance(spec, AttentionSpec):
        ir = spec.lower()
    elif isinstance(spec, AttentionIR):
        ir = spec
    else:
        if head_dim is None:
            raise ValueError("head_dim is required for the traced frontend")
        ir = trace(score_mod=score_mod, mask_mod=mask_mod, head_dim=head_dim, dtype=dtype, kv_heads=kv_heads, tables=tables)
    ir = optimize(ir)
    selected = "cuda" if backend == "auto" and cuda_available() else ("numpy" if backend == "auto" else backend)
    if selected not in {"numpy", "cuda"}:
        raise ValueError("backend must be auto, numpy, or cuda")
    kv_tile = default_kv_tile(arch, ir.layout.head_dim)
    source = generate_cuda(ir, kv_tile=kv_tile)
    key = ir.stable_hash(arch=arch, options={"backend": "cuda", "generator": 1})
    return CompiledAttention(ir=ir, arch=arch, backend=selected, source=source, cache_key=key, kv_tile=kv_tile, autotune=autotune and selected == "cuda")
