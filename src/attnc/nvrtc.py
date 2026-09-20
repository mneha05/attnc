"""Minimal ctypes NVRTC wrapper used by --dump-ptx (no CUDA SDK headers needed)."""

from __future__ import annotations

import ctypes
import ctypes.util


class NVRTCError(RuntimeError):
    pass


def _library() -> ctypes.CDLL:
    candidates = [ctypes.util.find_library("nvrtc"), "libnvrtc.so", "nvrtc64_130_0.dll", "nvrtc64_120_0.dll"]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            pass
    raise NVRTCError("NVRTC was not found; install a CUDA toolkit or the CuPy CUDA extra")


def compile_ptx(source: str, *, arch: str, name: str = "attnc.cu") -> bytes:
    lib = _library()
    program = ctypes.c_void_p()
    src = source.encode()
    result = lib.nvrtcCreateProgram(ctypes.byref(program), src, name.encode(), 0, None, None)
    if result != 0:
        raise NVRTCError(f"nvrtcCreateProgram failed with code {result}")
    compute = arch.replace("sm_", "compute_")
    opts = [f"--gpu-architecture={compute}".encode(), b"--std=c++17", b"--use_fast_math"]
    array = (ctypes.c_char_p * len(opts))(*opts)
    try:
        result = lib.nvrtcCompileProgram(program, len(opts), array)
        log_size = ctypes.c_size_t()
        lib.nvrtcGetProgramLogSize(program, ctypes.byref(log_size))
        log = ctypes.create_string_buffer(max(1, log_size.value))
        lib.nvrtcGetProgramLog(program, log)
        if result != 0:
            raise NVRTCError(log.value.decode(errors="replace"))
        size = ctypes.c_size_t()
        if lib.nvrtcGetPTXSize(program, ctypes.byref(size)) != 0:
            raise NVRTCError("nvrtcGetPTXSize failed")
        ptx = ctypes.create_string_buffer(size.value)
        if lib.nvrtcGetPTX(program, ptx) != 0:
            raise NVRTCError("nvrtcGetPTX failed")
        return ptx.raw.rstrip(b"\0")
    finally:
        lib.nvrtcDestroyProgram(ctypes.byref(program))

