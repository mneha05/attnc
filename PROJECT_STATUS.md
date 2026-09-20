# Project status

Validated in the build environment on 2026-09-20:

- Editable package installation succeeds on Python 3.12.
- All modules, tests, examples, and benchmark scripts compile as Python.
- 13 unit/property tests pass.
- The randomized differential test covers 160 variant/shape/seed cases.
- Both combinator and traced examples run through the NumPy backend.
- CLI IR and CUDA source dumping works.

Not validated in this environment:

- NVRTC compilation, CUDA execution, PTX inspection, and first-call autotuning.
- T4 or GB10 performance.
- PyTorch SDPA/FlexAttention benchmark cells.

Those checks require an NVIDIA GPU plus the `cuda`/`torch` extras. No runtime or
performance result in this repository should be inferred from static CUDA code
generation alone.

