# Benchmark protocol

Run each device separately and commit the JSON output with the exact GPU name,
driver, CUDA, PyTorch, and attnc revision recorded alongside it.

```bash
python benchmarks/bench.py --arch sm_75 --peak-gbps 320 --output results-t4.json
python benchmarks/bench.py --arch sm_121 --peak-gbps YOUR_MEASURED_BOARD_VALUE --output results-gb10.json
```

The default matrix covers decode at 1K/8K/32K KV length and prefill at 2K/8K,
across causal, sliding-window, GQA, softcap, ALiBi, and combined variants. SDPA
is reported only where the semantics can be matched exactly. A hand-written
baseline can be supplied as `--handwritten package.module:function`; the
function receives `(q, k, v, variant=NAME)`.

`effective_gbps` and `effective_percent_peak` count logical input/output tensor
bytes. They are useful for consistent comparisons, but they are not
hardware-counter measurements and must not be described as actual DRAM
bandwidth. Use Nsight Compute counters for a literal percent-of-peak claim.
