import numpy as np

from attnc import attention, compile

spec = (attention(head_dim=128, dtype="fp32")
        .causal()
        .sliding_window(4096)
        .gqa(kv_heads=2)
        .softcap(50.0))

# auto uses CUDA when CuPy and a device are available; NumPy otherwise.
kernel = compile(spec, arch="sm_121")
q = np.random.default_rng(0).normal(size=(1, 8, 4, 128)).astype(np.float32)
k = np.random.default_rng(1).normal(size=(1, 2, 16, 128)).astype(np.float32)
v = np.random.default_rng(2).normal(size=(1, 2, 16, 128)).astype(np.float32)
out = kernel(q, k, v)
print(out.shape, kernel.backend, kernel.cache_key)

