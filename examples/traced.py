import numpy as np

from attnc import compile, table

slopes = table([0.25, 0.125, 0.0625, 0.03125], name="slopes")


def alibi(score, b, h, q_idx, kv_idx):
    return score - slopes[h] * (q_idx - kv_idx)


def window_mask(b, h, q_idx, kv_idx):
    return (q_idx >= kv_idx) & ((q_idx - kv_idx) < 4096)


kernel = compile(score_mod=alibi, mask_mod=window_mask, head_dim=64, dtype="fp32")
rng = np.random.default_rng(4)
q = rng.normal(size=(1, 4, 8, 64)).astype(np.float32)
k = rng.normal(size=(1, 4, 8, 64)).astype(np.float32)
v = rng.normal(size=(1, 4, 8, 64)).astype(np.float32)
print(kernel(q, k, v).shape)

