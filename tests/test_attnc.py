from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from attnc import attention, compile, table, tanh
from attnc.codegen import generate_cuda
from attnc.interpreter import run_reference
from attnc.passes import TileState, classify_tile, infer_mask_bounds, optimize


def dense_expected(q, k, v, *, causal=False, window=None, cap=None, slopes=None):
    bsz, hq, nq, d = q.shape
    _, hkv, nk, _ = k.shape
    offset = max(nk - nq, 0)
    out = np.zeros_like(q, dtype=np.float32)
    for b in range(bsz):
        for h in range(hq):
            kh = h * hkv // hq
            scores = q[b, h].astype(np.float32) @ k[b, kh].astype(np.float32).T / math.sqrt(d)
            for qi in range(nq):
                qa = qi + offset
                for ki in range(nk):
                    allowed = (not causal or qa >= ki) and (window is None or qa - ki < window)
                    if not allowed:
                        scores[qi, ki] = -np.inf
                    else:
                        if slopes is not None: scores[qi, ki] -= slopes[h] * (qa - ki)
                        if cap is not None: scores[qi, ki] = cap * np.tanh(scores[qi, ki] / cap)
            probs = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            probs /= probs.sum(axis=-1, keepdims=True)
            out[b, h] = probs @ v[b, kh].astype(np.float32)
    return out


class FrontendTests(unittest.TestCase):
    def test_combinators_are_immutable_and_lower(self):
        base = attention(head_dim=16, dtype="fp32")
        spec = base.causal().sliding_window(8).gqa(2).softcap(30).alibi([0.1] * 4)
        self.assertFalse(base.use_causal)
        ir = optimize(spec.lower())
        self.assertEqual(ir.layout.kv_heads, 2)
        self.assertEqual(set(ir.features), {"causal", "sliding_window", "gqa", "softcap", "alibi"})
        self.assertEqual(ir.mask_bounds.window, 8)

    def test_stable_hash(self):
        ir = attention(head_dim=32).causal().lower()
        self.assertEqual(ir.stable_hash(arch="sm_80"), ir.stable_hash(arch="sm_80"))
        self.assertNotEqual(ir.stable_hash(arch="sm_80"), ir.stable_hash(arch="sm_121"))

    def test_invalid_options(self):
        with self.assertRaises(ValueError): attention(head_dim=0)
        with self.assertRaises(ValueError): attention(head_dim=16).sliding_window(0)
        with self.assertRaises(ValueError): attention(head_dim=16).softcap(-1)


class TracerTests(unittest.TestCase):
    def test_proxy_trace_with_captured_table(self):
        slopes = table([0.1, 0.2, 0.3, 0.4], "slopes")

        def score_mod(score, b, h, q_idx, kv_idx):
            return score - slopes[h] * (q_idx - kv_idx)

        def mask_mod(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & ((q_idx - kv_idx) < 4)

        kernel = compile(score_mod=score_mod, mask_mod=mask_mod, head_dim=8, dtype="fp32", backend="numpy")
        self.assertIn("slopes", kernel.ir.tables)
        self.assertEqual(kernel.ir.mask_bounds.window, 4)
        self.assertIn("table_slopes", kernel.source)

    def test_traced_softcap(self):
        def score_mod(score, b, h, q_idx, kv_idx):
            return 10.0 * tanh(score / 10.0)
        kernel = compile(score_mod=score_mod, head_dim=8, dtype="fp32", backend="numpy")
        self.assertIn("tanhf", kernel.source)


class PassTests(unittest.TestCase):
    def test_tile_classification(self):
        bounds = attention(head_dim=8).causal().sliding_window(4).lower().mask_bounds
        self.assertEqual(classify_tile(bounds, 8, 9, 0, 4), TileState.FULLY_MASKED)
        self.assertEqual(classify_tile(bounds, 8, 9, 5, 9), TileState.FULLY_UNMASKED)
        self.assertEqual(classify_tile(bounds, 8, 10, 5, 10), TileState.PARTIAL)

    def test_custom_mask_is_conservative(self):
        def mask(b, h, q, k): return ((q + k) >= 3)
        kernel = compile(mask_mod=mask, head_dim=8, dtype="fp32", backend="numpy")
        self.assertIsNone(kernel.ir.mask_bounds)
        self.assertIn("conservatively predicated", kernel.source)


class CorrectnessTests(unittest.TestCase):
    def test_160_differential_cases(self):
        variants = [
            dict(causal=True),
            dict(causal=True, window=3),
            dict(causal=True, cap=5.0),
            dict(causal=True, window=4, cap=8.0),
            dict(causal=True, window=5, slopes=[0.03, 0.07, 0.11, 0.15]),
        ]
        tested = 0
        for seed in range(32):
            rng = np.random.default_rng(seed)
            for opts in variants:
                b, hq, hkv, nq, nk, d = 1, 4, 2 if seed % 2 else 4, 1 + seed % 5, 5 + seed % 7, 8
                q = rng.normal(size=(b, hq, nq, d)).astype(np.float32)
                k = rng.normal(size=(b, hkv, nk, d)).astype(np.float32)
                v = rng.normal(size=(b, hkv, nk, d)).astype(np.float32)
                spec = attention(head_dim=d, dtype="fp32")
                if opts.get("causal"): spec = spec.causal()
                if opts.get("window"): spec = spec.sliding_window(opts["window"])
                if opts.get("cap"): spec = spec.softcap(opts["cap"])
                if opts.get("slopes"): spec = spec.alibi(opts["slopes"])
                if hkv != hq: spec = spec.gqa(hkv)
                got = compile(spec, backend="numpy")(q, k, v)
                expected = dense_expected(q, k, v, **opts)
                np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-5)
                tested += 1
        self.assertEqual(tested, 160)

    def test_decode_offset_attends_history(self):
        q = np.ones((1, 1, 1, 2), np.float32)
        k = np.ones((1, 1, 5, 2), np.float32)
        v = np.arange(10, dtype=np.float32).reshape(1, 1, 5, 2)
        out = compile(attention(head_dim=2, dtype="fp32").causal(), backend="numpy")(q, k, v)
        np.testing.assert_allclose(out[0, 0, 0], v[0, 0].mean(axis=0))

    def test_mask_composition_commutes(self):
        a = attention(head_dim=8).causal().sliding_window(4).lower()
        b = attention(head_dim=8).sliding_window(4).causal().lower()
        self.assertEqual(a.mask.to_dict(), b.mask.to_dict())

    def test_gqa_equals_mha_when_head_counts_match(self):
        rng = np.random.default_rng(9)
        q = rng.normal(size=(1, 2, 3, 8)).astype(np.float32)
        k = rng.normal(size=(1, 2, 3, 8)).astype(np.float32)
        v = rng.normal(size=(1, 2, 3, 8)).astype(np.float32)
        base = compile(attention(head_dim=8, dtype="fp32").causal(), backend="numpy")(q, k, v)
        gqa = compile(attention(head_dim=8, dtype="fp32").causal().gqa(2), backend="numpy")(q, k, v)
        np.testing.assert_array_equal(base, gqa)


class CodegenTests(unittest.TestCase):
    def test_fused_cuda_contains_expected_stages(self):
        ir = optimize(attention(head_dim=128).causal().sliding_window(4096).softcap(50).lower())
        source = generate_cuda(ir)
        for token in ("warp_sum", "classify_kv_tile", "tanhf", "new_m", "acc[i]", "q_offset"):
            self.assertIn(token, source)
        self.assertNotIn("scores[", source)

    def test_dump_cuda_and_ir_cache(self):
        kernel = compile(attention(head_dim=8, dtype="fp32").causal(), backend="numpy")
        self.assertTrue((kernel.cache_dir / "kernel.cu").exists())
        self.assertEqual(json.loads((kernel.cache_dir / "ir.json").read_text())["layout"]["head_dim"], 8)
        with tempfile.TemporaryDirectory() as tmp:
            path = kernel.dump_cuda(Path(tmp) / "out.cu")
            self.assertIn("attnc_attention", path.read_text())


if __name__ == "__main__":
    unittest.main()

