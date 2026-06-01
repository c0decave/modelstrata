"""Tests für tools/weight_stats.py — Statistik-Mathematik (numpy, kein gguf nötig)."""
import os
import sys
import unittest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

try:
    import numpy as np
    import weight_stats as ws
    HAVE = True
except Exception:
    HAVE = False


@unittest.skipUnless(HAVE, "numpy/weight_stats not importable")
class TestTensorStats(unittest.TestCase):
    def test_zeros(self):
        s = ws.tensor_stats(np.zeros((4, 4)))
        self.assertEqual(s["std"], 0.0)
        self.assertEqual(s["sparsity"], 1.0)     # all |w|<1e-6
        self.assertEqual(s["kurtosis"], 0.0)     # std==0 -> defined as 0
        self.assertEqual(s["l2"], 0.0)

    def test_constant(self):
        s = ws.tensor_stats(np.full((3, 3), 2.0))
        self.assertAlmostEqual(s["mean"], 2.0)
        self.assertEqual(s["std"], 0.0)
        self.assertEqual(s["min"], 2.0)
        self.assertEqual(s["max"], 2.0)

    def test_sparsity_half(self):
        a = np.array([0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(ws.tensor_stats(a)["sparsity"], 0.5)

    def test_l2_known(self):
        a = np.array([3.0, 4.0])
        self.assertAlmostEqual(ws.tensor_stats(a)["l2"], 5.0)

    def test_kurtosis_uniform_negative(self):
        # uniform distribution has excess kurtosis ≈ -1.2
        a = np.linspace(-1, 1, 100000)
        self.assertLess(ws.tensor_stats(a)["kurtosis"], -1.0)

    def test_empty(self):
        s = ws.tensor_stats(np.zeros((0,)))
        self.assertEqual(s["n"], 0)


@unittest.skipUnless(HAVE, "numpy/weight_stats not importable")
class TestOutlierChannels(unittest.TestCase):
    def test_one_giant_column(self):
        a = np.ones((10, 5))
        a[:, 2] *= 100.0                      # one column far above median
        self.assertGreater(ws.outlier_channel_frac(a), 0.0)
        self.assertAlmostEqual(ws.outlier_channel_frac(a), 1 / 5)

    def test_uniform_no_outliers(self):
        self.assertEqual(ws.outlier_channel_frac(np.ones((10, 5))), 0.0)

    def test_non_2d_zero(self):
        self.assertEqual(ws.outlier_channel_frac(np.ones((8,))), 0.0)

    def test_all_zero_columns(self):
        self.assertEqual(ws.outlier_channel_frac(np.zeros((10, 5))), 0.0)  # med==0 branch


@unittest.skipUnless(HAVE, "numpy/weight_stats not importable")
class TestTensorStatsEdge(unittest.TestCase):
    def test_1d_tensor(self):
        s = ws.tensor_stats(np.arange(10.0))
        for k in ("std", "l2", "sparsity", "kurtosis", "outlier"):
            self.assertIn(k, s)
        self.assertEqual(s["outlier"], 0.0)        # not 2-D

    def test_3d_moe_stack_flattened(self):
        s = ws.tensor_stats(np.ones((4, 8, 8)))    # MoE expert stack
        self.assertEqual(s["n"], 256)
        self.assertEqual(s["outlier"], 0.0)        # ndim != 2 -> 0

    def test_nan_does_not_crash(self):
        s = ws.tensor_stats(np.array([1.0, np.nan, 3.0]))
        self.assertEqual(s["n"], 3)                # no exception


@unittest.skipUnless(HAVE, "numpy/weight_stats not importable")
class TestAnalyzeModel(unittest.TestCase):
    def test_grouping_and_bias_exclusion(self):
        rows = [("token_embd.weight", np.ones((4, 4)), "F32"),       # global, ignored
                ("blk.0.attn_q.weight", np.eye(4), "F32"),
                ("blk.0.attn_q.bias", np.ones(4), "F32"),            # bias excluded
                ("blk.1.ffn_down.weight", np.ones((4, 8)), "Q6_K")]
        ws.iter_weights = lambda p, want=None: iter(rows)
        d = ws.analyze_model("m", "M")
        self.assertEqual(d["layers"], 2)
        self.assertIn("attn_q", d["cells"])
        self.assertIn(0, d["cells"]["attn_q"])
        self.assertNotIn("token_embd", d["cells"])     # globals not grouped
        # bias must not create its own metric entry beyond the weight cell
        self.assertEqual(set(d["cells"]["attn_q"][0]), set(ws.METRICS))


if __name__ == "__main__":
    unittest.main()
