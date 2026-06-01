"""Tests für tools/spectral.py — Spektral-Metriken (numpy)."""
import os
import sys
import unittest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

try:
    import numpy as np
    import spectral as sp
    HAVE = True
except Exception:
    HAVE = False


@unittest.skipUnless(HAVE, "numpy/spectral not importable")
class TestStableRank(unittest.TestCase):
    def test_identity(self):
        sv = np.linalg.svd(np.eye(10), compute_uv=False)
        self.assertAlmostEqual(sp.stable_rank(sv), 10.0, places=5)  # all sigma=1

    def test_rank_one(self):
        a = np.outer(np.arange(1, 6), np.arange(1, 4))   # rank-1 matrix
        sv = np.linalg.svd(a, compute_uv=False)
        self.assertAlmostEqual(sp.stable_rank(sv), 1.0, places=4)

    def test_zero(self):
        self.assertEqual(sp.stable_rank(np.zeros(5)), 0.0)


@unittest.skipUnless(HAVE, "numpy/spectral not importable")
class TestSpectralEntropy(unittest.TestCase):
    def test_identity_full_rank(self):
        sv = np.ones(8)
        self.assertAlmostEqual(sp.spectral_entropy(sv), 8.0, places=5)

    def test_rank_one(self):
        sv = np.array([5.0, 0.0, 0.0])
        self.assertAlmostEqual(sp.spectral_entropy(sv), 1.0, places=5)

    def test_zero(self):
        self.assertEqual(sp.spectral_entropy(np.zeros(4)), 0.0)


@unittest.skipUnless(HAVE, "numpy/spectral not importable")
class TestAlphaHill(unittest.TestCase):
    def test_too_few_returns_zero(self):
        self.assertEqual(sp.alpha_hill(np.ones(3)), 0.0)

    def test_positive_for_heavy_tail(self):
        # power-law-ish singular values -> finite positive alpha
        sv = 1.0 / np.sqrt(np.arange(1, 200))
        a = sp.alpha_hill(sv)
        self.assertGreater(a, 1.0)
        self.assertLess(a, 20.0)

    def test_matrix_spectral_skips_oversized(self):
        self.assertIsNone(sp.matrix_spectral(np.ones((100, 100)), max_dim=50))

    def test_matrix_spectral_runs(self):
        r = sp.matrix_spectral(np.random.RandomState(0).randn(20, 12))
        self.assertEqual(set(r), {"alpha", "stable_rank", "spectral_entropy"})
        self.assertGreater(r["stable_rank"], 1.0)

    def test_alpha_all_equal_returns_zero(self):
        self.assertEqual(sp.alpha_hill(np.ones(50)), 0.0)   # log(tail/xmin)=0 -> 0

    def test_alpha_n8_boundary(self):
        self.assertEqual(sp.alpha_hill(np.ones(7)), 0.0)    # <8 -> 0
        self.assertIsInstance(sp.alpha_hill(1.0 / np.sqrt(np.arange(1, 9))), float)

    def test_single_element_sv(self):
        self.assertAlmostEqual(sp.stable_rank(np.array([3.0])), 1.0)
        self.assertAlmostEqual(sp.spectral_entropy(np.array([3.0])), 1.0)

    def test_rank_one_matrix(self):
        r = sp.matrix_spectral(np.outer(np.arange(1, 9.0), np.arange(1, 6.0)))
        self.assertAlmostEqual(r["stable_rank"], 1.0, places=3)
        self.assertAlmostEqual(r["spectral_entropy"], 1.0, places=3)


@unittest.skipUnless(HAVE, "numpy/spectral not importable")
class TestSpectral3D(unittest.TestCase):
    def test_3d_moe_averaged(self):
        # regression for S1: MoE expert stacks must NOT be silently dropped
        a = np.stack([np.eye(6), np.eye(6) * 2.0, np.eye(6) * 0.5])  # [3,6,6]
        r = sp.matrix_spectral(a)
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r["stable_rank"], 6.0, places=3)     # each expert full-rank

    def test_analyze_model_includes_3d(self):
        rows = [("blk.0.attn_q.weight", np.eye(8), "F32"),
                ("blk.0.ffn_down_exps.weight", np.stack([np.eye(8)] * 4), "Q4_K")]
        sp.iter_weights = lambda p, want=None: iter(rows)
        d = sp.analyze_model("m", "M")
        self.assertIn("attn_q", d["cells"])
        self.assertIn("ffn_down_exps", d["cells"])                 # 3-D expert role present


if __name__ == "__main__":
    unittest.main()
