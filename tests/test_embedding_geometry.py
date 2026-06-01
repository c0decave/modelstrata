"""Tests für tools/embedding_geometry.py — PCA/Anisotropie/Histogramm (numpy)."""
import os
import sys
import unittest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

try:
    import numpy as np
    import embedding_geometry as eg
    HAVE = True
except Exception:
    HAVE = False


@unittest.skipUnless(HAVE, "numpy/embedding_geometry not importable")
class TestPCA2(unittest.TestCase):
    def test_elongated_cloud(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(500, 5))
        X[:, 0] *= 20.0                      # variance concentrated on axis 0
        coords, evr = eg.pca2(X)
        self.assertEqual(coords.shape, (500, 2))
        self.assertGreater(evr[0], 0.8)      # first PC dominates
        self.assertGreaterEqual(evr[0], evr[1])

    def test_shape_and_range(self):
        coords, evr = eg.pca2(np.eye(6))
        self.assertEqual(coords.shape, (6, 2))
        self.assertEqual(len(evr), 2)


@unittest.skipUnless(HAVE, "numpy/embedding_geometry not importable")
class TestAnisotropy(unittest.TestCase):
    def test_collinear_high(self):
        X = np.tile(np.array([1.0, 2.0, 3.0]), (200, 1))   # all same direction
        self.assertGreater(eg.anisotropy(X, n_pairs=1000), 0.99)

    def test_random_low(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(2000, 64))                    # isotropic gaussian
        self.assertLess(abs(eg.anisotropy(X, n_pairs=5000)), 0.1)

    def test_degenerate(self):
        self.assertEqual(eg.anisotropy(np.zeros((5, 4))), 0.0)


@unittest.skipUnless(HAVE, "numpy/embedding_geometry not importable")
class TestHistogram(unittest.TestCase):
    def test_counts_sum(self):
        h = eg.norm_histogram(np.linspace(0, 1, 100), bins=10)
        self.assertEqual(sum(h["counts"]), 100)
        self.assertEqual(len(h["edges"]), 11)

    def test_constant_input(self):
        h = eg.norm_histogram(np.full(50, 2.0), bins=8)   # degenerate, must not crash
        self.assertEqual(sum(h["counts"]), 50)


@unittest.skipUnless(HAVE, "numpy/embedding_geometry not importable")
class TestPCA2Degenerate(unittest.TestCase):
    def test_single_row(self):            # regression for S2 (crash on coords[:,1])
        coords, evr = eg.pca2(np.array([[1.0, 2.0, 3.0]]))
        self.assertEqual(coords.shape, (1, 2))
        self.assertEqual(len(evr), 2)

    def test_single_feature(self):
        coords, evr = eg.pca2(np.arange(10.0).reshape(10, 1))
        self.assertEqual(coords.shape, (10, 2))
        self.assertEqual(len(evr), 2)


@unittest.skipUnless(HAVE, "numpy/embedding_geometry not importable")
class TestAnalyzeModelEmbed(unittest.TestCase):
    def _stub(self, emb, toks):
        eg.load_weight = lambda p, n: (emb, "F32")
        eg.read_tokenizer = lambda p: {"tokens": toks, "tokenizer.ggml.model": "gpt2"}

    def test_near_zero_token_flagged(self):
        emb = np.ones((5, 4))
        emb[2] = 0.0                       # one near-zero "glitch" token
        self._stub(emb, ["a", "b", "ZERO", "d", "e"])
        d = eg.analyze_model("m", "M", sample=5)
        self.assertGreaterEqual(d["near_zero"], 1)
        self.assertEqual(d["low_norm"][0]["tok"], "ZERO")   # lowest norm first
        self.assertEqual(d["vocab"], 5)
        self.assertEqual(len(d["pca"]["evr"]), 2)

    def test_seed_neighbors_and_output_head_compare(self):
        emb = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        self._stub(emb, ["SQL", "sqlmap", "other", "tool"])
        d = eg.analyze_model("m", "M", sample=4)
        self.assertTrue(d["seed_neighbors"])
        self.assertEqual(d["seed_neighbors"][0]["seed"], "sql")
        self.assertEqual(d["seed_neighbors"][0]["neighbors"][0]["tok"], "sqlmap")
        self.assertEqual(d["output_head_compare"]["delta"], 0.0)

    def test_anisotropy_mixed_norms(self):
        X = np.zeros((5, 4)); X[0] = [1, 2, 3, 4]    # only one valid row -> idx<2
        self.assertEqual(eg.anisotropy(X), 0.0)


if __name__ == "__main__":
    unittest.main()
