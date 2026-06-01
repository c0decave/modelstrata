"""Tests für tools/model_diff.py — Diff-Mathematik & Gruppierungslogik."""
import os
import sys
import unittest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

try:
    import numpy as np
    import model_diff as md
    HAVE = True
except Exception:
    HAVE = False


@unittest.skipUnless(HAVE, "numpy/model_diff not importable")
class TestDiffMath(unittest.TestCase):
    def test_rel_delta_zero_for_identical(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(md.rel_delta(a, a), 0.0)

    def test_rel_delta_known(self):
        a = np.array([3.0, 4.0])          # ||a||=5
        b = np.array([3.0, 0.0])          # ||b-a||=4
        self.assertAlmostEqual(md.rel_delta(a, b), 4.0 / 5.0, places=6)

    def test_cosine_identical(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertAlmostEqual(md.cosine(a, a), 1.0, places=6)

    def test_cosine_opposite(self):
        a = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(md.cosine(a, -a), -1.0, places=6)

    def test_cosine_orthogonal(self):
        self.assertAlmostEqual(md.cosine(np.array([1.0, 0.0]),
                                         np.array([0.0, 1.0])), 0.0, places=6)


@unittest.skipUnless(HAVE, "numpy/model_diff not importable")
class TestDiffModels(unittest.TestCase):
    def _stub(self, mapping):
        def fake(path, want=None):
            for n, a, t in mapping[path]:
                if want and not want(n):
                    continue
                yield n, a, t
        md.iter_weights = fake

    def test_self_diff_is_zero(self):
        A = [("blk.0.attn_q.weight", np.eye(3), "F32"),
             ("blk.1.ffn_down.weight", np.ones((3, 4)), "F32")]
        self._stub({"m": A})
        d = md.diff_models("m", "m", "X", "X")
        self.assertEqual(d["matched"], 2)
        for role, layers in d["cells"].items():
            for l, s in layers.items():
                self.assertEqual(s["delta"], 0.0)
                self.assertAlmostEqual(s["cosine"], 1.0, places=6)

    def test_detects_change_and_ranks_top(self):
        A = [("blk.0.attn_q.weight", np.eye(2), "F32"),
             ("blk.0.ffn_down.weight", np.ones((2, 2)), "F32")]
        B = [("blk.0.attn_q.weight", np.eye(2), "F32"),          # unchanged
             ("blk.0.ffn_down.weight", np.ones((2, 2)) * 3, "F32")]  # changed a lot
        self._stub({"a": A, "b": B})
        d = md.diff_models("a", "b", "base", "ft")
        self.assertEqual(d["matched"], 2)
        self.assertEqual(d["cells"]["attn_q"][0]["delta"], 0.0)
        self.assertGreater(d["cells"]["ffn_down"][0]["delta"], 0.0)
        self.assertEqual(d["top"][0]["name"], "blk.0.ffn_down.weight")  # ranked first
        self.assertEqual(d["label"], "base → ft")

    def test_shape_mismatch_skipped(self):
        A = [("blk.0.attn_q.weight", np.eye(3), "F32")]
        B = [("blk.0.attn_q.weight", np.eye(4), "F32")]   # different shape
        self._stub({"a": A, "b": B})
        d = md.diff_models("a", "b")
        self.assertEqual(d["matched"], 0)
        self.assertEqual(d["shape_mismatch"], 1)

    def test_disjoint_names(self):
        self._stub({"a": [("blk.0.attn_q.weight", np.eye(3), "F32")],
                    "b": [("blk.0.attn_k.weight", np.eye(3), "F32")]})
        d = md.diff_models("a", "b")
        self.assertEqual(d["matched"], 0)
        self.assertEqual(d["cells"], {})
        self.assertEqual(d["top"], [])

    def test_partial_overlap(self):
        A = [("blk.0.attn_q.weight", np.eye(3), "F32"),
             ("blk.0.ffn_up.weight", np.ones((3, 3)), "F32")]   # B lacks ffn_up
        B = [("blk.0.attn_q.weight", np.eye(3) * 2, "F32"),
             ("blk.0.attn_v.weight", np.eye(3), "F32")]          # A lacks attn_v
        self._stub({"a": A, "b": B})
        d = md.diff_models("a", "b")
        self.assertEqual(d["matched"], 1)                        # only attn_q common
        self.assertIn("attn_q", d["cells"])

    def test_3d_expert_tensors_diffed(self):
        # regression for S1: MoE expert stacks must be diffed (ravel handles 3-D)
        A = [("blk.0.ffn_down_exps.weight", np.ones((4, 3, 3)), "F32")]
        B = [("blk.0.ffn_down_exps.weight", np.ones((4, 3, 3)) * 2, "F32")]
        self._stub({"a": A, "b": B})
        d = md.diff_models("a", "b")
        self.assertEqual(d["matched"], 1)
        self.assertGreater(d["cells"]["ffn_down_exps"][0]["delta"], 0.0)

    def test_zero_base_sentinel(self):
        self._stub({"a": [("blk.0.attn_q.weight", np.zeros((3, 3)), "F32")],
                    "b": [("blk.0.attn_q.weight", np.ones((3, 3)), "F32")]})
        d = md.diff_models("a", "b")
        self.assertGreater(d["cells"]["attn_q"][0]["delta"], 1e6)   # finite huge sentinel

    def test_top_truncated_to_15(self):
        A = [(f"blk.{i}.attn_q.weight", np.eye(3), "F32") for i in range(20)]
        B = [(f"blk.{i}.attn_q.weight", np.eye(3) * (1 + i), "F32") for i in range(20)]
        self._stub({"a": A, "b": B})
        d = md.diff_models("a", "b")
        self.assertEqual(d["matched"], 20)
        self.assertEqual(len(d["top"]), 15)


if __name__ == "__main__":
    unittest.main()
