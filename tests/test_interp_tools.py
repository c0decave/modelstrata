import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

from interp.activations import ActivationCache, ActivationKey, token_slice
from interp.logit_lens import project, softmax, topk
from interp.patching import patch, relative_delta


class TestActivationHelpers(unittest.TestCase):
    def test_cache_manifest_is_stable(self):
        c = ActivationCache()
        c.put(ActivationKey(1, "resid_pre", 2), np.zeros((4,), dtype=np.float32))
        c.put(ActivationKey(0, "resid_pre"), np.ones((2, 4), dtype=np.float32))
        self.assertEqual([m["label"] for m in c.manifest()],
                         ["layer.0.resid_pre", "layer.1.resid_pre.tok2"])
        self.assertEqual(c.manifest()[0]["shape"], [2, 4])

    def test_token_slice_modes(self):
        self.assertEqual(token_slice([1, 2, 3], last=True), [2])
        self.assertEqual(token_slice([1, 2, 3], positions=[0, -1]), [0, 2])
        with self.assertRaises(ValueError):
            token_slice([1], last=True, positions=[0])


class TestLogitLensHelpers(unittest.TestCase):
    def test_project_and_topk(self):
        hidden = np.array([1.0, 0.0])
        unemb = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
        logits = project(hidden, unemb)
        self.assertEqual(logits.tolist(), [1.0, 0.0, 2.0])
        probs = softmax(logits)
        self.assertAlmostEqual(float(probs.sum()), 1.0)
        top = topk(logits, tokens=["a", "b", "c"], k=2)
        self.assertEqual([x["token"] for x in top], ["c", "a"])

    def test_project_dim_mismatch(self):
        with self.assertRaises(ValueError):
            project(np.zeros(3), np.zeros((2, 4)))


class TestPatchingHelpers(unittest.TestCase):
    def test_patch_positions(self):
        clean = np.array([[1, 1], [2, 2], [3, 3]])
        corrupt = np.zeros((3, 2), dtype=int)
        out = patch(clean, corrupt, positions=[1])
        self.assertEqual(out.tolist(), [[0, 0], [2, 2], [0, 0]])
        out = patch(clean, corrupt, positions=[-1])
        self.assertEqual(out.tolist(), [[0, 0], [0, 0], [3, 3]])
        with self.assertRaises(IndexError):
            patch(clean, corrupt, positions=[3])

    def test_relative_delta(self):
        self.assertAlmostEqual(relative_delta(np.array([1.0, 0.0]),
                                             np.array([2.0, 0.0])), 1.0)


if __name__ == "__main__":
    unittest.main()
