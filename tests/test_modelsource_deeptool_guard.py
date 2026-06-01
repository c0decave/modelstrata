"""ROUND-1 Fix 2 — deep tools must guard EACH model so one bad model can't sink
the batch, and the failure is logged STRUCTURALLY (not a raw traceback, not
model="*").

Each tool's per-model loop must wrap analyze_model in try/except → emit an
``analyze_failed`` error for THAT model and continue, while still writing the
result JSON for the models that succeeded.

We force a raise independent of Fix 1 by monkeypatching analyze_model to raise
for one of two inputs, then asserting (a) the good model's entry IS in the
output JSON, (b) the bad model produced exactly one structured analyze_failed
error naming it, (c) main() did not abort (returns, JSON written).
"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(__file__)
TOOLS = os.path.join(HERE, "..", "tools")
sys.path.insert(0, TOOLS)
sys.path.insert(0, HERE)

try:
    import numpy as np  # noqa: F401
    import weight_stats as ws
    import spectral as sp
    import embedding_geometry as eg
    HAVE = True
except Exception:
    HAVE = False


@unittest.skipUnless(HAVE, "numpy/tools not importable")
class TestPerModelGuard(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = self._td.name
        self.out = os.path.join(self.tmp, "r.json")
        self.runlog = os.path.join(self.tmp, "run_log.json")

    def tearDown(self):
        self._td.cleanup()

    def _run_with_guard(self, mod, good_entry):
        """Run ``mod.main`` over [bad, good] with analyze_model monkeypatched to
        raise for 'bad' and return ``good_entry`` for 'good'. Returns the parsed
        output JSON and run-log entries."""
        orig = mod.analyze_model

        def fake_analyze(path, *a, **kw):
            if os.path.basename(path) == "bad":
                raise RuntimeError("kaboom in analyze")
            return good_entry(path)

        mod.analyze_model = fake_analyze
        argv = sys.argv
        sys.argv = [mod.__name__, os.path.join(self.tmp, "bad"),
                    os.path.join(self.tmp, "good"),
                    "-o", self.out, "--run-log", self.runlog]
        try:
            mod.main()
        finally:
            mod.analyze_model = orig
            sys.argv = argv
        with open(self.out) as f:
            out = json.load(f)
        with open(self.runlog) as f:
            entries = json.load(f)
        return out, entries

    def _assert_batch_survives(self, out, entries):
        labels = [e.get("label") for e in out]
        self.assertIn("good", labels, f"valid model missing from output: {out}")
        self.assertNotIn("bad", labels)
        failed = [e for e in entries
                  if e["code"] == "analyze_failed" and e["severity"] == "error"]
        self.assertEqual(len(failed), 1, f"expected one analyze_failed, got {entries}")
        self.assertEqual(failed[0]["model"], "bad")
        self.assertNotEqual(failed[0]["model"], "*")
        self.assertIn("kaboom", failed[0]["msg"])

    def test_weight_stats_guard(self):
        out, entries = self._run_with_guard(
            ws, lambda p: {"label": os.path.basename(p), "cells": {}})
        self._assert_batch_survives(out, entries)

    def test_spectral_guard(self):
        out, entries = self._run_with_guard(
            sp, lambda p: {"label": os.path.basename(p), "cells": {}})
        self._assert_batch_survives(out, entries)

    def test_embedding_guard(self):
        out, entries = self._run_with_guard(
            eg, lambda p: {"label": os.path.basename(p)})
        self._assert_batch_survives(out, entries)


if __name__ == "__main__":
    unittest.main()
