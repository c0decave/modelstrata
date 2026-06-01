"""Tests für tools/modelsource/log.py — strukturierter Logger (kein Verschlucken)."""
import io, os, sys, unittest
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from modelsource import log as L

class TestRunLog(unittest.TestCase):
    def setUp(self):
        self.rl = L.RunLog()
    def test_entry_goes_to_buffer_and_stderr(self):
        err = io.StringIO()
        self.rl.emit("warn", model="m1", stage="spectral",
                     code="arch_unmapped", msg="no map", stream=err)
        self.assertEqual(len(self.rl.entries), 1)
        e = self.rl.entries[0]
        self.assertEqual((e["severity"], e["model"], e["code"]), ("warn", "m1", "arch_unmapped"))
        self.assertIn("arch_unmapped", err.getvalue())   # NOT swallowed
    def test_counts_and_export(self):
        self.rl.emit("info", model="m1", stage="scan", code="ok", msg="x", stream=io.StringIO())
        self.rl.emit("error", model="m1", stage="weights", code="decode", msg="y", stream=io.StringIO())
        self.assertEqual(self.rl.counts(), {"info": 1, "warn": 0, "error": 1})
        self.assertEqual(self.rl.to_json()[0]["severity"], "info")
    def test_write_roundtrip(self):
        import json, tempfile, os
        self.rl.emit("info", model="m", stage="s", code="c", msg="hi", stream=io.StringIO())
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "run_log.json")
            self.rl.write(p)
            with open(p, encoding="utf-8") as fh:
                loaded = json.load(fh)
        self.assertEqual(loaded, self.rl.to_json())
    def test_bad_severity_raises(self):
        with self.assertRaises(ValueError):
            self.rl.emit("warning", model="m", stage="s", code="c", msg="x", stream=io.StringIO())
