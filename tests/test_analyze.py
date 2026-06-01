"""Tests für tools/analyze.py — Orchestrator-CLI (subprocess, stdlib only)."""
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(__file__)
TOOL = os.path.join(HERE, "..", "tools", "analyze.py")
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

import analyze


class TestAnalyzeCLI(unittest.TestCase):
    def test_no_targets_errors(self):
        r = subprocess.run([sys.executable, TOOL], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("nothing to analyze", r.stderr)

    def test_help_ok(self):
        r = subprocess.run([sys.executable, TOOL, "-h"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        self.assertIn("orchestrator", r.stdout.lower())


class TestAnalyzeTargets(unittest.TestCase):
    def test_scan_targets_include_uppercase_and_labels(self):
        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, "sub")
            os.makedirs(sub)
            keep = os.path.join(sub, "MODEL.GGUF")
            skip = os.path.join(sub, "notes.txt")
            with open(keep, "wb") as f:
                f.write(b"GGUF")
            with open(skip, "wb") as f:
                f.write(b"x")

            self.assertEqual(analyze._scan_gguf_targets([d]), [("MODEL.GGUF", keep)])

    def test_is_gguf_path_accepts_extensionless_magic(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "sha256-deadbeef")
            with open(p, "wb") as f:
                f.write(b"GGUF\x03\x00\x00\x00")
            self.assertTrue(analyze._is_gguf_path(p))

    def test_dedupe_targets_keeps_first_label(self):
        targets = [("first", "/tmp/a"), ("second", "/tmp/a"), ("b", "/tmp/b")]
        self.assertEqual(analyze._dedupe_targets(targets),
                         [("first", "/tmp/a"), ("b", "/tmp/b")])


if __name__ == "__main__":
    unittest.main()
