"""Task 15 — analyze.py owns ONE RunLog, writes reports/run_log.json with
stamped ts, and prints an honest degradation summary.

Deep tools run as subprocesses (their per-process RunLogs cannot be threaded
in-process), so analyze.py does the honest aggregation itself: it runs
modelsource.detect(path, log=rl) in-process per model and drives iter_weights
(with a reject-all want filter, so no heavy decode) so arch-level degradation
warnings (arch_unmapped, inventory_only, …) funnel into the ONE run log.

These tests exercise that aggregation end-to-end (subprocess invocation of
analyze.py, reading the produced run_log.json) AND the extracted summarize()
helper directly. numpy is needed only to write the safetensors fixtures.
"""
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(__file__)
TOOLS = os.path.join(HERE, "..", "tools")
sys.path.insert(0, TOOLS)
sys.path.insert(0, HERE)

try:
    import numpy as np  # noqa: F401
    HAVE_NP = True
except Exception:
    HAVE_NP = False

if HAVE_NP:
    from st_fixture import write_model


def _unmappable_cfg():
    # model_type that archmap.family_for() returns None for -> inventory-only,
    # and iter_weights() emits exactly one arch_unmapped warn.
    return {"architectures": ["MambaXyzForCausalLM"], "model_type": "mamba-xyz",
            "num_hidden_layers": 1, "hidden_size": 4}


def _tensors():
    # One tiny F32 tensor; never decoded under the reject-all want filter.
    arr = np.zeros((2, 2), dtype="<f4")
    return {"model.embed_tokens.weight": ("F32", [2, 2], arr)}


def _mappable_cfg():
    # A MAPPABLE llama config: archmap.family_for() resolves the standard
    # tensors, so precision is "exact" and NO arch_unmapped fires. The rotary
    # buffer below is the one tensor that has no canonical role.
    return {"model_type": "llama", "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 1, "num_attention_heads": 4,
            "hidden_size": 4, "tie_word_embeddings": False}


def _mappable_tensors():
    # One mappable weight + one UNMAPPABLE tensor (rotary buffer → no canonical
    # role → archmap.resolve() returns None → tensor_unmapped during
    # iter_weights, emitted by a DEEP-TOOL SUBPROCESS).
    return {
        "model.embed_tokens.weight": ("F32", [8, 4],
                                      np.arange(32, dtype="<f4").reshape(8, 4)),
        "model.layers.0.self_attn.q_proj.weight": (
            "F32", [4, 4], np.arange(16, dtype="<f4").reshape(4, 4)),
        "model.layers.0.self_attn.rotary_emb.inv_freq": (
            "F32", [2], np.arange(2, dtype="<f4")),
    }


@unittest.skipUnless(HAVE_NP, "numpy not installed")
class TestAnalyzeRunLogEndToEnd(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = self._td.name
        self.model_dir = os.path.join(self.root, "mamba_model")
        write_model(self.model_dir, _unmappable_cfg(), _tensors())
        self.out = os.path.join(self.root, "report")

    def tearDown(self):
        self._td.cleanup()

    def _run_analyze(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = TOOLS + os.pathsep + env.get("PYTHONPATH", "")
        cp = subprocess.run(
            [sys.executable, os.path.join(TOOLS, "analyze.py"),
             "--model", self.model_dir, "--deep",
             "--no-dashboard", "--out", self.out],
            capture_output=True, text=True, env=env)
        return cp

    def test_run_log_written_and_arch_unmapped_once(self):
        cp = self._run_analyze()
        self.assertEqual(cp.returncode, 0, msg=cp.stderr)

        log_path = os.path.join(self.out, "run_log.json")
        self.assertTrue(os.path.exists(log_path), "run_log.json not written")

        with open(log_path, encoding="utf-8") as fh:
            entries = json.load(fh)
        self.assertIsInstance(entries, list)
        self.assertTrue(entries, "run_log.json is empty")

        # EXACTLY one arch_unmapped for the mamba model.
        unmapped = [e for e in entries if e["code"] == "arch_unmapped"]
        self.assertEqual(len(unmapped), 1, f"expected one arch_unmapped, got {unmapped}")
        self.assertEqual(unmapped[0]["model"], "mamba_model")

        # Same message also went to stderr (no swallowing).
        self.assertIn("arch_unmapped", cp.stderr)

        # Every entry carries a non-empty ts (stamped by the app entry point).
        for e in entries:
            self.assertTrue(e["ts"], f"entry has empty ts: {e}")

    def test_summary_line_counts(self):
        cp = self._run_analyze()
        self.assertEqual(cp.returncode, 0, msg=cp.stderr)
        # The single mamba model is inventory-only, zero errors.
        # Summary format: "N ok (exact/approx) · M inventory-only · K errors"
        out = cp.stdout + cp.stderr
        self.assertIn("inventory-only", out)
        self.assertIn("mamba_model", out)

    def test_arch_unmapped_not_duplicated(self):
        """The merged deep-tool subprocess logs AND aggregate() must not BOTH
        record arch_unmapped for the same model — exactly one entry survives."""
        cp = self._run_analyze()
        self.assertEqual(cp.returncode, 0, msg=cp.stderr)
        with open(os.path.join(self.out, "run_log.json"), encoding="utf-8") as fh:
            entries = json.load(fh)
        unmapped = [e for e in entries if e["code"] == "arch_unmapped"]
        self.assertEqual(len(unmapped), 1,
                         f"expected exactly one arch_unmapped, got {unmapped}")


@unittest.skipUnless(HAVE_NP, "numpy not installed")
class TestSubprocessPerTensorWarning(unittest.TestCase):
    """A per-tensor warning (tensor_unmapped) raised INSIDE a deep-tool
    subprocess must reach the authoritative run_log.json — proving analyze.py
    captures and merges each subprocess's own RunLog."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = self._td.name
        self.model_dir = os.path.join(self.root, "llama_model")
        write_model(self.model_dir, _mappable_cfg(), _mappable_tensors())
        self.out = os.path.join(self.root, "report")

    def tearDown(self):
        self._td.cleanup()

    def test_subprocess_per_tensor_warning_reaches_run_log(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = TOOLS + os.pathsep + env.get("PYTHONPATH", "")
        cp = subprocess.run(
            [sys.executable, os.path.join(TOOLS, "analyze.py"),
             "--model", self.model_dir, "--deep",
             "--no-dashboard", "--out", self.out],
            capture_output=True, text=True, env=env)
        self.assertEqual(cp.returncode, 0, msg=cp.stderr)
        with open(os.path.join(self.out, "run_log.json"), encoding="utf-8") as fh:
            entries = json.load(fh)
        tu = [e for e in entries if e["code"] == "tensor_unmapped"]
        self.assertTrue(
            tu, "tensor_unmapped from a deep-tool subprocess did not reach "
                f"run_log.json; entries={entries}")
        self.assertTrue(any("rotary_emb.inv_freq" in e["msg"] for e in tu))
        # The mappable model is exact -> no arch_unmapped at all.
        self.assertFalse([e for e in entries if e["code"] == "arch_unmapped"])
        for e in entries:
            self.assertTrue(e["ts"], f"entry has empty ts: {e}")


@unittest.skipUnless(HAVE_NP, "numpy not installed")
class TestDashboardSeesRunLog(unittest.TestCase):
    """Integration: a full run WITH the dashboard built (no --no-dashboard) must
    embed the SAME run-log entries into dashboard.html that land in
    run_log.json. The bug: analyze.py built the dashboard BEFORE writing
    run_log.json (and without --run-log), so the Log tab's RUNLOG island was
    empty even though run_log.json had degradation entries."""

    RUNLOG_RE = re.compile(r"const RUNLOG = (\[.*?\]);")

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = self._td.name
        self.model_dir = os.path.join(self.root, "mamba_model")
        write_model(self.model_dir, _unmappable_cfg(), _tensors())
        self.out = os.path.join(self.root, "report")

    def tearDown(self):
        self._td.cleanup()

    def _run_analyze_with_dashboard(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = TOOLS + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run(
            [sys.executable, os.path.join(TOOLS, "analyze.py"),
             "--model", self.model_dir, "--deep",
             "--out", self.out],  # NO --no-dashboard: dashboard IS built
            capture_output=True, text=True, env=env)

    def test_dashboard_runlog_equals_run_log_json(self):
        cp = self._run_analyze_with_dashboard()
        self.assertEqual(cp.returncode, 0, msg=cp.stderr)

        # run_log.json exists, is non-empty and carries the degradation entry.
        log_path = os.path.join(self.out, "run_log.json")
        self.assertTrue(os.path.exists(log_path), "run_log.json not written")
        with open(log_path, encoding="utf-8") as fh:
            entries = json.load(fh)
        self.assertTrue(entries, "run_log.json is empty (clobbered?)")
        self.assertTrue([e for e in entries if e["code"] == "arch_unmapped"],
                        f"expected arch_unmapped in run_log.json, got {entries}")

        # the dashboard was built and embeds the RUNLOG data island.
        dash_path = os.path.join(self.out, "dashboard.html")
        self.assertTrue(os.path.exists(dash_path), "dashboard.html not built")
        with open(dash_path, encoding="utf-8") as fh:
            html = fh.read()
        m = self.RUNLOG_RE.search(html)
        self.assertIsNotNone(m, "RUNLOG data island not found in dashboard.html")
        dash_runlog = json.loads(m.group(1))

        # the dashboard now SEES the same entries that are in run_log.json
        # (same count, same set) — proving the Log tab is populated.
        self.assertTrue(dash_runlog,
                        "dashboard RUNLOG is EMPTY while run_log.json is not "
                        "— Log tab would show 0 rows (the bug)")
        self.assertEqual(len(dash_runlog), len(entries),
                         f"dashboard RUNLOG count {len(dash_runlog)} != "
                         f"run_log.json count {len(entries)}")
        key = lambda e: (e["model"], e["stage"], e["severity"], e["code"], e["msg"])
        self.assertEqual({key(e) for e in dash_runlog},
                         {key(e) for e in entries},
                         "dashboard RUNLOG entries differ from run_log.json")


class TestSummarizeHelper(unittest.TestCase):
    """summarize() must be a pure function of model precisions + rl.counts()."""

    def _import(self):
        import importlib
        import analyze
        importlib.reload(analyze)
        return analyze

    def test_summary_format_and_offenders(self):
        analyze = self._import()
        from modelsource import RunLog
        rl = RunLog()
        rl.emit("error", model="boom", stage="weights", code="x", msg="m",
                stream=io.StringIO())
        models = [
            {"name": "good", "precision": "exact"},
            {"name": "approxy", "precision": "approx"},
            {"name": "invo", "precision": "inventory-only"},
            {"name": "boom", "precision": None},   # failed to detect
        ]
        line = analyze.summarize(models, rl)
        # counts: 2 ok (exact+approx), 1 inventory-only, 1 error
        self.assertIn("2 ok", line)
        self.assertIn("1 inventory-only", line)
        self.assertIn("1 error", line)
        # offending names surfaced.
        self.assertIn("invo", line)
        self.assertIn("boom", line)

    def test_summarize_counts_unknown(self):
        """A model whose path detect() can't recognize has precision=None and
        only a warn (unknown_format) — no rl error entry. summarize() must
        still count it under errors/unknown AND list its name (it must not
        vanish)."""
        analyze = self._import()
        from modelsource import RunLog
        rl = RunLog()  # no error entries: the unknown model only warned
        models = [
            {"name": "good", "precision": "exact"},
            {"name": "ghost", "precision": None},   # unrecognized path
        ]
        line = analyze.summarize(models, rl)
        self.assertIn("1 ok", line)
        self.assertIn("1 error", line)   # the unknown model counts as one
        self.assertIn("ghost", line)     # and its name is surfaced

    def test_clean_run_zero_errors(self):
        analyze = self._import()
        from modelsource import RunLog
        rl = RunLog()
        models = [{"name": "a", "precision": "exact"},
                  {"name": "b", "precision": "approx"}]
        line = analyze.summarize(models, rl)
        self.assertIn("2 ok", line)
        self.assertIn("0 inventory-only", line)
        self.assertIn("0 error", line)


class TestAppendSynthRobust(unittest.TestCase):
    """Fix 2 — the models.json append loop must be per-model robust: if one
    model's synth_hf_entry raises, the others are still appended AND a single
    ``synth_failed`` error is logged (never abort the whole append)."""

    def _import(self):
        import importlib
        import analyze
        importlib.reload(analyze)
        return analyze

    def test_one_raise_does_not_abort_others(self):
        analyze = self._import()
        from modelsource import RunLog
        rl = RunLog()

        def fake_synth(path, rl=None):
            name = os.path.basename(str(path))
            if name == "boom":
                raise RuntimeError("kaboom in metadata()")
            return {"name": name, "label": name, "source": {"format": "x"}}

        orig = analyze.synth_hf_entry
        analyze.synth_hf_entry = fake_synth
        try:
            fleet = []
            analyze._append_synth_entries(["/m/boom", "/m/good"], fleet, rl)
        finally:
            analyze.synth_hf_entry = orig

        # the good model still made it in despite the first one raising
        names = [e["name"] for e in fleet]
        self.assertEqual(names, ["good"])
        # exactly one synth_failed error, naming the offending model
        failed = [e for e in rl.entries
                  if e["code"] == "synth_failed" and e["severity"] == "error"]
        self.assertEqual(len(failed), 1, f"expected one synth_failed, got {rl.entries}")
        self.assertEqual(failed[0]["model"], "boom")
        self.assertEqual(failed[0]["stage"], "fleet")
        self.assertIn("kaboom", failed[0]["msg"])


class TestMergeRunLog(unittest.TestCase):
    """Fix 7 (coverage) — unit-test _merge_run_log/_entry_key directly.

    _merge_run_log reads a deep tool's written run-log JSON from a path, merges
    NEW entries into rl (de-duped on _entry_key, which ignores ts), and deletes
    the file. We write a temp JSON and assert the merge semantics.
    """

    def _import(self):
        import importlib
        import analyze
        importlib.reload(analyze)
        return analyze

    def _entry(self, code, ts="", msg="m", model="x", stage="weights",
               severity="warn"):
        return {"ts": ts, "model": model, "stage": stage,
                "severity": severity, "code": code, "msg": msg}

    def _write(self, entries):
        td = tempfile.mkdtemp()
        self.addCleanup(lambda: os.path.exists(p) and os.remove(p))
        self.addCleanup(lambda: os.rmdir(td) if os.path.isdir(td) else None)
        p = os.path.join(td, "log.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(entries, fh)
        return p

    def test_entries_differing_only_in_ts_collapse(self):
        analyze = self._import()
        from modelsource import RunLog
        rl = RunLog()
        rl.entries.append(self._entry("arch_unmapped", ts="2026-01-01"))
        # incoming differs ONLY in ts → must NOT be added a second time
        p = self._write([self._entry("arch_unmapped", ts="2026-05-30")])
        analyze._merge_run_log(rl, p)
        same = [e for e in rl.entries if e["code"] == "arch_unmapped"]
        self.assertEqual(len(same), 1)
        # file deleted by the merge
        self.assertFalse(os.path.exists(p))

    def test_distinct_entries_survive(self):
        analyze = self._import()
        from modelsource import RunLog
        rl = RunLog()
        rl.entries.append(self._entry("arch_unmapped"))
        p = self._write([
            self._entry("tensor_unmapped", msg="a"),
            self._entry("bad_offset", severity="error", msg="b"),
        ])
        analyze._merge_run_log(rl, p)
        codes = [e["code"] for e in rl.entries]
        self.assertEqual(codes, ["arch_unmapped", "tensor_unmapped", "bad_offset"])

    def test_insertion_and_merge_order_preserved(self):
        analyze = self._import()
        from modelsource import RunLog
        rl = RunLog()
        rl.entries.append(self._entry("a", msg="1"))
        p = self._write([
            self._entry("a", msg="1"),   # dup of existing → dropped
            self._entry("b", msg="2"),   # new → appended (order kept)
            self._entry("c", msg="3"),   # new → appended after b
            self._entry("b", msg="2"),   # dup of just-merged b → dropped
        ])
        analyze._merge_run_log(rl, p)
        self.assertEqual([(e["code"], e["msg"]) for e in rl.entries],
                         [("a", "1"), ("b", "2"), ("c", "3")])

    def test_entry_key_ignores_ts(self):
        analyze = self._import()
        a = self._entry("x", ts="t1")
        b = self._entry("x", ts="t2")
        self.assertEqual(analyze._entry_key(a), analyze._entry_key(b))
        c = self._entry("x", ts="t1", msg="different")
        self.assertNotEqual(analyze._entry_key(a), analyze._entry_key(c))


if __name__ == "__main__":
    unittest.main()
