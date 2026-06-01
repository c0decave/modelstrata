"""Task 16 — Log tab in build_dashboard.py.

The dashboard gains a SIXTH "Log" tab that renders reports/run_log.json
(authoritative structured degradation list: [{ts,model,stage,severity,code,msg}]).
These tests assert the tab is ADDITIVE (5 originals untouched), renders all
entries, shows data-integrity-correct counts, and a warn+error badge — and that
a build WITHOUT a run_log still produces a valid dashboard (empty state, no crash).
"""
import os
import sys
import unittest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

import build_dashboard as bd


def tensor(name, dims, ttype, params):
    return {"name": name, "dims": dims, "type": ttype, "type_id": 0,
            "offset": 0, "params": params}


def base_info(meta=None, tensors=None, **kw):
    info = {
        "path": "/x/m.gguf", "label": "m.gguf", "file_size": 1000,
        "n_tensors": 0, "total_params": 1000, "gguf_version": 3,
        "quant_breakdown": {"Q6_K": 1}, "header_hex": "47475546",
        "header_ascii": "GGUF", "header_end": 100, "alignment": 32,
        "data_start": 128, "data_bytes": 872, "bits_per_weight": 6.5,
        "file_type_label": "Q6_K",
        "metadata": meta or {}, "tensors": tensors or [],
    }
    info.update(kw)
    return info


def _model(name, arch="qwen2"):
    meta = {"general.architecture": arch, "general.name": name}
    return bd.derive(base_info(meta, [tensor("token_embd.weight", [4, 4], "F32", 16)]))


def _run_log():
    # 1 info + 1 warn + 1 error across 2 models (modelA, modelB) + one "*"-scoped.
    return [
        {"ts": "2026-05-30T10:00:00", "model": "modelA", "stage": "detect",
         "severity": "info", "code": "ok_detect", "msg": "detected llama family"},
        {"ts": "2026-05-30T10:01:00", "model": "modelB", "stage": "weights",
         "severity": "warn", "code": "arch_unmapped", "msg": "rotary buffer unmapped"},
        {"ts": "2026-05-30T10:02:00", "model": "modelA", "stage": "tokenizer",
         "severity": "error", "code": "tok_missing", "msg": "tokenizer.json absent"},
    ]


class TestLogTab(unittest.TestCase):
    def _build(self, run_log=None):
        models = [_model("modelA"), _model("modelB", arch="llama")]
        return bd.render(models, run_log=run_log)

    # --- 5 existing tabs must remain present ---------------------------------
    EXISTING_TABS = ["Flotte", "Tokenizer", "Gewichte",
                     "Embedding &amp; Diff", "Glossar"]

    def test_five_existing_tabs_present(self):
        html = self._build(_run_log())
        for label in self.EXISTING_TABS:
            self.assertIn(f">{label}</div>", html,
                          f"existing tab {label!r} missing")

    def test_log_tab_button_and_panel_present(self):
        html = self._build(_run_log())
        # a tab button with data-tab="log"
        self.assertIn('data-tab="log"', html)
        # a tabpane with data-tab="log"
        self.assertIn('class="tabpane" data-tab="log"', html)

    def test_all_entries_rendered(self):
        html = self._build(_run_log())
        for e in _run_log():
            self.assertIn(e["model"], html)
            self.assertIn(e["stage"], html)
            self.assertIn(e["code"], html)
            self.assertIn(e["msg"], html)

    def test_counts_data_integrity(self):
        rl = _run_log()
        # independent count of the source — the rendered header MUST equal this
        info = sum(1 for e in rl if e["severity"] == "info")
        warn = sum(1 for e in rl if e["severity"] == "warn")
        err = sum(1 for e in rl if e["severity"] == "error")
        self.assertEqual((info, warn, err), (1, 1, 1))
        html = self._build(rl)
        # counts header: "1 info · 1 warn · 1 error" (the data island carries the
        # source array; the JS computes counts from it — assert the source counts
        # are present in the embedded data and the literal header template exists)
        self.assertIn("RUNLOG", html)  # data island name
        # the embedded run-log JSON carries exactly the 3 entries / severities
        import json
        marker = "const RUNLOG = "
        start = html.index(marker) + len(marker)
        end = html.index("];", start) + 1
        embedded = json.loads(html[start:end]
                              .replace("\\u003c", "<").replace("\\u003e", ">"))
        ei = sum(1 for e in embedded if e["severity"] == "info")
        ew = sum(1 for e in embedded if e["severity"] == "warn")
        ee = sum(1 for e in embedded if e["severity"] == "error")
        self.assertEqual((ei, ew, ee), (info, warn, err),
                         "rendered/embedded counts must equal source counts")

    def test_tab_badge_reflects_warn_plus_error(self):
        # warn(1)+error(1) = 2 -> the Log tab button badge must surface "2"
        html = self._build(_run_log())
        # the badge count is computed client-side from RUNLOG; ensure the
        # warn+error total (2) is derivable and the badge element exists.
        self.assertIn('id="log-badge"', html)
        # data integrity: warn+error from the embedded source == 2
        import json
        marker = "const RUNLOG = "
        start = html.index(marker) + len(marker)
        end = html.index("];", start) + 1
        embedded = json.loads(html[start:end]
                              .replace("\\u003c", "<").replace("\\u003e", ">"))
        we = sum(1 for e in embedded if e["severity"] in ("warn", "error"))
        self.assertEqual(we, 2)

    # --- empty / missing run_log: still renders, no crash --------------------
    def test_no_run_log_still_valid(self):
        html = self._build(run_log=None)
        self.assertIn('data-tab="log"', html)
        self.assertIn("RUNLOG", html)
        # empty data island
        self.assertIn("const RUNLOG = [];", html)
        # 5 originals still present
        for label in self.EXISTING_TABS:
            self.assertIn(f">{label}</div>", html)

    def test_empty_run_log_no_crash(self):
        html = self._build(run_log=[])
        self.assertIn("const RUNLOG = [];", html)
        self.assertIn('data-tab="log"', html)

    # --- i18n: EN label registered for the new tab ---------------------------
    def test_log_tab_i18n_en(self):
        html = self._build(_run_log())
        self.assertIn('data-i18n="tab-log"', html)
        self.assertIn('"tab-log":"Log"', html)

    # --- render() back-compat: callable without run_log ----------------------
    def test_render_without_run_log_kwarg(self):
        models = [_model("modelA")]
        html = bd.render(models)  # no run_log kwarg at all
        self.assertIn("const RUNLOG = [];", html)


if __name__ == "__main__":
    unittest.main()
