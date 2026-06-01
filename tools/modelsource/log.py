"""Strukturierter Run-Logger. EINE Wahrheit: jeder Eintrag geht nach stderr
UND in den Puffer (→ reports/run_log.json → Dashboard-Log-Tab). Nichts wird
verschluckt; Fallbacks/Skips MÜSSEN hier landen."""
import json, sys

SEVERITIES = ("info", "warn", "error")

class RunLog:
    def __init__(self):
        self.entries = []
    def emit(self, severity, *, model, stage, code, msg, ts="", stream=None):
        if severity not in SEVERITIES:
            raise ValueError(f"unknown severity {severity!r} (expected one of {SEVERITIES})")
        e = {"ts": ts, "model": model, "stage": stage,
             "severity": severity, "code": code, "msg": msg}
        self.entries.append(e)
        (stream or sys.stderr).write(f"[{severity}] {model}/{stage} {code}: {msg}\n")
        return e
    def counts(self):
        return {s: sum(1 for e in self.entries if e["severity"] == s) for s in SEVERITIES}
    def to_json(self):
        return list(self.entries)
    def write(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.entries, fh, ensure_ascii=False, indent=2)
