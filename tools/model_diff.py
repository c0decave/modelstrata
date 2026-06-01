#!/usr/bin/env python3
"""
model_diff.py — Ebene 7 ★: statischer Per-Tensor-Diff zweier Modelle (kein Lauf).

Für jeden gemeinsamen Block-Gewichts-Tensor:
  - delta  = ‖W_b − W_a‖_F / ‖W_a‖_F   (relative Veränderung)
  - cosine = <W_a, W_b> / (‖W_a‖·‖W_b‖) (Richtungs-Ähnlichkeit)
Gruppiert nach (Komponente, Layer) -> zeigt, WO ein Finetune/eine Abliteration
das Modell verändert hat, ganz ohne Forward-Pass (intro.md §5.4).

Schreibt reports/diff.json: [{label, label_a, label_b, note, roles, layers,
metrics, cells, top}]. note kennzeichnet Misch-Quantisierung (dann ist delta
durch Quant-Rauschen verfälscht — exakt nur bei gleicher Precision / HF-Gewichten).

  ~/.venv/bin/python model_diff.py BASE.gguf FINETUNE.gguf -o diff.json
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modelsource import detect, RunLog

BLK = re.compile(r"^(?P<pre>(?:[a-z]+\.)*)blk\.(?P<layer>\d+)\.(?P<role>.+?)\.(?P<suf>weight|bias)$")
METRICS = ["delta", "cosine"]


def iter_weights(path, want=None, log=None):
    """Tensor source — routed through modelsource.detect(). Same tuples as the
    old gguf_weights.iter_weights; SOURCE is whatever detect() picks. Unknown
    path -> empty iterator (logged in detect()). Monkeypatchable in tests."""
    src = detect(path, log=log)
    if src is None:
        return iter(())
    return src.iter_weights(want)


def _source_block(source):
    """source.metadata()['source'] for a REAL Source (passed by the CLI: detect
    once per model), else None. Called AFTER iter_weights() so the badged block
    reflects per-tensor warnings recorded DURING iteration. With no real
    ``source`` (direct callers that stub iter_weights and pass a fake path)
    returns None — no detect() on a fake path (avoids a misleading
    unknown_format warn for nothing)."""
    return source.metadata().get("source") if source is not None else None


def rel_delta(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    na = np.linalg.norm(a)
    return float(np.linalg.norm(b - a) / (na + 1e-12))


def cosine(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (den + 1e-12))


def _tensors(path, source, log):
    """Tensor iterator for ``path`` — the passed Source (CLI: detect once) or
    the module-level iter_weights (monkeypatchable; routes through detect)."""
    return source.iter_weights() if source is not None else iter_weights(path)


def _block_weights(path, source=None, log=None):
    """name -> fp32 weight array for block tensors (2-D and 3-D MoE experts).
    NOTE: the base model's block weights are held fully resident in RAM (fp32)
    while the finetune is streamed — size accordingly for large models."""
    out = {}
    for name, arr, _ in _tensors(path, source, log):
        m = BLK.match(name)
        if m and m.group("suf") == "weight" and arr.ndim >= 2:
            out[name] = arr        # rel_delta/cosine ravel -> any shape works
    return out


def diff_models(path_a, path_b, label_a=None, label_b=None, note="",
                *, source_a=None, source_b=None, log=None):
    A = _block_weights(path_a, source_a, log)   # hold base in memory, stream B
    cells, roles, max_layer, top = {}, [], -1, []
    mism = 0
    for name, arr_b, _ in _tensors(path_b, source_b, log):
        if name not in A:
            continue
        arr_a = A[name]
        if arr_a.shape != arr_b.shape:
            mism += 1
            continue
        m = BLK.match(name)
        layer = int(m.group("layer")); role = m.group("pre") + m.group("role")
        d = round(rel_delta(arr_a, arr_b), 6)
        c = round(cosine(arr_a, arr_b), 6)
        max_layer = max(max_layer, layer)
        cells.setdefault(role, {})[layer] = {"delta": d, "cosine": c}
        if role not in roles:
            roles.append(role)
        top.append({"name": name, "delta": d, "cosine": c})
    top.sort(key=lambda x: -x["delta"])
    la = label_a or os.path.basename(path_a)
    lb = label_b or os.path.basename(path_b)
    return {
        "label": f"{la} → {lb}", "label_a": la, "label_b": lb, "note": note,
        "metrics": METRICS, "roles": roles,
        "layers": max_layer + 1 if max_layer >= 0 else 0,
        "cells": cells, "top": top[:15], "shape_mismatch": mism,
        "matched": len(top),
        # honest precision labels for BOTH sides (dashboard badges each).
        "source_a": _source_block(source_a),
        "source_b": _source_block(source_b),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("finetune")
    ap.add_argument("-o", "--out", default="diff.json")
    ap.add_argument("--label-a"); ap.add_argument("--label-b")
    ap.add_argument("--note", default="")
    ap.add_argument("--run-log", default=None,
                    help="also write this tool's RunLog (per-tensor degradation) "
                         "to PATH — written even on partial/failed runs so the "
                         "orchestrator can merge an authoritative log")
    args = ap.parse_args()
    log = RunLog()
    out = []
    try:
        print(f"  diff {args.base}  ->  {args.finetune}", file=sys.stderr)
        # Guard the single diff: a raise (corrupt input, missing gguf package)
        # must produce a structured, identifiable analyze_failed + an honest
        # empty result JSON, never a raw traceback with no output.
        try:
            src_a = detect(args.base, log=log)  # once per model: feeds source + tensors
            src_b = detect(args.finetune, log=log)
            d = diff_models(args.base, args.finetune, args.label_a, args.label_b, args.note,
                            source_a=src_a, source_b=src_b, log=log)
            out = [d]
            print(f"[diff: {args.out}  matched={d['matched']} mismatch={d['shape_mismatch']}]",
                  file=sys.stderr)
        except Exception as e:  # noqa: BLE001 — defense-in-depth on top of Fix 1
            log.emit("error",
                     model=f"{os.path.basename(args.base)}→{os.path.basename(args.finetune)}",
                     stage="model_diff", code="analyze_failed", msg=str(e))
        with open(args.out, "w") as f:
            json.dump(out, f, default=str)
    finally:
        if args.run_log:
            log.write(args.run_log)


if __name__ == "__main__":
    main()
