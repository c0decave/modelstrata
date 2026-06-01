#!/usr/bin/env python3
"""
weight_stats.py — Ebene 4: statische Gewichts-Statistik (dequantisiert, kein Lauf).

Pro Block-Tensor: mean/std/min/max, L2-Norm, Sparsity, Kurtosis und (für 2D)
Outlier-Channel-Anteil. Gruppiert nach (Komponente, Layer) wie die Quant-Heatmap.
Schreibt reports/weight_stats.json: Liste von {label, arch, layers, metrics, cells}.

  ~/.venv/bin/python weight_stats.py MODELL.gguf [MODELL2.gguf ...] -o out.json

Hinweis: Werte sind aus dequantisierten (Q*_K) Gewichten -> "approx". Exakt nur
auf un-quantisierten HF-Gewichten.
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


def iter_weights(path, want=None, log=None):
    """Tensor source for this tool — routed through modelsource.detect().

    Yields ``(canonical_name, fp32 ndarray, dtype_label)`` exactly like the old
    gguf_weights.iter_weights, but the SOURCE is now whatever detect() picks
    (GGUF / HF safetensors / inventory). detect() returning None (unknown path)
    degrades to an empty iterator — logged inside detect(), never a crash.
    Tests may monkeypatch this symbol to inject fixed tensors."""
    src = detect(path, log=log)
    if src is None:
        return iter(())
    return src.iter_weights(want)

# metrics exposed in the dashboard heatmap selector
METRICS = ["std", "l2", "sparsity", "kurtosis", "outlier"]


def tensor_stats(a):
    """Pure numpy: summary stats for a weight tensor (any shape)."""
    flat = a.reshape(-1).astype(np.float64)
    n = flat.size
    if n == 0:
        return {m: 0.0 for m in ("mean", "std", "min", "max", "l2",
                                 "sparsity", "kurtosis", "outlier")} | {"n": 0}
    mean = float(flat.mean())
    std = float(flat.std())
    out = {
        "n": int(n),
        "mean": mean,
        "std": std,
        "min": float(flat.min()),
        "max": float(flat.max()),
        "l2": float(np.linalg.norm(flat)),
        "sparsity": float(np.mean(np.abs(flat) < 1e-6)),
        "kurtosis": float(np.mean(((flat - mean) / std) ** 4) - 3.0) if std > 0 else 0.0,
        "outlier": outlier_channel_frac(a),
    }
    return out


def outlier_channel_frac(a, k=6.0):
    """Fraction of input-channels (columns) whose L2 norm exceeds k×median.
    Heuristic for the 'massive activations' precursor. 0.0 for non-2D tensors."""
    if a.ndim != 2:
        return 0.0
    col = np.linalg.norm(a.astype(np.float64), axis=0)
    med = np.median(col)
    if med <= 0:
        return 0.0
    return float(np.mean(col > k * med))


def _tensors_for(path, source):
    """Tensor iterator for ``path``.

    ``source`` (a modelsource.Source) is passed by the CLI so detect() runs
    ONCE per model and feeds BOTH iter_weights() AND the source block. When
    omitted (unit tests calling analyze_model directly with a stubbed,
    monkeypatched module-level ``iter_weights`` and a fake path), tensors come
    from that module-level wrapper — and no detect() is run on the fake path."""
    return source.iter_weights() if source is not None else iter_weights(path)


def _source_block(source):
    """source.metadata()['source'] for a REAL Source, else None.

    Called AFTER iter_weights() so the badged block reflects per-tensor
    warnings recorded DURING iteration (tensor_unmapped, dtype_unsupported,
    bad_offset, shard_missing, no_output_weight). With no real ``source``
    (stubbed unit-test seam, fake path) returns None — no detect() on a fake
    path (that would emit a misleading unknown_format warn for nothing)."""
    return source.metadata().get("source") if source is not None else None


def analyze_model(path, label=None, *, source=None, log=None):
    cells = {}          # role -> {layer -> {metric: value}}
    roles = []
    max_layer = -1
    for name, arr, _ in _tensors_for(path, source):
        m = BLK.match(name)
        if not m:
            continue
        if m.group("suf") != "weight":   # 1-D biases excluded (match heatmap)
            continue
        layer = int(m.group("layer"))
        role = m.group("pre") + m.group("role")
        max_layer = max(max_layer, layer)
        st = tensor_stats(arr)
        cells.setdefault(role, {})[layer] = {k: round(st[k], 6) for k in METRICS}
        if role not in roles:
            roles.append(role)
    return {
        "label": label or os.path.basename(path),
        "path": path,
        "metrics": METRICS,
        "roles": roles,
        "layers": max_layer + 1 if max_layer >= 0 else 0,
        "cells": cells,
        # Captured AFTER the loop so iteration-time warnings are reflected.
        "source": _source_block(source),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="GGUF file(s)")
    ap.add_argument("-o", "--out", default="weight_stats.json")
    ap.add_argument("--label", action="append", default=[],
                    help="explicit label per model (same order)")
    ap.add_argument("--run-log", default=None,
                    help="also write this tool's RunLog (per-tensor degradation) "
                         "to PATH — written even on partial/failed runs so the "
                         "orchestrator can merge an authoritative log")
    args = ap.parse_args()
    log = RunLog()
    out = []
    try:
        for i, p in enumerate(args.models):
            label = args.label[i] if i < len(args.label) else None
            print(f"  analyzing {p} ...", file=sys.stderr)
            # Per-model guard: one model raising (corrupt input, missing gguf
            # package, etc.) must NOT abort the batch or drop the other models'
            # output. Log a structured, identifiable analyze_failed and skip
            # just this model.
            try:
                src = detect(p, log=log)    # once per model: feeds source + tensors
                out.append(analyze_model(p, label, source=src, log=log))
            except Exception as e:  # noqa: BLE001 — defense-in-depth on top of Fix 1
                log.emit("error", model=os.path.basename(p), stage="weight_stats",
                         code="analyze_failed", msg=str(e))
                continue
        with open(args.out, "w") as f:
            json.dump(out, f, default=str)
        print(f"[weight_stats: {args.out}  {len(out)} models]", file=sys.stderr)
    finally:
        if args.run_log:
            log.write(args.run_log)


if __name__ == "__main__":
    main()
