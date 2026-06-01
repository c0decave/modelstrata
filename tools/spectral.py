#!/usr/bin/env python3
"""
spectral.py — Ebene 5: Spektral-Analyse pro Gewichtsmatrix (WeightWatcher-Stil).

Pro 2D-Gewicht: Singulärwerte -> Stable Rank, Spektral-Entropie (effektiver Rang)
und Heavy-Tail-Exponent alpha (Hill-Schätzer auf den Eigenwerten der Korrelations-
matrix). WeightWatcher/Martin & Mahoney interpretieren alpha datenfrei als
grobe Layer-Diagnose: etwa 2-6 ist ein plausibler Bereich für gut korrelierte
trainierte Layer, >6 ein Warnsignal. Diese Implementierung ist bewusst nur eine
einfache SVD/Hill-Variante, nicht die komplette WeightWatcher-Pipeline.
Gruppiert nach (Komponente, Layer).

Schreibt reports/spectral.json (gleiche Form wie weight_stats: label/roles/layers/
metrics/cells). Werte aus dequantisierten Gewichten -> "approx".

  ~/.venv/bin/python spectral.py MODELL.gguf -o spectral.json [--max-dim 8192]
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
METRICS = ["alpha", "stable_rank", "spectral_entropy"]


def iter_weights(path, want=None, log=None):
    """Tensor source — routed through modelsource.detect(). Same tuples as the
    old gguf_weights.iter_weights; SOURCE is whatever detect() picks. Unknown
    path -> empty iterator (logged in detect()). Monkeypatchable in tests."""
    src = detect(path, log=log)
    if src is None:
        return iter(())
    return src.iter_weights(want)


def _tensors_for(path, source):
    """Tensor iterator — the passed Source (CLI: detect once per model) or the
    module-level iter_weights (monkeypatchable in tests). With no real
    ``source`` (stubbed seam, fake path) no detect() is run on the fake path."""
    return source.iter_weights() if source is not None else iter_weights(path)


def _source_block(source):
    """source.metadata()['source'] for a REAL Source, else None. Called AFTER
    iter_weights() so the badged block reflects per-tensor warnings recorded
    DURING iteration (tensor_unmapped, dtype_unsupported, bad_offset,
    shard_missing, no_output_weight). No detect() on a fake path."""
    return source.metadata().get("source") if source is not None else None


def stable_rank(sv):
    """||W||_F^2 / sigma_max^2  ∈ [1, rank]. 1 = rank-1-artig, hoch = volle Nutzung."""
    sv = np.asarray(sv, dtype=np.float64)
    smax = sv.max()
    if smax <= 0:
        return 0.0
    return float((sv ** 2).sum() / (smax ** 2))


def spectral_entropy(sv):
    """Effektiver Rang via Shannon-Entropie der normierten Singulärwert-'Energie'."""
    sv = np.asarray(sv, dtype=np.float64)
    e = sv ** 2
    tot = e.sum()
    if tot <= 0:
        return 0.0
    p = e / tot
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))   # perplexity = effective rank


def alpha_hill(sv, tail_frac=0.5):
    """Heavy-Tail-Exponent der ESD (eigenvalues = sv^2) via Hill-Schätzer auf dem
    oberen Tail. Kleinere alpha = schwererer Tail."""
    eig = np.sort(np.asarray(sv, dtype=np.float64) ** 2)[::-1]
    eig = eig[eig > 0]
    if eig.size < 8:
        return 0.0
    k = max(4, int(eig.size * tail_frac))
    tail = eig[:k]
    xmin = tail[-1]
    if xmin <= 0:
        return 0.0
    logs = np.log(tail / xmin)
    s = logs.sum()
    if s <= 0:
        return 0.0
    return float(1.0 + k / s)


def matrix_spectral(a, max_dim=None):
    a = a.astype(np.float64)
    if a.ndim == 3:
        # MoE expert stack [n_expert, out, in]: metrics per expert, then averaged
        res = [matrix_spectral(a[i], max_dim) for i in range(a.shape[0])]
        res = [r for r in res if r]
        if not res:
            return None
        return {k: round(float(np.mean([r[k] for r in res])), 4) for k in res[0]}
    if a.ndim != 2:
        return None
    if max_dim and min(a.shape) > max_dim:
        return None                       # skip oversized matrices (CPU cost)
    sv = np.linalg.svd(a, compute_uv=False)
    return {
        "alpha": round(alpha_hill(sv), 4),
        "stable_rank": round(stable_rank(sv), 4),
        "spectral_entropy": round(spectral_entropy(sv), 4),
    }


def analyze_model(path, label=None, max_dim=None, *, source=None, log=None):
    cells, roles, max_layer = {}, [], -1
    for name, arr, _ in _tensors_for(path, source):
        m = BLK.match(name)
        if not m or m.group("suf") != "weight" or arr.ndim < 2:
            continue   # 2-D weights and 3-D MoE expert stacks (handled in matrix_spectral)
        layer = int(m.group("layer"))
        role = m.group("pre") + m.group("role")
        sp = matrix_spectral(arr, max_dim)
        if sp is None:
            continue
        max_layer = max(max_layer, layer)
        cells.setdefault(role, {})[layer] = sp
        if role not in roles:
            roles.append(role)
    return {"label": label or os.path.basename(path), "path": path,
            "metrics": METRICS, "roles": roles,
            "layers": max_layer + 1 if max_layer >= 0 else 0, "cells": cells,
            # Captured AFTER the loop so iteration-time warnings are reflected.
            "source": _source_block(source)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("-o", "--out", default="spectral.json")
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--max-dim", type=int, default=None,
                    help="skip matrices whose smaller dim exceeds this (CPU budget)")
    ap.add_argument("--run-log", default=None,
                    help="also write this tool's RunLog (per-tensor degradation) "
                         "to PATH — written even on partial/failed runs so the "
                         "orchestrator can merge an authoritative log")
    args = ap.parse_args()
    log = RunLog()
    out = []
    try:
        for i, p in enumerate(args.models):
            print(f"  spectral {p} ...", file=sys.stderr)
            # Per-model guard: one model raising must not abort the batch nor
            # drop the others' output. Structured, identifiable analyze_failed.
            try:
                src = detect(p, log=log)
                out.append(analyze_model(p, args.label[i] if i < len(args.label) else None,
                                         args.max_dim, source=src, log=log))
            except Exception as e:  # noqa: BLE001 — defense-in-depth on top of Fix 1
                log.emit("error", model=os.path.basename(p), stage="spectral",
                         code="analyze_failed", msg=str(e))
                continue
        with open(args.out, "w") as f:
            json.dump(out, f, default=str)
        print(f"[spectral: {args.out}  {len(out)} models]", file=sys.stderr)
    finally:
        if args.run_log:
            log.write(args.run_log)


if __name__ == "__main__":
    main()
