#!/usr/bin/env python3
"""
embedding_geometry.py — Ebene 6: statische Embedding-Geometrie (kein Lauf).

Die Token-Embedding-Matrix ist statisch. Daraus:
  - Per-Token-L2-Norm-Histogramm + Low-Norm-Tokens (ECHTE Glitch-Token-Kandidaten,
    à la SolidGoldMagikarp — Tokens mit ~0-Norm-Embedding können untertrainiert/selten gesehen sein)
  - Anisotropie (mittlerer Cosine zufälliger Token-Paare)
  - 2D-PCA einer Token-Stichprobe (für einen Scatter; hover = Token)

Schreibt reports/embedding.json. Werte aus dequantisiertem token_embd -> "approx".

  ~/.venv/bin/python embedding_geometry.py MODELL.gguf -o embedding.json [--sample 4000]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modelsource import detect, RunLog
from tokenizer_forensics import read_tokenizer, decode_token


def load_weight(path, name, log=None):
    """Load a single canonical tensor as (fp32 ndarray, dtype_label) via
    modelsource.detect(), or (None, None). Replaces the old direct
    gguf_weights.load_weight; SOURCE is whatever detect() picks. Unknown path
    -> (None, None) (logged in detect()). Monkeypatchable in tests."""
    src = detect(path, log=log)
    if src is None:
        return None, None
    for _n, arr, dt in src.iter_weights(lambda x: x == name):
        return arr, dt
    return None, None


def _source_block(source):
    """source.metadata()['source'] for a REAL Source (passed by the CLI: detect
    once per model), else None. With no real ``source`` (direct unit-test
    callers stub the module-level load_weight and pass a fake path) returns None
    — no detect() on a fake path (avoids a misleading unknown_format warn)."""
    return source.metadata().get("source") if source is not None else None


def pca2(X):
    """Project rows of X onto their top-2 principal components.
    Returns (coords Nx2, explained_variance_ratio[2])."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu
    # economy SVD; columns of Vt are principal axes
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt[:2].T
    # degenerate input (1 row / 1 feature) yields <2 components -> pad to 2 cols
    if coords.shape[1] < 2:
        coords = np.column_stack([coords, np.zeros(len(coords))])
    var = S ** 2
    evr = (var[:2] / var.sum()).tolist() if var.sum() > 0 else []
    evr = (evr + [0.0, 0.0])[:2]
    return coords, evr


def anisotropy(X, n_pairs=20000, seed=0):
    """Mean cosine similarity of random token pairs. High = embeddings clump in
    a narrow cone (anisotropic); near 0 = well spread."""
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1)
    ok = norms > 0
    idx = np.where(ok)[0]
    if idx.size < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    a = rng.choice(idx, n_pairs)
    b = rng.choice(idx, n_pairs)
    Xa, Xb = X[a], X[b]
    cos = (Xa * Xb).sum(1) / (np.linalg.norm(Xa, axis=1) * np.linalg.norm(Xb, axis=1))
    return float(np.mean(cos))


def norm_histogram(norms, bins=40):
    counts, edges = np.histogram(norms, bins=bins)
    return {"counts": counts.tolist(), "edges": [round(float(e), 4) for e in edges]}


DEFAULT_SEEDS = ("sql", "cwe", "exploit", "payload", "refusal", "system", "tool", "password")


def _cosine_rows(a, b, eps=1e-12):
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return np.sum(a * b, axis=1) / np.maximum(na * nb, eps)


def _head_compare(emb, out):
    if out is None or emb.shape != out.shape:
        return None
    e = emb.astype(np.float64)
    o = out.astype(np.float64)
    cos = _cosine_rows(e, o)
    return {
        "mean_row_cosine": round(float(np.mean(cos)), 6),
        "min_row_cosine": round(float(np.min(cos)), 6),
        "delta": round(float(np.linalg.norm(o - e) / (np.linalg.norm(e) + 1e-12)), 6),
    }


def _seed_neighbors(emb, toks, model, seeds=DEFAULT_SEEDS, topk=8):
    if not toks:
        return []
    decoded = [decode_token(str(t), model) for t in toks[:emb.shape[0]]]
    norm = np.linalg.norm(emb.astype(np.float64), axis=1)
    ok = norm > 0
    X = emb.astype(np.float64)
    rows = []
    for seed in seeds:
        sid = None
        needle = seed.lower()
        for i, tok in enumerate(decoded):
            if tok.strip().lower() == needle:
                sid = i
                break
        if sid is None:
            for i, tok in enumerate(decoded):
                if needle in tok.lower():
                    sid = i
                    break
        if sid is None or sid >= len(ok) or not ok[sid]:
            continue
        sims = (X @ X[sid]) / np.maximum(norm * norm[sid], 1e-12)
        order = np.argsort(-sims)
        neigh = []
        for j in order:
            if j == sid or not ok[j]:
                continue
            neigh.append({"id": int(j), "tok": decoded[int(j)], "cosine": round(float(sims[j]), 5)})
            if len(neigh) >= topk:
                break
        rows.append({"seed": seed, "id": int(sid), "tok": decoded[sid], "neighbors": neigh})
    return rows


def analyze_model(path, label=None, sample=4000, seed=0, *, source=None, log=None):
    # Tensor source routed through modelsource. The CLI passes a detected
    # ``source`` so detect() runs once per model and feeds BOTH the embedding
    # tensor AND the source block. Without a real ``source`` (direct callers,
    # incl. unit tests that monkeypatch the module-level load_weight) tensors
    # come from that module-level loader and the source block is None — no
    # detect() on a fake path.
    if source is not None:
        emb = out_head = None
        for n, arr, _d in source.iter_weights(lambda x: x in ("token_embd.weight", "output.weight")):
            if n == "token_embd.weight":
                emb = arr
            elif n == "output.weight":
                out_head = arr
    else:
        emb, _ = load_weight(path, "token_embd.weight")
        out_head, _ = load_weight(path, "output.weight")
    if emb is None:
        # No embedding tensor (inventory-only model, or an unmapped-arch HF that
        # yields no token_embd.weight). Match weight_stats/spectral: DEGRADE
        # gracefully — log it and return a minimal honest entry — instead of
        # raising, which would abort the whole batch in main() and drop the
        # output of the VALID models alongside it. No fabricated embedding data.
        name = label or os.path.basename(path)
        if log is not None:
            log.emit("warn", model=name, stage="weights", code="no_embedding",
                     msg="no token_embd.weight → embedding analysis skipped")
        return {
            "label": name, "path": path,
            "vocab": None, "dim": None,
            "norm_mean": None, "norm_min": None, "norm_max": None,
            "near_zero": None, "anisotropy": None,
            "histogram": None, "low_norm": [], "pca": None,
            "seed_neighbors": [], "output_head_compare": None,
            "source": _source_block(source),
        }
    # logical shape is [vocab, dim]
    vocab, dim = emb.shape
    norms = np.linalg.norm(emb.astype(np.float64), axis=1)

    # Vocab: a real Source exposes tokenizer() (None ⇒ no/forensics-skipped
    # vocab, surfaced inside the source — token strings degrade to "#id", not a
    # crash). Only the stubbed/bare-path branch falls back to the GGUF reader.
    if source is not None:
        tk = source.tokenizer() or {}
    else:
        tk = read_tokenizer(path)
    toks = tk.get("tokens") or []
    model = tk.get("tokenizer.ggml.model", "")

    def tokstr(i):
        return decode_token(str(toks[i]), model) if i < len(toks) else f"#{i}"

    # low-norm (glitch) candidates: smallest norms
    order = np.argsort(norms)
    low = [{"id": int(i), "tok": tokstr(int(i)), "norm": round(float(norms[i]), 5)}
           for i in order[:40]]

    # 2D PCA of an evenly-spaced sample (deterministic)
    step = max(1, vocab // sample)
    sidx = np.arange(0, vocab, step)[:sample]
    coords, evr = pca2(emb[sidx])
    pts = [{"x": round(float(coords[j, 0]), 3), "y": round(float(coords[j, 1]), 3),
            "tok": tokstr(int(sidx[j])), "norm": round(float(norms[sidx[j]]), 3)}
           for j in range(len(sidx))]

    return {
        "label": label or os.path.basename(path),
        "path": path, "vocab": int(vocab), "dim": int(dim),
        "norm_mean": round(float(norms.mean()), 5),
        "norm_min": round(float(norms.min()), 5),
        "norm_max": round(float(norms.max()), 5),
        "near_zero": int((norms < 1e-3).sum()),
        "anisotropy": round(anisotropy(emb, seed=seed), 4),
        "histogram": norm_histogram(norms),
        "low_norm": low,
        "pca": {"evr": [round(e, 4) for e in evr], "points": pts},
        "seed_neighbors": _seed_neighbors(emb, toks, model),
        "output_head_compare": _head_compare(emb, out_head),
        "source": _source_block(source),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("-o", "--out", default="embedding.json")
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--run-log", default=None,
                    help="also write this tool's RunLog (per-tensor degradation) "
                         "to PATH — written even on partial/failed runs so the "
                         "orchestrator can merge an authoritative log")
    args = ap.parse_args()
    log = RunLog()
    out = []
    try:
        for i, p in enumerate(args.models):
            print(f"  embedding {p} ...", file=sys.stderr)
            # Per-model guard: one model raising must not abort the batch nor
            # drop the others' output. Structured, identifiable analyze_failed.
            try:
                src = detect(p, log=log)    # once per model: feeds source + tensors
                out.append(analyze_model(p, args.label[i] if i < len(args.label) else None,
                                         args.sample, source=src, log=log))
            except Exception as e:  # noqa: BLE001 — defense-in-depth on top of Fix 1
                log.emit("error", model=os.path.basename(p), stage="embedding",
                         code="analyze_failed", msg=str(e))
                continue
        with open(args.out, "w") as f:
            json.dump(out, f, default=str)
        print(f"[embedding: {args.out}  {len(out)} models]", file=sys.stderr)
    finally:
        if args.run_log:
            log.write(args.run_log)


if __name__ == "__main__":
    main()
