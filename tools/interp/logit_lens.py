"""Numerical helpers for logit-lens style projections."""

import numpy as np


def softmax(logits):
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def project(hidden, unembedding, bias=None):
    """Project hidden states to vocabulary logits.

    ``hidden`` shape: ``(..., d_model)``. ``unembedding`` shape:
    ``(vocab, d_model)``. This matches common LM-head storage and avoids any
    framework dependency.
    """
    h = np.asarray(hidden, dtype=np.float64)
    w = np.asarray(unembedding, dtype=np.float64)
    if h.shape[-1] != w.shape[-1]:
        raise ValueError(f"hidden dim {h.shape[-1]} != unembedding dim {w.shape[-1]}")
    logits = h @ w.T
    if bias is not None:
        logits = logits + np.asarray(bias, dtype=np.float64)
    return logits


def topk(logits, tokens=None, k=5):
    """Return top-k token ids/tokens and probabilities for one logit vector."""
    if k <= 0:
        raise ValueError("k must be positive")
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("topk expects a single 1-D logit vector")
    k = min(k, x.size)
    probs = softmax(x)
    idx = np.argpartition(-probs, k - 1)[:k]
    idx = idx[np.argsort(-probs[idx])]
    return [
        {"id": int(i), "token": tokens[i] if tokens is not None and i < len(tokens) else None,
         "prob": float(probs[i]), "logit": float(x[i])}
        for i in idx
    ]

