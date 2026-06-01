"""Activation patching math helpers."""

import numpy as np


def patch(clean, corrupted, *, positions=None):
    """Return ``corrupted`` with selected token positions copied from ``clean``.

    Arrays are expected to have token position on axis 0. If ``positions`` is
    omitted, the whole activation tensor is copied.
    """
    c = np.asarray(clean)
    x = np.array(corrupted, copy=True)
    if c.shape != x.shape:
        raise ValueError(f"shape mismatch: clean {c.shape} vs corrupted {x.shape}")
    if positions is None:
        return np.array(c, copy=True)
    for p in positions:
        if p < 0:
            p += x.shape[0]
        if p < 0 or p >= x.shape[0]:
            raise IndexError(f"token position out of range: {p}")
        x[p] = c[p]
    return x


def relative_delta(a, b, eps=1e-12):
    """Relative Frobenius delta ``||b-a|| / max(||a||, eps)``."""
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        raise ValueError(f"shape mismatch: {aa.shape} vs {bb.shape}")
    return float(np.linalg.norm(bb - aa) / max(np.linalg.norm(aa), eps))
