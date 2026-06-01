"""Activation-cache helpers for phase-2 interpretability experiments.

This module deliberately avoids importing torch at module import time. The pure
Python helpers are enough for tests and for JSON-safe cache manifests; real
model hooks can pass tensors/arrays through ``as_array`` at the edge.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActivationKey:
    """Stable identifier for one cached activation tensor."""

    layer: int
    stream: str
    token: int | None = None

    def label(self):
        base = f"layer.{self.layer}.{self.stream}"
        return base if self.token is None else f"{base}.tok{self.token}"


def as_array(x):
    """Return ``x`` as a detached numpy array.

    Torch tensors are supported if torch is installed, but torch is never
    imported here. Objects with ``detach``/``cpu`` methods are handled by duck
    typing; everything else goes through ``np.asarray``.
    """
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


class ActivationCache:
    """Tiny in-memory cache keyed by ``ActivationKey``.

    Values are copied on write so later model operations cannot mutate cached
    baselines under our feet.
    """

    def __init__(self):
        self._data = {}

    def put(self, key, value):
        if not isinstance(key, ActivationKey):
            raise TypeError("key must be ActivationKey")
        self._data[key] = np.array(as_array(value), copy=True)

    def get(self, key):
        return self._data[key]

    def keys(self):
        return sorted(self._data, key=lambda k: (k.layer, k.stream, -1 if k.token is None else k.token))

    def manifest(self):
        return [
            {"label": k.label(), "layer": k.layer, "stream": k.stream,
             "token": k.token, "shape": list(v.shape), "dtype": str(v.dtype)}
            for k, v in ((k, self._data[k]) for k in self.keys())
        ]


def token_slice(tokens, *, last=False, positions=None):
    """Resolve token positions for cache/patch operations.

    ``last=True`` selects the final token. ``positions`` selects explicit
    integer positions. Exactly one mode must be provided.
    """
    n = len(tokens)
    if last == (positions is not None):
        raise ValueError("choose exactly one of last=True or positions=[...]")
    if last:
        if n == 0:
            raise ValueError("cannot select last token from an empty sequence")
        return [n - 1]
    out = []
    for p in positions:
        if p < 0:
            p += n
        if p < 0 or p >= n:
            raise IndexError(f"token position out of range: {p}")
        out.append(p)
    return out

