#!/usr/bin/env python3
"""
gguf_weights.py — Ebene-4–7-Fundament (Weg A): GGUF-Tensoren dequantisieren.

GGUFReader liefert für quantisierte Tensoren ROHE Bytes (uint8); diese müssen
über gguf.quants.dequantize zu fp32 entpackt werden. GGUF-Dims sind in ne-
Reihenfolge (umgekehrt zur logischen Form), daher reshape(reversed(shape)).

Braucht: numpy + gguf  (auf dem Host: ~/.venv).
Reines IO/Dequant — die Analyse-Mathematik liegt in weight_stats/spectral/...
"""
import numpy as np

try:
    import gguf
    from gguf.quants import dequantize
    _GGUF_IMPORT_ERROR = None
except Exception as _e:                 # surfaced, not swallowed
    gguf = None
    _GGUF_IMPORT_ERROR = repr(_e)


def _logical_shape(t):
    return tuple(int(x) for x in reversed(t.shape))


def iter_weights(path, want=None):
    """Yield (name, fp32 ndarray in logical shape, ggml_type_name).

    want: optional predicate(name) -> bool to select a subset.
    """
    if gguf is None:
        raise RuntimeError(
            "gguf package not available (install in the host venv): " + str(_GGUF_IMPORT_ERROR))
    reader = gguf.GGUFReader(path)
    for t in reader.tensors:
        if want is not None and not want(t.name):
            continue
        arr = dequantize(np.array(t.data), t.tensor_type)
        yield t.name, arr.reshape(_logical_shape(t)).astype(np.float32), t.tensor_type.name


def load_weight(path, name):
    """Return (fp32 ndarray, ggml_type_name) for a single tensor, or (None, None)."""
    for n, a, ty in iter_weights(path, lambda x: x == name):
        return a, ty
    return None, None
