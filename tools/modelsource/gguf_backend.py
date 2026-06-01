"""GGUF backend for the uniform ``Source`` interface.

Thin wrapper around the existing GGUF tooling — it does NOT reimplement any
parsing or dequant logic:

  * ``metadata()`` delegates to ``gguf_inspect.parse_gguf`` (pure-stdlib,
    header-only) and stamps a ``source`` block onto the result.
  * ``iter_weights()`` delegates to ``gguf_weights.iter_weights`` (needs the
    ``gguf`` PyPI package at runtime). GGUF tensor names are ALREADY canonical
    (``blk.N.<role>.weight``), so this is a straight pass-through.
  * ``tokenizer()`` returns ``None`` here — GGUF tokenizer extraction stays in
    ``tokenizer_forensics``.
"""

import gguf_inspect
import gguf_weights

from modelsource.base import Source

# ggml tensor types that carry full (un-quantized) precision. Anything else
# present in a model's tensors means values are reconstructed approximately.
_EXACT_TYPES = frozenset({"F32", "F16", "BF16"})


class GGUFSource(Source):
    def __init__(self, path):
        self.path = path

    @staticmethod
    def _is_exact(meta):
        """True iff every tensor type in the model is un-quantized.

        ``gguf_inspect`` exposes the per-type tensor counts in
        ``meta["quant_breakdown"]`` (e.g. ``{"F32": 3, "Q6_K": 2}``). The model
        is "exact" only when all of those types are F32/F16/BF16; any K-quant /
        Q* / IQ* type present makes it "approx".
        """
        breakdown = meta.get("quant_breakdown") or {}
        return all(t in _EXACT_TYPES for t in breakdown)

    def metadata(self):
        meta = gguf_inspect.parse_gguf(self.path)
        arch = meta.get("metadata", {}).get("general.architecture")
        meta["source"] = self.source_block(
            fmt="gguf",
            precision="exact" if self._is_exact(meta) else "approx",
            arch=arch,
            mapped=True,                 # GGUF names are already canonical
            warnings=[],
        )
        return meta

    def iter_weights(self, want=None):
        # names already canonical (blk.N.<role>.weight) — pass through unchanged
        return gguf_weights.iter_weights(self.path, want)

    def tokenizer(self):
        # GGUF tokenizer extraction lives in tokenizer_forensics, not here.
        return None
