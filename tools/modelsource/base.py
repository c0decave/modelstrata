"""Abstract base class for model-format backends.

Every format backend (GGUF, HF safetensors, inventory) implements ``Source``.
This module defines ONLY the interface plus the concrete ``source_block``
helper; no real backend logic lives here.
"""

import abc


class Source(abc.ABC):
    """Single interface every format backend implements."""

    @abc.abstractmethod
    def metadata(self):
        """Return a dict in the gguf_inspect schema plus a ``source`` block."""
        raise NotImplementedError

    @abc.abstractmethod
    def iter_weights(self, want=None):
        """Yield ``(canonical_name, fp32 ndarray, dtype_label)`` tuples."""
        raise NotImplementedError

    @abc.abstractmethod
    def tokenizer(self):
        """Return the canonical vocab dict, or ``None``."""
        raise NotImplementedError

    def source_block(self, *, fmt, precision, arch, mapped, warnings=None):
        """Build the per-model source descriptor stamped into metadata.

        ``list(warnings or [])`` makes a fresh copy so a caller's mutable list
        cannot later leak changes into the returned block.
        """
        return {
            "format": fmt,
            "precision": precision,
            "arch": arch,
            "mapped": mapped,
            "warnings": list(warnings or []),
        }
