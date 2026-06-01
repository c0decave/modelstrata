"""modelsource — uniform ``Source`` backends plus the ``detect()`` dispatcher.

``detect(path, ...)`` is the SINGLE entry point the rest of the pipeline calls.
It picks a backend from the path: file extension (lowercased), GGUF magic bytes
for extensionless Ollama blobs, and directory contents for model dirs. It never
fully parses a file and never raises — anything unrecognized (including a path
that does not exist) emits exactly one ``unknown_format`` warning and returns
None.
"""
import glob
import os

from modelsource.base import Source
from modelsource.gguf_backend import GGUFSource
from modelsource.hf_backend import HFSource
from modelsource.inventory import InventorySource
from modelsource.log import RunLog

__all__ = [
    "detect",
    "Source",
    "GGUFSource",
    "HFSource",
    "InventorySource",
    "RunLog",
]


def _is_hf_dir(path):
    """True iff ``path`` is an HF model dir: a ``config.json`` plus at least one
    of (a ``*.safetensors`` shard OR a ``model.safetensors.index.json``)."""
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    if os.path.isfile(os.path.join(path, "model.safetensors.index.json")):
        return True
    return bool(glob.glob(os.path.join(path, "*.safetensors")))


def _pytorch_bin_in_dir(path):
    """Return the .bin/.pth file to inventory for a pytorch model DIR, or None.

    A real HF pytorch model is a directory with ``config.json`` plus a torch
    ``*.bin``/``*.pth`` (and NO safetensors). We point ``InventorySource`` at
    the actual weights file so its sibling-``config.json`` arch lookup and
    pickle opcode scan work. Selection is deterministic: prefer
    ``pytorch_model.bin``, else the first shard named by a
    ``pytorch_model.bin.index.json``, else the first ``*.bin``/``*.pth``.
    Returns None for dirs with no torch weights (caller → unknown_format).
    """
    if not os.path.isfile(os.path.join(path, "config.json")):
        return None
    # safetensors always wins — handled by the _is_hf_dir branch, never here.
    if _is_hf_dir(path):
        return None

    preferred = os.path.join(path, "pytorch_model.bin")
    if os.path.isfile(preferred):
        return preferred

    index_path = os.path.join(path, "pytorch_model.bin.index.json")
    if os.path.isfile(index_path):
        try:
            import json
            with open(index_path, encoding="utf-8") as f:
                weight_map = (json.load(f) or {}).get("weight_map") or {}
            for shard in weight_map.values():
                shard_path = os.path.join(path, shard)
                if os.path.isfile(shard_path):
                    return shard_path
        except Exception:  # noqa: BLE001 — odd index, fall through to glob
            pass

    bins = sorted(glob.glob(os.path.join(path, "*.bin"))
                  + glob.glob(os.path.join(path, "*.pth")))
    return bins[0] if bins else None


def _has_gguf_magic(path):
    """True for extensionless GGUF files such as Ollama blob-store objects."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF"
    except OSError:
        return False


def detect(path, *, log=None, model=None):
    """Return the right ``Source`` subclass for ``path``, or None if unknown.

    Dispatch is by path/extension/dir-contents only — no file is parsed here.
    Unknown or nonexistent paths log one ``unknown_format`` warning + return None.
    """
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".gguf" or _has_gguf_magic(path):
            return GGUFSource(path)
        if ext in (".bin", ".pth", ".onnx"):
            return InventorySource(path, log=log, model=model)
    elif os.path.isdir(path):
        if _is_hf_dir(path):
            return HFSource(path, log=log, model=model)
        # HF pytorch model dir: config.json + a torch .bin/.pth and no
        # safetensors. Inventory the actual weights file (sibling config.json
        # gives the arch). Odd dirs (no torch weights) fall through to None.
        bin_path = _pytorch_bin_in_dir(path)
        if bin_path is not None:
            return InventorySource(
                bin_path, log=log,
                model=(model or os.path.basename(os.path.normpath(path))))

    (log or RunLog()).emit(
        "warn",
        model=(model or os.path.basename(os.path.normpath(path))),
        stage="detect",
        code="unknown_format",
        msg=f"unrecognized model path: {path}",
    )
    return None
