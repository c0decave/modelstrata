"""Minimaler safetensors-/HF-Modell-Writer für Tests (kein torch/safetensors-Paket)."""
import json, os, struct
import numpy as np

# safetensors dtype name -> numpy dtype (for the bytes we actually write)
_NP = {"F32": "<f4", "F16": "<f2", "I64": "<i8", "I32": "<i4"}

def build_st(tensors):
    """tensors: dict name -> (dtype_name, shape, np.ndarray-or-None).
    Returns bytes of a valid single-file safetensors blob.
    For BF16 pass raw uint16-backed bytes via a (dtype, shape, ndarray) where
    ndarray.tobytes() already holds the bf16 bytes; otherwise we encode from _NP."""
    header, blob, off = {}, bytearray(), 0
    for name, (dt, shape, arr) in tensors.items():
        if arr is None:
            n = 1
            for d in shape: n *= d
            raw = b"\x00" * (n * (2 if dt in ("F16","BF16") else 4))
        else:
            raw = np.asarray(arr).tobytes()
        header[name] = {"dtype": dt, "shape": list(shape),
                        "data_offsets": [off, off + len(raw)]}
        blob += raw; off += len(raw)
    hjson = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(hjson)) + hjson + bytes(blob)

def write_model(d, config, tensors, tokenizer=None):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w") as f: json.dump(config, f)
    with open(os.path.join(d, "model.safetensors"), "wb") as f: f.write(build_st(tensors))
    if tokenizer is not None:
        with open(os.path.join(d, "tokenizer.json"), "w") as f: json.dump(tokenizer, f)
    return d

def write_sharded_model(d, config, shards, tokenizer=None):
    """Write a sharded HF model.

    ``shards``: ordered dict ``{shard_filename: {tensor_name: (dtype, shape, ndarray)}}``.
    Writes each shard via ``build_st``, a ``model.safetensors.index.json`` whose
    ``weight_map`` maps every tensor_name -> its shard_filename, and ``config.json``.
    Returns ``d``."""
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w") as f: json.dump(config, f)
    weight_map = {}
    for shard_name, tensors in shards.items():
        with open(os.path.join(d, shard_name), "wb") as f: f.write(build_st(tensors))
        for name in tensors: weight_map[name] = shard_name
    index = {"metadata": {"total_size": 0}, "weight_map": weight_map}
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f: json.dump(index, f)
    if tokenizer is not None:
        with open(os.path.join(d, "tokenizer.json"), "w") as f: json.dump(tokenizer, f)
    return d
