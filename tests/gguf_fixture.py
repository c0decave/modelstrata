"""Minimaler GGUF-Writer für Tests — erzeugt eine gültige GGUF-Datei im Speicher
oder auf der Platte, ohne echte Modellgewichte."""
import struct

MAGIC = 0x46554747
# GGUF value types
T_UINT32, T_FLOAT32, T_STRING, T_ARRAY, T_UINT64, T_BOOL = 4, 6, 8, 9, 10, 7


def _str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _val(vtype, value):
    if vtype == T_UINT32:
        return struct.pack("<I", value)
    if vtype == T_UINT64:
        return struct.pack("<Q", value)
    if vtype == T_FLOAT32:
        return struct.pack("<f", value)
    if vtype == T_BOOL:
        return struct.pack("<?", value)
    if vtype == T_STRING:
        return _str(value)
    if vtype == T_ARRAY:
        elem_t, items = value
        out = struct.pack("<I", elem_t) + struct.pack("<Q", len(items))
        for it in items:
            out += _val(elem_t, it)
        return out
    raise ValueError(vtype)


def build_gguf(kvs, tensors, version=3, pad=64):
    """
    kvs: list of (key, vtype, value)
    tensors: list of (name, dims_list, type_id, offset)
    Returns bytes of a valid GGUF file (header + metadata + tensor dir + pad).
    """
    out = struct.pack("<I", MAGIC) + struct.pack("<I", version)
    out += struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(kvs))
    for key, vtype, value in kvs:
        out += _str(key) + struct.pack("<I", vtype) + _val(vtype, value)
    for name, dims, type_id, offset in tensors:
        out += _str(name) + struct.pack("<I", len(dims))
        for d in dims:
            out += struct.pack("<Q", d)
        out += struct.pack("<I", type_id) + struct.pack("<Q", offset)
    out += b"\x00" * pad   # stand-in for the tensor data section
    return out


def write_gguf(path, kvs, tensors, **kw):
    data = build_gguf(kvs, tensors, **kw)
    with open(path, "wb") as f:
        f.write(data)
    return path
