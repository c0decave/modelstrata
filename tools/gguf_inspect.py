#!/usr/bin/env python3
"""
gguf_inspect.py — statischer GGUF-Inspektor (pure Python, keine Dependencies).

Liest NUR den GGUF-Header (Metadaten + Tensor-Verzeichnis). Die Tensor-Daten
selbst werden nie geladen -> auch 26-GB-Modelle sind in Sekunden inspiziert.

Kann:
  - einzelne .gguf-Dateien lesen
  - ganze Verzeichnisse rekursiv scannen (--scan DIR)
  - Ollama-Blobs über die Manifeste auflösen (--ollama DIR)

Ausgabe: lesbarer Report auf stdout, optional JSON via --json OUT.

GGUF-Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""
import argparse
import json
import os
import struct
import sys

MAGIC = 0x46554747  # "GGUF" little-endian

# GGUF metadata value types
(GT_UINT8, GT_INT8, GT_UINT16, GT_INT16, GT_UINT32, GT_INT32, GT_FLOAT32,
 GT_BOOL, GT_STRING, GT_ARRAY, GT_UINT64, GT_INT64, GT_FLOAT64) = range(13)

_FIXED = {
    GT_UINT8: ("<B", 1), GT_INT8: ("<b", 1),
    GT_UINT16: ("<H", 2), GT_INT16: ("<h", 2),
    GT_UINT32: ("<I", 4), GT_INT32: ("<i", 4),
    GT_FLOAT32: ("<f", 4), GT_BOOL: ("<?", 1),
    GT_UINT64: ("<Q", 8), GT_INT64: ("<q", 8),
    GT_FLOAT64: ("<d", 8),
}

# GGML tensor (quant) types -> name + block size (elements per block, bytes per block)
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4),
    27: ("I64", 1, 8), 28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
}


# Sanity caps against malformed/adversarial files that declare absurd lengths
# (the file is untrusted). Real vocab/merges are <1M; strings well under 1 GiB.
MAX_ARRAY = 100_000_000
MAX_STR = 1 << 30
MAX_TENSOR_DIMS = 16


class Reader:
    def __init__(self, f):
        self.f = f

    def raw(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("unexpected EOF")
        return b

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.raw(8))[0]

    def gstr(self):
        n = self.u64()
        if n > MAX_STR:
            raise ValueError(f"string length {n} exceeds cap {MAX_STR}")
        return self.raw(n).decode("utf-8", "replace")

    def value(self, vtype):
        if vtype in _FIXED:
            fmt, size = _FIXED[vtype]
            return struct.unpack(fmt, self.raw(size))[0]
        if vtype == GT_STRING:
            return self.gstr()
        if vtype == GT_ARRAY:
            elem_t = self.u32()
            n = self.u64()
            if n > MAX_ARRAY:
                raise ValueError(f"array length {n} exceeds cap {MAX_ARRAY}")
            # Arrays can be huge (tokenizer vocab). Read fully but summarize later.
            return [self.value(elem_t) for _ in range(n)], elem_t, n
        raise ValueError(f"unknown value type {vtype}")


def _summarize_meta(key, val):
    """Collapse giant arrays so the report stays readable; keep scalars verbatim."""
    if isinstance(val, tuple) and len(val) == 3 and isinstance(val[0], list):
        items, elem_t, n = val
        head = items[:24]
        return {"_array": True, "len": n, "elem_type": elem_t, "sample": head}
    return val


# llama.cpp general.file_type enum -> label
FILE_TYPE = {
    0: "ALL_F32", 1: "MOSTLY_F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0",
    8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M",
    13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
    18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
    23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
}


def parse_gguf(path):
    info = {"path": path, "file_size": os.path.getsize(path)}
    with open(path, "rb") as f:
        # raw first bytes for hex/ascii display (magic, version, counts, ...)
        raw_head = f.read(64)
        info["header_hex"] = raw_head.hex()
        info["header_ascii"] = "".join(
            chr(b) if 32 <= b < 127 else "." for b in raw_head)
        f.seek(0)

        r = Reader(f)
        magic = r.u32()
        if magic != MAGIC:
            raise ValueError(f"not a GGUF file (magic={magic:#x})")
        info["gguf_version"] = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        info["n_tensors"] = n_tensors
        info["n_kv"] = n_kv

        meta = {}
        for _ in range(n_kv):
            key = r.gstr()
            vtype = r.u32()
            meta[key] = _summarize_meta(key, r.value(vtype))
        info["metadata"] = meta

        tensors = []
        quant_counts = {}
        total_params = 0
        for _ in range(n_tensors):
            name = r.gstr()
            ndim = r.u32()
            if ndim > MAX_TENSOR_DIMS:
                raise ValueError(
                    f"tensor {name!r} ndim {ndim} exceeds cap {MAX_TENSOR_DIMS}")
            dims = [r.u64() for _ in range(ndim)]
            ttype = r.u32()
            offset = r.u64()
            tname = GGML_TYPES.get(ttype, (f"TYPE_{ttype}", 0, 0))[0]
            nparams = 1
            for d in dims:
                nparams *= d
            total_params += nparams
            quant_counts[tname] = quant_counts.get(tname, 0) + 1
            tensors.append({"name": name, "dims": dims, "type": tname,
                            "type_id": ttype, "offset": offset, "params": nparams})
        info["tensors"] = tensors
        info["total_params"] = total_params
        info["quant_breakdown"] = quant_counts

        # tensor data layout: header+kv+dir end here, data starts at next
        # `alignment` boundary (general.alignment, default 32)
        hdr_end = f.tell()
        align = meta.get("general.alignment", 32)
        if not isinstance(align, int) or isinstance(align, bool) or align <= 0:
            align = 32
        data_start = ((hdr_end + align - 1) // align) * align
        info["header_end"] = hdr_end
        info["alignment"] = align
        info["data_start"] = data_start
        info["data_bytes"] = max(0, info["file_size"] - data_start)
        # effective bits-per-weight (real quant level incl. F32 norms/embeds)
        info["bits_per_weight"] = (info["data_bytes"] * 8 / total_params
                                   if total_params else None)
        ft = meta.get("general.file_type")
        info["file_type_label"] = FILE_TYPE.get(ft) if isinstance(ft, int) else None
    return info


# ---- pretty printing -------------------------------------------------------

# metadata keys that matter most for "what is this model"
KEY_PREFIXES_ARCH = (".context_length", ".embedding_length", ".block_count",
                     ".feed_forward_length", ".attention.head_count",
                     ".attention.head_count_kv", ".attention.layer_norm_rms_epsilon",
                     ".rope.freq_base", ".rope.dimension_count", ".vocab_size",
                     ".expert_count", ".expert_used_count")

HEADLINE_KEYS = ("general.architecture", "general.name", "general.basename",
                 "general.finetune", "general.size_label", "general.quantization_version",
                 "general.file_type", "general.license",
                 "tokenizer.ggml.model", "tokenizer.ggml.pre")


def human(n):
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else str(n)
        n /= 1000
    return f"{n:.1f}P"


def human_bytes(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PiB"


def print_report(info, label=None):
    meta = info["metadata"]
    arch = meta.get("general.architecture", "?")
    print("=" * 78)
    print(f"# {label or info['path']}")
    print(f"  file: {info['path']}  ({human_bytes(info['file_size'])})")
    print(f"  gguf v{info['gguf_version']}  tensors={info['n_tensors']}  "
          f"kv={info['n_kv']}  params~{human(info['total_params'])}")
    print(f"  quant: " + ", ".join(f"{k}×{v}" for k, v in
                                    sorted(info["quant_breakdown"].items(),
                                           key=lambda x: -x[1])))
    print("  --- headline metadata ---")
    for k in HEADLINE_KEYS:
        if k in meta:
            print(f"    {k:32s} = {meta[k]}")
    print("  --- architecture params ---")
    for k in sorted(meta):
        if any(k.endswith(suf) for suf in KEY_PREFIXES_ARCH):
            print(f"    {k:42s} = {meta[k]}")
    ct = meta.get("tokenizer.chat_template")
    if isinstance(ct, str):
        preview = ct.replace("\n", "\\n")[:160]
        print(f"  --- chat_template ({len(ct)} chars) ---")
        print(f"    {preview}...")
    # tokenizer size
    toks = meta.get("tokenizer.ggml.tokens")
    if isinstance(toks, dict) and toks.get("_array"):
        print(f"  --- tokenizer: {toks['len']} tokens ---")
    print()


def discover_ollama(ollama_dir):
    """Map ollama model:tag -> gguf blob path via manifests."""
    out = {}
    man_root = os.path.join(ollama_dir, "manifests")
    blob_root = os.path.join(ollama_dir, "blobs")
    for dirpath, _, files in os.walk(man_root):
        for fn in files:
            mpath = os.path.join(dirpath, fn)
            try:
                with open(mpath) as fh:
                    man = json.load(fh)
            except Exception:
                continue
            rel = os.path.relpath(dirpath, man_root)
            name = f"{os.path.basename(rel)}:{fn}"
            for layer in (man.get("layers") or []):
                if layer.get("mediaType", "").endswith(".model"):
                    digest = layer["digest"].replace(":", "-")
                    blob = os.path.join(blob_root, digest)
                    if os.path.exists(blob):
                        out[name] = blob
    return out


def main():
    ap = argparse.ArgumentParser(description="static GGUF inspector")
    ap.add_argument("paths", nargs="*", help=".gguf files")
    ap.add_argument("--scan", action="append", default=[],
                    help="recursively scan DIR for *.gguf")
    ap.add_argument("--ollama", action="append", default=[],
                    help="ollama models dir (resolves blobs via manifests)")
    ap.add_argument("--json", dest="json_out", help="write full JSON report to file")
    args = ap.parse_args()

    targets = []  # (label, path)
    for p in args.paths:
        targets.append((None, p))
    for d in args.scan:
        for dp, _, files in os.walk(d):
            for fn in files:
                if fn.lower().endswith(".gguf"):
                    targets.append((None, os.path.join(dp, fn)))
    for od in args.ollama:
        for name, blob in sorted(discover_ollama(od).items()):
            targets.append((f"ollama:{name}", blob))

    if not targets:
        ap.error("no targets — give .gguf paths, --scan DIR, or --ollama DIR")

    reports = []
    for label, path in targets:
        try:
            info = parse_gguf(path)
            info["label"] = label
            reports.append(info)
            print_report(info, label)
        except Exception as e:
            print(f"!! {label or path}: {type(e).__name__}: {e}\n", file=sys.stderr)

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(reports, fh, indent=2, default=str)
        print(f"[json written: {args.json_out}  ({len(reports)} models)]")


if __name__ == "__main__":
    main()
