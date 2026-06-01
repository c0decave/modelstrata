"""Inventory-only backend for formats we surface for BREADTH, not weight math.

``InventorySource`` covers PyTorch ``.bin``/``.pth`` and ONNX ``.onnx``. For
these we deliberately do NO weight decoding: ``precision="inventory-only"`` and
``iter_weights()`` yields nothing. The job is an HONEST best-effort tensor-name
inventory plus a ``source`` block, and — critically — to NEVER raise into the
batch: every parse failure degrades to a coarser inventory with a logged
warning.

SECURITY: PyTorch ``.bin``/``.pth`` are pickles. We NEVER ``pickle.load`` or
``torch.load`` them (that would execute arbitrary code). The only pickle
inspection here is ``pickletools.genops``, which is a pure OPCODE DISASSEMBLER:
it walks the pickle bytecode and reports opcodes/args WITHOUT executing any of
them. ONNX is parsed by a shallow, bounded protobuf field scan — no eval, no
external package.
"""

import os
import pickletools
import zipfile


from modelsource.base import Source
from modelsource.log import RunLog

# Extension -> coarse format label for the source block.
_PYTORCH_EXTS = (".bin", ".pth")
_ONNX_EXTS = (".onnx",)

# Memory-exhaustion (zip-bomb) cap: the UNCOMPRESSED size of the ``data.pkl``
# member we opcode-scan. A pickle of tensor *metadata* (state_dict keys) is
# tiny; the raw tensor storages are SEPARATE zip members we never read. A small
# highly-compressible .bin can otherwise decompress to GBs of RSS. 64 MiB is
# ample headroom for legitimate metadata pickles.
_MAX_PKL_BYTES = 64 * 1024 * 1024

# ONNX protobufs can be model-sized. The shallow stdlib scanner reads bytes
# into memory, so cap the file before scanning and fall back to a file-level
# inventory for unusually large inputs.
_MAX_ONNX_BYTES = 256 * 1024 * 1024

_ONNX_DTYPES = {
    1: "FLOAT",
    2: "UINT8",
    3: "INT8",
    4: "UINT16",
    5: "INT16",
    6: "INT32",
    7: "INT64",
    8: "STRING",
    9: "BOOL",
    10: "FLOAT16",
    11: "DOUBLE",
    12: "UINT32",
    13: "UINT64",
    14: "COMPLEX64",
    15: "COMPLEX128",
    16: "BFLOAT16",
    17: "FLOAT8E4M3FN",
    18: "FLOAT8E4M3FNUZ",
    19: "FLOAT8E5M2",
    20: "FLOAT8E5M2FNUZ",
    21: "UINT4",
    22: "INT4",
    23: "FLOAT4E2M1",
}


class InventorySource(Source):
    """Header/inventory-only ``Source`` for .bin/.pth and .onnx."""

    def __init__(self, path, *, log=None, model=None):
        self.path = path
        # A logger is ALWAYS present so fallbacks are never swallowed.
        self.log = log or RunLog()
        self.model = model or os.path.basename(os.path.normpath(path))
        self.warnings = []

    def _warn(self, stage, code, msg):
        """Recoverable fallback: human string into source.warnings AND a
        structured RunLog entry (stderr + buffer). Never swallow."""
        self.warnings.append(msg)
        self.log.emit("warn", model=self.model, stage=stage, code=code, msg=msg)

    def _info(self, stage, code, msg):
        """Informational note: RunLog only (not surfaced in source.warnings)."""
        self.log.emit("info", model=self.model, stage=stage, code=code, msg=msg)

    def _fmt(self):
        ext = os.path.splitext(self.path)[1].lower()
        if ext in _ONNX_EXTS:
            return "onnx"
        # .bin/.pth and anything else routed here are treated as pytorch.
        return "pytorch"

    # --- PyTorch .bin/.pth -------------------------------------------------

    def _pytorch_tensors(self):
        """Best-effort tensor inventory for a torch ``.bin``/``.pth``.

        torch.save writes a ZIP archive containing a ``*/data.pkl`` pickle.
        We list members (always works) and try to recover state_dict KEYS by
        disassembling that pickle with ``pickletools.genops`` — an opcode scan
        that NEVER executes the pickle. Shapes can't be recovered reliably from
        the opcode stream, so dims/type stay None (honest). ANY failure (not a
        zip, no data.pkl, disassembler error, legacy non-zip pickle) degrades
        to a file-/member-level inventory with a ``pickle_unparsed`` warning.
        """
        if not zipfile.is_zipfile(self.path):
            # Legacy non-zip pickle (old torch). Do NOT unpickle — file-level
            # inventory only.
            self._warn("scan", "pickle_unparsed",
                       "legacy non-zip pickle → file inventory only")
            return [self._file_entry()]

        try:
            with zipfile.ZipFile(self.path) as zf:
                members = zf.namelist()
                pkl_info = next(
                    (zi for zi in zf.infolist()
                     if zi.filename.endswith("data.pkl")), None)
                if pkl_info is None:
                    raise ValueError("no data.pkl in archive")
                # Zip-bomb guard: check the DECLARED uncompressed size BEFORE
                # reading. Oversized → do NOT read the pkl; fall back to the
                # archive member-list inventory (still useful, bounded memory).
                if pkl_info.file_size > _MAX_PKL_BYTES:
                    self._warn(
                        "scan", "pickle_too_large",
                        f"data.pkl {pkl_info.file_size} bytes exceeds cap "
                        f"({_MAX_PKL_BYTES}) → member-list inventory only")
                    return [{"name": m, "dims": None, "type": None}
                            for m in members]
                blob = zf.read(pkl_info.filename)
            names = self._scan_pickle_names(blob)
            if not names:
                raise ValueError("no state_dict-like names found in pickle")
            return [{"name": n, "dims": None, "type": None} for n in names]
        except Exception as e:  # noqa: BLE001 — never raise into the batch
            self._warn("scan", "pickle_unparsed", f"could not parse pickle: {e}")
            # Fall back to the archive member list (still useful), or a single
            # file entry if even that is unavailable.
            try:
                with zipfile.ZipFile(self.path) as zf:
                    members = zf.namelist()
                return [{"name": m, "dims": None, "type": None}
                        for m in members]
            except Exception:  # noqa: BLE001
                return [self._file_entry()]

    @staticmethod
    def _scan_pickle_names(blob):
        """Collect state_dict-like string args from a pickle's OPCODE stream.

        Uses ``pickletools.genops`` — a disassembler that yields
        ``(opcode, arg, pos)`` WITHOUT executing the pickle. We keep string
        args from the *UNICODE string* opcodes (BINUNICODE / SHORT_BINUNICODE /
        UNICODE / and their byte-string cousins) that look like state_dict keys
        (contain a dot). Order-preserving + de-duplicated.
        """
        wanted = {
            "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8", "UNICODE",
            "STRING", "BINSTRING", "SHORT_BINSTRING",
        }
        seen = []
        seen_set = set()
        for opcode, arg, _pos in pickletools.genops(blob):
            if opcode.name not in wanted or not isinstance(arg, str):
                continue
            # state_dict keys look like "model.layers.0.weight" — require a dot
            # to avoid scooping up class names / persistent-id junk.
            if "." in arg and arg not in seen_set:
                seen.append(arg)
                seen_set.add(arg)
        return seen

    # --- ONNX .onnx --------------------------------------------------------

    def _onnx_tensors(self):
        """Best-effort tensor inventory for an ``.onnx`` (protobuf ModelProto).

        WITHOUT the ``onnx`` package, we do a SHALLOW, bounded protobuf field
        scan: descend ModelProto.graph (field 7) -> GraphProto.initializer
        (field 5, TensorProto) -> TensorProto.name/dims/data_type. This is
        best-effort only; on ANY uncertainty/error or when no initializer is
        found, degrade to a file-level inventory entry with an ``onnx_shallow``
        warning.
        """
        try:
            size = os.path.getsize(self.path)
        except OSError:
            size = None
        if size is not None and size > _MAX_ONNX_BYTES:
            self._warn(
                "scan", "onnx_too_large",
                f"onnx file {size} bytes exceeds cap ({_MAX_ONNX_BYTES}) "
                "→ file inventory only")
            return [self._file_entry()]
        try:
            tensors = self._scan_onnx_initializers()
        except Exception:  # noqa: BLE001 — never raise into the batch
            tensors = None
        if not tensors:
            self._warn("scan", "onnx_shallow",
                       "onnx parsed shallowly / not parsed → file inventory only")
            return [self._file_entry()]
        return tensors

    def _scan_onnx_initializers(self):
        """Return initializer inventories via shallow protobuf descent, or None."""
        with open(self.path, "rb") as f:
            data = f.read()
        graph = _pb_first_subfield(data, 7)          # ModelProto.graph
        if graph is None:
            return None
        tensors = []
        seen = set()
        for init in _pb_subfields(graph, 5):         # GraphProto.initializer
            tensor = _parse_onnx_tensor(init)
            if not tensor or not tensor["name"] or tensor["name"] in seen:
                continue
            tensors.append(tensor)
            seen.add(tensor["name"])
        return tensors or None

    # --- shared ------------------------------------------------------------

    def _file_entry(self):
        """A single coarse inventory row naming the file itself."""
        return {"name": os.path.basename(self.path), "dims": None, "type": None}

    def metadata(self):
        fmt = self._fmt()
        if fmt == "onnx":
            tensors = self._onnx_tensors()
        else:
            tensors = self._pytorch_tensors()

        arch = self._sibling_arch()

        return {
            "path": self.path,
            "tensors": tensors,
            "quant_breakdown": _dtype_breakdown(tensors),
            "source": self.source_block(
                fmt=fmt,
                precision="inventory-only",
                arch=arch,
                mapped=False,
                warnings=self.warnings,
            ),
        }

    def _sibling_arch(self):
        """Read arch from a sibling ``config.json`` if present, else None."""
        cfg_path = os.path.join(os.path.dirname(self.path), "config.json")
        if not os.path.exists(cfg_path):
            return None
        try:
            import json
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            archs = cfg.get("architectures") or []
            if archs:
                return archs[0]
            return cfg.get("model_type")
        except Exception:  # noqa: BLE001 — config is optional, never crash
            self._warn("scan", "config_unparsed",
                       "sibling config.json present but unreadable")
            return None

    def iter_weights(self, want=None):
        """Inventory-only: emit ONE info note and yield nothing. Never raises."""
        self._info("weights", "inventory_only",
                   f"{self._fmt()} is inventory-only: no weight math")
        return
        yield  # pragma: no cover — makes this a generator

    def tokenizer(self):
        """Out of scope for the inventory backend."""
        return None


# --- shallow protobuf helpers --------------------------------------------
#
# These read just enough wire format to descend named message/string fields and
# scalar varints. They are tolerant: a malformed buffer raises, and callers
# catch it to degrade. No protobuf package, no code execution.

def _read_varint(buf, pos):
    """Return (value, new_pos). Raises on truncation/overlong varint."""
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("truncated varint")
        b = buf[pos]
        result |= (b & 0x7F) << shift
        pos += 1
        if not (b & 0x80):
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def _iter_fields(buf):
    """Yield (field_number, wire_type, payload) over ``buf``.

    Length-delimited fields carry bytes; varint fields carry int values; 64-bit
    and 32-bit fields are skipped with payload None. Wire type 3/4 (groups,
    deprecated) abort the scan (raise) so we degrade rather than guess.
    """
    pos = 0
    n = len(buf)
    while pos < n:
        tag, pos = _read_varint(buf, pos)
        field_number = tag >> 3
        wire_type = tag & 0x7
        if wire_type == 0:          # varint
            value, pos = _read_varint(buf, pos)
            yield field_number, wire_type, value
        elif wire_type == 1:        # 64-bit
            pos += 8
            if pos > n:
                raise ValueError("truncated 64-bit field")
            yield field_number, wire_type, None
        elif wire_type == 2:        # length-delimited
            length, pos = _read_varint(buf, pos)
            end = pos + length
            if end > n:
                raise ValueError("truncated length-delimited field")
            yield field_number, wire_type, buf[pos:end]
            pos = end
        elif wire_type == 5:        # 32-bit
            pos += 4
            if pos > n:
                raise ValueError("truncated 32-bit field")
            yield field_number, wire_type, None
        else:                       # 3/4 groups — unsupported
            raise ValueError(f"unsupported wire type {wire_type}")


def _pb_first_subfield(buf, field_number):
    """First length-delimited payload for ``field_number``, or None."""
    for fn, wt, payload in _iter_fields(buf):
        if fn == field_number and wt == 2:
            return payload
    return None


def _pb_subfields(buf, field_number):
    """All length-delimited payloads for ``field_number`` (repeated field)."""
    out = []
    for fn, wt, payload in _iter_fields(buf):
        if fn == field_number and wt == 2:
            out.append(payload)
    return out


def _pb_varint_fields(buf, field_number):
    """All varint payloads for ``field_number`` (repeated scalar field)."""
    out = []
    for fn, wt, payload in _iter_fields(buf):
        if fn == field_number and wt == 0:
            out.append(payload)
    return out


def _pb_first_varint_field(buf, field_number):
    """First varint payload for ``field_number``, or None."""
    vals = _pb_varint_fields(buf, field_number)
    return vals[0] if vals else None


def _packed_varints(payload):
    """Decode packed repeated varints from a length-delimited payload."""
    vals = []
    pos = 0
    while pos < len(payload):
        value, pos = _read_varint(payload, pos)
        vals.append(value)
    return vals


def _parse_onnx_tensor(init):
    """Parse the TensorProto fields we can safely use: name, dims, data_type."""
    name_bytes = _pb_first_subfield(init, 8)  # TensorProto.name
    if name_bytes is None:
        return None
    try:
        name = name_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None

    dims = list(_pb_varint_fields(init, 1))   # TensorProto.dims, unpacked
    for payload in _pb_subfields(init, 1):    # tolerate packed encodings too
        dims.extend(_packed_varints(payload))
    data_type = _pb_first_varint_field(init, 2)
    dtype = _ONNX_DTYPES.get(data_type)
    if dtype is None and data_type is not None:
        dtype = f"ONNX_TYPE_{data_type}"
    return {"name": name, "dims": dims or None, "type": dtype}


def _dtype_breakdown(tensors):
    out = {}
    for t in tensors or []:
        dtype = t.get("type")
        if dtype:
            out[dtype] = out.get(dtype, 0) + 1
    return out
