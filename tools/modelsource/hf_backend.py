"""HF safetensors backend for the uniform ``Source`` interface.

Header parsing is pure stdlib (``struct`` + ``json``). A safetensors file is::

    [8-byte LE uint64 header_len][header JSON][raw tensor bytes]

where the header JSON maps ``tensor_name -> {"dtype", "shape", "data_offsets"}``
plus an optional ``"__metadata__"`` key (ignored). The metadata is therefore
fully readable WITHOUT torch. Weight decoding uses numpy, supports fp32/fp16/
bf16, and degrades explicitly for unsupported dtypes.
"""

import json
import os
import struct

import numpy as np

from modelsource import archmap, naming
from modelsource.base import Source
from modelsource.log import RunLog

# safetensors dtype -> bytes per element (for offset-length validation).
_ITEMSIZE = {"F32": 4, "F16": 2, "BF16": 2}

# Sanity cap against malformed/adversarial files declaring an absurd header
# length (the file is untrusted). Real safetensors headers are well under this.
_MAX_HEADER = 1 << 28  # 256 MiB


_EXACT_FP = {"F32", "F16", "BF16"}


class _Degrade(Exception):
    """Internal sentinel: a config/header/index read failed and was ALREADY
    logged as a structured error. Public methods (metadata/iter_weights) catch
    this at their boundary and DEGRADE — they never let it (or the underlying
    parse exception) escape. Carries no new message; the structured _error was
    emitted at the failure site."""


def bytes_to_fp32(raw, dtype, shape):
    """Decode raw safetensors tensor bytes -> fp32 ndarray of ``shape``.

    bf16 is decoded by hand (numpy has no native bfloat16): widen each
    uint16 to uint32 and shift left 16 bits, then reinterpret as float32.
    Unsupported dtypes (fp8, int) raise ``NotImplementedError`` so callers
    can mark the model inventory-only and LOG it — never silently skip.
    """
    if dtype == "F32":
        arr = np.frombuffer(raw, dtype="<f4")
    elif dtype == "F16":
        arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    elif dtype == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        arr = (u16 << 16).view(np.float32)
    else:
        raise NotImplementedError(
            f"dtype {dtype!r}: fp8/int weights are inventory-only (no fp32 decode)")
    return np.ascontiguousarray(arr, dtype=np.float32).reshape(shape)


def _read_st_header_len(path):
    """Return ``(header_dict, header_len)`` for a safetensors file.

    ``header_len`` is the byte length of the JSON directory; the raw tensor
    region therefore begins at ``8 + header_len``. Pure struct+json — never
    touches tensor bytes. ``__metadata__`` is stripped from the returned dict.
    """
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: truncated safetensors (no header length)")
        (hlen,) = struct.unpack("<Q", prefix)
        if hlen == 0 or hlen > _MAX_HEADER:
            raise ValueError(f"{path}: header length {hlen} out of range")
        raw = f.read(hlen)
        if len(raw) != hlen:
            raise ValueError(f"{path}: truncated safetensors header")
    header = json.loads(raw.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"{path}: safetensors header is not an object")
    header.pop("__metadata__", None)
    return header, hlen


class HFSource(Source):
    """Read an HF safetensors model *directory* (metadata layer only)."""

    def __init__(self, path, *, log=None, model=None):
        self.path = path
        # A logger is ALWAYS present so warnings are never swallowed, even when
        # the caller injects none.
        self.log = log or RunLog()
        self.model = model or os.path.basename(os.path.normpath(path))
        self.warnings = []

    def _warn(self, stage, code, msg):
        """Record a recoverable skip/fallback: human string in source.warnings
        AND a structured RunLog entry (stderr + buffer). Never swallow."""
        self.warnings.append(msg)
        self.log.emit("warn", model=self.model, stage=stage, code=code, msg=msg)

    def _error(self, stage, code, msg):
        """Record an error-level skip (e.g. invalid offsets) — same dual sink."""
        self.warnings.append(msg)
        self.log.emit("error", model=self.model, stage=stage, code=code, msg=msg)

    def _config(self):
        """Read ``config.json``. A missing/corrupt config (FileNotFoundError,
        JSONDecodeError, decode error) is logged as a structured ``bad_config``
        and re-raised as the internal ``_Degrade`` sentinel — the public method
        catches it and returns/yields a degraded result rather than crashing
        the batch."""
        try:
            with open(os.path.join(self.path, "config.json"),
                      encoding="utf-8") as f:
                config = json.load(f)
            if not isinstance(config, dict):
                raise ValueError("config.json is not an object")
            return config
        except (OSError, ValueError, UnicodeDecodeError,
                AttributeError, TypeError) as exc:
            self._error("config", "bad_config",
                        f"config.json missing/unreadable → inventory-only: {exc}")
            raise _Degrade from exc

    @staticmethod
    def _arch(config):
        archs = config.get("architectures") or []
        if archs:
            return archs[0]
        return config.get("model_type")

    def _index_path(self):
        return os.path.join(self.path, "model.safetensors.index.json")

    def _safe_header_len(self, path):
        """``_read_st_header_len`` wrapped so a truncated/non-UTF8/non-JSON
        safetensors header is logged as a structured ``bad_header`` and raised
        as the ``_Degrade`` sentinel — never an uncaught ValueError/
        UnicodeDecodeError out of a public method."""
        try:
            return _read_st_header_len(path)
        except (OSError, ValueError, UnicodeDecodeError,
                AttributeError, TypeError) as exc:
            self._error("weights", "bad_header",
                        f"{os.path.basename(path)}: unreadable safetensors "
                        f"header → skipped: {exc}")
            raise _Degrade from exc

    def _safe_header(self, path):
        """Header-dict-only variant of ``_safe_header_len`` (same degrade)."""
        header, _ = self._safe_header_len(path)
        return header

    @staticmethod
    def _group_by_shard(weight_map):
        """tensor_name -> shard_file  =>  ordered {shard_file: [tensor_name, ...]}.
        Deterministic: shards sorted, names kept in weight_map order."""
        by_shard = {}
        for name, shard in weight_map.items():
            by_shard.setdefault(shard, []).append(name)
        return {s: by_shard[s] for s in sorted(by_shard)}

    def _weight_map(self):
        """Return the ``weight_map`` dict from ``model.safetensors.index.json``.

        Degrades (logged, never crashes) to ``None`` if the index is missing
        or malformed (``bad_index``). The caller distinguishes "no index ⇒
        single-file" from "broken index" by checking ``os.path.exists`` first.
        """
        try:
            with open(self._index_path(), encoding="utf-8") as f:
                index = json.load(f)
            if not isinstance(index, dict):
                raise ValueError("index.json is not an object")
        except (OSError, ValueError, UnicodeDecodeError,
                AttributeError, TypeError) as exc:
            self._error("weights", "bad_index",
                        f"index.json unreadable → no weights: {exc}")
            return None
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            self._error("weights", "bad_index",
                        "index.json missing weight_map → no weights")
            return None
        return weight_map

    def _merged_header(self):
        """Build the unified safetensors tensor directory for this model.

        Returns ``(header, names)`` where ``header`` maps ``name ->
        tensor-descriptor`` (``dtype``/``shape``/…) for EITHER a single
        ``model.safetensors`` OR a sharded model, and ``names`` is the full
        set of *declared* tensor names (the ``weight_map`` keys for the
        sharded case — authoritative even if a shard is absent; the header
        keys for the single-file case). For the sharded case it reads
        ``model.safetensors.index.json`` ONCE and merges the headers of every
        referenced shard that exists (union of all shard directories).
        Malformed/missing pieces degrade with a logged ``bad_index``/
        ``shard_missing`` — never crash. Returning the declared name set here
        lets ``metadata`` reuse the single index read for tied detection
        instead of re-reading (and re-logging ``bad_index``).
        """
        index_path = self._index_path()
        if os.path.exists(index_path):
            weight_map = self._weight_map()
            if weight_map is None:
                return {}, set()
            names = set(weight_map)
            # Group tensor names by shard, then read each existing shard's
            # header once and merge. Missing shards degrade per-shard.
            by_shard = self._group_by_shard(weight_map)
            merged = {}
            for shard, shard_names in by_shard.items():
                shard_path = os.path.join(self.path, shard)
                if not os.path.exists(shard_path):
                    self._error(
                        "weights", "shard_missing",
                        f"shard {shard} referenced by index.json is absent → "
                        f"{len(shard_names)} tensors skipped")
                    continue
                try:
                    header = self._safe_header(shard_path)
                except _Degrade:
                    # This shard's header is unreadable (already logged
                    # bad_header). Keep the good shards visible in metadata
                    # instead of degrading the whole sharded model.
                    continue
                for name in shard_names:
                    td = header.get(name)
                    if td is not None:
                        merged[name] = td
            return merged, names

        header = self._safe_header(
            os.path.join(self.path, "model.safetensors"))
        return header, set(header)

    def _tensor_names(self):
        """Full set of tensor names declared by the model.

        Sharded: the ``weight_map`` keys (authoritative even if a shard is
        absent). Single file: the header keys. Used for tied-embedding
        detection (is ``lm_head.weight`` shipped at all?)."""
        index_path = self._index_path()
        if os.path.exists(index_path):
            weight_map = self._weight_map()
            return set(weight_map) if weight_map is not None else set()
        return set(self._safe_header(
            os.path.join(self.path, "model.safetensors")))

    def _is_tied(self, config, names=None):
        """True iff the LM head is tied to the input embedding.

        Two conditions must both hold: the arch is mappable (only then can we
        alias a canonical ``output.weight``) AND no ``lm_head.weight`` tensor
        is shipped. The HF ``tie_word_embeddings`` flag is the corroborating
        signal that the absent head is *intentionally* tied rather than an
        honest gap:

          * flag true  + lm_head absent → tied (alias the embedding).
          * flag false + lm_head absent → NOT tied; an honest missing output
            projection, surfaced at emit time as ``no_output_weight`` rather
            than a silently fabricated output.
          * flag absent + lm_head absent → conservatively NOT tied for most
            families (we do not fabricate an output the config never authorized)
            — EXCEPT gemma, which ties by architecture default and often omits
            the flag (e.g. gemma3_text / unsloth gemma-3-1b). For gemma an absent
            flag means tied, so we alias the embedding instead of logging a false
            ``no_output_weight`` gap.

        A shipped ``lm_head.weight`` always wins (never tied).
        """
        family = archmap.family_for(config)
        if family is None:
            return False
        if names is None:
            names = self._tensor_names()
        if "lm_head.weight" in names:
            return False
        flag = config.get("tie_word_embeddings")
        if flag is None:
            # gemma's documented default is tied; other families stay conservative.
            return family == "gemma"
        return bool(flag)

    def _degraded_metadata(self):
        """Minimal honest metadata when config/header/index couldn't be read.

        The structured error was ALREADY logged at the failure site; this just
        surfaces an inventory-only block (with the recorded warnings) so the
        caller gets a usable dict instead of an exception."""
        return {
            "path": self.path,
            "metadata": {"general.architecture": None},
            "tensors": [],
            "quant_breakdown": {},
            "tied": False,
            "n_layers": None,
            "n_heads": None,
            "n_kv_heads": None,
            "hidden": None,
            "source": self.source_block(
                fmt="safetensors", precision="inventory-only",
                arch=None, mapped=False, warnings=self.warnings),
        }

    def metadata(self):
        try:
            return self._metadata()
        except _Degrade:
            # config/header/index unreadable (already logged) → degrade, never
            # raise out of the public method.
            return self._degraded_metadata()

    def _metadata(self):
        config = self._config()
        # Read the index ONCE: _merged_header returns both the merged tensor
        # directory AND the declared-name set, so the tied check below reuses
        # that single read instead of re-reading (which would log bad_index a
        # second time on a malformed index).
        header, names = self._merged_header()

        tensors = []
        quant_breakdown = {}
        for name, td in header.items():
            if not isinstance(td, dict):
                # Untrusted input: a descriptor that is valid JSON but not an
                # object (string/number/list) would raise on td.get(...). Skip
                # it (logged) so one bad descriptor never sinks metadata().
                self._error("weights", "bad_header",
                            f"{name}: descriptor is not an object")
                continue
            dtype = td.get("dtype")
            # Match gguf_inspect's tensor-record key names (name/dims/type) so
            # consumers read the same schema. safetensors `shape` is already the
            # logical shape, so dims = shape directly (no ne-order reversal).
            tensors.append({"name": name, "dims": td.get("shape"), "type": dtype})
            quant_breakdown[dtype] = quant_breakdown.get(dtype, 0) + 1

        arch = self._arch(config)
        mapped = archmap.family_for(config) is not None
        exact_fp = bool(tensors) and all(t in _EXACT_FP for t in quant_breakdown)
        precision = "exact" if mapped and exact_fp else "inventory-only"

        return {
            "path": self.path,
            # mirror gguf_inspect's nesting: arch lives under metadata.*
            "metadata": {"general.architecture": arch},
            "tensors": tensors,
            "quant_breakdown": quant_breakdown,
            "tied": self._is_tied(config, names),
            "n_layers": config.get("num_hidden_layers"),
            "n_heads": config.get("num_attention_heads"),
            "n_kv_heads": (config.get("num_key_value_heads")
                           or config.get("num_attention_heads")),
            "hidden": config.get("hidden_size"),
            "ffn": (config.get("intermediate_size")
                    or config.get("ffn_dim")),
            "ctx": (config.get("max_position_embeddings")
                    or config.get("seq_length")
                    or config.get("max_sequence_length")),
            "source": self.source_block(
                fmt="safetensors",
                precision=precision,
                arch=arch,
                mapped=mapped,
                # Reflect warnings recorded so far (e.g. by a prior
                # iter_weights run). source_block makes a defensive copy.
                warnings=self.warnings,
            ),
        }

    def _emit_tensor(self, name, td, base, blob, family, want):
        """Process ONE tensor from an already-read shard/file ``blob``.

        ``blob`` is the full bytes of the file containing this tensor and
        ``base = 8 + header_len`` of THAT file (so ``blob[base:]`` is its raw
        tensor region). Does resolve→(None⇒warn+return)→canonical→``want``
        filter→offset-validate→dtype-check→decode and yields
        ``(canonical_name, fp32 ndarray, dtype_label)``. Shared by the
        single-file and sharded paths so the per-tensor logic lives once.
        """
        region = len(blob) - base  # size of this file's tensor byte region

        # Untrusted input: a per-tensor descriptor that is valid JSON but not an
        # object (a string/number/list) would raise AttributeError on td.get(...)
        # and abort the batch. Degrade per-tensor — one malformed descriptor must
        # never sink the good tensors in the same file.
        if not isinstance(td, dict):
            self._error("weights", "bad_header",
                        f"{name}: descriptor is not an object")
            return

        r = archmap.resolve(family, name)
        if r is None:
            self._warn("weights", "tensor_unmapped",
                       f"no canonical role for {name}")
            return
        role_str, layer = r
        # Derive the component from the HF tensor-name suffix so a q_proj.bias
        # yields blk.N.attn_q.bias (GGUF parity), not a dropped tensor. resolve()
        # is deliberately weight/bias-agnostic; the suffix decides here.
        comp = "bias" if name.endswith(".bias") else "weight"
        if role_str in naming.Role.__members__:
            cname = naming.canonical(naming.Role[role_str], layer, comp)
        else:
            # MoE expert/shared roles carry their already-canonical GGUF stem
            # (e.g. ffn_gate_exps.e3). Keeping the expert id in the role avoids
            # overwriting separate HF expert tensors while still preserving the
            # blk.N.<role>.weight shape downstream.
            if layer is None:
                cname = f"{role_str}.{comp}"
            else:
                cname = f"blk.{layer}.{role_str}.{comp}"

        if want is not None and not want(cname):
            return

        dtype = td.get("dtype")
        dims = td.get("shape")
        offsets = td.get("data_offsets")

        # Validate the per-tensor header descriptor on UNTRUSTED input BEFORE
        # using it: a missing/non-list shape, or data_offsets that are absent /
        # not a 2-tuple of ints, would otherwise raise (TypeError/ValueError)
        # out of iter_weights and abort the whole batch. Degrade per-tensor —
        # same style as the bad_offset branch below — so one malformed
        # descriptor never sinks the good tensors in the same file.
        if not (isinstance(dims, (list, tuple))
                and all(isinstance(d, int) for d in dims)
                and isinstance(offsets, (list, tuple)) and len(offsets) == 2
                and all(isinstance(o, int) for o in offsets)):
            self._error("weights", "bad_header",
                        f"{name}: malformed header descriptor {td!r}")
            return
        s, e = offsets

        # Offset validation on UNTRUSTED input — before any slicing.
        itemsize = _ITEMSIZE.get(dtype)
        expected = None
        if itemsize is not None:
            n = 1
            for dpart in dims:
                n *= dpart
            expected = n * itemsize
        if not (0 <= s <= e <= region) or (
                expected is not None and (e - s) != expected):
            self._error("weights", "bad_offset",
                        f"{name}: offsets {[s, e]} invalid for {dtype} {dims}")
            return

        if dtype not in _EXACT_FP:
            self._warn("weights", "dtype_unsupported",
                       f"{name}: {dtype} not decodable → skipped "
                       f"(inventory-only weight)")
            return

        raw = blob[base + s:base + e]
        arr = bytes_to_fp32(raw, dtype, dims)
        yield (cname, arr, dtype)

    def iter_weights(self, want=None):
        """Yield ``(canonical_name, fp32 ndarray, dtype_label)`` for every
        mappable, exactly-representable tensor. Every skip (unmapped arch,
        unmapped tensor, bad offsets, undecodable dtype, missing shard) is
        surfaced via ``_warn``/``_error`` — never silent.

        Public boundary: a config/header/index read that fails raises the
        internal ``_Degrade`` sentinel (already logged as a structured error);
        we swallow it here and stop yielding, so a corrupt model degrades to an
        empty iterator instead of raising into the batch.
        """
        try:
            yield from self._iter_weights(want)
        except _Degrade:
            return

    def _iter_weights(self, want=None):
        """Inner generator (see ``iter_weights``).

        Supports BOTH a single ``model.safetensors`` and sharded models
        described by ``model.safetensors.index.json`` (whose ``weight_map``
        maps each tensor name to its shard file). Each shard is read into
        memory once and its blob reused for all its tensors. Shards are
        bounded (~≤5GB) and the host has ample RAM, so a full read is an
        acceptable trade-off vs. mmap complexity.
        """
        config = self._config()
        family = archmap.family_for(config)
        if family is None:
            arch = self._arch(config)
            self._warn("weights", "arch_unmapped",
                       f"architecture not in archmap → inventory-only: {arch}")
            return

        # Resolve the declared tensor-name set ONCE (a malformed index must log
        # exactly one bad_index). Sharded: weight_map keys; single: header keys.
        # Reused for tied detection AND, for shards, the iteration plan below.
        index_path = self._index_path()
        is_sharded = os.path.exists(index_path)
        if is_sharded:
            weight_map = self._weight_map()
            if weight_map is None:
                return  # bad_index already logged
            names = set(weight_map)
        else:
            st_path = os.path.join(self.path, "model.safetensors")
            single_header, single_hlen = self._safe_header_len(st_path)
            names = set(single_header)

        # Tied-embedding state: when the model ships no lm_head.weight we alias
        # the decoded input embedding as output.weight. Capture it as it streams
        # past so the alias is the SAME array (Part B). Determined up-front from
        # the declared tensor names so a filtered-out embedding is still known.
        tied = self._is_tied(config, names)
        # No lm_head.weight AND not tied (tie_word_embeddings false OR absent) ⇒
        # honest gap: the model ships no output projection. Surface it once
        # rather than silently producing nothing — every other missing-output
        # case logs something. _is_tied() already returns False here.
        honest_gap = "lm_head.weight" not in names and not tied
        embd_arr = embd_dtype = None

        def _scan(emitted):
            """Pass tuples through, capturing the embedding for aliasing."""
            nonlocal embd_arr, embd_dtype
            for cname, arr, dt in emitted:
                if tied and cname == "token_embd.weight":
                    embd_arr, embd_dtype = arr, dt
                yield (cname, arr, dt)

        if is_sharded:
            # Group tensor names by their shard filename (deterministic order).
            by_shard = self._group_by_shard(weight_map)

            for shard, shard_names in by_shard.items():
                shard_path = os.path.join(self.path, shard)
                if not os.path.exists(shard_path):
                    self._error(
                        "weights", "shard_missing",
                        f"shard {shard} referenced by index.json is absent → "
                        f"{len(shard_names)} tensors skipped")
                    continue
                try:
                    header, hlen = self._safe_header_len(shard_path)
                except _Degrade:
                    # This shard's header is unreadable (already logged
                    # bad_header) — skip it and keep processing other shards.
                    continue
                base = 8 + hlen
                with open(shard_path, "rb") as f:
                    blob = f.read()
                for name in shard_names:
                    td = header.get(name)
                    if td is None:
                        self._error(
                            "weights", "tensor_missing_in_shard",
                            f"{name} mapped to {shard} but absent from its "
                            f"header → skipped")
                        continue
                    yield from _scan(self._emit_tensor(
                        name, td, base, blob, family, want))
        else:
            # Single-file path (header already read above for the name set).
            base = 8 + single_hlen
            with open(st_path, "rb") as f:
                blob = f.read()
            for name, td in single_header.items():
                yield from _scan(self._emit_tensor(
                    name, td, base, blob, family, want))

        if honest_gap:
            self._warn("weights", "no_output_weight",
                       "no lm_head.weight and not tied → no output projection")

        if not tied:
            return

        # Alias the input embedding as the (untied-on-disk) output projection.
        want_output = want is None or want("output.weight")
        if not want_output:
            return
        if embd_arr is None:
            # The embedding was filtered out or never decoded, so the tied
            # output cannot be produced under this filter. Not a crash, not a
            # degradation of the model itself — surface as info.
            self.log.emit("info", model=self.model, stage="weights",
                          code="tied_embeddings",
                          msg="tied output.weight requested but the input "
                              "embedding was not decoded → not produced")
            return
        self.log.emit("info", model=self.model, stage="weights",
                      code="tied_embeddings",
                      msg="no lm_head.weight: aliasing token_embd.weight as "
                          "output.weight (tied embeddings)")
        yield ("output.weight", embd_arr, embd_dtype)

    # GGUF token_type enum (mirrors tokenizer_forensics.TT) — the canonical
    # vocab structure encodes types as these ints so the SAME forensics code
    # runs on GGUF and HF models. We emit only the types HF distinguishes:
    #   NORMAL=1 (plain vocab), CONTROL=3 (special added token),
    #   USER_DEFINED=4 (non-special added token).
    _TT_NORMAL = 1
    _TT_CONTROL = 3
    _TT_USER_DEFINED = 4

    def tokenizer(self):
        """Parse ``tokenizer.json`` into the canonical vocab dict consumed by
        ``tokenizer_forensics.analyze`` (id-ordered ``tokens`` + parallel
        ``token_type`` ints + ``tokenizer.ggml.model``), or ``None``.

        Missing ``tokenizer.json`` ⇒ one ``no_tokenizer`` warning + ``None``
        (caller skips forensics — surfaced, never silent). An unparseable or
        unrecognized vocab shape degrades the same way (warn + ``None``).
        """
        tok_path = os.path.join(self.path, "tokenizer.json")
        if not os.path.exists(tok_path):
            self._warn("tokenizer", "no_tokenizer",
                       "no tokenizer.json → forensics skipped")
            return None

        # Parse + build under one guard: a malformed tokenizer.json (invalid
        # JSON, or a vocab id that isn't an int) must degrade exactly like an
        # unrecognized shape — one bad_vocab warning + None, never a crash.
        try:
            with open(tok_path, encoding="utf-8") as f:
                tj = json.load(f)

            model = tj.get("model") or {}
            vocab = model.get("vocab")

            # vocab -> (token_string, id) pairs. BPE: dict token->id. Unigram: a
            # list of [token, score] pairs whose position IS the id.
            if isinstance(vocab, dict):
                pairs = vocab.items()
            elif isinstance(vocab, list):
                pairs = []
                for i, entry in enumerate(vocab):
                    # Unigram entry is [token, score]; tolerate a bare token too.
                    tok = entry[0] if isinstance(entry, (list, tuple)) else entry
                    pairs.append((tok, i))
            else:
                self._warn("tokenizer", "bad_vocab",
                           "tokenizer.json model.vocab missing/unrecognized → "
                           "forensics skipped")
                return None

            # Build an id→token map AND an id→token_type map from BOTH the base
            # vocab and added_tokens. Chat-template specials (<|im_start|> etc.)
            # live ONLY in added_tokens at ids PAST the base vocab — they must be
            # kept (placed at their id) so forensics buckets them as special.
            by_id = {}
            type_by_id = {}
            for tok, tid in pairs:
                by_id[int(tid)] = str(tok)

            for at in tj.get("added_tokens") or []:
                aid = at.get("id")
                if not isinstance(aid, int) or aid < 0:
                    continue
                content = at.get("content")
                # Place the added token at its id (covers ids past base vocab);
                # only override an existing token string if content is present.
                if content is not None:
                    by_id[aid] = str(content)
                elif aid not in by_id:
                    by_id[aid] = ""
                # special → CONTROL (forensics' "special" bucket),
                # non-special added → USER_DEFINED.
                type_by_id[aid] = (
                    self._TT_CONTROL if at.get("special")
                    else self._TT_USER_DEFINED)

            if not by_id:
                self._warn("tokenizer", "empty_vocab",
                           "tokenizer.json vocab is empty → forensics skipped")
                return None

            # n spans the UNION of base + added ids so nothing past the base
            # vocab is dropped.
            n = max(by_id) + 1
            tokens = []
            token_type = []
            gaps = 0
            for i in range(n):
                if i in by_id:
                    tokens.append(by_id[i])
                    token_type.append(type_by_id.get(i, self._TT_NORMAL))
                else:
                    # A real id gap (no base or added token): fill empty/NORMAL
                    # and surface it once — phantom slots are never silent.
                    tokens.append("")
                    token_type.append(self._TT_NORMAL)
                    gaps += 1

            if gaps:
                self._warn("tokenizer", "vocab_gaps",
                           f"{gaps} vocab id gap(s) → filled empty")

            merges = [str(m) for m in model.get("merges") or []]
            return {
                "tokens": tokens,
                "token_type": token_type,
                # forensics' decode path keys off this; HF byte-level BPE matches
                # gpt2's reversible byte encoding (Ġ etc.), unigram uses ▁ spaces.
                "tokenizer.ggml.model": "gpt2" if isinstance(vocab, dict) else "llama",
                "merges": merges,
                "merges_n": len(merges) or None,
            }
        except (OSError, UnicodeDecodeError, ValueError, TypeError) as e:
            # JSONDecodeError (invalid JSON) and int()/coercion failures both
            # subclass ValueError; UnicodeDecodeError covers non-UTF8 files;
            # TypeError guards odd entry shapes. Degrade.
            self._warn("tokenizer", "bad_vocab",
                       f"unparseable tokenizer.json: {e}")
            return None
