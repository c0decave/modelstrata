#!/usr/bin/env python3
"""
tokenizer_forensics.py — Ebene 3: statische Tokenizer-Forensik (kein Modell-Lauf).

Liest das VOLLE Vokabular (nicht nur eine Stichprobe) aus jedem GGUF und berechnet:
  - token_type-Verteilung (NORMAL/CONTROL/USER_DEFINED/UNUSED/BYTE/UNKNOWN)
  - Glitch-Token-Kandidaten (UNUSED + reservierte Platzhalter-Muster)
  - Skript-/Sprachabdeckung (mit Rück-Decodierung der gpt2-Byte-Kodierung)
  - Vokab-Overlap (Jaccard) zwischen allen Modellen -> gemeinsame Herkunft

Gibt reports/tokenizer_forensics.json aus (ohne die vollen Vokabeln einzubetten;
nur Zusammenfassungen + die Overlap-Matrix).

  python3 tokenizer_forensics.py --scan /path/to/models --ollama /usr/share/ollama/.ollama/models -o tf.json
"""
import argparse
import json
import re
import sys
import unicodedata

from gguf_inspect import Reader, MAGIC, GT_ARRAY, discover_ollama
import os

# GGUF token_type enum
TT = {1: "NORMAL", 2: "UNKNOWN", 3: "CONTROL", 4: "USER_DEFINED",
      5: "UNUSED", 6: "BYTE"}

# Cap for the full token lists embedded in the report (special/reserved/
# unknown). Functional-special and unknown lists are tiny (well under this);
# reserved can be large on big vocabs, so we bound the JSON size. The
# *_truncated flag tells the dashboard when a list was clipped.
FULL_CAP = 4000

# reserved/placeholder slots = static candidates for "likely-untrained" tokens.
# NOTE: this is NOT true glitch-token detection (à la SolidGoldMagikarp); that
# needs embedding norms (Ebene 6). Functional special tokens (<|im_start|> etc.)
# are tracked separately and are NOT counted here.
RESERVED_RE = re.compile(
    r"(<unused\d+>|<extra_id_\d+>|<reserved[^>]*>|\[PAD\d*\]|<pad>|madeupword\d+"
    # placeholder/reserved slots that ship typed CONTROL/USER_DEFINED but are
    # unused fillers, not functional chat/tool tokens — keep them OUT of the
    # functional-special list (else they bury the real tokens & inflate counts):
    r"|<\|reserved_special_token_\d+\|>"   # Llama-3 (~250 slots)
    r"|<dummy\d+>"                          # DeepSeek-coder
    r"|\[control_\d+\]"                     # Mistral v3
    r"|<SPECIAL_\d+>)", re.I)               # Mistral / tekken
# functional special / control tokens (intentional, used) — not glitches
SPECIAL_RE = re.compile(r"^<\|.*\|>$|^<(s|/s|unk|mask|eos|bos|pad|cls|sep)>$", re.I)


def _gpt2_byte_decoder():
    """Reverse of GPT-2 bytes_to_unicode: remapped-unicode-char -> original byte."""
    bs = list(range(ord("!"), ord("~") + 1)) + \
        list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTE_DEC = _gpt2_byte_decoder()


def decode_token(tok, model):
    """Recover human text from a stored token string."""
    if model == "gpt2":
        try:
            raw = bytes(BYTE_DEC[ch] for ch in tok if ch in BYTE_DEC)
            return raw.decode("utf-8", "replace")
        except Exception:
            return tok
    # sentencepiece/llama: U+2581 is space
    return tok.replace("▁", " ")


def script_of(ch):
    o = ord(ch)
    if ch.isdigit():
        return "digit"
    if (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF
            or 0x20000 <= o <= 0x2EBEF):   # incl. CJK Ext B–F (above U+1F000)
        return "CJK"
    if 0x3040 <= o <= 0x30FF:
        return "Kana"
    if 0xAC00 <= o <= 0xD7AF:
        return "Hangul"
    if 0x0400 <= o <= 0x04FF:
        return "Cyrillic"
    if 0x0600 <= o <= 0x06FF:
        return "Arabic"
    if 0x0590 <= o <= 0x05FF:
        return "Hebrew"
    if 0x0370 <= o <= 0x03FF:
        return "Greek"
    if 0x0900 <= o <= 0x097F:
        return "Devanagari"
    if o > 0x1F000:
        return "emoji/symbol"
    if ("a" <= ch.lower() <= "z") or 0x00C0 <= o <= 0x024F:
        return "Latin"
    if ch.isspace():
        return "space"
    cat = unicodedata.category(ch)
    if cat.startswith("P"):
        return "punct"
    return "other"


def read_tokenizer(path):
    with open(path, "rb") as f:
        r = Reader(f)
        if r.u32() != MAGIC:
            raise ValueError("not GGUF")
        r.u32()            # version
        r.u64()            # n_tensors
        n_kv = r.u64()
        out = {"merges": [], "merges_n": None}
        tokens = ttype = None
        for _ in range(n_kv):
            key = r.gstr()
            vt = r.u32()
            val = r.value(vt)
            is_arr = isinstance(val, tuple) and len(val) == 3
            if key == "tokenizer.ggml.tokens" and is_arr:
                tokens = [str(t) for t in val[0]]   # coerce: tokens must be str
            elif key == "tokenizer.ggml.token_type" and is_arr:
                ttype = val[0]
            elif key == "tokenizer.ggml.merges" and is_arr:
                out["merges"] = [str(m) for m in val[0]]
                out["merges_n"] = val[2]
            elif key in ("general.architecture", "tokenizer.ggml.model",
                         "tokenizer.ggml.pre"):
                out[key] = val
            elif key.startswith("tokenizer.ggml.") and key.endswith("_token_id"):
                out[key] = val
        out["tokens"] = tokens or []
        out["token_type"] = ttype or []
    return out


def _is_gguf_path(path):
    if str(path).lower().endswith(".gguf"):
        return True
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF"
    except OSError:
        return False


def read_tokenizer_any(path, *, label=None, log=None):
    """Read tokenizer data from GGUF or any modelsource-backed model.

    GGUF keeps its specialized reader because GGUFSource intentionally leaves
    tokenizer extraction here. HF safetensors reuse HFSource.tokenizer(), which
    returns the same canonical ``tokens`` / ``token_type`` shape consumed by
    ``analyze`` below. Inventory-only formats return ``None`` and are skipped.
    """
    if _is_gguf_path(path):
        return read_tokenizer(path)
    if os.path.isdir(path):
        return read_hf_tokenizer(path)
    return None


def read_hf_tokenizer(path):
    """Parse an HF ``tokenizer.json`` into the canonical tokenizer dict.

    This mirrors ``HFSource.tokenizer`` but stays local and stdlib-only so
    tokenizer forensics remain cheap even without numpy/torch installed.
    """
    tok_path = os.path.join(path, "tokenizer.json")
    if not os.path.exists(tok_path):
        return None
    try:
        with open(tok_path, encoding="utf-8") as f:
            tj = json.load(f)
        model = tj.get("model") or {}
        vocab = model.get("vocab")
        if isinstance(vocab, dict):
            pairs = vocab.items()
            ggml_model = "gpt2"
        elif isinstance(vocab, list):
            pairs = []
            for i, entry in enumerate(vocab):
                tok = entry[0] if isinstance(entry, (list, tuple)) else entry
                pairs.append((tok, i))
            ggml_model = "llama"
        else:
            return None

        by_id = {}
        type_by_id = {}
        for tok, tid in pairs:
            by_id[int(tid)] = str(tok)
        for at in tj.get("added_tokens") or []:
            aid = at.get("id")
            if not isinstance(aid, int) or aid < 0:
                continue
            content = at.get("content")
            by_id[aid] = str(content) if content is not None else by_id.get(aid, "")
            type_by_id[aid] = 3 if at.get("special") else 4
        if not by_id:
            return None
        tokens = []
        token_type = []
        for i in range(max(by_id) + 1):
            tokens.append(by_id.get(i, ""))
            token_type.append(type_by_id.get(i, 1))
        merges = [str(m) for m in model.get("merges") or []]
        return {
            "tokens": tokens,
            "token_type": token_type,
            "tokenizer.ggml.model": ggml_model,
            "merges": merges,
            "merges_n": len(merges) or None,
        }
    except (OSError, UnicodeDecodeError, ValueError, TypeError):
        return None


def analyze(tk):
    tokens = [str(t) for t in tk["tokens"]]   # robust against non-string elements
    ttype = tk["token_type"]
    model = tk.get("tokenizer.ggml.model", "")
    type_counts = {}
    for t in ttype:
        type_counts[TT.get(t, str(t))] = type_counts.get(TT.get(t, str(t)), 0) + 1

    # distinguish functional special tokens from reserved/unused placeholder slots.
    # NOTE: reserved is checked first, so placeholder-named tokens (<pad>, [PAD])
    # count as reserved even if also CONTROL-typed — they are placeholders, not
    # content-bearing special tokens like <|im_start|>.
    reserved = []
    special = []
    unknown = []
    for i, tok in enumerate(tokens):
        tok = str(tok)
        ty = ttype[i] if i < len(ttype) else 1
        tn = TT.get(ty)
        if tn == "UNUSED" or RESERVED_RE.search(tok):
            reserved.append(tok)
        elif tn in ("CONTROL", "USER_DEFINED") or SPECIAL_RE.match(tok):
            special.append(tok)
        elif tn == "UNKNOWN":
            unknown.append(tok)
    # script histogram over a representative slice (decode is costly for 260k)
    script = {}
    step = max(1, len(tokens) // 40000)   # cap at ~40k decoded tokens
    sampled = 0
    for tok in tokens[::step]:
        sampled += 1
        seen = set()
        for ch in decode_token(tok, model):
            seen.add(script_of(ch))
        for s in seen:
            script[s] = script.get(s, 0) + 1
    return {
        "vocab_size": len(tokens),
        "merges_n": tk.get("merges_n"),
        "tokenizer_model": model,
        "pre": tk.get("tokenizer.ggml.pre", ""),
        "type_counts": type_counts,
        "reserved_count": len(reserved),
        "reserved_sample": reserved[:40],
        "reserved": reserved[:FULL_CAP],
        "reserved_truncated": len(reserved) > FULL_CAP,
        "special_count": len(special),
        "special_sample": special[:40],
        "special": special[:FULL_CAP],
        "special_truncated": len(special) > FULL_CAP,
        "unknown_count": len(unknown),
        "unknown": unknown[:FULL_CAP],
        "unknown_truncated": len(unknown) > FULL_CAP,
        "script_sampled": sampled,
        "script_hist": dict(sorted(script.items(), key=lambda x: -x[1])),
    }


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return round(inter / (len(a) + len(b) - inter), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="append", default=[])
    ap.add_argument("--ollama", action="append", default=[])
    ap.add_argument("--model", action="append", default=[],
                    help="explicit model path: GGUF file or HF model dir")
    ap.add_argument("--hf", action="append", default=[],
                    help="explicit HF model dir (alias of --model)")
    ap.add_argument("-o", "--out", default="tokenizer_forensics.json")
    args = ap.parse_args()

    targets = []  # (label, path)
    for m in list(args.model) + list(args.hf):
        targets.append((os.path.basename(os.path.normpath(str(m))), m))
    for d in args.scan:
        for dp, _, files in os.walk(d):
            for fn in files:
                if fn.lower().endswith(".gguf"):
                    targets.append((fn, os.path.join(dp, fn)))
    for od in args.ollama:
        for name, blob in sorted(discover_ollama(od).items()):
            targets.append((f"ollama:{name}", blob))

    models = {}
    vocab_sets = {}
    for label, path in targets:
        try:
            tk = read_tokenizer_any(path, label=label)
            if tk is None:
                print(f"!! {label}: tokenizer unavailable/skipped", file=sys.stderr)
                continue
            models[label] = analyze(tk)
            vocab_sets[label] = set(tk["tokens"])
            print(f"  {label}: {len(tk['tokens'])} tokens, "
                  f"{models[label]['reserved_count']} reserved, "
                  f"{models[label]['special_count']} special", file=sys.stderr)
        except Exception as e:
            print(f"!! {label}: {type(e).__name__}: {e}", file=sys.stderr)

    # pairwise Jaccard overlap matrix
    labels = list(vocab_sets)
    mat = [[jaccard(vocab_sets[a], vocab_sets[b]) for b in labels] for a in labels]

    out = {"models": models, "overlap": {"labels": labels, "jaccard": mat}}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"[tokenizer_forensics: {args.out}  {len(models)} models]", file=sys.stderr)


if __name__ == "__main__":
    main()
