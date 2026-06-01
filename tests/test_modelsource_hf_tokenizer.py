"""Tests für HFSource.tokenizer() — HF tokenizer.json -> canonical vocab dict.

The canonical structure is exactly what ``tokenizer_forensics.analyze`` consumes:
a dict with id-ordered ``tokens`` + parallel ``token_type`` (GGUF enum ints) +
``tokenizer.ggml.model``. The cross-check below feeds the returned structure
straight through the REAL ``analyze`` to prove the contract matches (not a guess).
"""
import io
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, HERE)

from modelsource.hf_backend import HFSource
from modelsource.log import RunLog

# Real consumer — proves the structure HFSource emits is the one forensics reads.
import tokenizer_forensics

try:
    import numpy as np  # noqa: F401
    HAVE_NP = True
except Exception:
    HAVE_NP = False

if HAVE_NP:
    from st_fixture import write_model


def _cfg():
    return {"architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "num_hidden_layers": 1, "num_attention_heads": 2,
            "num_key_value_heads": 1, "hidden_size": 4}


def _tensors():
    return {"model.embed_tokens.weight": ("F32", [4, 4], np.zeros((4, 4), "<f4"))}


def _tokenizer_json():
    # BPE dict-vocab form: token_string -> id (NOT id-ordered on disk).
    return {
        "model": {
            "type": "BPE",
            "vocab": {
                "<tok0>": 0,
                "tok1": 1,
                "Ġthe": 2,   # 'Ġthe' — gpt2 byte-level space marker
                "<unused3>": 3,
            },
        },
        "added_tokens": [
            {"id": 0, "content": "<tok0>", "special": True},
        ],
    }


@unittest.skipUnless(HAVE_NP, "numpy not installed")
class TestHFTokenizer(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = self._td.name

    def tearDown(self):
        self._td.cleanup()

    def test_vocab_is_id_ordered(self):
        d = write_model(os.path.join(self.tmp, "m"), _cfg(), _tensors(),
                        tokenizer=_tokenizer_json())
        tk = HFSource(d).tokenizer()
        self.assertIsNotNone(tk)
        # tokens must be id-ordered (position == id), as analyze() assumes.
        self.assertEqual(tk["tokens"], ["<tok0>", "tok1", "Ġthe", "<unused3>"])

    def test_token_type_parallel_and_special_flagged(self):
        d = write_model(os.path.join(self.tmp, "m"), _cfg(), _tensors(),
                        tokenizer=_tokenizer_json())
        tk = HFSource(d).tokenizer()
        self.assertEqual(len(tk["token_type"]), len(tk["tokens"]))
        # id 0 is a special added token -> CONTROL (GGUF enum 3); plain vocab
        # tokens -> NORMAL (1).
        self.assertEqual(tk["token_type"][0], 3)   # <tok0> special
        self.assertEqual(tk["token_type"][1], 1)   # tok1 normal
        self.assertEqual(tk["token_type"][2], 1)   # Ġthe normal

    def test_structure_keys_match_consumer(self):
        d = write_model(os.path.join(self.tmp, "m"), _cfg(), _tensors(),
                        tokenizer=_tokenizer_json())
        tk = HFSource(d).tokenizer()
        # Keys analyze()/read_tokenizer() read off the structure.
        self.assertIn("tokens", tk)
        self.assertIn("token_type", tk)
        self.assertIn("tokenizer.ggml.model", tk)
        self.assertIn("merges", tk)

    def test_feeds_real_forensics_analyze(self):
        """CROSS-CHECK: run the real tokenizer_forensics.analyze on our output."""
        d = write_model(os.path.join(self.tmp, "m"), _cfg(), _tensors(),
                        tokenizer=_tokenizer_json())
        tk = HFSource(d).tokenizer()
        res = tokenizer_forensics.analyze(tk)
        self.assertEqual(res["vocab_size"], 4)
        # <tok0> is a special (CONTROL) functional token.
        self.assertIn("<tok0>", res["special"])
        # <unused3> matches RESERVED_RE -> reserved bucket, NOT special.
        self.assertIn("<unused3>", res["reserved"])
        self.assertNotIn("<unused3>", res["special"])

    def test_missing_tokenizer_returns_none_and_warns(self):
        # No tokenizer.json written.
        d = write_model(os.path.join(self.tmp, "m"), _cfg(), _tensors())
        log = RunLog()
        err = io.StringIO()
        # Route stderr through the buffer so the test stays quiet.
        src = HFSource(d, log=log)
        sys_err = sys.stderr
        sys.stderr = err
        try:
            tk = src.tokenizer()
        finally:
            sys.stderr = sys_err
        self.assertIsNone(tk)
        # exactly one no_tokenizer warning, in both sinks
        no_tok = [e for e in log.entries if e["code"] == "no_tokenizer"]
        self.assertEqual(len(no_tok), 1)
        self.assertEqual(no_tok[0]["severity"], "warn")
        self.assertEqual(len(src.warnings), 1)
        self.assertIn("no_tokenizer", err.getvalue())

    def test_added_special_beyond_base_vocab_kept(self):
        """Chat-template specials live ONLY in added_tokens at ids PAST the base
        vocab — they must NOT be dropped, and must bucket as special."""
        tj = {
            "model": {
                "type": "BPE",
                "vocab": {"a": 0, "b": 1, "c": 2},  # base vocab ids 0-2 only
            },
            "added_tokens": [
                {"id": 3, "content": "<|im_start|>", "special": True},
                {"id": 4, "content": "<|im_end|>", "special": True},
            ],
        }
        d = write_model(os.path.join(self.tmp, "beyond"), _cfg(), _tensors(),
                        tokenizer=tj)
        tk = HFSource(d).tokenizer()
        self.assertIsNotNone(tk)
        # union of base vocab + added tokens -> length >= 5 (ids 0..4)
        self.assertGreaterEqual(len(tk["tokens"]), 5)
        # the id-3 added special is present at index 3 with CONTROL type
        self.assertEqual(tk["tokens"][3], "<|im_start|>")
        self.assertEqual(tk["token_type"][3], 3)   # CONTROL/special
        self.assertEqual(tk["tokens"][4], "<|im_end|>")
        self.assertEqual(tk["token_type"][4], 3)
        # CROSS-CHECK: real forensics buckets it as special.
        res = tokenizer_forensics.analyze(tk)
        self.assertIn("<|im_start|>", res["special"])
        self.assertIn("<|im_end|>", res["special"])

    def test_malformed_json_warns_and_none(self):
        """Invalid JSON in tokenizer.json -> warn + None, never raises."""
        d = write_model(os.path.join(self.tmp, "bad"), _cfg(), _tensors())
        # Write a broken tokenizer.json manually (write_model would JSON-encode).
        with open(os.path.join(d, "tokenizer.json"), "w") as f:
            f.write("{not json")
        log = RunLog()
        err = io.StringIO()
        src = HFSource(d, log=log)
        sys_err = sys.stderr
        sys.stderr = err
        try:
            tk = src.tokenizer()   # must NOT raise
        finally:
            sys.stderr = sys_err
        self.assertIsNone(tk)
        bad = [e for e in log.entries if e["code"] == "bad_vocab"]
        self.assertEqual(len(bad), 1)
        self.assertEqual(bad[0]["severity"], "warn")
        self.assertEqual(len(src.warnings), 1)
        self.assertIn("bad_vocab", err.getvalue())

    def test_non_utf8_tokenizer_warns_and_none(self):
        """A tokenizer.json that exists but is not UTF-8 must degrade cleanly."""
        d = write_model(os.path.join(self.tmp, "badutf8"), _cfg(), _tensors())
        with open(os.path.join(d, "tokenizer.json"), "wb") as f:
            f.write(b"\xff\xfe\x00not utf8")
        log = RunLog()
        err = io.StringIO()
        src = HFSource(d, log=log)
        sys_err = sys.stderr
        sys.stderr = err
        try:
            tk = src.tokenizer()   # must NOT raise UnicodeDecodeError
        finally:
            sys.stderr = sys_err
        self.assertIsNone(tk)
        bad = [e for e in log.entries if e["code"] == "bad_vocab"]
        self.assertEqual(len(bad), 1)
        self.assertEqual(bad[0]["severity"], "warn")
        self.assertEqual(len(src.warnings), 1)
        self.assertIn("bad_vocab", err.getvalue())

    def test_noninteger_vocab_id_warns(self):
        """A vocab id that isn't an int -> warn + None, no crash."""
        tj = {"model": {"type": "BPE", "vocab": {"a": "not-an-int", "b": 1}}}
        d = write_model(os.path.join(self.tmp, "nonint"), _cfg(), _tensors(),
                        tokenizer=tj)
        log = RunLog()
        err = io.StringIO()
        src = HFSource(d, log=log)
        sys_err = sys.stderr
        sys.stderr = err
        try:
            tk = src.tokenizer()
        finally:
            sys.stderr = sys_err
        self.assertIsNone(tk)
        bad = [e for e in log.entries if e["code"] == "bad_vocab"]
        self.assertEqual(len(bad), 1)
        self.assertEqual(len(src.warnings), 1)

    def test_vocab_gaps_filled_and_warned(self):
        # An id gap (no token at id 1) → that slot filled empty/NORMAL AND a
        # single vocab_gaps warn surfaces it (phantom slots never silent).
        tj = {"model": {"type": "BPE",
                        "vocab": {"a": 0, "c": 2, "d": 3}}}  # id 1 missing
        d = write_model(os.path.join(self.tmp, "gap"), _cfg(), _tensors(),
                        tokenizer=tj)
        log = RunLog()
        err = io.StringIO()
        src = HFSource(d, log=log)
        sys_err = sys.stderr
        sys.stderr = err
        try:
            tk = src.tokenizer()
        finally:
            sys.stderr = sys_err
        self.assertIsNotNone(tk)
        # n spans ids 0..3; the missing id 1 is filled empty/NORMAL.
        self.assertEqual(tk["tokens"], ["a", "", "c", "d"])
        self.assertEqual(tk["token_type"][1], 1)   # NORMAL
        gaps = [e for e in log.entries if e["code"] == "vocab_gaps"]
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["severity"], "warn")
        self.assertEqual(len(src.warnings), 1)
        self.assertIn("vocab_gaps", err.getvalue())

    def test_empty_vocab_warns_none(self):
        # An empty model.vocab dict → one empty_vocab warn + None.
        tj = {"model": {"type": "BPE", "vocab": {}}}
        d = write_model(os.path.join(self.tmp, "empty"), _cfg(), _tensors(),
                        tokenizer=tj)
        log = RunLog()
        err = io.StringIO()
        src = HFSource(d, log=log)
        sys_err = sys.stderr
        sys.stderr = err
        try:
            tk = src.tokenizer()
        finally:
            sys.stderr = sys_err
        self.assertIsNone(tk)
        empty = [e for e in log.entries if e["code"] == "empty_vocab"]
        self.assertEqual(len(empty), 1)
        self.assertEqual(empty[0]["severity"], "warn")
        self.assertEqual(len(src.warnings), 1)
        self.assertIn("empty_vocab", err.getvalue())

    def test_unigram_list_form_handled(self):
        """Unigram vocab is a list of [token, score] pairs."""
        tj = {
            "model": {
                "type": "Unigram",
                "vocab": [["<unk>", 0.0], ["the", -1.5], ["▁a", -2.0]],
            },
            "added_tokens": [{"id": 0, "content": "<unk>", "special": True}],
        }
        d = write_model(os.path.join(self.tmp, "u"), _cfg(), _tensors(),
                        tokenizer=tj)
        tk = HFSource(d).tokenizer()
        self.assertIsNotNone(tk)
        self.assertEqual(tk["tokens"], ["<unk>", "the", "▁a"])
        self.assertEqual(tk["token_type"][0], 3)   # special
        # Cross-check it still flows through the real consumer.
        res = tokenizer_forensics.analyze(tk)
        self.assertEqual(res["vocab_size"], 3)


if __name__ == "__main__":
    unittest.main()
