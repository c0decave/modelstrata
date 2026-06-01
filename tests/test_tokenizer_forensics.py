"""Tests für tools/tokenizer_forensics.py — Klassifikation & Forensik."""
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, HERE)

import tokenizer_forensics as tf
from st_fixture import write_model


class TestGpt2ByteDecode(unittest.TestCase):
    def test_space_marker(self):
        # In GPT-2 byte-level BPE, U+0120 ('Ġ') represents a leading space.
        self.assertEqual(tf.decode_token("Ġabc", "gpt2"), " abc")

    def test_plain_ascii(self):
        self.assertEqual(tf.decode_token("hello", "gpt2"), "hello")

    def test_sentencepiece_space(self):
        # SentencePiece/llama uses U+2581 for space.
        self.assertEqual(tf.decode_token("▁the", "llama"), " the")


class TestScriptOf(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(tf.script_of("A"), "Latin")
        self.assertEqual(tf.script_of("z"), "Latin")
        self.assertEqual(tf.script_of("中"), "CJK")
        self.assertEqual(tf.script_of("5"), "digit")
        self.assertEqual(tf.script_of("д"), "Cyrillic")
        self.assertEqual(tf.script_of("א"), "Hebrew")
        self.assertEqual(tf.script_of("あ"), "Kana")
        self.assertEqual(tf.script_of("\U00020000"), "CJK")   # CJK Ext B, not emoji


class TestJaccard(unittest.TestCase):
    def test_overlap(self):
        self.assertEqual(tf.jaccard({"a", "b"}, {"b", "c"}), round(1 / 3, 4))

    def test_identical(self):
        self.assertEqual(tf.jaccard({"a", "b", "c"}, {"a", "b", "c"}), 1.0)

    def test_disjoint(self):
        self.assertEqual(tf.jaccard({"a"}, {"b"}), 0.0)

    def test_empty(self):
        self.assertEqual(tf.jaccard(set(), {"a"}), 0.0)


class TestClassificationRegex(unittest.TestCase):
    def test_reserved_matches_placeholders(self):
        self.assertTrue(tf.RESERVED_RE.search("<unused42>"))
        self.assertTrue(tf.RESERVED_RE.search("[PAD]"))
        self.assertTrue(tf.RESERVED_RE.search("<extra_id_7>"))

    def test_reserved_does_not_match_functional_special(self):
        # the key fix: chat/control tokens are NOT reserved/glitch
        self.assertIsNone(tf.RESERVED_RE.search("<|im_start|>"))
        self.assertIsNone(tf.RESERVED_RE.search("<|endoftext|>"))

    def test_special_matches_functional(self):
        self.assertTrue(tf.SPECIAL_RE.match("<|im_start|>"))
        self.assertTrue(tf.SPECIAL_RE.match("<s>"))
        self.assertTrue(tf.SPECIAL_RE.match("</s>"))

    def test_reserved_matches_numbered_pad(self):
        self.assertTrue(tf.RESERVED_RE.search("[PAD1]"))
        self.assertTrue(tf.RESERVED_RE.search("[PAD151669]"))

    def test_reserved_matches_placeholder_families(self):
        # placeholder/reserved slots from real fleets that were previously
        # MIS-counted as functional special tokens (they are typed CONTROL/
        # USER_DEFINED but are unused slots, not chat/tool tokens):
        self.assertTrue(tf.RESERVED_RE.search("<|reserved_special_token_0|>"))   # Llama-3
        self.assertTrue(tf.RESERVED_RE.search("<|reserved_special_token_250|>"))
        self.assertTrue(tf.RESERVED_RE.search("<dummy00001>"))                   # DeepSeek
        self.assertTrue(tf.RESERVED_RE.search("[control_8]"))                    # Mistral v3
        self.assertTrue(tf.RESERVED_RE.search("<SPECIAL_31>"))                   # Mistral/tekken

    def test_placeholders_do_not_match_real_functional_tokens(self):
        # the new reserved patterns must NOT swallow genuine tool/chat tokens
        for t in ("<|tool_call|>", "<|object_ref_start|>", "[TOOL_CALLS]",
                  "<|vision_start|>", "[gMASK]", "<|im_start|>"):
            self.assertIsNone(tf.RESERVED_RE.search(t), t)


class TestAnalyze(unittest.TestCase):
    def test_counts_split_reserved_vs_special(self):
        tokens = ["hello", "<unused0>", "<unused1>", "<|im_start|>", "world", "[PAD]"]
        # token_type: 1=NORMAL, 3=CONTROL, 5=UNUSED
        ttype = [1, 1, 5, 3, 1, 1]
        res = tf.analyze({"tokens": tokens, "token_type": ttype,
                          "tokenizer.ggml.model": "gpt2", "merges_n": 10})
        self.assertEqual(res["vocab_size"], 6)
        # <unused0> (regex), <unused1> (UNUSED type), [PAD] (regex) -> 3 reserved
        self.assertEqual(res["reserved_count"], 3)
        # <|im_start|> (CONTROL type + pattern) -> 1 special
        self.assertEqual(res["special_count"], 1)
        self.assertIn("<unused0>", res["reserved_sample"])
        self.assertIn("<|im_start|>", res["special_sample"])

    def test_reserved_placeholders_not_counted_as_functional(self):
        # Llama-3 reserved_special_token / DeepSeek dummy / Mistral control are
        # typed CONTROL but are placeholders -> reserved, NOT functional special.
        tokens = ["<|im_start|>", "<|reserved_special_token_0|>", "<dummy00001>",
                  "[control_8]", "<SPECIAL_31>", "<|object_ref_start|>"]
        ttype = [3, 3, 4, 3, 4, 3]  # all CONTROL/USER_DEFINED
        res = tf.analyze({"tokens": tokens, "token_type": ttype,
                          "tokenizer.ggml.model": "gpt2", "merges_n": 0})
        # only the two genuine functional tokens remain special
        self.assertEqual(res["special_count"], 2)
        self.assertIn("<|im_start|>", res["special_sample"])
        self.assertIn("<|object_ref_start|>", res["special_sample"])
        # the four placeholders are now reserved
        self.assertEqual(res["reserved_count"], 4)
        self.assertIn("<|reserved_special_token_0|>", res["reserved_sample"])

    def test_non_string_tokens_no_crash(self):
        # regression: malformed vocab with non-string elements must not crash
        res = tf.analyze({"tokens": [123, "ok", None], "token_type": [1, 1, 1],
                          "tokenizer.ggml.model": "gpt2", "merges_n": 0})
        self.assertEqual(res["vocab_size"], 3)

    def test_unknown_collected_and_full_lists(self):
        # token_type 2 = UNKNOWN tokens that are NOT named special tokens are
        # collected in `unknown`. (A typed-UNKNOWN token like "<unk>" that also
        # matches SPECIAL_RE is intentionally bucketed as a functional special
        # token by name — that path is covered separately below.)
        tokens = ["hi", "�", "￾", "<|im_start|>", "<unused0>"]
        ttype = [1, 2, 2, 3, 5]  # NORMAL, UNKNOWN, UNKNOWN, CONTROL, UNUSED
        res = tf.analyze({"tokens": tokens, "token_type": ttype,
                          "tokenizer.ggml.model": "gpt2", "merges_n": 0})
        self.assertEqual(res["unknown_count"], 2)
        self.assertEqual(res["unknown"], ["�", "￾"])
        self.assertFalse(res["unknown_truncated"])
        # full lists present and consistent with counts
        self.assertEqual(res["special"], ["<|im_start|>"])
        self.assertEqual(res["reserved"], ["<unused0>"])
        self.assertFalse(res["special_truncated"])
        self.assertFalse(res["reserved_truncated"])

    def test_named_unk_token_is_special_not_unknown(self):
        # "<unk>" matches SPECIAL_RE -> classified as a functional special token
        # even when its token_type is UNKNOWN(2). Documents the precedence.
        res = tf.analyze({"tokens": ["a", "<unk>"], "token_type": [1, 2],
                          "tokenizer.ggml.model": "gpt2", "merges_n": 0})
        self.assertIn("<unk>", res["special"])
        self.assertEqual(res["unknown_count"], 0)

    def test_full_list_truncation_flag(self):
        # more than FULL_CAP reserved tokens -> list capped, flag set
        n = tf.FULL_CAP + 5
        tokens = [f"<unused{i}>" for i in range(n)]
        ttype = [5] * n
        res = tf.analyze({"tokens": tokens, "token_type": ttype,
                          "tokenizer.ggml.model": "gpt2", "merges_n": 0})
        self.assertEqual(res["reserved_count"], n)
        self.assertEqual(len(res["reserved"]), tf.FULL_CAP)
        self.assertTrue(res["reserved_truncated"])


class TestHFTokenizerInput(unittest.TestCase):
    def test_read_tokenizer_any_hf_dir(self):
        with tempfile.TemporaryDirectory() as d:
            write_model(
                d,
                {"model_type": "llama", "architectures": ["LlamaForCausalLM"],
                 "num_hidden_layers": 1, "hidden_size": 4,
                 "num_attention_heads": 2, "num_key_value_heads": 1,
                 "intermediate_size": 8, "max_position_embeddings": 32},
                {"model.embed_tokens.weight": (
                    "F32", [4, 4], np.zeros((4, 4), dtype="<f4"))},
                tokenizer={
                    "model": {"vocab": {"hello": 0, "world": 1},
                              "merges": ["h e"]},
                    "added_tokens": [
                        {"id": 2, "content": "<|im_start|>", "special": True},
                        {"id": 3, "content": "<unused0>", "special": False},
                    ],
                },
            )
            tk = tf.read_tokenizer_any(d, label="hf")
            self.assertEqual(tk["tokens"], ["hello", "world", "<|im_start|>", "<unused0>"])
            self.assertEqual(tk["merges"], ["h e"])
            res = tf.analyze(tk)
            self.assertEqual(res["special_count"], 1)
            self.assertEqual(res["reserved_count"], 1)

    def test_hf_tokenizer_reader_is_stdlib_local_path(self):
        # Documents the design: tokenizer_forensics should not need the
        # modelsource/HF weight stack just to parse tokenizer.json.
        self.assertFalse(hasattr(tf, "detect"))


if __name__ == "__main__":
    unittest.main()
