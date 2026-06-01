"""Tests für tools/gguf_inspect.py — der GGUF-Header-Parser."""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, HERE)

import gguf_inspect as gi
from gguf_fixture import write_gguf, T_UINT32, T_STRING, T_ARRAY, T_FLOAT32

# ggml tensor type ids
F32, Q6_K = 0, 14


def make_model(path):
    kvs = [
        ("general.architecture", T_STRING, "llama"),
        ("general.name", T_STRING, "Test Model"),
        ("general.file_type", T_UINT32, 18),            # -> Q6_K
        ("llama.block_count", T_UINT32, 2),
        ("llama.embedding_length", T_UINT32, 8),
        ("llama.attention.head_count", T_UINT32, 4),
        ("llama.attention.head_count_kv", T_UINT32, 2),
        ("llama.context_length", T_UINT32, 1024),
        ("general.alignment", T_UINT32, 32),
        ("tokenizer.ggml.tokens", T_ARRAY, (T_STRING, ["a", "b", "c", "d"])),
    ]
    tensors = [
        ("token_embd.weight", [4, 8], F32, 0),        # 32 params
        ("blk.0.attn_q.weight", [8, 8], Q6_K, 0),     # 64
        ("blk.0.attn_q.bias",   [8],    F32, 0),      # 8
        ("blk.1.ffn_down.weight", [8, 16], Q6_K, 0),  # 128
        ("output.weight", [8, 4], F32, 0),            # 32
    ]
    return write_gguf(path, kvs, tensors)


class TestGGUFInspect(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = make_model(os.path.join(self.tmp, "m.gguf"))
        self.info = gi.parse_gguf(self.path)

    def test_magic_and_version(self):
        self.assertEqual(self.info["gguf_version"], 3)
        self.assertTrue(self.info["header_hex"].startswith("47475546"))  # "GGUF"
        self.assertTrue(self.info["header_ascii"].startswith("GGUF"))

    def test_counts(self):
        self.assertEqual(self.info["n_tensors"], 5)
        self.assertEqual(self.info["n_kv"], 10)

    def test_total_params(self):
        # 32 + 64 + 8 + 128 + 32
        self.assertEqual(self.info["total_params"], 264)

    def test_quant_breakdown(self):
        self.assertEqual(self.info["quant_breakdown"], {"F32": 3, "Q6_K": 2})

    def test_file_type_label(self):
        self.assertEqual(self.info["file_type_label"], "Q6_K")

    def test_array_summarized(self):
        toks = self.info["metadata"]["tokenizer.ggml.tokens"]
        self.assertTrue(toks["_array"])
        self.assertEqual(toks["len"], 4)
        self.assertEqual(toks["sample"][:2], ["a", "b"])

    def test_alignment_and_data_offset(self):
        self.assertEqual(self.info["alignment"], 32)
        self.assertEqual(self.info["data_start"] % 32, 0)
        self.assertGreaterEqual(self.info["data_start"], self.info["header_end"])
        self.assertGreater(self.info["data_bytes"], 0)

    def test_bits_per_weight_positive(self):
        self.assertIsInstance(self.info["bits_per_weight"], float)
        self.assertGreater(self.info["bits_per_weight"], 0)

    def test_tensor_dims_and_types(self):
        by = {t["name"]: t for t in self.info["tensors"]}
        self.assertEqual(by["blk.0.attn_q.weight"]["type"], "Q6_K")
        self.assertEqual(by["blk.0.attn_q.weight"]["dims"], [8, 8])
        self.assertEqual(by["blk.0.attn_q.weight"]["params"], 64)

    def test_alignment_zero_no_crash(self):
        # regression: general.alignment==0 must not ZeroDivisionError
        kvs = [("general.architecture", T_STRING, "llama"),
               ("general.alignment", T_UINT32, 0)]
        p = write_gguf(os.path.join(self.tmp, "z.gguf"), kvs,
                       [("blk.0.attn_q.weight", [4, 4], Q6_K, 0)])
        info = gi.parse_gguf(p)
        self.assertEqual(info["alignment"], 32)        # fell back to default
        self.assertGreaterEqual(info["data_bytes"], 0)

    def test_rejects_non_gguf(self):
        bad = os.path.join(self.tmp, "bad.gguf")
        with open(bad, "wb") as f:
            f.write(b"NOPE" + b"\x00" * 32)
        with self.assertRaises(ValueError):
            gi.parse_gguf(bad)

    def test_alignment_nonint_falls_back(self):
        kvs = [("general.architecture", T_STRING, "llama"),
               ("general.alignment", T_STRING, "weird")]
        p = write_gguf(os.path.join(self.tmp, "a.gguf"), kvs,
                       [("blk.0.attn_q.weight", [4, 4], Q6_K, 0)])
        self.assertEqual(gi.parse_gguf(p)["alignment"], 32)

    def test_array_length_cap_raises(self):
        # regression: a declared array length beyond MAX_ARRAY must raise, not OOM
        import struct
        from gguf_fixture import MAGIC
        data = (struct.pack("<I", MAGIC) + struct.pack("<I", 3)
                + struct.pack("<Q", 0) + struct.pack("<Q", 1))   # 0 tensors, 1 kv
        key = b"x"
        data += (struct.pack("<Q", len(key)) + key
                 + struct.pack("<I", 9)        # value type ARRAY
                 + struct.pack("<I", 4)        # elem type UINT32
                 + struct.pack("<Q", 10 ** 12))  # absurd length, no items follow
        p = os.path.join(self.tmp, "big.gguf")
        with open(p, "wb") as f:
            f.write(data)
        with self.assertRaises(ValueError):
            gi.parse_gguf(p)

    def test_tensor_ndim_cap_raises(self):
        # regression: a malicious tensor directory can declare an absurd ndim.
        # Reject it before looping over a huge dimension list.
        import struct
        from gguf_fixture import MAGIC
        name = b"evil.weight"
        data = (struct.pack("<I", MAGIC) + struct.pack("<I", 3)
                + struct.pack("<Q", 1) + struct.pack("<Q", 0)
                + struct.pack("<Q", len(name)) + name
                + struct.pack("<I", gi.MAX_TENSOR_DIMS + 1))
        p = os.path.join(self.tmp, "ndim.gguf")
        with open(p, "wb") as f:
            f.write(data)
        with self.assertRaisesRegex(ValueError, "ndim"):
            gi.parse_gguf(p)


if __name__ == "__main__":
    unittest.main()
