"""Tests für tools/modelsource/gguf_backend.py — der GGUFSource-Wrapper.

metadata()/precision laufen rein über den stdlib-Header-Parser + Fixture und
brauchen kein gguf-Paket. iter_weights() braucht gguf -> per skipUnless geguardet.
"""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, HERE)

from modelsource.gguf_backend import GGUFSource
from gguf_fixture import write_gguf, T_UINT32, T_STRING, T_ARRAY

try:
    import gguf  # noqa: F401
    HAVE_GGUF = True
except Exception:
    HAVE_GGUF = False

# ggml tensor type ids
F32, F16, BF16, Q6_K = 0, 1, 30, 14


def _kvs(arch="llama", file_type=18):
    return [
        ("general.architecture", T_STRING, arch),
        ("general.name", T_STRING, "Test Model"),
        ("general.file_type", T_UINT32, file_type),
        ("llama.block_count", T_UINT32, 2),
        ("llama.embedding_length", T_UINT32, 8),
        ("general.alignment", T_UINT32, 32),
        ("tokenizer.ggml.tokens", T_ARRAY, (T_STRING, ["a", "b", "c", "d"])),
    ]


def _write_exact(path):
    # all weight tensors un-quantized (F32/F16/BF16)
    tensors = [
        ("token_embd.weight", [4, 8], F32, 0),
        ("blk.0.attn_q.weight", [8, 8], F16, 0),
        ("blk.0.attn_q.bias", [8], F32, 0),
        ("blk.1.ffn_down.weight", [8, 16], BF16, 0),
        ("output.weight", [8, 4], F32, 0),
    ]
    return write_gguf(path, _kvs(file_type=0), tensors)


def _write_quant(path):
    # at least one K-quant weight tensor -> approx
    tensors = [
        ("token_embd.weight", [4, 8], F32, 0),
        ("blk.0.attn_q.weight", [8, 8], Q6_K, 0),
        ("blk.0.attn_q.bias", [8], F32, 0),
        ("blk.1.ffn_down.weight", [8, 16], Q6_K, 0),
        ("output.weight", [8, 4], F32, 0),
    ]
    return write_gguf(path, _kvs(file_type=18), tensors)


class TestGGUFBackendMetadata(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp = self._td.name

    def test_metadata_arch_and_source_block(self):
        p = _write_quant(os.path.join(self.tmp, "q.gguf"))
        meta = GGUFSource(p).metadata()
        # arch is surfaced both in the gguf_inspect metadata and the source block
        self.assertEqual(meta["metadata"]["general.architecture"], "llama")
        sb = meta["source"]
        self.assertEqual(sb["format"], "gguf")
        self.assertIs(sb["mapped"], True)
        self.assertEqual(sb["arch"], "llama")
        self.assertEqual(sb["warnings"], [])

    def test_precision_exact_for_unquantized(self):
        p = _write_exact(os.path.join(self.tmp, "e.gguf"))
        sb = GGUFSource(p).metadata()["source"]
        self.assertEqual(sb["precision"], "exact")

    def test_precision_approx_for_quantized(self):
        p = _write_quant(os.path.join(self.tmp, "q.gguf"))
        sb = GGUFSource(p).metadata()["source"]
        self.assertEqual(sb["precision"], "approx")

    def test_tokenizer_returns_none(self):
        p = _write_exact(os.path.join(self.tmp, "e.gguf"))
        self.assertIsNone(GGUFSource(p).tokenizer())


class TestIsExactHelper(unittest.TestCase):
    """_is_exact works directly on a gguf_inspect-style metadata dict."""

    def _meta(self, breakdown):
        return {"quant_breakdown": breakdown}

    def test_pure_f32(self):
        self.assertTrue(GGUFSource._is_exact(self._meta({"F32": 5})))

    def test_f16_bf16_mix(self):
        self.assertTrue(GGUFSource._is_exact(self._meta({"F16": 3, "BF16": 2, "F32": 1})))

    def test_any_quant_is_approx(self):
        self.assertFalse(GGUFSource._is_exact(self._meta({"F32": 3, "Q6_K": 2})))

    def test_q4k_is_approx(self):
        self.assertFalse(GGUFSource._is_exact(self._meta({"Q4_K": 10})))


class TestIterWeightsPassthrough(unittest.TestCase):
    @unittest.skipUnless(HAVE_GGUF, "gguf not installed")
    def test_iter_weights_canonical_passthrough(self):
        # build a real-ish exact GGUF; gguf.GGUFReader must be able to read it.
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        tmp = td.name
        p = _write_exact(os.path.join(tmp, "e.gguf"))
        names = [name for name, _arr, _ty in GGUFSource(p).iter_weights()]
        # names are already canonical blk.N.* / token_embd.* — unchanged
        self.assertIn("blk.0.attn_q.weight", names)
        self.assertTrue(any(n.startswith("blk.") for n in names))


if __name__ == "__main__":
    unittest.main()
