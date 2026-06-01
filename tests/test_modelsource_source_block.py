"""Source-block freshness: per-tensor warnings recorded DURING iter_weights()
must appear in the report's ``source.warnings``.

weight_stats.py and spectral.py used to snapshot the source block BEFORE
consuming iter_weights(), so warnings (tensor_unmapped, dtype_unsupported,
bad_offset, shard_missing, no_output_weight) raised during iteration were
silently dropped from the badged block. These tests build a REAL HF safetensors
model containing one UNMAPPABLE tensor and assert the badged block reflects what
iteration actually did (non-empty warnings, with the tensor_unmapped string).

Runs LOCALLY (stdlib + numpy, no gguf package needed)."""
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, HERE)

from st_fixture import write_model
from modelsource import detect
from modelsource.hf_backend import HFSource

# An llamalike config: archmap maps the standard tensors, but NOT the rotary
# buffer below — that one tensor drives the tensor_unmapped warning.
CONFIG = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "hidden_size": 4,
    "tie_word_embeddings": False,
}

# One mappable weight + one UNMAPPABLE tensor (rotary buffer has no canonical
# role → archmap.resolve() returns None → tensor_unmapped during iter_weights).
UNMAPPABLE = "model.layers.0.self_attn.rotary_emb.inv_freq"


def _build_hf(d):
    tensors = {
        "model.embed_tokens.weight": ("F32", (8, 4),
                                      np.arange(32, dtype="<f4").reshape(8, 4)),
        "model.layers.0.self_attn.q_proj.weight": (
            "F32", (4, 4), np.arange(16, dtype="<f4").reshape(4, 4)),
        UNMAPPABLE: ("F32", (2,), np.arange(2, dtype="<f4")),
    }
    return write_model(d, CONFIG, tensors)


class TestSourceBlockFreshness(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp = self._td.name
        self.hdir = os.path.join(self.tmp, "hf")
        _build_hf(self.hdir)
        # Sanity: detect() picks the HF backend, and that backend really does
        # raise tensor_unmapped for the rotary buffer during iteration.
        src = detect(self.hdir)
        self.assertIsInstance(src, HFSource)
        list(src.iter_weights())
        self.assertTrue(any("tensor_unmapped" in code or UNMAPPABLE in msg
                            for code, msg in
                            [(e["code"], e["msg"]) for e in src.log.entries]))

    def _assert_unmapped_warning(self, block):
        self.assertIsNotNone(block)
        warns = block["warnings"]
        self.assertTrue(warns, "source.warnings must be non-empty after an "
                               "unmappable tensor")
        self.assertTrue(any(UNMAPPABLE in w for w in warns),
                        f"expected the unmapped tensor in warnings, got {warns}")

    def test_weight_stats_source_block_has_iteration_warnings(self):
        import weight_stats as ws
        d = ws.analyze_model(self.hdir, source=detect(self.hdir))
        self._assert_unmapped_warning(d["source"])

    def test_spectral_source_block_has_iteration_warnings(self):
        import spectral as sp
        d = sp.analyze_model(self.hdir, source=detect(self.hdir))
        self._assert_unmapped_warning(d["source"])

    def test_model_diff_source_block_has_iteration_warnings(self):
        # model_diff already captures the block AFTER fully consuming the tensor
        # stream — guard it stays correct.
        import model_diff as md
        d = md.diff_models(self.hdir, self.hdir, "A", "B",
                           source_a=detect(self.hdir), source_b=detect(self.hdir))
        self._assert_unmapped_warning(d["source_a"])
        self._assert_unmapped_warning(d["source_b"])

    # NOTE: embedding_geometry deliberately requests ONLY token_embd.weight and
    # breaks after the first match, so it never drives the full tensor stream
    # past the unmappable tensor — its source block legitimately lacks the
    # tensor_unmapped warning. That is correct-by-design (it captures the block
    # AFTER its single-tensor iteration), so there is no freshness bug to guard.


if __name__ == "__main__":
    unittest.main()
