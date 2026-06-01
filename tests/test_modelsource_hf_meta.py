"""Tests für tools/modelsource/hf_backend.py — der HFSource-Metadaten-Layer.

metadata() liest NUR den safetensors-Header (struct+json) + config.json. Das
braucht weder torch noch das safetensors-Paket. Die Fixture schreibt echte
Tensor-Bytes via numpy -> per HAVE_NP geguardet.
"""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, HERE)

from modelsource.hf_backend import HFSource

try:
    import numpy as np
    HAVE_NP = True
except Exception:
    HAVE_NP = False

if HAVE_NP:
    from st_fixture import write_model


def _llama_cfg():
    return {"architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "hidden_size": 8,
            "intermediate_size": 32, "max_position_embeddings": 4096}


def _llama_tensors():
    return {
        "model.embed_tokens.weight": ("F32", [10, 8], np.zeros((10, 8), "<f4")),
        "model.layers.0.self_attn.q_proj.weight": ("F32", [8, 8], np.zeros((8, 8), "<f4")),
    }


@unittest.skipUnless(HAVE_NP, "numpy not installed")
class TestHFMetadata(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = self._td.name

    def tearDown(self):
        self._td.cleanup()

    def test_arch_in_metadata_block(self):
        d = write_model(os.path.join(self.tmp, "m"), _llama_cfg(), _llama_tensors())
        meta = HFSource(d).metadata()
        self.assertEqual(meta["metadata"]["general.architecture"], "LlamaForCausalLM")

    def test_config_dims(self):
        d = write_model(os.path.join(self.tmp, "m"), _llama_cfg(), _llama_tensors())
        meta = HFSource(d).metadata()
        self.assertEqual(meta["n_layers"], 2)
        self.assertEqual(meta["n_heads"], 4)
        self.assertEqual(meta["n_kv_heads"], 2)
        self.assertEqual(meta["hidden"], 8)
        self.assertEqual(meta["ffn"], 32)
        self.assertEqual(meta["ctx"], 4096)

    def test_tensor_directory(self):
        d = write_model(os.path.join(self.tmp, "m"), _llama_cfg(), _llama_tensors())
        meta = HFSource(d).metadata()
        by_name = {t["name"]: t for t in meta["tensors"]}
        self.assertEqual(set(by_name), {
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight"})
        self.assertEqual(by_name["model.embed_tokens.weight"]["dims"], [10, 8])
        self.assertEqual(by_name["model.embed_tokens.weight"]["type"], "F32")
        self.assertEqual(by_name["model.layers.0.self_attn.q_proj.weight"]["dims"], [8, 8])

    def test_quant_breakdown(self):
        d = write_model(os.path.join(self.tmp, "m"), _llama_cfg(), _llama_tensors())
        meta = HFSource(d).metadata()
        self.assertEqual(meta["quant_breakdown"], {"F32": 2})

    def test_source_block_mapped_exact(self):
        d = write_model(os.path.join(self.tmp, "m"), _llama_cfg(), _llama_tensors())
        meta = HFSource(d).metadata()
        self.assertEqual(meta["source"], {
            "format": "safetensors",
            "precision": "exact",
            "arch": "LlamaForCausalLM",
            "mapped": True,
            "warnings": [],
        })

    def test_unmappable_arch_inventory_only(self):
        cfg = {"architectures": ["MambaXyzForCausalLM"], "model_type": "mamba-xyz",
               "num_hidden_layers": 1, "hidden_size": 4}
        tensors = {"backbone.embeddings.weight": ("F32", [4, 4], np.zeros((4, 4), "<f4"))}
        d = write_model(os.path.join(self.tmp, "mamba"), cfg, tensors)
        meta = HFSource(d).metadata()
        sb = meta["source"]
        self.assertIs(sb["mapped"], False)
        self.assertEqual(sb["precision"], "inventory-only")
        self.assertEqual(sb["arch"], "MambaXyzForCausalLM")

    def test_arch_falls_back_to_model_type(self):
        cfg = {"model_type": "llama", "num_hidden_layers": 1,
               "num_attention_heads": 2, "hidden_size": 4}
        tensors = {"model.embed_tokens.weight": ("F32", [4, 4], np.zeros((4, 4), "<f4"))}
        d = write_model(os.path.join(self.tmp, "noarchs"), cfg, tensors)
        meta = HFSource(d).metadata()
        self.assertEqual(meta["source"]["arch"], "llama")

    def test_tokenizer_missing_returns_none(self):
        # No tokenizer.json in this model dir → tokenizer() degrades to None
        # (full parsing behaviour lives in test_modelsource_hf_tokenizer.py).
        d = write_model(os.path.join(self.tmp, "m"), _llama_cfg(), _llama_tensors())
        self.assertIsNone(HFSource(d).tokenizer())


if __name__ == "__main__":
    unittest.main()
