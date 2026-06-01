"""Tests für sharded safetensors via model.safetensors.index.json.

Cross-Shard-Iteration: jeder Tensor wird aus SEINEM Shard gelesen und liefert
den korrekten kanonischen Namen + exakte fp32-Werte. ``want`` filtert über
Shards hinweg. Fehlender Shard → genau EIN ``shard_missing``-Error, die
vorhandenen Tensoren werden weiter geliefert, kein Crash.

Echte Tensor-Bytes via numpy → per HAVE_NP geguardet.
"""
import json
import os
import sys
import tempfile
import unittest
from collections import OrderedDict

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, HERE)

from modelsource.hf_backend import HFSource
from modelsource.log import RunLog

try:
    import numpy as np
    HAVE_NP = True
except Exception:
    HAVE_NP = False

if HAVE_NP:
    from st_fixture import write_sharded_model, write_model


def _llama_cfg():
    return {"architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "num_hidden_layers": 1, "num_attention_heads": 4,
            "num_key_value_heads": 2, "hidden_size": 8}


@unittest.skipUnless(HAVE_NP, "numpy not installed")
class TestHFShards(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = self._td.name

    def tearDown(self):
        self._td.cleanup()

    def _shards(self):
        embd = np.arange(10 * 8, dtype="<f4").reshape(10, 8)
        q = (np.arange(8 * 8, dtype="<f4").reshape(8, 8) * 0.5)
        shards = OrderedDict()
        shards["model-00001-of-00002.safetensors"] = {
            "model.embed_tokens.weight": ("F32", [10, 8], embd)}
        shards["model-00002-of-00002.safetensors"] = {
            "model.layers.0.self_attn.q_proj.weight": ("F32", [8, 8], q)}
        return shards, embd, q

    def test_cross_shard_iteration_yields_both_with_values(self):
        shards, embd, q = self._shards()
        d = write_sharded_model(os.path.join(self.tmp, "m"), _llama_cfg(), shards)
        rl = RunLog()
        src = HFSource(d, log=rl)
        out = list(src.iter_weights())
        by_name = {name: (arr, ty) for name, arr, ty in out}

        self.assertIn("token_embd.weight", by_name)
        self.assertIn("blk.0.attn_q.weight", by_name)

        a_embd, ty_embd = by_name["token_embd.weight"]
        self.assertEqual(ty_embd, "F32")
        self.assertEqual(a_embd.dtype, np.float32)
        self.assertEqual(a_embd.shape, (10, 8))
        np.testing.assert_array_equal(a_embd, embd)

        a_q, _ = by_name["blk.0.attn_q.weight"]
        self.assertEqual(a_q.shape, (8, 8))
        np.testing.assert_array_equal(a_q, q)

    def test_want_filters_across_shards(self):
        shards, _, _ = self._shards()
        d = write_sharded_model(os.path.join(self.tmp, "m"), _llama_cfg(), shards)
        src = HFSource(d, log=RunLog())
        out = list(src.iter_weights(want=lambda n: n == "blk.0.attn_q.weight"))
        self.assertEqual([name for name, _a, _t in out], ["blk.0.attn_q.weight"])

    def test_missing_shard_one_error_present_still_yielded(self):
        shards, embd, _ = self._shards()
        d = write_sharded_model(os.path.join(self.tmp, "m"), _llama_cfg(), shards)
        # Delete the second shard; its tensor must be skipped (one error), the
        # first shard's tensor must still be yielded, no crash.
        os.remove(os.path.join(d, "model-00002-of-00002.safetensors"))
        rl = RunLog()
        src = HFSource(d, log=rl)
        out = list(src.iter_weights())
        names = [name for name, _a, _t in out]
        self.assertEqual(names, ["token_embd.weight"])
        np.testing.assert_array_equal(out[0][1], embd)

        missing = [e for e in rl.entries if e["code"] == "shard_missing"]
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["severity"], "error")
        self.assertEqual(sum("shard_missing" == e["code"] for e in rl.entries), 1)
        # the missing shard error is surfaced in src.warnings; this fixture also
        # ships no lm_head and isn't tied → one honest no_output_weight warn too.
        no_out = [e for e in rl.entries if e["code"] == "no_output_weight"]
        self.assertEqual(len(no_out), 1)

    def test_unknown_family_empty_one_arch_warning(self):
        cfg = {"architectures": ["MambaXyzForCausalLM"], "model_type": "mamba-xyz",
               "num_hidden_layers": 1, "hidden_size": 4}
        shards = OrderedDict()
        shards["model-00001-of-00001.safetensors"] = {
            "backbone.embeddings.weight": ("F32", [4, 4], np.zeros((4, 4), "<f4"))}
        d = write_sharded_model(os.path.join(self.tmp, "mamba"), cfg, shards)
        rl = RunLog()
        src = HFSource(d, log=rl)
        self.assertEqual(list(src.iter_weights()), [])
        arch_w = [e for e in rl.entries if e["code"] == "arch_unmapped"]
        self.assertEqual(len(arch_w), 1)
        self.assertEqual(len(src.warnings), 1)

    def test_index_names_missing_tensor_is_skipped_not_crash(self):
        # Build a valid 2-shard model, then corrupt the index.json so its
        # weight_map references an extra tensor name that exists in NO shard
        # header (mapped to shard 1, whose header lacks it). Untrusted input:
        # the absent tensor must be skipped with one error, the valid tensors
        # must still be yielded, and iteration must NOT crash.
        shards, embd, q = self._shards()
        d = write_sharded_model(os.path.join(self.tmp, "m"), _llama_cfg(), shards)
        idx_path = os.path.join(d, "model.safetensors.index.json")
        with open(idx_path) as f:
            index = json.load(f)
        index["weight_map"]["model.embed_tokens.bogus"] = \
            "model-00001-of-00002.safetensors"
        with open(idx_path, "w") as f:
            json.dump(index, f)

        rl = RunLog()
        src = HFSource(d, log=rl)
        out = list(src.iter_weights())  # must not raise
        names = sorted(name for name, _a, _t in out)
        self.assertEqual(names, ["blk.0.attn_q.weight", "token_embd.weight"])
        by_name = {name: arr for name, arr, _ in out}
        np.testing.assert_array_equal(by_name["token_embd.weight"], embd)
        np.testing.assert_array_equal(by_name["blk.0.attn_q.weight"], q)

        missing = [e for e in rl.entries if e["code"] == "tensor_missing_in_shard"]
        self.assertEqual(len(missing), 1)
        self.assertIn("model.embed_tokens.bogus", missing[0]["msg"])
        self.assertTrue(
            any("model.embed_tokens.bogus" in w for w in src.warnings))

    def test_index_without_weight_map(self):
        # An index.json without a weight_map -> no weights, one bad_index
        # error, no crash.
        shards, _, _ = self._shards()
        d = write_sharded_model(os.path.join(self.tmp, "m"), _llama_cfg(), shards)
        idx_path = os.path.join(d, "model.safetensors.index.json")
        with open(idx_path, "w") as f:
            json.dump({}, f)

        rl = RunLog()
        src = HFSource(d, log=rl)
        self.assertEqual(list(src.iter_weights()), [])  # must not raise
        bad = [e for e in rl.entries if e["code"] == "bad_index"]
        self.assertEqual(len(bad), 1)

    def test_single_file_path_still_works_after_refactor(self):
        embd = np.arange(10 * 8, dtype="<f4").reshape(10, 8)
        tensors = {"model.embed_tokens.weight": ("F32", [10, 8], embd)}
        d = write_model(os.path.join(self.tmp, "single"), _llama_cfg(), tensors)
        src = HFSource(d, log=RunLog())
        out = list(src.iter_weights())
        by_name = {name: arr for name, arr, _ in out}
        self.assertIn("token_embd.weight", by_name)
        np.testing.assert_array_equal(by_name["token_embd.weight"], embd)


if __name__ == "__main__":
    unittest.main()
