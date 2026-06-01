"""Tests für Task 10 — tied embeddings + sharded-aware metadata().

Part A (Regressionsschutz): metadata() darf bei einem sharded HF-Modell (kein
``model.safetensors``, nur Shards + index.json) NICHT crashen und muss die
Tensoren aus ALLEN Shards in seiner ``tensors``-Directory vereinen.

Part B (tied embeddings): Modelle ohne ``lm_head.weight`` binden den Output an
das Input-Embedding. metadata() meldet ``tied: True``; iter_weights() liefert
zusätzlich ein aliasiertes ``output.weight`` == Embedding und loggt GENAU EIN
``tied_embeddings``-info (NICHT in source.warnings, da info, keine Degradation).

Echte Tensor-Bytes via numpy → per HAVE_NP geguardet.
"""
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
    from st_fixture import write_model, write_sharded_model


def _llama_cfg(**over):
    cfg = {"architectures": ["LlamaForCausalLM"], "model_type": "llama",
           "num_hidden_layers": 1, "num_attention_heads": 4,
           "num_key_value_heads": 2, "hidden_size": 8}
    cfg.update(over)
    return cfg


@unittest.skipUnless(HAVE_NP, "numpy not installed")
class TestTiedEmbeddings(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = self._td.name

    def tearDown(self):
        self._td.cleanup()

    # ---- tied: single file, no lm_head ------------------------------------
    def _tied_single(self, **cfg_over):
        embd = np.arange(10 * 8, dtype="<f4").reshape(10, 8) * 0.25
        tensors = {
            "model.embed_tokens.weight": ("F32", [10, 8], embd),
            "model.layers.0.self_attn.q_proj.weight":
                ("F32", [8, 8], np.zeros((8, 8), "<f4")),
        }
        cfg_kw = {"tie_word_embeddings": True}
        cfg_kw.update(cfg_over)
        cfg = _llama_cfg(**cfg_kw)
        d = write_model(os.path.join(self.tmp, "tied"), cfg, tensors)
        return d, embd

    def test_metadata_tied_true(self):
        d, _ = self._tied_single()
        meta = HFSource(d).metadata()
        self.assertIs(meta["tied"], True)

    def test_iter_yields_aliased_output_equal_to_embedding(self):
        d, embd = self._tied_single()
        rl = RunLog()
        src = HFSource(d, log=rl)
        out = list(src.iter_weights())
        by_name = {name: arr for name, arr, _ in out}
        self.assertIn("token_embd.weight", by_name)
        self.assertIn("output.weight", by_name)
        np.testing.assert_array_equal(by_name["output.weight"], embd)
        np.testing.assert_array_equal(
            by_name["output.weight"], by_name["token_embd.weight"])

    def test_exactly_one_tied_embeddings_info_not_in_warnings(self):
        d, _ = self._tied_single()
        rl = RunLog()
        src = HFSource(d, log=rl)
        list(src.iter_weights())
        tied = [e for e in rl.entries if e["code"] == "tied_embeddings"]
        self.assertEqual(len(tied), 1)
        self.assertEqual(tied[0]["severity"], "info")
        # info, keine Degradation → NICHT in source.warnings
        self.assertFalse(any("tied" in w.lower() for w in src.warnings))

    def test_want_filters_aliased_output(self):
        d, _ = self._tied_single()
        src = HFSource(d, log=RunLog())
        out = list(src.iter_weights(want=lambda n: n == "token_embd.weight"))
        names = [name for name, _a, _t in out]
        self.assertIn("token_embd.weight", names)
        self.assertNotIn("output.weight", names)

    def test_gemma_absent_flag_no_lm_head_is_tied(self):
        # Gemma ties embeddings by architecture default even when the config
        # OMITS tie_word_embeddings (real case: unsloth gemma-3-1b → model_type
        # "gemma3_text", flag absent, no lm_head.weight). Must alias the output
        # and NOT emit a false no_output_weight gap.
        embd = np.arange(10 * 8, dtype="<f4").reshape(10, 8) * 0.25
        tensors = {
            "model.embed_tokens.weight": ("F32", [10, 8], embd),
            "model.layers.0.self_attn.q_proj.weight":
                ("F32", [8, 8], np.zeros((8, 8), "<f4")),
        }
        cfg = {"architectures": ["Gemma3ForCausalLM"], "model_type": "gemma3_text",
               "num_hidden_layers": 1, "num_attention_heads": 4,
               "num_key_value_heads": 2, "hidden_size": 8}  # NO tie_word_embeddings
        d = write_model(os.path.join(self.tmp, "gemma"), cfg, tensors)
        rl = RunLog()
        src = HFSource(d, log=rl)
        self.assertIs(src.metadata()["tied"], True)
        names = [n for n, _a, _t in src.iter_weights()]
        self.assertIn("output.weight", names)
        self.assertFalse([e for e in rl.entries if e["code"] == "no_output_weight"])

    def test_tie_false_but_lm_head_absent_is_honest_warn(self):
        # tie_word_embeddings: false WITH a missing lm_head → honest warn,
        # not a silently fabricated output.
        d, _ = self._tied_single(tie_word_embeddings=False)
        rl = RunLog()
        src = HFSource(d, log=rl)
        meta = src.metadata()
        self.assertIs(meta["tied"], False)
        out = list(src.iter_weights())
        names = [name for name, _a, _t in out]
        self.assertNotIn("output.weight", names)
        no_out = [e for e in rl.entries if e["code"] == "no_output_weight"]
        self.assertEqual(len(no_out), 1)
        self.assertEqual(no_out[0]["severity"], "warn")

    def test_tie_flag_absent_lm_head_absent_warns_once(self):
        # Fix 5: a mappable model with NO lm_head.weight and NO
        # tie_word_embeddings flag (absent) → conservatively NOT tied, AND the
        # missing output projection must not be silent: exactly one
        # no_output_weight warn, tied is False, no output.weight yielded.
        embd = np.arange(10 * 8, dtype="<f4").reshape(10, 8) * 0.25
        tensors = {
            "model.embed_tokens.weight": ("F32", [10, 8], embd),
            "model.layers.0.self_attn.q_proj.weight":
                ("F32", [8, 8], np.zeros((8, 8), "<f4")),
        }
        # _llama_cfg() does NOT set tie_word_embeddings → flag absent.
        d = write_model(os.path.join(self.tmp, "absent"), _llama_cfg(), tensors)
        rl = RunLog()
        src = HFSource(d, log=rl)
        meta = src.metadata()
        self.assertIs(meta["tied"], False)
        out = list(src.iter_weights())
        names = [name for name, _a, _t in out]
        self.assertNotIn("output.weight", names)
        no_out = [e for e in rl.entries if e["code"] == "no_output_weight"]
        self.assertEqual(len(no_out), 1)
        self.assertEqual(no_out[0]["severity"], "warn")
        self.assertEqual(
            no_out[0]["msg"],
            "no lm_head.weight and not tied → no output projection")

    def test_lm_head_present_emits_no_no_output_weight(self):
        # Confirm the lm_head-present path emits NO no_output_weight.
        embd = np.arange(10 * 8, dtype="<f4").reshape(10, 8) * 0.25
        head = np.full((10, 8), 7.0, dtype="<f4")
        tensors = {
            "model.embed_tokens.weight": ("F32", [10, 8], embd),
            "lm_head.weight": ("F32", [10, 8], head),
        }
        d = write_model(os.path.join(self.tmp, "head"),
                        _llama_cfg(tie_word_embeddings=False), tensors)
        rl = RunLog()
        list(HFSource(d, log=rl).iter_weights())
        self.assertFalse(
            [e for e in rl.entries if e["code"] == "no_output_weight"])

    def test_tied_path_emits_no_no_output_weight(self):
        # Confirm the genuinely-tied path emits NO no_output_weight.
        d, _ = self._tied_single()  # tie_word_embeddings=True
        rl = RunLog()
        list(HFSource(d, log=rl).iter_weights())
        self.assertFalse(
            [e for e in rl.entries if e["code"] == "no_output_weight"])

    # ---- NOT tied: lm_head present ----------------------------------------
    def test_lm_head_present_not_tied_real_output(self):
        embd = np.arange(10 * 8, dtype="<f4").reshape(10, 8) * 0.25
        head = np.full((10, 8), 7.0, dtype="<f4")
        tensors = {
            "model.embed_tokens.weight": ("F32", [10, 8], embd),
            "lm_head.weight": ("F32", [10, 8], head),
        }
        d = write_model(os.path.join(self.tmp, "untied"),
                        _llama_cfg(tie_word_embeddings=False), tensors)
        rl = RunLog()
        src = HFSource(d, log=rl)
        meta = src.metadata()
        self.assertIs(meta["tied"], False)
        out = list(src.iter_weights())
        by_name = {name: arr for name, arr, _ in out}
        self.assertIn("output.weight", by_name)
        # the real lm_head, NOT the aliased embedding
        np.testing.assert_array_equal(by_name["output.weight"], head)
        self.assertFalse(
            any(e["code"] == "tied_embeddings" for e in rl.entries))

    # ---- Part A: sharded metadata regression guard ------------------------
    def _shards(self):
        embd = np.arange(10 * 8, dtype="<f4").reshape(10, 8)
        q = np.arange(8 * 8, dtype="<f4").reshape(8, 8) * 0.5
        shards = OrderedDict()
        shards["model-00001-of-00002.safetensors"] = {
            "model.embed_tokens.weight": ("F32", [10, 8], embd)}
        shards["model-00002-of-00002.safetensors"] = {
            "model.layers.0.self_attn.q_proj.weight": ("F32", [8, 8], q),
            "model.norm.weight": ("F16", [8], np.zeros(8, "<f2"))}
        return shards, embd, q

    def test_sharded_metadata_no_crash_union_directory(self):
        shards, _, _ = self._shards()
        d = write_sharded_model(os.path.join(self.tmp, "sh"),
                                _llama_cfg(), shards)
        meta = HFSource(d).metadata()  # must not crash
        names = {t["name"] for t in meta["tensors"]}
        self.assertEqual(names, {
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.norm.weight"})
        # quant_breakdown counts tensors from BOTH shards
        self.assertEqual(meta["quant_breakdown"], {"F32": 2, "F16": 1})

    def test_sharded_metadata_dims_from_each_shard(self):
        shards, _, _ = self._shards()
        d = write_sharded_model(os.path.join(self.tmp, "sh"),
                                _llama_cfg(), shards)
        by_name = {t["name"]: t for t in HFSource(d).metadata()["tensors"]}
        self.assertEqual(by_name["model.embed_tokens.weight"]["dims"], [10, 8])
        self.assertEqual(
            by_name["model.layers.0.self_attn.q_proj.weight"]["dims"], [8, 8])
        self.assertEqual(by_name["model.norm.weight"]["dims"], [8])

    def test_sharded_missing_lm_head_is_tied(self):
        # weight_map lacks lm_head → tied True; aliased output across shards.
        shards, embd, _ = self._shards()
        d = write_sharded_model(os.path.join(self.tmp, "sh"),
                                _llama_cfg(tie_word_embeddings=True), shards)
        rl = RunLog()
        src = HFSource(d, log=rl)
        self.assertIs(src.metadata()["tied"], True)
        out = list(src.iter_weights())
        by_name = {name: arr for name, arr, _ in out}
        self.assertIn("token_embd.weight", by_name)
        self.assertIn("output.weight", by_name)
        np.testing.assert_array_equal(by_name["output.weight"], embd)
        tied = [e for e in rl.entries if e["code"] == "tied_embeddings"]
        self.assertEqual(len(tied), 1)

    def test_sharded_with_lm_head_not_tied(self):
        embd = np.arange(10 * 8, dtype="<f4").reshape(10, 8)
        head = np.full((10, 8), 3.0, dtype="<f4")
        shards = OrderedDict()
        shards["model-00001-of-00002.safetensors"] = {
            "model.embed_tokens.weight": ("F32", [10, 8], embd)}
        shards["model-00002-of-00002.safetensors"] = {
            "lm_head.weight": ("F32", [10, 8], head)}
        d = write_sharded_model(os.path.join(self.tmp, "sh"),
                                _llama_cfg(), shards)
        self.assertIs(HFSource(d).metadata()["tied"], False)

    # ---- Part A degrade: malformed index / missing shard ------------------
    def test_metadata_missing_shard_degrades_no_crash(self):
        shards, _, _ = self._shards()
        d = write_sharded_model(os.path.join(self.tmp, "sh"),
                                _llama_cfg(), shards)
        os.remove(os.path.join(d, "model-00002-of-00002.safetensors"))
        rl = RunLog()
        src = HFSource(d, log=rl)
        meta = src.metadata()  # must not crash
        names = {t["name"] for t in meta["tensors"]}
        # only the surviving shard's tensor is present
        self.assertEqual(names, {"model.embed_tokens.weight"})
        self.assertTrue(
            any(e["code"] == "shard_missing" for e in rl.entries))

    def test_metadata_bad_index_degrades_no_crash(self):
        import json
        shards, _, _ = self._shards()
        d = write_sharded_model(os.path.join(self.tmp, "sh"),
                                _llama_cfg(), shards)
        with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
            json.dump({}, f)
        rl = RunLog()
        src = HFSource(d, log=rl)
        meta = src.metadata()  # must not crash
        self.assertEqual(meta["tensors"], [])
        self.assertTrue(any(e["code"] == "bad_index" for e in rl.entries))

    def test_metadata_bad_index_logs_once(self):
        # A malformed index (no weight_map) must yield EXACTLY ONE bad_index
        # entry per metadata() call — not one from _merged_header() and a
        # second from the tied-embedding name-set re-reading the index.
        import json
        shards, _, _ = self._shards()
        d = write_sharded_model(os.path.join(self.tmp, "sh"),
                                _llama_cfg(), shards)
        with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
            json.dump({}, f)
        rl = RunLog()
        src = HFSource(d, log=rl)
        src.metadata()
        bad = [e for e in rl.entries if e["code"] == "bad_index"]
        self.assertEqual(len(bad), 1)


if __name__ == "__main__":
    unittest.main()
