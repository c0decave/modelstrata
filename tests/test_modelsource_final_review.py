"""Final-review fixes for the modelsource feature.

Three regressions, TDD (fail-first then pass):

  Fix 1 — embedding_geometry.main() must DEGRADE gracefully on an
    embedding-less model (inventory-only .bin / unmapped HF) instead of raising
    and aborting the whole batch. The VALID model in the same batch must keep
    its full entry; the embedding-less one appears as a minimal honest degraded
    entry (no pca / null vocab+dim); embedding.json IS written; EXACTLY ONE
    ``no_embedding`` warning is logged.

  Fix 2 — build_dashboard.derive() must not KeyError on a synth (non-GGUF)
    tensor record that lacks ``params``.

  Fix 3 — an HF synth entry with tied=True must surface tied truthfully through
    derive() (the card data shows tied), while a GGUF entry's gguf-convention
    tied value stays unchanged.
"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, HERE)

import build_dashboard as bd
import analyze

try:
    import numpy as np
    from st_fixture import write_model
    HAVE_NP = True
except Exception:
    HAVE_NP = False


# Reuse the badge test's tiny gguf-schema raw-entry builder.
def tensor(name, dims, ttype, params=None):
    t = {"name": name, "dims": dims, "type": ttype, "type_id": 0, "offset": 0}
    if params is not None:
        t["params"] = params
    return t


def base_info(meta=None, tensors=None, **kw):
    info = {
        "path": "/x/m.gguf", "label": "m.gguf", "file_size": 1000,
        "n_tensors": 0, "total_params": 1000, "gguf_version": 3,
        "quant_breakdown": {"Q6_K": 1}, "header_hex": "47475546",
        "header_ascii": "GGUF", "header_end": 100, "alignment": 32,
        "data_start": 128, "data_bytes": 872, "bits_per_weight": 6.5,
        "file_type_label": "Q6_K",
        "metadata": meta or {}, "tensors": tensors or [],
    }
    info.update(kw)
    return info


# A mappable HF model that DOES carry a real token_embd (valid embedding side).
_HF_CFG = {"architectures": ["LlamaForCausalLM"], "model_type": "llama",
           "num_hidden_layers": 1, "num_attention_heads": 4,
           "num_key_value_heads": 2, "hidden_size": 4,
           "tie_word_embeddings": False}


# ---- Fix 1: embedding_geometry degrades gracefully, does not abort batch -----
@unittest.skipUnless(HAVE_NP, "numpy not installed")
class TestEmbeddingDegradesGracefully(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = self._td.name
        # valid HF model with a real embedding
        self.hdir = os.path.join(self.tmp, "valid-hf")
        embd = np.arange(8 * 4, dtype="<f4").reshape(8, 4) * 0.1
        write_model(self.hdir, _HF_CFG,
                    {"model.embed_tokens.weight": ("F32", [8, 4], embd),
                     "model.layers.0.self_attn.q_proj.weight":
                         ("F32", [4, 4], np.zeros((4, 4), "<f4"))})
        # inventory-only .bin (no token_embd.weight ever yielded)
        self.binp = os.path.join(self.tmp, "inv.bin")
        with open(self.binp, "wb") as f:
            f.write(b"not a real torch zip")   # legacy/unparsable -> inventory

    def tearDown(self):
        self._td.cleanup()

    def _run_main(self, models, outp, runlogp):
        import embedding_geometry as eg
        import importlib
        importlib.reload(eg)   # restore real module-level load_weight/read_tokenizer
        argv = ["embedding_geometry", *models, "-o", outp, "--sample", "8",
                "--run-log", runlogp]
        old = sys.argv
        sys.argv = argv
        try:
            eg.main()
        finally:
            sys.argv = old

    def test_batch_with_embeddingless_model_still_writes_json(self):
        outp = os.path.join(self.tmp, "embedding.json")
        runlogp = os.path.join(self.tmp, "rl.json")
        # batch: valid HF FIRST, embedding-less .bin SECOND
        self._run_main([self.hdir, self.binp], outp, runlogp)

        self.assertTrue(os.path.isfile(outp),
                        "embedding.json must be written despite the embedding-less model")
        with open(outp) as f:
            out = json.load(f)
        self.assertEqual(len(out), 2)
        by_label = {e["label"]: e for e in out}

        valid = by_label["valid-hf"]
        self.assertEqual(valid["vocab"], 8)
        self.assertEqual(valid["dim"], 4)
        self.assertIn("pca", valid)
        self.assertTrue(valid["pca"]["points"])

        degraded = by_label["inv.bin"]
        # honest degraded entry: null vocab/dim, no pca points, carries source
        self.assertIsNone(degraded.get("vocab"))
        self.assertIsNone(degraded.get("dim"))
        self.assertIsNone(degraded.get("pca"))
        self.assertIn("source", degraded)

        # exactly one no_embedding warning logged
        with open(runlogp) as f:
            entries = json.load(f)
        no_emb = [e for e in entries if e["code"] == "no_embedding"]
        self.assertEqual(len(no_emb), 1)
        self.assertEqual(no_emb[0]["severity"], "warn")

    def test_embeddingless_first_does_not_drop_valid_entry(self):
        # order-independence: even with the bad model FIRST, the valid model
        # still gets its full entry (the raise used to abort before reaching it).
        outp = os.path.join(self.tmp, "embedding2.json")
        runlogp = os.path.join(self.tmp, "rl2.json")
        self._run_main([self.binp, self.hdir], outp, runlogp)
        with open(outp) as f:
            out = json.load(f)
        by_label = {e["label"]: e for e in out}
        self.assertIn("valid-hf", by_label)
        self.assertEqual(by_label["valid-hf"]["vocab"], 8)
        self.assertIsNone(by_label["inv.bin"].get("vocab"))


# ---- Fix 2: derive() KeyError hardening on params-less synth tensor ----------
class TestDeriveParamsHardening(unittest.TestCase):
    def test_blk_tensor_without_params_does_not_keyerror(self):
        info = base_info(
            {"general.architecture": "llama"},
            tensors=[tensor("blk.0.attn_q.weight", [4, 4], "F32")],  # no params key
        )
        d = bd.derive(info)            # must NOT raise KeyError
        html = bd.render([d])
        self.assertIn("MODELS", html)


# ---- Fix 3: HF synth forwards tied; derive() prefers explicit tied -----------
@unittest.skipUnless(HAVE_NP, "numpy not installed")
class TestHFTiedForwarded(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = self._td.name

    def tearDown(self):
        self._td.cleanup()

    def test_synth_hf_tied_true_surfaces_in_card(self):
        mdir = os.path.join(self.tmp, "tied-hf")
        embd = np.arange(10 * 4, dtype="<f4").reshape(10, 4)
        write_model(
            mdir,
            {"architectures": ["LlamaForCausalLM"], "model_type": "llama",
             "num_hidden_layers": 1, "num_attention_heads": 4,
             "num_key_value_heads": 2, "hidden_size": 4,
             "tie_word_embeddings": True},
            {"model.embed_tokens.weight": ("F32", [10, 4], embd),
             "model.layers.0.self_attn.q_proj.weight":
                 ("F32", [4, 4], np.zeros((4, 4), "<f4"))},
        )
        entry = analyze.synth_hf_entry(mdir)
        self.assertIs(entry["tied"], True)           # forwarded onto synth entry
        d = bd.derive(entry)
        self.assertIs(d["tied_embeddings"], True)     # surfaced in the card data

    def test_gguf_tied_value_unchanged(self):
        # gguf entry with a separate output.weight -> NOT tied; the gguf-convention
        # recompute must be untouched (no explicit top-level tied to prefer).
        info = base_info(
            {"general.architecture": "llama"},
            tensors=[tensor("token_embd.weight", [10, 4], "F32", 40),
                     tensor("output.weight", [10, 4], "F32", 40)],
        )
        self.assertNotIn("tied", info)
        d = bd.derive(info)
        self.assertIs(d["tied_embeddings"], False)

    def test_gguf_tied_true_when_no_output_weight(self):
        # gguf entry WITHOUT output.weight but WITH token_embd -> tied True via
        # the convention; unchanged by the explicit-tied preference.
        info = base_info(
            {"general.architecture": "llama"},
            tensors=[tensor("token_embd.weight", [10, 4], "F32", 40)],
        )
        d = bd.derive(info)
        self.assertIs(d["tied_embeddings"], True)


if __name__ == "__main__":
    unittest.main()
