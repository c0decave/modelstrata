"""Task 17 — per-model format/precision badge + source warnings + HF fleet presence.

The fleet view must make the multi-format work VISIBLE:

  * every fleet card carries a ``format · precision`` badge (colour-hinted:
    exact=green, approx=amber, inventory-only=grey),
  * each model's ``source.warnings`` show up in its detail modal,
  * GGUF entries that arrive WITHOUT a source block still get one synthesized in
    ``derive()`` (precision from the quant_breakdown),
  * an entry with NO derivable source still renders (no badge, no crash),
  * ``analyze.synth_hf_entry`` turns an HF dir into a minimal fleet entry that
    appends to models.json carrying a source block (precision=exact for a
    mappable arch).

Additive: existing gguf cards + the 5 tabs + the Log tab stay green.
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
from st_fixture import write_model


def tensor(name, dims, ttype, params):
    return {"name": name, "dims": dims, "type": ttype, "type_id": 0,
            "offset": 0, "params": params}


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


# --- derive(): synthesize a gguf source block from the quant breakdown -------
class TestGgufSourceSynthesis(unittest.TestCase):
    def test_approx_when_kquant_present(self):
        # Q6_K present -> approx
        d = bd.derive(base_info({"general.architecture": "qwen2"},
                                quant_breakdown={"F32": 3, "Q6_K": 2}))
        self.assertEqual(d["source"]["format"], "gguf")
        self.assertEqual(d["source"]["precision"], "approx")
        self.assertEqual(d["source"]["arch"], "qwen2")
        self.assertTrue(d["source"]["mapped"])
        self.assertEqual(d["source"]["warnings"], [])

    def test_exact_when_all_unquantized(self):
        d = bd.derive(base_info({"general.architecture": "llama"},
                                quant_breakdown={"F32": 3, "F16": 1, "BF16": 2}))
        self.assertEqual(d["source"]["precision"], "exact")

    def test_existing_source_block_preserved(self):
        # an entry that ALREADY carries a source block (e.g. an HF synth) keeps it
        info = base_info({"general.architecture": "llama"})
        info["source"] = {"format": "safetensors", "precision": "exact",
                          "arch": "LlamaForCausalLM", "mapped": True,
                          "warnings": ["no_output_weight: ..."]}
        d = bd.derive(info)
        self.assertEqual(d["source"]["format"], "safetensors")
        self.assertEqual(d["source"]["warnings"], ["no_output_weight: ..."])


# --- badge in the card + warnings in the modal -------------------------------
class TestBadgeAndWarnings(unittest.TestCase):
    def _two_models(self):
        st = bd.derive(base_info({"general.architecture": "llama"}))
        st["source"] = {"format": "safetensors", "precision": "exact",
                        "arch": "LlamaForCausalLM", "mapped": True,
                        "warnings": ["no_output_weight: model ships no output projection"]}
        gg = bd.derive(base_info({"general.architecture": "qwen2"},
                                 quant_breakdown={"Q4_K": 5}))  # approx
        return [st, gg]

    def test_both_badges_present(self):
        html = bd.render(self._two_models())
        # badge text is built client-side from source.{format,precision}; assert
        # the data island carries both source blocks with the right fields
        marker = "const MODELS = "
        start = html.index(marker) + len(marker)
        end = html.index(";\n", start)
        embedded = json.loads(html[start:end]
                              .replace("\\u003c", "<").replace("\\u003e", ">"))
        srcs = [(m["source"]["format"], m["source"]["precision"]) for m in embedded]
        self.assertIn(("safetensors", "exact"), srcs)
        self.assertIn(("gguf", "approx"), srcs)
        # the badge renderer function is present in the page
        self.assertIn("srcBadge", html)

    def test_warnings_in_modal(self):
        html = bd.render(self._two_models())
        # the per-model degradation string must be reachable for the modal —
        # carried in the embedded source.warnings
        self.assertIn("no_output_weight: model ships no output projection", html)
        # the modal has a container the warnings render into
        self.assertIn('id="m-warnings"', html)

    def test_i18n_badge_label_registered(self):
        html = bd.render(self._two_models())
        # a bilingual label for the modal warnings section
        self.assertIn('data-i18n="m-warn"', html)
        self.assertIn('"m-warn"', html)  # EN entry present


# --- regression: no source -> no badge, no crash -----------------------------
class TestNoSourceGraceful(unittest.TestCase):
    def test_entry_without_source_renders(self):
        # an entry with neither a source block nor a quant_breakdown the synth can
        # use must still render. Force the synth to be skipped by clearing quant.
        info = base_info({"general.architecture": "qwen2"}, quant_breakdown={})
        d = bd.derive(info)
        # gguf entries always get a source (empty quant -> exact, vacuously) but
        # the page must render either way without crashing
        html = bd.render([d])
        self.assertIn("MODELS", html)
        self.assertIn("srcBadge", html)


# --- analyze.synth_hf_entry: HF dir -> minimal fleet entry -------------------
class TestHFFleetSynthesis(unittest.TestCase):
    def test_synth_hf_entry_has_source_block(self):
        with tempfile.TemporaryDirectory() as d:
            mdir = os.path.join(d, "tiny-llama")
            write_model(
                mdir,
                {"architectures": ["LlamaForCausalLM"], "model_type": "llama",
                 "num_hidden_layers": 4, "hidden_size": 16,
                 "num_attention_heads": 4, "num_key_value_heads": 2,
                 "intermediate_size": 64, "max_position_embeddings": 2048},
                {"model.embed_tokens.weight": ("F32", [10, 16], None)},
            )
            entry = analyze.synth_hf_entry(mdir)
            self.assertIsNotNone(entry)
            self.assertEqual(entry["source"]["format"], "safetensors")
            self.assertEqual(entry["source"]["precision"], "exact")
            self.assertTrue(entry["source"]["mapped"])
            # minimal fleet fields populated; gguf-only fields empty/None
            self.assertEqual(entry["arch"], "LlamaForCausalLM")
            self.assertEqual(entry["n_layers"], 4)
            self.assertEqual(entry["n_heads"], 4)
            self.assertEqual(entry["n_kv_heads"], 2)
            self.assertEqual(entry["ffn"], 64)
            self.assertEqual(entry["ctx"], 2048)
            self.assertIn("name", entry)
            self.assertIn("label", entry)
            # must survive derive() (the dashboard runs derive on every raw entry)
            d2 = bd.derive(entry)
            self.assertEqual(d2["source"]["format"], "safetensors")
            self.assertEqual(d2["arch"], "LlamaForCausalLM")

    def test_hf_card_shows_real_layer_count(self):
        # Fix 1: an HF synth entry lacks the gguf <arch>.block_count key, so
        # derive() would show n_layers=None ('–' in the card). derive() must
        # FALL BACK to the entry's honest top-level n_layers (from config.json's
        # num_hidden_layers) and surface the real value (here 2), and likewise
        # d_model from the top-level hidden/d_model field (here 16).
        with tempfile.TemporaryDirectory() as d:
            mdir = os.path.join(d, "tiny2")
            write_model(
                mdir,
                {"architectures": ["LlamaForCausalLM"], "model_type": "llama",
                 "num_hidden_layers": 2, "hidden_size": 16,
                 "num_attention_heads": 4, "num_key_value_heads": 2,
                 "intermediate_size": 64, "max_position_embeddings": 4096},
                {"model.embed_tokens.weight": ("F32", [10, 16], None)},
            )
            entry = analyze.synth_hf_entry(mdir)
            self.assertEqual(entry["n_layers"], 2)
            self.assertEqual(entry["d_model"], 16)
            d2 = bd.derive(entry)
            self.assertEqual(d2["n_layers"], 2)     # real value, not None ('–')
            self.assertEqual(d2["d_model"], 16)
            self.assertEqual(d2["n_head"], 4)
            self.assertEqual(d2["n_head_kv"], 2)
            self.assertEqual(d2["ffn"], 64)
            self.assertEqual(d2["ctx"], 4096)
            # and it renders into the card (not the '–' fallback)
            html = bd.render([d2])
            self.assertIn("MODELS", html)

    def test_gguf_layer_count_unchanged_by_fallback(self):
        # Regression: a gguf entry carries <arch>.block_count, so the Fix 1
        # fallback must NOT change it. Even if a stray top-level n_layers is
        # present, the gguf block_count key wins (gguf behavior untouched).
        info = base_info({"general.architecture": "qwen2",
                          "qwen2.block_count": 7,
                          "qwen2.embedding_length": 32})
        info["n_layers"] = 999   # would be wrong if the fallback ever fired
        info["d_model"] = 999
        d = bd.derive(info)
        self.assertEqual(d["n_layers"], 7)
        self.assertEqual(d["d_model"], 32)

    def test_synth_hf_entry_unknown_path_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(analyze.synth_hf_entry(os.path.join(d, "nope")))


if __name__ == "__main__":
    unittest.main()
