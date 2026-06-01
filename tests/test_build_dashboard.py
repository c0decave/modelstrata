"""Tests für tools/build_dashboard.py — derive() & dedup-Logik."""
import os
import sys
import unittest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

import build_dashboard as bd


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


class TestDerive(unittest.TestCase):
    def _qwen_like(self):
        meta = {
            "general.architecture": "qwen2",
            "general.name": "Q",
            "qwen2.block_count": 2,
            "qwen2.embedding_length": 16,
            "qwen2.attention.head_count": 8,
            "qwen2.attention.head_count_kv": 2,
            "qwen2.feed_forward_length": 64,
            "qwen2.context_length": 4096,
            "qwen2.rope.freq_base": 1000000.0,
        }
        tensors = [
            tensor("token_embd.weight", [100, 16], "F32", 1600),
            tensor("blk.0.attn_q.weight", [16, 16], "Q6_K", 256),
            tensor("blk.0.attn_q.bias", [16], "F32", 16),       # must NOT win cell
            tensor("blk.0.ffn_down.weight", [16, 64], "Q6_K", 1024),
            tensor("output.weight", [16, 100], "F32", 1600),    # untied
            tensor("v.blk.0.attn_q.weight", [8, 8], "F16", 64),  # vision tower
        ]
        return base_info(meta, tensors)

    def test_basic_arch_fields(self):
        d = bd.derive(self._qwen_like())
        self.assertEqual(d["arch"], "qwen2")
        self.assertEqual(d["n_layers"], 2)
        self.assertEqual(d["d_model"], 16)
        self.assertEqual(d["ffn"], 64)
        self.assertEqual(d["gqa"], 4.0)               # 8/2
        self.assertEqual(d["ffn_ratio"], 4.0)         # 64/16

    def test_head_dim_fallback(self):
        d = bd.derive(self._qwen_like())
        self.assertEqual(d["head_dim"], 2)            # 16/8, no key_length

    def test_kv_cache_gqa(self):
        d = bd.derive(self._qwen_like())
        # 2 * layers(2) * kv_heads(2) * head_dim(2) * ctx(4096) * 2 bytes
        self.assertEqual(d["kv_cache"], 2 * 2 * 2 * 2 * 4096 * 2)
        self.assertEqual(d["kv_cache_kind"], "GQA/MHA, obere Schranke")

    def test_weight_preferred_over_bias(self):
        d = bd.derive(self._qwen_like())
        # attn_q cell must be the weight's Q6_K, not the bias' F32
        self.assertEqual(d["grid"]["cells"]["attn_q"][0], "Q6_K")
        self.assertEqual(d["bias_count"], 1)

    def test_tower_prefix_no_collision(self):
        d = bd.derive(self._qwen_like())
        self.assertIn("attn_q", d["grid"]["cells"])       # text tower
        self.assertIn("v.attn_q", d["grid"]["cells"])     # vision tower distinct
        self.assertEqual(d["grid"]["cells"]["v.attn_q"][0], "F16")

    def test_tied_embeddings_false_when_output_present(self):
        d = bd.derive(self._qwen_like())
        self.assertFalse(d["tied_embeddings"])

    def test_tied_embeddings_true_when_no_output(self):
        info = self._qwen_like()
        info["tensors"] = [t for t in info["tensors"] if t["name"] != "output.weight"]
        d = bd.derive(info)
        self.assertTrue(d["tied_embeddings"])

    def test_scalarize_array_head_count(self):
        # gemma-style per-layer head_count_kv stored as array
        info = self._qwen_like()
        info["metadata"]["qwen2.attention.head_count_kv"] = {
            "_array": True, "len": 2, "elem_type": 4, "sample": [2, 2]}
        d = bd.derive(info)
        self.assertEqual(d["n_head_kv"], 2)           # max of sample

    def test_mla_kv_cache(self):
        meta = {
            "general.architecture": "glm4moelite",
            "glm4moelite.block_count": 4,
            "glm4moelite.context_length": 1000,
            "glm4moelite.attention.head_count": 8,
            "glm4moelite.attention.head_count_kv": 8,
            "glm4moelite.attention.kv_lora_rank": 512,
            "glm4moelite.rope.dimension_count": 64,
            "glm4moelite.expert_count": 64,
            "glm4moelite.expert_used_count": 4,
        }
        d = bd.derive(base_info(meta, []))
        self.assertEqual(d["kv_cache_kind"], "MLA (komprimierter Latent)")
        # layers(4) * (512+64) * ctx(1000) * 2
        self.assertEqual(d["kv_cache"], 4 * 576 * 1000 * 2)
        self.assertEqual(d["moe"]["expert_used_count"], 4)

    def test_kv_cache_asymmetric_key_value(self):
        # regression: when key_length != value_length, both must be counted
        info = self._qwen_like()
        info["metadata"]["qwen2.attention.key_length"] = 128
        info["metadata"]["qwen2.attention.value_length"] = 64
        d = bd.derive(info)
        # layers(2) * kv_heads(2) * (128+64) * ctx(4096) * 2
        self.assertEqual(d["kv_cache"], 2 * 2 * (128 + 64) * 4096 * 2)

    def test_bool_head_kv_rejected_no_bogus_kv_cache(self):
        # regression: a boolean per-layer array must not leak into numeric math.
        # num() rejects bool -> n_head_kv None -> kv_cache None (not a bogus number)
        info = self._qwen_like()
        info["metadata"]["qwen2.attention.head_count_kv"] = {
            "_array": True, "len": 2, "elem_type": 7, "sample": [True, True]}
        d = bd.derive(info)
        self.assertIsNone(d["n_head_kv"])
        self.assertIsNone(d["kv_cache"])
        self.assertIsNone(d["gqa"])

    def test_kv_cache_none_when_missing_head_kv(self):
        info = self._qwen_like()
        del info["metadata"]["qwen2.attention.head_count_kv"]
        d = bd.derive(info)
        self.assertIsNone(d["n_head_kv"])
        self.assertIsNone(d["kv_cache"])

    def test_fingerprint_dedup(self):
        a = bd.derive(self._qwen_like())
        b = bd.derive(self._qwen_like())
        self.assertEqual(bd.fingerprint(a), bd.fingerprint(b))

    def test_fingerprint_keeps_same_shape_different_names_distinct(self):
        a_info = self._qwen_like()
        b_info = self._qwen_like()
        b_info["metadata"]["general.name"] = "Q finetune"
        self.assertNotEqual(
            bd.fingerprint(bd.derive(a_info)),
            bd.fingerprint(bd.derive(b_info)))

    def test_js_embed_escapes_script_breakout(self):
        out = bd._js_embed({"x": "</script><img src=x onerror=alert(1)>"})
        self.assertNotIn("</script>", out)          # cannot close the data island
        self.assertNotIn("<img", out)
        self.assertIn("\\u003c", out)               # escaped to inert unicode

    def test_render_neutralizes_malicious_name(self):
        info = self._qwen_like()
        info["metadata"]["general.name"] = "</script><img src=x onerror=alert(1)>"
        html = bd.render([bd.derive(info)])
        # the breakout sequence must never appear literally in the output
        self.assertNotIn("</script><img", html)

    def test_dashboard_is_self_contained_no_external_font_links(self):
        html = bd.render([bd.derive(self._qwen_like())])
        self.assertNotIn("fonts.googleapis.com", html)
        self.assertNotIn("fonts.gstatic.com", html)

    def test_tutorial_mode_is_embedded_and_uses_fictitious_model(self):
        html = bd.render([bd.derive(self._qwen_like())])
        self.assertIn('id="tutorial-start"', html)
        self.assertIn('id="quiz-start"', html)
        self.assertIn('id="tour-panel"', html)
        self.assertIn("TUTORIAL_MODEL", html)
        self.assertIn("Strata-Tutor-7B", html)
        self.assertIn("Tutorial-Datensatz: fiktiv", html)
        self.assertIn("openModelModal(TUTORIAL_MODEL)", html)
        self.assertIn('id="tour-quiz"', html)
        self.assertIn("Mini-Check", html)
        self.assertIn("Antwort zeigen", html)
        self.assertIn("renderTutorialStep()", html)

    def test_quiz_mode_embeds_100_learning_questions(self):
        html = bd.render([bd.derive(self._qwen_like())])
        start = html.index("const QUIZ_BANK = [")
        end = html.index("];", start)
        quiz_block = html[start:end]
        self.assertEqual(100, quiz_block.count("qa("))
        self.assertIn("Was misst Jaccard?", quiz_block)
        self.assertIn("Was ist PCA?", quiz_block)
        self.assertIn("Was ist ein Glitch-Token?", quiz_block)
        self.assertIn("Was ist ein Tensor in modelstrata?", quiz_block)
        self.assertIn("Was ist ein Health-Score?", quiz_block)
        self.assertIn("renderQuizQuestion()", html)
        self.assertIn("startQuiz", html)

    def test_quiz_answers_avoid_overclaiming(self):
        html = bd.render([bd.derive(self._qwen_like())])
        self.assertIn("gleichem Testset/Setup", html)
        self.assertIn("nur Hinweise, keine Beweise", html)
        self.assertIn("koennen auf wenige starke Ausreisser hindeuten", html)
        self.assertIn("Quantisierung oder Formatunterschiede muessen mitbedacht werden", html)
        self.assertIn("Datenbytes mal 8 geteilt durch Parameterzahl", html)
        self.assertIn("mit hoeherer Praezision gespeichert", html)
        self.assertIn("Bei aktivem Byte-Fallback", html)
        self.assertIn("1.0 bedeutet gleiche Menge, 0.0 keine gemeinsamen Elemente", html)
        self.assertIn("Wahrscheinlichkeitsverteilungen oder Antwortmuster", html)
        self.assertNotIn("hoeher quantisiert", html)
        self.assertNotIn("Loesungswege eines Teacher-Modells", html)
        self.assertNotIn("Matrix-L2-Norm", html)

    def test_static_compare_tab_renders_report_data(self):
        compare = {
            "schema": 5,
            "coverage": {"model_count": 2, "tokenizer_models": 2},
            "lineage_pairs": [{"a": "base", "b": "ft", "score": 0.92,
                               "class": "probable same base / close finetune",
                               "signals": {"architecture": 1.0}}],
            "architecture_pairs": [{"a": "base", "b": "ft", "compat_core": True,
                                    "diffs": {}}],
            "health_summary": [{"model": "base", "score": 84, "class": "warnings",
                                "errors": 0, "warnings": 2, "infos": 1,
                                "top_issues": [{"section": "quant", "code": "sensitive_role_aggressive_quant"}]}],
            "context_pairs": [{"a": "base", "b": "ft", "score": 0.66,
                               "ctx": [4096, 8192], "rope_compat": 0.5,
                               "kv_cache_ratio": 0.5,
                               "notes": ["different configured context length"],
                               "diffs": {"rope_freq_base": [10000, 500000]}}],
            "architecture_clusters": [{"size": 2, "models": ["base", "ft"],
                                       "signature": {"arch": "qwen2", "layers": 2, "d_model": 8},
                                       "formats": ["gguf"], "ctx_range": [128, 128]}],
            "quant_pairs": [{"a": "base", "b": "ft", "breakdown_jaccard": 0.75,
                             "role_quant_diffs": {"ffn_down": ["Q4_K", "Q6_K"]}}],
            "tensor_pairs": [{"a": "base", "b": "ft", "name_jaccard": 1.0,
                              "role_jaccard": 1.0, "shared_names": 3,
                              "same_shape_shared_ratio": 1.0,
                              "shape_mismatch_count": 0}],
            "anomalies": {"weight_stats": [{"model": "base", "anomalies": [
                {"metric": "l2", "layer": 1, "role": "attn_q", "robust_z": 8}
            ]}], "spectral": []},
            "diff_explanations": [{"label": "base -> ft", "matched": 4,
                                   "top_roles": [{"role": "ffn_down", "mean_delta": 0.2}],
                                   "notes": ["largest mean delta role: ffn_down"]}],
            "chat_template_pairs": [{"a": "base", "b": "ft", "score": 0.91,
                                     "class": "likely prompt-compatible",
                                     "marker_jaccard": 1.0,
                                     "special_id_same_ratio": 0.8,
                                     "shared_markers": ["system", "user"],
                                     "special_mismatches": {"eos": [2, 99]}}],
        }
        html = bd.render([bd.derive(self._qwen_like())], compare=compare)
        self.assertIn('data-tab="compare"', html)
        self.assertIn("const COMPARE", html)
        self.assertIn("renderCompare()", html)
        self.assertIn("Statische Vergleiche", html)
        self.assertIn("Health ${esc(h.score)}", html)
        self.assertIn("top_issues", html)
        self.assertIn("Tensor ${pct(p.name_jaccard)}", html)
        self.assertIn("Cluster <span class=\"score\">", html)
        self.assertIn("Context ${pct(p.score)}", html)
        self.assertIn("RoPE ${pct(p.rope_compat)}", html)
        self.assertIn("prompt ${pct(p.score)}", html)
        self.assertIn("special IDs ${pct(p.special_id_same_ratio)}", html)

    def test_learning_copy_avoids_overclaiming_heuristics(self):
        html = bd.render([bd.derive(self._qwen_like())])
        # Content-accuracy guardrails: these are educational explanations, so
        # keep heuristic diagnostics framed as heuristics rather than labels.
        self.assertNotIn("2–4 = gut trainiert", html)
        self.assertNotIn("2–4 = well trained", html)
        self.assertNotIn("Higher = longer contexts addressable", html)
        self.assertNotIn("F32/F16 = full precision", html)
        self.assertNotIn("usually untrained", html)
        self.assertNotIn("meist untrainiert", html)
        self.assertNotIn("single activation direction", html)
        self.assertNotIn("Lifts guardrails", html)
        self.assertIn("not the full WeightWatcher pipeline", html)
        self.assertIn("Higher alone does not guarantee", html)
        self.assertIn("name/type alone is not proof", html)

    def test_glossary_covers_core_learning_terms(self):
        html = bd.render([bd.derive(self._qwen_like())])
        for term in (
            "Jaccard-Ähnlichkeit",
            "PCA (Hauptkomponenten)",
            "Glitch-Token",
            "Tensor",
            "Health-Score",
            "Lineage-Score",
            "Config↔Tensor-Invarianten",
            "Tokenizer↔Embedding-Konsistenz",
            "Chat-Template-Lint",
        ):
            self.assertIn(term, html)
        for term in (
            "Jaccard similarity",
            "PCA (principal components)",
            "Glitch token",
            "Health score",
            "Lineage score",
            "Config↔tensor invariants",
        ):
            self.assertIn(term, html)
        self.assertIn('["compare","Statische Vergleiche (Ebene 8)"]', html)
        self.assertIn('compare:"Static comparisons (level 8)"', html)

    def test_hf_fallback_fields_render_as_real_values(self):
        info = base_info({"general.architecture": "LlamaForCausalLM"}, [],
                         source={"format": "safetensors", "precision": "exact",
                                 "arch": "LlamaForCausalLM", "mapped": True,
                                 "warnings": []},
                         n_layers=2, hidden=16, n_heads=4, n_kv_heads=2,
                         ffn=64, ctx=4096, quant_breakdown={"F32": 1})
        d = bd.derive(info)
        self.assertEqual(d["n_layers"], 2)
        self.assertEqual(d["d_model"], 16)
        self.assertEqual(d["n_head"], 4)
        self.assertEqual(d["n_head_kv"], 2)
        self.assertEqual(d["ffn"], 64)
        self.assertEqual(d["ctx"], 4096)


if __name__ == "__main__":
    unittest.main()
