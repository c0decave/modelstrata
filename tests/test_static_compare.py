"""Tests for tools/static_compare.py."""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

import static_compare as sc


def _raw(label, name, qtype="Q6_K", arch="llama"):
    return {
        "path": f"/models/{label}.gguf",
        "label": label,
        "file_size": 1000,
        "n_tensors": 3,
        "total_params": 256,
        "gguf_version": 3,
        "quant_breakdown": {qtype: 2, "F16": 1},
        "header_hex": "47475546",
        "header_ascii": "GGUF",
        "header_end": 100,
        "alignment": 32,
        "data_start": 128,
        "data_bytes": 872,
        "bits_per_weight": 5.5,
        "file_type_label": qtype,
        "metadata": {
            "general.architecture": arch,
            "general.name": name,
            "tokenizer.chat_template": "{% for message in messages %}{{ message['role'] }} tool_call image{% endfor %}",
            "tokenizer.ggml.bos_token_id": 1,
            "tokenizer.ggml.eos_token_id": 2,
            "tokenizer.ggml.unknown_token_id": 0,
            "tokenizer.ggml.add_bos_token": True,
            "tokenizer.ggml.add_eos_token": False,
            f"{arch}.block_count": 2,
            f"{arch}.embedding_length": 8,
            f"{arch}.attention.head_count": 4,
            f"{arch}.attention.head_count_kv": 2,
            f"{arch}.feed_forward_length": 32,
            f"{arch}.context_length": 128,
            f"{arch}.rope.freq_base": 10000,
        },
        "tensors": [
            {"name": "token_embd.weight", "dims": [16, 8], "type": "F16", "params": 128},
            {"name": "blk.0.attn_q.weight", "dims": [8, 8], "type": qtype, "params": 64},
            {"name": "blk.0.ffn_down.weight", "dims": [8, 32], "type": qtype, "params": 256},
        ],
    }


def _tokdir(parent, name, vocab, merges=None):
    d = os.path.join(parent, name)
    os.makedirs(d)
    with open(os.path.join(d, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump({"model": {"vocab": vocab, "merges": merges or []}, "added_tokens": [
            {"id": max(vocab.values()) + 1, "content": "<|im_start|>", "special": True}
        ]}, f)
    return d


class TestStaticCompare(unittest.TestCase):
    def test_build_report_computes_real_pair_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            a_dir = _tokdir(tmp, "a", {"hello": 0, "world": 1},
                            merges=["h e", "he l"])
            b_dir = _tokdir(tmp, "b", {"hello": 0, "other": 1},
                            merges=["h e", "x y"])
            raw = [_raw("a", "A"), _raw("b", "B", qtype="Q4_K")]
            report = sc.build_report(raw, [("a", a_dir), ("b", b_dir)])

        self.assertEqual(report["coverage"]["model_count"], 2)
        self.assertEqual(report["schema"], 5)
        self.assertEqual(report["coverage"]["tokenizer_models"], 2)
        self.assertEqual(len(report["context_pairs"]), 1)
        self.assertEqual(report["context_pairs"][0]["score"], 1.0)
        self.assertEqual(report["architecture_clusters"][0]["size"], 2)
        self.assertEqual(len(report["architecture_pairs"]), 1)
        self.assertTrue(report["architecture_pairs"][0]["compat_core"])
        self.assertEqual(len(report["tensor_pairs"]), 1)
        self.assertEqual(report["tensor_pairs"][0]["name_jaccard"], 1.0)
        self.assertEqual(report["tensor_pairs"][0]["same_shape_shared_ratio"], 1.0)
        self.assertLess(report["tensor_pairs"][0]["same_type_shared_ratio"], 1.0)
        self.assertEqual(len(report["tokenizer_pairs"]), 1)
        self.assertGreater(report["tokenizer_pairs"][0]["vocab_jaccard"], 0)
        self.assertEqual(report["tokenizer_pairs"][0]["shared_merges"], 1)
        self.assertEqual(report["tokenizer_pairs"][0]["merge_jaccard"], round(1 / 3, 4))
        self.assertLess(report["quant_pairs"][0]["breakdown_jaccard"], 1.0)
        self.assertEqual(len(report["lineage_pairs"]), 1)
        self.assertIn("architecture", report["lineage_pairs"][0]["signals"])
        self.assertIn("context_rope", report["lineage_pairs"][0]["signals"])
        self.assertIn("tensor_names", report["lineage_pairs"][0]["signals"])
        self.assertIn("chat_template", report["lineage_pairs"][0]["signals"])
        self.assertEqual(report["chat_template_pairs"][0]["exact_template"], True)
        self.assertEqual(report["chat_template_pairs"][0]["special_id_same_ratio"], 1.0)
        self.assertEqual(report["chat_templates"][0]["markers"]["tool"], True)
        self.assertIn("health_summary", report)
        self.assertEqual(len(report["tensor_config_checks"]), 2)

    def test_chat_template_pairs_compare_markers_and_special_ids(self):
        a = _raw("a", "A")
        b = _raw("b", "B")
        b["metadata"]["tokenizer.chat_template"] = "{{ '<start_of_turn>user' }}{{ '<start_of_turn>model' }}"
        b["metadata"]["tokenizer.ggml.eos_token_id"] = 99
        models = sc.dedupe_models([a, b])
        pairs = sc.chat_template_pairs(models)
        self.assertEqual(len(pairs), 1)
        self.assertLess(pairs[0]["marker_jaccard"], 1.0)
        self.assertIn("tool", pairs[0]["markers"][0])
        self.assertNotIn("tool", pairs[0]["markers"][1])
        self.assertEqual(pairs[0]["special_mismatches"]["eos"], [2, 99])
        self.assertEqual(pairs[0]["special_id_same_ratio"], round(4 / 5, 4))

    def test_context_pairs_surface_rope_and_kv_differences(self):
        a = _raw("a", "A")
        b = _raw("b", "B")
        b["metadata"]["llama.context_length"] = 512
        b["metadata"]["llama.rope.freq_base"] = 500000
        b["metadata"]["llama.attention.sliding_window"] = 128
        models = sc.dedupe_models([a, b])
        profiles = {sc.model_key(m): sc.context_profile(m) for m in models}
        pairs = sc.context_pairs(profiles)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["ctx"], [128, 512])
        self.assertEqual(pairs[0]["ctx_ratio"], 0.25)
        self.assertIn("rope_freq_base", pairs[0]["diffs"])
        self.assertIn("sliding_window", pairs[0]["diffs"])
        self.assertIn("different RoPE parametrization", pairs[0]["notes"])

    def test_tensor_pairs_surface_shape_mismatches_and_missing_names(self):
        a = _raw("a", "A")
        b = _raw("b", "B")
        b["tensors"][1]["dims"] = [16, 4]
        b["tensors"].append(
            {"name": "blk.0.attn_k.weight", "dims": [8, 8], "type": "Q6_K", "params": 64})
        models = sc.dedupe_models([a, b])
        pairs = sc.tensor_pairs(models)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["shape_mismatch_count"], 1)
        self.assertEqual(pairs[0]["shape_mismatch_sample"][0]["name"], "blk.0.attn_q.weight")
        self.assertIn("blk.0.attn_k.weight", pairs[0]["only_b_sample"])

    def test_static_health_checks_find_real_inconsistencies(self):
        bad = _raw("bad", "Bad", qtype="Q4_K")
        bad["metadata"]["llama.context_length"] = 256
        bad["metadata"]["llama.vocab_size"] = 99
        bad["metadata"]["tokenizer.ggml.eos_token_id"] = 100
        bad["metadata"]["tokenizer.ggml.add_bos_token"] = True
        bad["metadata"]["tokenizer.chat_template"] = "{{ bos_token }} {{ message['content'] }}"
        bad["tensors"][0]["dims"] = [16, 8]          # embedding rows != vocab
        bad["tensors"][1]["dims"] = [7, 8]           # attn_q shape invariant broken
        bad["tensors"] = [t for t in bad["tensors"] if not t["name"].startswith("blk.0.ffn_down")]
        models = sc.dedupe_models([bad])

        tcfg = sc.tensor_config_checks(models)[0]["issues"]
        self.assertTrue(any(i["code"] == "shape_invariant" for i in tcfg))
        self.assertTrue(any(i["code"] == "role_layer_gap" for i in tcfg))

        temb = sc.tokenizer_embedding_checks(models)[0]["issues"]
        self.assertTrue(any(i["code"] == "vocab_embedding_mismatch" for i in temb))
        self.assertTrue(any(i["code"] == "special_id_oob" for i in temb))

        lint = sc.chat_template_lints(models)[0]["issues"]
        self.assertTrue(any(i["code"] == "chat_missing_user" for i in lint))
        self.assertTrue(any(i["code"] == "possible_double_bos" for i in lint))

        qdiag = sc.quant_diagnostics(models)[0]["issues"]
        self.assertTrue(any(i["code"] == "sensitive_role_aggressive_quant" for i in qdiag))

        summary = sc.health_summary(
            ("tensor_config", sc.tensor_config_checks(models)),
            ("tokenizer_embedding", sc.tokenizer_embedding_checks(models)),
            ("chat_template", sc.chat_template_lints(models)),
            ("quant", sc.quant_diagnostics(models)),
        )
        self.assertLess(summary[0]["score"], 100)
        self.assertGreaterEqual(summary[0]["warnings"] + summary[0]["errors"], 1)

    def test_metric_anomalies_and_diff_explanations(self):
        wstats = [{
            "label": "m",
            "cells": {"attn_q": {0: {"l2": 1}, 1: {"l2": 1}, 2: {"l2": 1}, 3: {"l2": 100}}},
        }]
        an = sc.metric_anomalies(wstats)
        self.assertEqual(an[0]["anomalies"][0]["layer"], 3)

        diff = [{
            "label": "a -> b",
            "matched": 2,
            "shape_mismatch": 1,
            "cells": {"ffn_down": {0: {"delta": 0.5, "cosine": 0.9}},
                      "attn_q": {0: {"delta": 0.1, "cosine": 0.99}}},
            "top": [],
        }]
        ex = sc.diff_explanations(diff)
        self.assertEqual(ex[0]["top_roles"][0]["role"], "ffn_down")
        self.assertIn("shape mismatches", ex[0]["notes"][0])

    def test_role_aggregates_moe_and_embedding_summaries(self):
        wstats = [{
            "label": "moe",
            "cells": {
                "ffn_gate_exps.e0": {0: {"l2": 2.0}},
                "ffn_gate_exps.e1": {0: {"l2": 4.0}},
                "attn_q": {0: {"l2": 1.0}},
            },
        }]
        ag = sc.role_aggregates(wstats)
        self.assertEqual(ag[0]["roles"][0]["metrics"]["l2"], 2.0)
        moe = sc.moe_analysis(wstats, [])
        self.assertEqual(moe[0]["model"], "moe")
        self.assertIn("ffn_gate_exps.e0", moe[0]["expert_roles"])

        emb = sc.embedding_summaries([{
            "label": "m",
            "seed_neighbors": [{"seed": "sql", "neighbors": []}],
            "output_head_compare": {"delta": 0.1},
            "near_zero": 1,
            "anisotropy": 0.02,
        }])
        self.assertEqual(emb[0]["output_head_compare"]["delta"], 0.1)


if __name__ == "__main__":
    unittest.main()
