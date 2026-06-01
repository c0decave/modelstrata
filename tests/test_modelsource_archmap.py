import os, sys, unittest
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

from modelsource.archmap import resolve, family_for
from modelsource import naming


class T(unittest.TestCase):
    def test_llama_qwen(self):
        self.assertEqual(family_for({"model_type": "qwen2"}), "llamalike")
        self.assertEqual(resolve("llamalike", "model.layers.5.self_attn.q_proj.weight"), ("ATTN_Q", 5))
        self.assertEqual(resolve("llamalike", "model.layers.5.mlp.gate_proj.weight"), ("FFN_GATE", 5))
        self.assertEqual(resolve("llamalike", "model.embed_tokens.weight"), ("TOK_EMBD", None))
        self.assertEqual(resolve("llamalike", "lm_head.weight"), ("OUTPUT", None))
        self.assertEqual(resolve("llamalike", "model.norm.weight"), ("OUTPUT_NORM", None))

    def test_more_llamalike_roles(self):
        self.assertEqual(resolve("llamalike", "model.layers.0.self_attn.k_proj.weight"), ("ATTN_K", 0))
        self.assertEqual(resolve("llamalike", "model.layers.0.self_attn.v_proj.weight"), ("ATTN_V", 0))
        self.assertEqual(resolve("llamalike", "model.layers.0.self_attn.o_proj.weight"), ("ATTN_OUT", 0))
        self.assertEqual(resolve("llamalike", "model.layers.0.mlp.up_proj.weight"), ("FFN_UP", 0))
        self.assertEqual(resolve("llamalike", "model.layers.0.mlp.down_proj.weight"), ("FFN_DOWN", 0))
        self.assertEqual(resolve("llamalike", "model.layers.0.input_layernorm.weight"), ("ATTN_NORM", 0))
        self.assertEqual(resolve("llamalike", "model.layers.0.post_attention_layernorm.weight"), ("FFN_NORM", 0))

    def test_attention_biases_map_same_role(self):
        # Qwen2/Qwen2.5 ship q/k/v_proj.bias — resolve must map them to the
        # SAME role as the weight (bias-vs-weight is decided downstream by the
        # tensor-name suffix, not by resolve()).
        self.assertEqual(resolve("llamalike", "model.layers.0.self_attn.q_proj.bias"), ("ATTN_Q", 0))
        self.assertEqual(resolve("llamalike", "model.layers.0.self_attn.k_proj.bias"), ("ATTN_K", 0))
        self.assertEqual(resolve("llamalike", "model.layers.0.self_attn.v_proj.bias"), ("ATTN_V", 0))
        self.assertEqual(resolve("llamalike", "model.layers.3.self_attn.o_proj.bias"), ("ATTN_OUT", 3))

    def test_family_from_architectures(self):
        self.assertEqual(family_for({"architectures": ["LlamaForCausalLM"]}), "llamalike")
        self.assertEqual(family_for({"model_type": "qwen3"}), "llamalike")
        self.assertEqual(family_for({"model_type": "mistral"}), "llamalike")

    def test_unknown_returns_none(self):
        self.assertIsNone(family_for({"model_type": "mamba-xyz"}))
        self.assertIsNone(family_for({}))
        self.assertIsNone(resolve("llamalike", "totally.unknown.tensor"))
        self.assertIsNone(resolve("no-such-family", "model.norm.weight"))

    # --- MoE families: common expert/router tensors are mappable ------------
    def test_moe_model_type_is_none(self):
        self.assertEqual(family_for({"model_type": "qwen2_moe"}), "moe")
        self.assertEqual(family_for({"model_type": "qwen3_moe"}), "moe")
        self.assertEqual(family_for({"model_type": "mixtral"}), "moe")

    def test_moe_arch_name_is_none(self):
        self.assertEqual(family_for({"architectures": ["Qwen2MoeForCausalLM"]}), "moe")
        self.assertEqual(family_for({"architectures": ["MixtralForCausalLM"]}), "moe")

    def test_moe_config_key_is_none(self):
        self.assertEqual(family_for({"model_type": "qwen2", "num_experts": 60}), "moe")
        self.assertEqual(
            family_for({"model_type": "qwen2", "num_local_experts": 8}), "moe")

    def test_moe_takes_precedence_over_family_regex(self):
        # Qwen2MoeForCausalLM also matches the family regex base — MoE wins.
        self.assertEqual(family_for({"model_type": "qwen2_moe",
                                     "architectures": ["Qwen2MoeForCausalLM"]}), "moe")

    def test_qwen_moe_expert_and_router_roles(self):
        self.assertEqual(resolve("moe", "model.layers.0.mlp.gate.weight"), ("FFN_GATE_INP", 0))
        self.assertEqual(resolve("moe", "model.layers.0.mlp.shared_expert_gate.weight"),
                         ("ffn_gate_shared_inp", 0))
        self.assertEqual(resolve("moe", "model.layers.0.mlp.experts.7.gate_proj.weight"),
                         ("ffn_gate_exps.e7", 0))
        self.assertEqual(resolve("moe", "model.layers.0.mlp.experts.7.up_proj.weight"),
                         ("ffn_up_exps.e7", 0))
        self.assertEqual(resolve("moe", "model.layers.0.mlp.experts.7.down_proj.weight"),
                         ("ffn_down_exps.e7", 0))

    def test_mixtral_moe_expert_and_router_roles(self):
        self.assertEqual(resolve("moe", "model.layers.2.block_sparse_moe.gate.weight"),
                         ("FFN_GATE_INP", 2))
        self.assertEqual(resolve("moe", "model.layers.2.block_sparse_moe.experts.3.w1.weight"),
                         ("ffn_gate_exps.e3", 2))
        self.assertEqual(resolve("moe", "model.layers.2.block_sparse_moe.experts.3.w3.weight"),
                         ("ffn_up_exps.e3", 2))
        self.assertEqual(resolve("moe", "model.layers.2.block_sparse_moe.experts.3.w2.weight"),
                         ("ffn_down_exps.e3", 2))

    # --- Fix 4: tighten incidental-substring arch matching -----------------
    def test_incidental_substring_arch_is_none(self):
        # "SomeNonLlamaThing" merely CONTAINS "llama" — it is not a real family
        # arch class name and must NOT be misclassified as llamalike.
        self.assertIsNone(family_for({"architectures": ["SomeNonLlamaThing"]}))

    def test_real_family_arch_names_map(self):
        self.assertEqual(
            family_for({"architectures": ["LlamaForCausalLM"]}), "llamalike")
        self.assertEqual(
            family_for({"architectures": ["Qwen2ForCausalLM"]}), "llamalike")
        self.assertEqual(
            family_for({"architectures": ["Qwen3ForCausalLM"]}), "llamalike")
        self.assertEqual(
            family_for({"architectures": ["MistralForCausalLM"]}), "llamalike")


    # --- Gemma family (gemma / gemma2 / gemma3 text) -----------------------
    def test_gemma_family_recognized(self):
        for mt in ("gemma", "gemma2", "gemma3"):
            self.assertEqual(family_for({"model_type": mt}), "gemma", mt)
        self.assertEqual(family_for({"architectures": ["Gemma3ForCausalLM"]}), "gemma")
        self.assertEqual(family_for({"architectures": ["Gemma2ForCausalLM"]}), "gemma")

    def test_gemma_multimodal_and_moe_are_inventory_only(self):
        # Gemma3 multimodal (vision tower + language_model. prefix) is NOT mapped
        # here — honest inventory-only rather than dropping the vision tower.
        self.assertIsNone(family_for({"architectures": ["Gemma3ForConditionalGeneration"]}))
        # REALISTIC multimodal config: top-level model_type is "gemma3" AND the
        # arch is ConditionalGeneration. The model_type must NOT short-circuit to
        # "gemma" (would falsely badge precision=exact while every tensor, nested
        # under language_model.*, is unmapped). The multimodal arch vetoes first.
        self.assertIsNone(family_for(
            {"model_type": "gemma3", "architectures": ["Gemma3ForConditionalGeneration"]}))
        # defensive: unknown/gemma MoE layouts remain inventory-only until
        # their router/expert naming is explicitly mapped.
        self.assertIsNone(family_for({"model_type": "gemma3", "num_experts": 8}))

    def test_unknown_moe_family_is_inventory_only(self):
        self.assertIsNone(family_for({"model_type": "some_moe", "num_experts": 8}))

    def test_other_conditional_generation_archs_are_inventory_only(self):
        # Any *ForConditionalGeneration (multimodal/encoder-decoder) is nested /
        # has towers we cannot map -> inventory-only, never falsely exact.
        self.assertIsNone(family_for(
            {"model_type": "qwen2", "architectures": ["Qwen2_5_VLForConditionalGeneration"]}))

    def test_gemma_core_tensors(self):
        self.assertEqual(resolve("gemma", "model.layers.2.self_attn.q_proj.weight"), ("ATTN_Q", 2))
        self.assertEqual(resolve("gemma", "model.layers.2.mlp.gate_proj.weight"), ("FFN_GATE", 2))
        self.assertEqual(resolve("gemma", "model.embed_tokens.weight"), ("TOK_EMBD", None))
        self.assertEqual(resolve("gemma", "model.norm.weight"), ("OUTPUT_NORM", None))

    def test_gemma_sandwich_norms_distinct_from_llama(self):
        # The key correctness point: in gemma2/3 post_attention_layernorm is a
        # SEPARATE post-attn norm, and the pre-FFN norm is pre_feedforward_layernorm
        # (mirrors llama.cpp gemma stems). Reusing llama's mapping would mislabel.
        self.assertEqual(resolve("gemma", "model.layers.0.input_layernorm.weight"), ("ATTN_NORM", 0))
        self.assertEqual(resolve("gemma", "model.layers.0.post_attention_layernorm.weight"), ("ATTN_POST_NORM", 0))
        self.assertEqual(resolve("gemma", "model.layers.0.pre_feedforward_layernorm.weight"), ("FFN_NORM", 0))
        self.assertEqual(resolve("gemma", "model.layers.0.post_feedforward_layernorm.weight"), ("FFN_POST_NORM", 0))

    def test_gemma3_qk_norm(self):
        self.assertEqual(resolve("gemma", "model.layers.7.self_attn.q_norm.weight"), ("ATTN_Q_NORM", 7))
        self.assertEqual(resolve("gemma", "model.layers.7.self_attn.k_norm.weight"), ("ATTN_K_NORM", 7))

    def test_gemma_new_roles_render_gguf_stems(self):
        # parity with llama.cpp gemma GGUF tensor names
        c = naming.canonical
        self.assertEqual(c(naming.Role.ATTN_POST_NORM, 3), "blk.3.attn_post_norm.weight")
        self.assertEqual(c(naming.Role.FFN_POST_NORM, 3), "blk.3.ffn_post_norm.weight")
        self.assertEqual(c(naming.Role.ATTN_Q_NORM, 3), "blk.3.attn_q_norm.weight")
        self.assertEqual(c(naming.Role.ATTN_K_NORM, 3), "blk.3.attn_k_norm.weight")


if __name__ == "__main__":
    unittest.main()
