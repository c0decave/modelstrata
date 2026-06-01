"""Tests für HFSource.iter_weights — archmap→kanonische Namen, fp32-Decode,
Offset-Validierung, lückenlose No-Swallow-Logs.

Echte Tensor-Bytes via numpy → per HAVE_NP geguardet. Jeder Skip MUSS sichtbar
sein: genau ein RunLog-Eintrag UND ein Eintrag in src.warnings.
"""
import json
import os
import struct
import sys
import tempfile
import unittest

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
    from st_fixture import write_model


def _llama_cfg():
    return {"architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "num_hidden_layers": 1, "num_attention_heads": 4,
            "num_key_value_heads": 2, "hidden_size": 8}


@unittest.skipUnless(HAVE_NP, "numpy not installed")
class TestHFIterWeights(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = self._td.name

    def tearDown(self):
        self._td.cleanup()

    def _build(self):
        # Known values so we can assert exact fp32 round-trip.
        embd = np.arange(10 * 8, dtype="<f4").reshape(10, 8)
        q = (np.arange(8 * 8, dtype="<f4").reshape(8, 8) * 0.5)
        # unmappable: rotary_emb.inv_freq has no canonical role
        inv = np.arange(4, dtype="<f4")
        tensors = {
            "model.embed_tokens.weight": ("F32", [10, 8], embd),
            "model.layers.0.self_attn.q_proj.weight": ("F32", [8, 8], q),
            "model.layers.0.self_attn.rotary_emb.inv_freq": ("F32", [4], inv),
            # ship a real output projection so the model has no honest output
            # gap (the focus here is unmapped/offset/dtype handling, not the
            # no_output_weight warning, which is exercised in test_*_tied.py).
            "lm_head.weight": ("F32", [10, 8], np.zeros((10, 8), "<f4")),
        }
        d = write_model(os.path.join(self.tmp, "m"), _llama_cfg(), tensors)
        return d, embd, q

    def test_yields_canonical_fp32_with_values(self):
        d, embd, q = self._build()
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
        self.assertEqual(a_q.dtype, np.float32)
        self.assertEqual(a_q.shape, (8, 8))
        np.testing.assert_array_equal(a_q, q)

    def test_unmappable_tensor_warned_once_and_skipped(self):
        d, _, _ = self._build()
        rl = RunLog()
        src = HFSource(d, log=rl)
        names = [name for name, _a, _t in src.iter_weights()]

        # not yielded
        self.assertNotIn("model.layers.0.self_attn.rotary_emb.inv_freq", names)

        # exactly one tensor_unmapped — in BOTH the logger and src.warnings
        unmapped = [e for e in rl.entries if e["code"] == "tensor_unmapped"]
        self.assertEqual(len(unmapped), 1)
        self.assertIn("inv_freq", unmapped[0]["msg"])
        self.assertEqual(
            sum("inv_freq" in w for w in src.warnings), 1)
        self.assertEqual(len(src.warnings), 1)

    def test_unknown_family_empty_and_one_arch_warning(self):
        cfg = {"architectures": ["MambaXyzForCausalLM"], "model_type": "mamba-xyz",
               "num_hidden_layers": 1, "hidden_size": 4}
        tensors = {"backbone.embeddings.weight": ("F32", [4, 4],
                                                  np.zeros((4, 4), "<f4"))}
        d = write_model(os.path.join(self.tmp, "mamba"), cfg, tensors)
        rl = RunLog()
        src = HFSource(d, log=rl)
        out = list(src.iter_weights())
        self.assertEqual(out, [])
        arch_w = [e for e in rl.entries if e["code"] == "arch_unmapped"]
        self.assertEqual(len(arch_w), 1)
        self.assertEqual(len(src.warnings), 1)

    def test_want_filters_on_canonical_name(self):
        d, _, q = self._build()
        rl = RunLog()
        src = HFSource(d, log=rl)
        out = list(src.iter_weights(want=lambda n: n == "blk.0.attn_q.weight"))
        self.assertEqual([name for name, _a, _t in out], ["blk.0.attn_q.weight"])

    def test_default_logger_records_warnings(self):
        # Even without an injected logger, warnings must not be swallowed.
        d, _, _ = self._build()
        src = HFSource(d)
        list(src.iter_weights())
        self.assertEqual(len(src.warnings), 1)
        self.assertEqual(
            src.log.entries[-1]["code"], "tensor_unmapped")

    def test_metadata_source_reflects_iter_warnings(self):
        d, _, _ = self._build()
        rl = RunLog()
        src = HFSource(d, log=rl)
        list(src.iter_weights())
        sb = src.metadata()["source"]
        self.assertEqual(len(sb["warnings"]), 1)
        self.assertIn("inv_freq", sb["warnings"][0])

    def test_attention_bias_yields_canonical_bias_no_unmapped(self):
        # Qwen2-like: q_proj ships BOTH .weight and .bias. The bias must yield a
        # canonical blk.0.attn_q.bias (1-D) and produce NO tensor_unmapped warn.
        embd = np.arange(10 * 8, dtype="<f4").reshape(10, 8)
        q = np.arange(8 * 8, dtype="<f4").reshape(8, 8) * 0.5
        qb = np.arange(8, dtype="<f4") + 0.25
        tensors = {
            "model.embed_tokens.weight": ("F32", [10, 8], embd),
            "model.layers.0.self_attn.q_proj.weight": ("F32", [8, 8], q),
            "model.layers.0.self_attn.q_proj.bias": ("F32", [8], qb),
            # real output projection → no honest no_output_weight gap here.
            "lm_head.weight": ("F32", [10, 8], np.zeros((10, 8), "<f4")),
        }
        d = write_model(os.path.join(self.tmp, "qwen"), _llama_cfg(), tensors)
        rl = RunLog()
        src = HFSource(d, log=rl)
        out = list(src.iter_weights())
        by_name = {name: arr for name, arr, _ty in out}

        self.assertIn("blk.0.attn_q.weight", by_name)
        self.assertIn("blk.0.attn_q.bias", by_name)
        self.assertEqual(by_name["blk.0.attn_q.bias"].ndim, 1)
        self.assertEqual(by_name["blk.0.attn_q.bias"].shape, (8,))
        np.testing.assert_array_equal(by_name["blk.0.attn_q.bias"], qb)

        # NO tensor_unmapped warning for the bias (parity with GGUF).
        unmapped = [e for e in rl.entries if e["code"] == "tensor_unmapped"]
        self.assertEqual(unmapped, [])
        self.assertEqual(src.warnings, [])

    def test_moe_expert_tensors_yield_distinct_canonical_roles(self):
        cfg = {"architectures": ["Qwen2MoeForCausalLM"], "model_type": "qwen2_moe",
               "num_hidden_layers": 1, "num_attention_heads": 4,
               "num_key_value_heads": 2, "hidden_size": 8,
               "num_experts": 2, "tie_word_embeddings": False}
        tensors = {
            "model.embed_tokens.weight": ("F32", [10, 8], np.zeros((10, 8), "<f4")),
            "model.layers.0.self_attn.q_proj.weight": ("F32", [8, 8], np.zeros((8, 8), "<f4")),
            "model.layers.0.mlp.gate.weight": ("F32", [2, 8], np.zeros((2, 8), "<f4")),
            "model.layers.0.mlp.shared_expert_gate.weight": (
                "F32", [1, 8], np.zeros((1, 8), "<f4")),
            "model.layers.0.mlp.experts.0.gate_proj.weight": ("F32", [16, 8], np.zeros((16, 8), "<f4")),
            "model.layers.0.mlp.experts.1.gate_proj.weight": ("F32", [16, 8], np.zeros((16, 8), "<f4")),
            "model.layers.0.mlp.experts.1.down_proj.weight": ("F32", [8, 16], np.zeros((8, 16), "<f4")),
            "lm_head.weight": ("F32", [10, 8], np.zeros((10, 8), "<f4")),
        }
        d = write_model(os.path.join(self.tmp, "moe"), cfg, tensors)
        rl = RunLog()
        src = HFSource(d, log=rl)
        names = [name for name, _a, _t in src.iter_weights()]

        self.assertIn("blk.0.ffn_gate_inp.weight", names)
        self.assertIn("blk.0.ffn_gate_shared_inp.weight", names)
        self.assertIn("blk.0.ffn_gate_exps.e0.weight", names)
        self.assertIn("blk.0.ffn_gate_exps.e1.weight", names)
        self.assertIn("blk.0.ffn_down_exps.e1.weight", names)
        self.assertEqual([e for e in rl.entries if e["code"] == "arch_unmapped"], [])
        self.assertEqual([e for e in rl.entries if e["code"] == "tensor_unmapped"], [])
        self.assertEqual(src.metadata()["source"]["precision"], "exact")

    def test_bad_offset_errors_and_skips_no_crash(self):
        # Hand-craft a header whose data_offsets exceed the file region.
        embd = np.arange(10 * 8, dtype="<f4").reshape(10, 8)
        good = embd.tobytes()
        header = {
            "model.embed_tokens.weight": {
                "dtype": "F32", "shape": [10, 8],
                "data_offsets": [0, len(good)]},
            # claims far more bytes than exist
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype": "F32", "shape": [8, 8],
                "data_offsets": [len(good), len(good) + 999999]},
        }
        hjson = json.dumps(header).encode("utf-8")
        blob = struct.pack("<Q", len(hjson)) + hjson + good
        d = os.path.join(self.tmp, "bad")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(_llama_cfg(), f)
        with open(os.path.join(d, "model.safetensors"), "wb") as f:
            f.write(blob)

        rl = RunLog()
        src = HFSource(d, log=rl)
        names = [name for name, _a, _t in src.iter_weights()]
        self.assertEqual(names, ["token_embd.weight"])
        bad = [e for e in rl.entries if e["code"] == "bad_offset"]
        self.assertEqual(len(bad), 1)
        self.assertEqual(bad[0]["severity"], "error")
        # one bad_offset error; the model also legitimately has no output
        # projection (no lm_head, not tied) → one honest no_output_weight warn.
        self.assertIn("q_proj", bad[0]["msg"])
        no_out = [e for e in rl.entries if e["code"] == "no_output_weight"]
        self.assertEqual(len(no_out), 1)

    def test_offset_wrong_length_skips(self):
        # data_offsets are IN-RANGE but (e - s) != prod(dims) * itemsize → one
        # bad_offset error, that tensor skipped, valid siblings still yielded.
        # (The bad_offset test above only covers OUT-OF-range offsets.)
        embd = np.arange(10 * 8, dtype="<f4").reshape(10, 8)
        good = embd.tobytes()
        # q claims 8x8 F32 (256 bytes) but its in-range region is only 4 bytes.
        filler = b"\x00\x00\x00\x00"
        header = {
            "model.embed_tokens.weight": {
                "dtype": "F32", "shape": [10, 8],
                "data_offsets": [0, len(good)]},
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype": "F32", "shape": [8, 8],
                # in-range (within the file region) but only 4 bytes, not 256
                "data_offsets": [len(good), len(good) + len(filler)]},
        }
        hjson = json.dumps(header).encode("utf-8")
        blob = struct.pack("<Q", len(hjson)) + hjson + good + filler
        d = os.path.join(self.tmp, "wronglen")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(_llama_cfg(), f)
        with open(os.path.join(d, "model.safetensors"), "wb") as f:
            f.write(blob)

        rl = RunLog()
        src = HFSource(d, log=rl)
        names = [name for name, _a, _t in src.iter_weights()]
        # the valid embedding still yields; the wrong-length q is skipped
        self.assertEqual(names, ["token_embd.weight"])
        bad = [e for e in rl.entries if e["code"] == "bad_offset"]
        self.assertEqual(len(bad), 1)
        self.assertEqual(bad[0]["severity"], "error")
        self.assertIn("q_proj", bad[0]["msg"])

    def test_unsupported_dtype_tensor_skipped(self):
        # A header tensor with an unsupported dtype (I8) alongside F32 ones →
        # one dtype_unsupported warn, that tensor skipped, F32 tensors yielded.
        embd = np.arange(10 * 8, dtype="<f4").reshape(10, 8)
        good = embd.tobytes()
        # A MAPPABLE I8 tensor (k_proj) so it reaches the dtype branch rather
        # than being dropped earlier as unmapped. _ITEMSIZE has no I8 entry, so
        # the length check is skipped and only the in-range check applies — the
        # 4 bytes here are in-range, so the dtype_unsupported branch is hit.
        k = b"\x00" * 4
        header = {
            "model.embed_tokens.weight": {
                "dtype": "F32", "shape": [10, 8],
                "data_offsets": [0, len(good)]},
            "model.layers.0.self_attn.k_proj.weight": {
                "dtype": "I8", "shape": [2, 2],
                "data_offsets": [len(good), len(good) + len(k)]},
        }
        hjson = json.dumps(header).encode("utf-8")
        blob = struct.pack("<Q", len(hjson)) + hjson + good + k
        d = os.path.join(self.tmp, "i8")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(_llama_cfg(), f)
        with open(os.path.join(d, "model.safetensors"), "wb") as f:
            f.write(blob)

        rl = RunLog()
        src = HFSource(d, log=rl)
        names = [name for name, _a, _t in src.iter_weights()]
        self.assertEqual(names, ["token_embd.weight"])
        unsup = [e for e in rl.entries if e["code"] == "dtype_unsupported"]
        self.assertEqual(len(unsup), 1)
        self.assertEqual(unsup[0]["severity"], "warn")
        self.assertIn("I8", unsup[0]["msg"])
        self.assertEqual(src.metadata()["source"]["precision"], "inventory-only")


if __name__ == "__main__":
    unittest.main()
