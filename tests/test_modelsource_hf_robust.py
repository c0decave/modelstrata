"""ROUND-1 robustness — HFSource MUST NEVER raise out of metadata()/iter_weights().

A corrupt or non-standard HF model (missing/corrupt config.json, truncated
safetensors header, a malformed per-tensor header descriptor) must DEGRADE
gracefully — exactly like the inventory backend — by logging a structured
``_error(...)`` and returning/yielding a degraded (never crashing) result. One
bad tensor descriptor must NOT prevent the good tensors in the same file from
being yielded.

numpy is needed only to write the safetensors fixtures.
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
    from st_fixture import build_st, write_model


def _llama_cfg():
    return {"architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "num_hidden_layers": 1, "num_attention_heads": 4,
            "num_key_value_heads": 2, "hidden_size": 8}


def _write_raw_st(d, config, header):
    """Write a config.json plus a model.safetensors whose header is EXACTLY the
    given (possibly malformed) descriptor dict + a tiny raw tensor region."""
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(config, f)
    body = (np.arange(10 * 8, dtype="<f4").reshape(10, 8).tobytes()
            if HAVE_NP else b"\x00" * 320)
    hjson = json.dumps(header).encode("utf-8")
    with open(os.path.join(d, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hjson)) + hjson + body)
    return d


@unittest.skipUnless(HAVE_NP, "numpy not installed")
class TestHFNeverRaisesConfig(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = self._td.name

    def tearDown(self):
        self._td.cleanup()

    def test_missing_config_degrades(self):
        # An HF-ish dir: a model.safetensors but NO config.json.
        d = os.path.join(self.tmp, "noconfig")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "model.safetensors"), "wb") as f:
            hjson = json.dumps({}).encode("utf-8")
            f.write(struct.pack("<Q", len(hjson)) + hjson)
        rl = RunLog()
        src = HFSource(d, log=rl)
        meta = src.metadata()                  # must NOT raise
        self.assertEqual(list(src.iter_weights()), [])  # must NOT raise
        self.assertEqual(meta["source"]["precision"], "inventory-only")
        bad = [e for e in rl.entries if e["code"] == "bad_config"]
        self.assertTrue(bad, f"expected a bad_config error, got {rl.entries}")
        self.assertEqual(bad[0]["severity"], "error")

    def test_corrupt_config_degrades(self):
        d = os.path.join(self.tmp, "badconfig")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w") as f:
            f.write("{ this is : not json,, ]")
        with open(os.path.join(d, "model.safetensors"), "wb") as f:
            hjson = json.dumps({}).encode("utf-8")
            f.write(struct.pack("<Q", len(hjson)) + hjson)
        rl = RunLog()
        src = HFSource(d, log=rl)
        meta = src.metadata()
        self.assertEqual(list(src.iter_weights()), [])
        self.assertEqual(meta["source"]["precision"], "inventory-only")
        self.assertTrue([e for e in rl.entries if e["code"] == "bad_config"])


@unittest.skipUnless(HAVE_NP, "numpy not installed")
class TestHFNeverRaisesHeader(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = self._td.name

    def tearDown(self):
        self._td.cleanup()

    def test_truncated_header_degrades(self):
        # Declare a huge header_len but write almost nothing after it.
        d = os.path.join(self.tmp, "trunc")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(_llama_cfg(), f)
        with open(os.path.join(d, "model.safetensors"), "wb") as f:
            f.write(struct.pack("<Q", 4096) + b"{")   # header_len >> file
        rl = RunLog()
        src = HFSource(d, log=rl)
        meta = src.metadata()                 # must NOT raise
        self.assertEqual(list(src.iter_weights()), [])
        self.assertEqual(meta["source"]["precision"], "inventory-only")
        self.assertTrue([e for e in rl.entries if e["code"] == "bad_header"],
                        f"expected bad_header, got {rl.entries}")

    def test_descriptor_missing_shape_skips_that_tensor_only(self):
        # One GOOD tensor + one tensor whose descriptor has NO "shape".
        good = np.arange(10 * 8, dtype="<f4").reshape(10, 8).tobytes()
        header = {
            "model.embed_tokens.weight": {
                "dtype": "F32", "shape": [10, 8],
                "data_offsets": [0, len(good)]},
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype": "F32",       # NO shape
                "data_offsets": [len(good), len(good)]},
        }
        d = os.path.join(self.tmp, "noshape")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(_llama_cfg(), f)
        hjson = json.dumps(header).encode("utf-8")
        with open(os.path.join(d, "model.safetensors"), "wb") as f:
            f.write(struct.pack("<Q", len(hjson)) + hjson + good)
        rl = RunLog()
        src = HFSource(d, log=rl)
        names = [n for n, _a, _t in src.iter_weights()]   # must NOT raise
        # the GOOD tensor is still yielded despite the malformed sibling
        self.assertIn("token_embd.weight", names)
        bad = [e for e in rl.entries if e["code"] == "bad_header"]
        self.assertEqual(len(bad), 1, f"expected one bad_header, got {rl.entries}")
        self.assertEqual(bad[0]["severity"], "error")
        # metadata() also survives
        self.assertIsInstance(src.metadata(), dict)

    def test_descriptor_missing_data_offsets_skips_that_tensor_only(self):
        good = np.arange(10 * 8, dtype="<f4").reshape(10, 8).tobytes()
        header = {
            "model.embed_tokens.weight": {
                "dtype": "F32", "shape": [10, 8],
                "data_offsets": [0, len(good)]},
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype": "F32", "shape": [8, 8]},   # NO data_offsets
        }
        d = _write_raw_st(os.path.join(self.tmp, "nooff"), _llama_cfg(), header)
        rl = RunLog()
        src = HFSource(d, log=rl)
        names = [n for n, _a, _t in src.iter_weights()]
        self.assertIn("token_embd.weight", names)
        bad = [e for e in rl.entries if e["code"] == "bad_header"]
        self.assertEqual(len(bad), 1, f"expected one bad_header, got {rl.entries}")

    def test_descriptor_data_offsets_wrong_arity_skips_that_tensor_only(self):
        good = np.arange(10 * 8, dtype="<f4").reshape(10, 8).tobytes()
        header = {
            "model.embed_tokens.weight": {
                "dtype": "F32", "shape": [10, 8],
                "data_offsets": [0, len(good)]},
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype": "F32", "shape": [8, 8],
                "data_offsets": [len(good)]},   # arity != 2
        }
        d = _write_raw_st(os.path.join(self.tmp, "arity"), _llama_cfg(), header)
        rl = RunLog()
        src = HFSource(d, log=rl)
        names = [n for n, _a, _t in src.iter_weights()]
        self.assertIn("token_embd.weight", names)
        bad = [e for e in rl.entries if e["code"] == "bad_header"]
        self.assertEqual(len(bad), 1, f"expected one bad_header, got {rl.entries}")


def _write_raw_st_header_bytes(d, config, header_bytes):
    """Write a config.json plus a model.safetensors whose header region is the
    EXACT bytes given (used to craft a header that is valid JSON but NOT an
    object — a list/string/number — which json.dump of a dict can't produce)."""
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(config, f)
    body = b"\x00" * 320
    with open(os.path.join(d, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)) + header_bytes + body)
    return d


@unittest.skipUnless(HAVE_NP, "numpy not installed")
class TestHFNeverRaisesWrongType(unittest.TestCase):
    """Valid JSON but the WRONG top-level TYPE (list/string/number where an
    object is required) must DEGRADE — not raise AttributeError/TypeError out
    of metadata()/iter_weights()."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = self._td.name

    def tearDown(self):
        self._td.cleanup()

    def test_header_is_json_array_degrades(self):
        # model.safetensors header is a JSON array, not an object.
        hbytes = json.dumps([1, 2, 3]).encode("utf-8")
        d = _write_raw_st_header_bytes(
            os.path.join(self.tmp, "harr"), _llama_cfg(), hbytes)
        rl = RunLog()
        src = HFSource(d, log=rl)
        meta = src.metadata()                          # must NOT raise
        self.assertIsInstance(meta, dict)
        self.assertEqual(meta["source"]["precision"], "inventory-only")
        self.assertEqual(list(src.iter_weights()), [])  # must NOT raise
        self.assertTrue([e for e in rl.entries if e["code"] == "bad_header"],
                        f"expected bad_header, got {rl.entries}")

    def test_config_is_json_array_degrades(self):
        # config.json is a JSON list, not an object.
        d = os.path.join(self.tmp, "carr")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(["LlamaForCausalLM"], f)
        with open(os.path.join(d, "model.safetensors"), "wb") as f:
            hjson = json.dumps({}).encode("utf-8")
            f.write(struct.pack("<Q", len(hjson)) + hjson)
        rl = RunLog()
        src = HFSource(d, log=rl)
        meta = src.metadata()                          # must NOT raise
        self.assertIsInstance(meta, dict)
        self.assertEqual(meta["source"]["precision"], "inventory-only")
        self.assertEqual(list(src.iter_weights()), [])  # must NOT raise
        self.assertTrue([e for e in rl.entries if e["code"] == "bad_config"],
                        f"expected bad_config, got {rl.entries}")

    def test_index_is_json_array_degrades(self):
        # model.safetensors.index.json is a JSON list, not an object (sharded).
        d = os.path.join(self.tmp, "iarr")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(_llama_cfg(), f)
        with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
            json.dump(["not", "an", "object"], f)
        rl = RunLog()
        src = HFSource(d, log=rl)
        meta = src.metadata()                          # must NOT raise
        self.assertIsInstance(meta, dict)
        # A broken index ⇒ no tensors surfaced (config still maps the arch, so
        # this degrades to inventory-empty rather than crashing).
        self.assertEqual(meta["tensors"], [])
        self.assertEqual(list(src.iter_weights()), [])  # must NOT raise
        self.assertTrue([e for e in rl.entries if e["code"] == "bad_index"],
                        f"expected bad_index, got {rl.entries}")

    def test_sharded_metadata_skips_corrupt_shard_only(self):
        # A corrupt shard header must not make metadata() throw away tensors
        # from the other readable shards. iter_weights() already had this
        # per-shard behavior; metadata() should match it.
        d = os.path.join(self.tmp, "onebadshard")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(_llama_cfg(), f)
        good_name = "model.embed_tokens.weight"
        bad_name = "model.layers.0.self_attn.q_proj.weight"
        good = np.arange(10 * 8, dtype="<f4").reshape(10, 8)
        with open(os.path.join(d, "good.safetensors"), "wb") as f:
            f.write(build_st({good_name: ("F32", [10, 8], good)}))
        with open(os.path.join(d, "bad.safetensors"), "wb") as f:
            f.write(struct.pack("<Q", 4096) + b"{")  # truncated header
        with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": {
                good_name: "good.safetensors",
                bad_name: "bad.safetensors",
            }}, f)

        rl = RunLog()
        src = HFSource(d, log=rl)
        meta = src.metadata()                         # must NOT raise
        names = {t["name"] for t in meta["tensors"]}
        self.assertIn(good_name, names)
        self.assertNotIn(bad_name, names)
        self.assertEqual(meta["source"]["precision"], "exact")
        bad = [e for e in rl.entries if e["code"] == "bad_header"]
        self.assertEqual(len(bad), 1, f"expected one bad_header, got {rl.entries}")

    def test_descriptor_is_json_string_skips_that_tensor_only(self):
        # One GOOD tensor + one whose descriptor is a JSON string (not object).
        good = np.arange(10 * 8, dtype="<f4").reshape(10, 8).tobytes()
        header = {
            "model.embed_tokens.weight": {
                "dtype": "F32", "shape": [10, 8],
                "data_offsets": [0, len(good)]},
            "model.layers.0.self_attn.q_proj.weight": "not-an-object",
        }
        d = os.path.join(self.tmp, "tdstr")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(_llama_cfg(), f)
        hjson = json.dumps(header).encode("utf-8")
        with open(os.path.join(d, "model.safetensors"), "wb") as f:
            f.write(struct.pack("<Q", len(hjson)) + hjson + good)
        rl = RunLog()
        src = HFSource(d, log=rl)
        names = [n for n, _a, _t in src.iter_weights()]   # must NOT raise
        self.assertIn("token_embd.weight", names)
        bad = [e for e in rl.entries if e["code"] == "bad_header"]
        self.assertEqual(len(bad), 1, f"expected one bad_header, got {rl.entries}")
        self.assertEqual(bad[0]["severity"], "error")
        self.assertIsInstance(src.metadata(), dict)     # metadata() also survives


if __name__ == "__main__":
    unittest.main()
