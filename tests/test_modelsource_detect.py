"""Tests für tools/modelsource/__init__.py — der detect() Format-Dispatcher.

detect() ist der EINZIGE Einstiegspunkt, den der Rest der Pipeline aufruft. Er
entscheidet anhand von Pfad / Endung / GGUF-Magic / Verzeichnis-Inhalt, welcher
Source-Backend zuständig ist, und gibt für Unbekanntes None zurück (plus genau
EINE unknown_format-Warnung) — er wirft NIE.
"""
import json
import os
import pickle
import sys
import tempfile
import unittest
import zipfile

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, HERE)

from modelsource import detect
from modelsource.gguf_backend import GGUFSource
from modelsource.hf_backend import HFSource
from modelsource.inventory import InventorySource
from modelsource.log import RunLog

from st_fixture import write_model, write_sharded_model

_CFG = {"architectures": ["LlamaForCausalLM"], "hidden_size": 4}
_T = {"model.embed_tokens.weight": ("F32", [2, 2], None)}


def _write_torch_zip(path):
    """Minimal torch.save-style ZIP: archive/data.pkl holding a dict[str,int]."""
    state = {"model.layers.0.weight": 1, "model.embed.weight": 2}
    blob = pickle.dumps(state, protocol=2)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", blob)
        zf.writestr("archive/data/0", b"\x00\x00\x00\x00")


class DetectTest(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp = self._td.name

    def _touch(self, name):
        p = os.path.join(self.tmp, name)
        with open(p, "wb") as f:
            f.write(b"")
        return p

    def test_gguf_file(self):
        p = self._touch("model.gguf")
        self.assertIsInstance(detect(p), GGUFSource)

    def test_gguf_uppercase_ext(self):
        p = self._touch("MODEL.GGUF")
        self.assertIsInstance(detect(p), GGUFSource)

    def test_gguf_extensionless_magic(self):
        p = os.path.join(self.tmp, "sha256-deadbeef")
        with open(p, "wb") as f:
            f.write(b"GGUF\x03\x00\x00\x00")
        self.assertIsInstance(detect(p), GGUFSource)

    def test_hf_dir_single(self):
        d = write_model(os.path.join(self.tmp, "hf"), _CFG, _T)
        self.assertIsInstance(detect(d), HFSource)

    def test_hf_dir_sharded(self):
        shards = {"model-00001-of-00001.safetensors": _T}
        d = write_sharded_model(os.path.join(self.tmp, "hfshard"), _CFG, shards)
        self.assertIsInstance(detect(d), HFSource)

    def test_pytorch_bin_dir_is_inventory(self):
        # A real HF pytorch model: a DIRECTORY with config.json +
        # pytorch_model.bin and NO safetensors. detect() must route it to
        # InventorySource pointed at the .bin (so sibling-config arch + pickle
        # scan work), not fall through to unknown_format.
        d = os.path.join(self.tmp, "ptdir")
        os.makedirs(d)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"architectures": ["LlamaForCausalLM"],
                       "model_type": "llama"}, f)
        _write_torch_zip(os.path.join(d, "pytorch_model.bin"))

        log = RunLog()
        src = detect(d, log=log)
        self.assertIsInstance(src, InventorySource)
        meta = src.metadata()
        self.assertEqual(meta["source"]["format"], "pytorch")
        self.assertEqual(meta["source"]["precision"], "inventory-only")
        self.assertEqual(meta["source"]["arch"], "LlamaForCausalLM")
        self.assertEqual(list(src.iter_weights()), [])
        # No unknown_format warning emitted for this routed dir.
        self.assertEqual(
            [e for e in log.entries if e["code"] == "unknown_format"], [])

    def test_pytorch_bin_dir_via_index(self):
        # config.json + pytorch_model.bin.index.json (sharded pytorch), no
        # safetensors → InventorySource pointed at the first shard.
        d = os.path.join(self.tmp, "ptidx")
        os.makedirs(d)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"architectures": ["LlamaForCausalLM"]}, f)
        _write_torch_zip(os.path.join(d, "pytorch_model-00001-of-00001.bin"))
        with open(os.path.join(d, "pytorch_model.bin.index.json"), "w") as f:
            json.dump({"weight_map": {
                "model.layers.0.weight": "pytorch_model-00001-of-00001.bin"}}, f)
        src = detect(d)
        self.assertIsInstance(src, InventorySource)
        self.assertTrue(src.path.endswith("pytorch_model-00001-of-00001.bin"))

    def test_safetensors_wins_over_bin_in_dir(self):
        # A dir with config.json + BOTH a safetensors AND a .bin → HFSource
        # (safetensors wins, never inventory).
        d = write_model(os.path.join(self.tmp, "both"), _CFG, _T)
        _write_torch_zip(os.path.join(d, "pytorch_model.bin"))
        self.assertIsInstance(detect(d), HFSource)

    def test_odd_dir_returns_none(self):
        # config.json but no bin/pth/safetensors → unknown_format + None.
        d = os.path.join(self.tmp, "odd")
        os.makedirs(d)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"architectures": ["X"]}, f)
        log = RunLog()
        self.assertIsNone(detect(d, log=log))
        self.assertEqual(
            len([e for e in log.entries if e["code"] == "unknown_format"]), 1)

    def test_bin_is_inventory_pytorch(self):
        p = self._touch("weights.bin")
        src = detect(p)
        self.assertIsInstance(src, InventorySource)
        self.assertEqual(src._fmt(), "pytorch")

    def test_pth_is_inventory_pytorch(self):
        p = self._touch("weights.pth")
        src = detect(p)
        self.assertIsInstance(src, InventorySource)
        self.assertEqual(src._fmt(), "pytorch")

    def test_onnx_is_inventory(self):
        p = self._touch("model.onnx")
        src = detect(p)
        self.assertIsInstance(src, InventorySource)
        self.assertEqual(src._fmt(), "onnx")

    def test_unknown_ext_returns_none_one_warning(self):
        p = self._touch("notes.txt")
        log = RunLog()
        self.assertIsNone(detect(p, log=log))
        warns = [e for e in log.entries if e["code"] == "unknown_format"]
        self.assertEqual(len(warns), 1)
        self.assertEqual(warns[0]["severity"], "warn")
        self.assertEqual(warns[0]["stage"], "detect")

    def test_nonexistent_path_returns_none_no_raise(self):
        log = RunLog()
        missing = os.path.join(self.tmp, "does-not-exist.gguf")
        self.assertIsNone(detect(missing, log=log))
        self.assertEqual(
            len([e for e in log.entries if e["code"] == "unknown_format"]), 1)

    def test_unknown_uses_model_label_when_given(self):
        p = self._touch("notes.txt")
        log = RunLog()
        detect(p, log=log, model="mymodel")
        warns = [e for e in log.entries if e["code"] == "unknown_format"]
        self.assertEqual(warns[0]["model"], "mymodel")

    def test_unknown_without_log_does_not_raise(self):
        p = self._touch("notes.txt")
        # No log passed: a fresh RunLog must be used internally; no exception.
        self.assertIsNone(detect(p))


if __name__ == "__main__":
    unittest.main()
