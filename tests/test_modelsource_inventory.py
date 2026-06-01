"""Tests für tools/modelsource/inventory.py — der InventorySource-Breite-Pfad.

Dieser Backend macht NUR Inventar (kein Gewichts-Rechnen): PyTorch .bin/.pth
(ZIP + pickletools-Opcode-Scan, NIE pickle.load) und ONNX .onnx (flacher
Protobuf-Namens-Scan, sonst Datei-Inventar). Er darf NIE in den Batch werfen;
jeder Fallback wird geloggt. Komplett numpy-frei.

Der PyTorch-ZIP-Fixture legt eine echte ``archive/data.pkl`` an, deren Inhalt
mit ``pickle.dumps`` eines reinen dict[str,int] erzeugt wird (KEINE Tensoren,
damit der reine Opcode-Scan die dotted keys findet — ohne je zu entpickeln).
"""
import io
import os
import pickle
import struct
import sys
import tempfile
import unittest
import zipfile

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, HERE)

from modelsource.inventory import InventorySource


def _write_torch_zip(path):
    """Minimaler torch.save-artiger ZIP: archive/data.pkl mit echtem pickle
    eines dict[str,int] (state_dict-Keys als BINUNICODE-Strings)."""
    state = {"model.layers.0.weight": 1, "model.embed.weight": 2}
    blob = pickle.dumps(state, protocol=2)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", blob)
        # plausible torch member (we never read it)
        zf.writestr("archive/data/0", b"\x00\x00\x00\x00")


def _proto_varint(field_number, wire_type, length):
    """(tag varint, length varint) for a length-delimited protobuf field."""
    tag = (field_number << 3) | wire_type
    return _varint(tag) + _varint(length)


def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _ld(field_number, payload):
    """Encode one length-delimited (wire type 2) protobuf field."""
    tag = (field_number << 3) | 2
    return _varint(tag) + _varint(len(payload)) + payload


def _v(field_number, value):
    """Encode one varint (wire type 0) protobuf field."""
    tag = (field_number << 3) | 0
    return _varint(tag) + _varint(value)


def _write_onnx_with_initializer(path, init_name, dims=None, data_type=1, packed_dims=False):
    """Hand-roll a tiny ONNX ModelProto with one graph initializer named
    ``init_name``. ModelProto.graph = field 7; GraphProto.initializer =
    field 5 (TensorProto); TensorProto fields: dims=1, data_type=2, name=8."""
    dims = list(dims or [])
    dim_blob = (
        _ld(1, b"".join(_varint(d) for d in dims)) if packed_dims
        else b"".join(_v(1, d) for d in dims)
    )
    tensor = (
        dim_blob
        + _v(2, data_type)                          # TensorProto.data_type
        + _ld(8, init_name.encode("utf-8"))          # TensorProto.name
    )
    graph = _ld(5, tensor)                           # GraphProto.initializer
    model = _ld(7, graph)                            # ModelProto.graph
    with open(path, "wb") as f:
        f.write(model)


class TestPyTorchZip(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = self._td.name

    def tearDown(self):
        self._td.cleanup()

    def test_metadata_source_block(self):
        p = os.path.join(self.tmp, "pytorch_model.bin")
        _write_torch_zip(p)
        meta = InventorySource(p).metadata()
        src = meta["source"]
        self.assertEqual(src["format"], "pytorch")
        self.assertEqual(src["precision"], "inventory-only")
        self.assertIs(src["mapped"], False)

    def test_dotted_names_extracted(self):
        p = os.path.join(self.tmp, "pytorch_model.bin")
        _write_torch_zip(p)
        meta = InventorySource(p).metadata()
        names = {t["name"] for t in meta["tensors"]}
        self.assertIn("model.layers.0.weight", names)
        self.assertIn("model.embed.weight", names)
        # honest: dims/type unknown for pickle-scanned names
        for t in meta["tensors"]:
            if t["name"] in ("model.layers.0.weight", "model.embed.weight"):
                self.assertIsNone(t["dims"])
                self.assertIsNone(t["type"])

    def test_iter_weights_empty_and_one_info(self):
        p = os.path.join(self.tmp, "pytorch_model.bin")
        _write_torch_zip(p)
        src = InventorySource(p)
        self.assertEqual(list(src.iter_weights()), [])
        infos = [e for e in src.log.entries
                 if e["severity"] == "info" and e["code"] == "inventory_only"]
        self.assertEqual(len(infos), 1)

    def test_tokenizer_none(self):
        p = os.path.join(self.tmp, "pytorch_model.bin")
        _write_torch_zip(p)
        self.assertIsNone(InventorySource(p).tokenizer())


class TestPickleUnparsable(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = self._td.name

    def tearDown(self):
        self._td.cleanup()

    def test_no_data_pkl_falls_back_to_member_list(self):
        # A valid zip with NO data.pkl member → no crash; degrade to the archive
        # member-list inventory with a pickle_unparsed warning.
        p = os.path.join(self.tmp, "pytorch_model.bin")
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("archive/data/0", b"\x00\x00\x00\x00")
            zf.writestr("archive/version", b"3")
        src = InventorySource(p)
        meta = src.metadata()  # must not crash
        warns = [e for e in src.log.entries
                 if e["severity"] == "warn" and e["code"] == "pickle_unparsed"]
        self.assertEqual(len(warns), 1)
        # member-list inventory: the zip members are listed (no dotted
        # state_dict keys, since no pickle was scanned).
        names = {t["name"] for t in meta["tensors"]}
        self.assertIn("archive/data/0", names)
        self.assertIn("archive/version", names)
        self.assertEqual(meta["source"]["format"], "pytorch")

    def test_sibling_config_unreadable_warned(self):
        # A .bin whose sibling config.json is unreadable (invalid JSON) →
        # arch None + a logged config_unparsed warn, no raise.
        p = os.path.join(self.tmp, "pytorch_model.bin")
        _write_torch_zip(p)
        with open(os.path.join(self.tmp, "config.json"), "w") as f:
            f.write("{ this is : not valid json ]")
        src = InventorySource(p)
        meta = src.metadata()  # must not crash
        self.assertIsNone(meta["source"]["arch"])
        warns = [e for e in src.log.entries
                 if e["severity"] == "warn" and e["code"] == "config_unparsed"]
        self.assertEqual(len(warns), 1)

    def test_non_zip_bin_degrades(self):
        p = os.path.join(self.tmp, "pytorch_model.bin")
        with open(p, "wb") as f:
            f.write(os.urandom(256))
        src = InventorySource(p)
        meta = src.metadata()  # must not crash
        self.assertEqual(meta["source"]["format"], "pytorch")
        warns = [e for e in src.log.entries
                 if e["severity"] == "warn" and e["code"] == "pickle_unparsed"]
        self.assertEqual(len(warns), 1)
        # file-level inventory entry present
        self.assertTrue(len(meta["tensors"]) >= 1)

    def test_iter_weights_empty_on_garbage(self):
        p = os.path.join(self.tmp, "pytorch_model.bin")
        with open(p, "wb") as f:
            f.write(os.urandom(256))
        self.assertEqual(list(InventorySource(p).iter_weights()), [])


class TestZipBombCap(unittest.TestCase):
    """ROUND-1 Fix 3 — a torch .bin whose data.pkl decompresses past the cap
    must NOT be read into memory: degrade to the archive member-list inventory
    with exactly one ``pickle_too_large`` warning. Never raise, never OOM.

    We avoid crafting an actual multi-GB bomb by monkeypatching the cap to a
    tiny value and using a normal (but larger-than-tiny-cap) pkl."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = self._td.name

    def tearDown(self):
        self._td.cleanup()

    def test_oversized_pkl_falls_back_to_member_list(self):
        from modelsource import inventory
        p = os.path.join(self.tmp, "pytorch_model.bin")
        state = {"model.layers.0.weight": 1, "model.embed.weight": 2}
        blob = pickle.dumps(state, protocol=2)
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("archive/data.pkl", blob)
            zf.writestr("archive/data/0", b"\x00\x00\x00\x00")

        orig = inventory._MAX_PKL_BYTES
        inventory._MAX_PKL_BYTES = 1   # force the cap to trip
        try:
            src = InventorySource(p)
            meta = src.metadata()      # must NOT raise / OOM
        finally:
            inventory._MAX_PKL_BYTES = orig

        warns = [e for e in src.log.entries
                 if e["severity"] == "warn" and e["code"] == "pickle_too_large"]
        self.assertEqual(len(warns), 1, f"expected one pickle_too_large, got {src.log.entries}")
        # fell back to the archive member-list inventory (the pkl was NOT scanned
        # for state_dict keys → those dotted names are absent)
        names = {t["name"] for t in meta["tensors"]}
        self.assertIn("archive/data.pkl", names)
        self.assertNotIn("model.layers.0.weight", names)
        self.assertEqual(meta["source"]["format"], "pytorch")

    def test_normal_pkl_still_scanned(self):
        # Under the real (large) cap, a normal pkl is still opcode-scanned.
        p = os.path.join(self.tmp, "pytorch_model.bin")
        _write_torch_zip(p)
        src = InventorySource(p)
        meta = src.metadata()
        names = {t["name"] for t in meta["tensors"]}
        self.assertIn("model.layers.0.weight", names)
        self.assertFalse([e for e in src.log.entries
                          if e["code"] == "pickle_too_large"])


class TestOnnx(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = self._td.name

    def tearDown(self):
        self._td.cleanup()

    def test_initializer_name_listed(self):
        p = os.path.join(self.tmp, "model.onnx")
        _write_onnx_with_initializer(p, "encoder.weight", dims=[2, 3], data_type=1)
        meta = InventorySource(p).metadata()
        self.assertEqual(meta["source"]["format"], "onnx")
        self.assertEqual(meta["source"]["precision"], "inventory-only")
        self.assertIs(meta["source"]["mapped"], False)
        by_name = {t["name"]: t for t in meta["tensors"]}
        self.assertIn("encoder.weight", by_name)
        self.assertEqual(by_name["encoder.weight"]["dims"], [2, 3])
        self.assertEqual(by_name["encoder.weight"]["type"], "FLOAT")
        self.assertEqual(meta["quant_breakdown"], {"FLOAT": 1})

    def test_initializer_packed_dims_supported(self):
        p = os.path.join(self.tmp, "model.onnx")
        _write_onnx_with_initializer(
            p, "decoder.weight", dims=[4, 5], data_type=10, packed_dims=True)
        meta = InventorySource(p).metadata()
        by_name = {t["name"]: t for t in meta["tensors"]}
        self.assertEqual(by_name["decoder.weight"]["dims"], [4, 5])
        self.assertEqual(by_name["decoder.weight"]["type"], "FLOAT16")

    def test_garbage_onnx_degrades(self):
        p = os.path.join(self.tmp, "model.onnx")
        with open(p, "wb") as f:
            f.write(b"\xff\xff\xff not a protobuf \x00\x01")
        src = InventorySource(p)
        meta = src.metadata()  # must not crash
        self.assertEqual(meta["source"]["format"], "onnx")
        warns = [e for e in src.log.entries
                 if e["severity"] == "warn" and e["code"] == "onnx_shallow"]
        self.assertEqual(len(warns), 1)
        self.assertTrue(len(meta["tensors"]) >= 1)

    def test_oversized_onnx_falls_back_to_file_inventory(self):
        # The stdlib ONNX scanner reads the protobuf bytes, so it must check a
        # file-size cap first. Force the cap tiny to exercise the fallback
        # without creating a large fixture.
        from modelsource import inventory
        p = os.path.join(self.tmp, "large.onnx")
        _write_onnx_with_initializer(p, "encoder.weight")

        orig = inventory._MAX_ONNX_BYTES
        inventory._MAX_ONNX_BYTES = 1
        try:
            src = InventorySource(p)
            meta = src.metadata()      # must NOT read/scan past the cap
        finally:
            inventory._MAX_ONNX_BYTES = orig

        warns = [e for e in src.log.entries
                 if e["severity"] == "warn" and e["code"] == "onnx_too_large"]
        self.assertEqual(len(warns), 1, f"expected onnx_too_large, got {src.log.entries}")
        self.assertEqual(meta["tensors"], [{
            "name": "large.onnx", "dims": None, "type": None}])

    def test_iter_weights_empty(self):
        p = os.path.join(self.tmp, "model.onnx")
        _write_onnx_with_initializer(p, "encoder.weight")
        src = InventorySource(p)
        self.assertEqual(list(src.iter_weights()), [])
        infos = [e for e in src.log.entries
                 if e["severity"] == "info" and e["code"] == "inventory_only"]
        self.assertEqual(len(infos), 1)


if __name__ == "__main__":
    unittest.main()
