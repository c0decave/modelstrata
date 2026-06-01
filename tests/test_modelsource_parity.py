"""Parität: dasselbe winzige Modell als GGUF und als HF-safetensors muss von der
Pipeline IDENTISCH behandelt werden (gleiche kanonische Tensornamen + (role,layer)-
Gruppierung; bei vorhandenem gguf-Paket zusätzlich identische weight_stats-Struktur
und Per-Zelle-Statistik).

Naming-/Grouping-Parität läuft LOKAL (nur stdlib + numpy, kein gguf-Paket nötig):
die GGUF-Seite liest ihre kanonischen Namen aus dem Tensor-Directory der Metadaten
(gguf_inspect, stdlib); die HF-Seite normalisiert über archmap→naming.canonical.

Stats-Parität ist mit @skipUnless(HAVE_GGUF) bewacht — lokal SKIP, auf dem Host läuft
sie (F32-Fixtures ⇒ exakt, keine Quant-Toleranz nötig)."""
import os
import re
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, HERE)

import gguf_inspect
from gguf_fixture import write_gguf, T_STRING, T_UINT32
from st_fixture import write_model
from modelsource import detect
from modelsource.hf_backend import HFSource

try:
    import gguf  # noqa: F401
    HAVE_GGUF = True
except Exception:
    HAVE_GGUF = False

# GGML type id for F32 (matches gguf_inspect.GGML_TYPES); fixtures stay F32 for
# exactness (no quant noise to tolerate).
F32 = 0

BLK = re.compile(
    r"^(?P<pre>(?:[a-z]+\.)*)blk\.(?P<layer>\d+)\.(?P<role>.+?)\.(?P<suf>weight|bias)$")


# The two parallel name lists: same logical tensors, two naming conventions.
# (canonical GGUF name, equivalent HF name, logical shape)
PAIRS = [
    ("token_embd.weight",        "model.embed_tokens.weight",                       (8, 4)),
    ("blk.0.attn_q.weight",      "model.layers.0.self_attn.q_proj.weight",          (4, 4)),
    ("blk.0.attn_k.weight",      "model.layers.0.self_attn.k_proj.weight",          (4, 4)),
    ("blk.0.attn_v.weight",      "model.layers.0.self_attn.v_proj.weight",          (4, 4)),
    ("blk.0.attn_output.weight", "model.layers.0.self_attn.o_proj.weight",          (4, 4)),
    ("blk.0.ffn_gate.weight",    "model.layers.0.mlp.gate_proj.weight",             (6, 4)),
    ("blk.0.ffn_up.weight",      "model.layers.0.mlp.up_proj.weight",               (6, 4)),
    ("blk.0.ffn_down.weight",    "model.layers.0.mlp.down_proj.weight",             (4, 6)),
    ("blk.1.attn_q.weight",      "model.layers.1.self_attn.q_proj.weight",          (4, 4)),
    ("blk.1.ffn_down.weight",    "model.layers.1.mlp.down_proj.weight",             (4, 6)),
]

CONFIG = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "hidden_size": 4,
    "tie_word_embeddings": False,
}


def _grouping(names):
    """Set of (role, layer) for block-weight tensors + set of all names."""
    grp = set()
    for n in names:
        m = BLK.match(n)
        if m and m.group("suf") == "weight":
            grp.add((m.group("pre") + m.group("role"), int(m.group("layer"))))
    return grp


def _build_gguf(path):
    # Canonical names live in the GGUF tensor directory; dims in ne-order
    # (reversed logical). Offsets are irrelevant for name/dir reading.
    tensors = []
    off = 0
    for cname, _hf, shape in PAIRS:
        ne = list(reversed(shape))            # GGUF stores ne (reversed logical)
        tensors.append((cname, ne, F32, off))
        off += 64
    kvs = [("general.architecture", T_STRING, "llama")]
    return write_gguf(path, kvs, tensors)


def _build_hf(d):
    tensors = {}
    for _c, hf, shape in PAIRS:
        arr = np.arange(int(np.prod(shape)), dtype="<f4").reshape(shape)
        tensors[hf] = ("F32", shape, arr)
    return write_model(d, CONFIG, tensors)


class TestNamingParity(unittest.TestCase):
    """Runs locally — no gguf package required."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp = self._td.name

    def test_canonical_names_and_grouping_identical(self):
        # GGUF side: read canonical names straight from the metadata directory.
        gpath = os.path.join(self.tmp, "m.gguf")
        _build_gguf(gpath)
        gmeta = gguf_inspect.parse_gguf(gpath)
        gguf_names = {t["name"] for t in gmeta["tensors"]}

        # HF side: archmap normalization via the real backend (numpy only).
        hdir = os.path.join(self.tmp, "hf")
        _build_hf(hdir)
        src = detect(hdir)
        self.assertIsInstance(src, HFSource)
        hf_names = {name for name, _arr, _dt in src.iter_weights()}

        # The canonical name SETS must match exactly.
        self.assertEqual(gguf_names, hf_names)
        # And the (role, layer) grouping derived from each side.
        self.assertEqual(_grouping(gguf_names), _grouping(hf_names))

    def test_detect_picks_right_backend(self):
        gpath = os.path.join(self.tmp, "m.gguf")
        _build_gguf(gpath)
        self.assertEqual(type(detect(gpath)).__name__, "GGUFSource")
        hdir = os.path.join(self.tmp, "hf")
        _build_hf(hdir)
        self.assertEqual(type(detect(hdir)).__name__, "HFSource")


class TestToolRoutingHF(unittest.TestCase):
    """Each weight tool must source tensors via detect().iter_weights and stamp the
    Source.metadata()['source'] block onto its report entry. Runs LOCALLY on the
    HF fixture (numpy only — no gguf package needed)."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp = self._td.name
        self.hdir = os.path.join(self.tmp, "hf")
        _build_hf(self.hdir)

    def test_weight_stats_routes_through_modelsource(self):
        # Pass an explicit detected Source — the CLI path. (Other tool tests
        # permanently monkeypatch the module-level iter_weights/load_weight, so
        # exercising the source= path keeps this test order-independent.)
        import weight_stats as ws
        d = ws.analyze_model(self.hdir, source=detect(self.hdir))
        # archmap normalization produced canonical block cells from HF names.
        self.assertIn("attn_q", d["cells"])
        self.assertEqual(d["layers"], 2)
        # honest precision label from Source.metadata()['source'].
        self.assertIsNotNone(d.get("source"))
        self.assertEqual(d["source"]["format"], "safetensors")
        self.assertEqual(d["source"]["precision"], "exact")

    def test_spectral_routes_through_modelsource(self):
        import spectral as sp
        d = sp.analyze_model(self.hdir, source=detect(self.hdir))
        self.assertIn("attn_q", d["cells"])
        self.assertEqual(d["source"]["format"], "safetensors")

    def test_embedding_routes_through_modelsource(self):
        import embedding_geometry as eg
        d = eg.analyze_model(self.hdir, sample=8, source=detect(self.hdir))
        self.assertEqual(d["vocab"], 8)
        self.assertEqual(d["dim"], 4)
        self.assertEqual(d["source"]["format"], "safetensors")

    def test_model_diff_routes_through_modelsource(self):
        import model_diff as md
        d = md.diff_models(self.hdir, self.hdir, "X", "Y",
                           source_a=detect(self.hdir), source_b=detect(self.hdir))
        # self-diff: every matched cell delta 0.
        self.assertGreater(d["matched"], 0)
        for layers in d["cells"].values():
            for cell in layers.values():
                self.assertEqual(cell["delta"], 0.0)
        # diff carries BOTH source blocks.
        self.assertEqual(d["source_a"]["format"], "safetensors")
        self.assertEqual(d["source_b"]["format"], "safetensors")

    def test_unknown_path_degrades(self):
        # detect() returns None for an unknown path; the tool's own iter_weights
        # wrapper then yields nothing rather than crashing. Asserted directly on
        # the wrapper so a sibling test's monkeypatch can't mask it.
        import importlib
        import weight_stats as ws
        importlib.reload(ws)             # restore the real iter_weights wrapper
        unknown = os.path.join(self.tmp, "nope.xyz")
        self.assertIsNone(detect(unknown))
        self.assertEqual(list(ws.iter_weights(unknown)), [])


@unittest.skipUnless(HAVE_GGUF, "gguf package not installed (runs on host only)")
class TestStatsParity(unittest.TestCase):
    """Build the SAME numeric F32 tensors both ways, run weight_stats grouping on
    each, and assert identical roles/layers + per-cell stats. Needs gguf to write
    a real numeric GGUF and to dequantize it back."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp = self._td.name

    def _write_real_gguf(self, path):
        """Write a numeric F32 GGUF with the canonical PAIRS tensors using the
        gguf package (only available on the host)."""
        import gguf as gg
        w = gg.GGUFWriter(path, "llama")
        for cname, _hf, shape in PAIRS:
            arr = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
            w.add_tensor(cname, arr)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        return path

    def test_weight_stats_structure_and_values_match(self):
        import weight_stats as ws

        gpath = os.path.join(self.tmp, "m.gguf")
        self._write_real_gguf(gpath)
        hdir = os.path.join(self.tmp, "hf")
        _build_hf(hdir)

        g = ws.analyze_model(gpath)
        h = ws.analyze_model(hdir)

        self.assertEqual(set(g["roles"]), set(h["roles"]))
        self.assertEqual(g["layers"], h["layers"])
        self.assertEqual(set(g["cells"]), set(h["cells"]))
        for role, layers in g["cells"].items():
            self.assertEqual(set(layers), set(h["cells"][role]))
            for layer, cell in layers.items():
                hcell = h["cells"][role][layer]
                for metric, val in cell.items():
                    self.assertAlmostEqual(
                        val, hcell[metric], places=4,
                        msg=f"{role}[{layer}].{metric}")


if __name__ == "__main__":
    unittest.main()
