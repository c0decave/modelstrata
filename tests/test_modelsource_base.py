import os, sys, unittest
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

import abc
from modelsource.base import Source


class _Mini(Source):
    def metadata(self): return {}
    def iter_weights(self, want=None): return iter(())
    def tokenizer(self): return None


class T(unittest.TestCase):
    def test_abc_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            Source()

    def test_source_block(self):
        m = _Mini()
        sb = m.source_block(fmt="safetensors", precision="exact", arch="llama", mapped=True)
        self.assertEqual(sb, {"format": "safetensors", "precision": "exact",
                              "arch": "llama", "mapped": True, "warnings": []})

    def test_source_block_warnings_copied(self):
        m = _Mini()
        warns = ["w1"]
        sb = m.source_block(fmt="gguf", precision="approx", arch="qwen2",
                            mapped=True, warnings=warns)
        self.assertEqual(sb["warnings"], ["w1"])
        warns.append("w2")                       # caller's list mutated...
        self.assertEqual(sb["warnings"], ["w1"]) # ...must NOT leak into the block (it's copied)


if __name__ == "__main__":
    unittest.main()
