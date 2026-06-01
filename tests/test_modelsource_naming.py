"""Tests für tools/modelsource/naming.py — Role enum + canonical GGUF name renderer."""
import os, sys, unittest
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from modelsource.naming import Role, canonical

class T(unittest.TestCase):
    def test_layered(self):
        self.assertEqual(canonical(Role.ATTN_Q, 0), "blk.0.attn_q.weight")
        self.assertEqual(canonical(Role.FFN_DOWN, 12), "blk.12.ffn_down.weight")
        self.assertEqual(canonical(Role.ATTN_NORM, 3), "blk.3.attn_norm.weight")
    def test_global(self):
        self.assertEqual(canonical(Role.TOK_EMBD), "token_embd.weight")
        self.assertEqual(canonical(Role.OUTPUT), "output.weight")
        self.assertEqual(canonical(Role.OUTPUT_NORM), "output_norm.weight")
    def test_moe_exps(self):
        self.assertEqual(canonical(Role.FFN_DOWN_EXPS, 2), "blk.2.ffn_down_exps.weight")
    def test_layered_requires_layer(self):
        with self.assertRaises(ValueError):
            canonical(Role.ATTN_Q)            # no layer -> error, never a silent default
