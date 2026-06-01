"""Tests für die reine bf16/fp16/fp32 -> fp32 Decode-Hilfe in hf_backend.

numpy hat keinen nativen bfloat16-Typ; bf16 wird per Hand dekodiert (uint16
nach uint32 verbreitern, 16 Bit nach links schieben, als float32 reinterpretieren).
Bekannte Werte werden exakt geprüft (0x3F80 -> 1.0). numpy ist hier installiert.
"""
import os
import sys
import unittest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

try:
    import numpy as np
    HAVE_NP = True
except Exception:
    HAVE_NP = False

from modelsource.hf_backend import bytes_to_fp32


@unittest.skipUnless(HAVE_NP, "numpy not installed")
class T(unittest.TestCase):
    def test_bf16_known(self):
        # bf16: 1.0=0x3F80, 2.0=0x4000, -2.0=0xC000
        raw = np.array([0x3F80, 0x4000, 0xC000], dtype="<u2").tobytes()
        out = bytes_to_fp32(raw, "BF16", (3,))
        np.testing.assert_allclose(out, [1.0, 2.0, -2.0])
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.shape, (3,))

    def test_f32_roundtrip(self):
        a = np.array([[1.5, -3.25], [0.0, 7.0]], dtype="<f4")
        out = bytes_to_fp32(a.tobytes(), "F32", (2, 2))
        np.testing.assert_array_equal(out, a)

    def test_f16(self):
        a = np.array([1.0, -0.5, 256.0], dtype="<f2")
        out = bytes_to_fp32(a.tobytes(), "F16", (3,))
        np.testing.assert_allclose(out, [1.0, -0.5, 256.0])
        self.assertEqual(out.dtype, np.float32)

    def test_unsupported_dtype_raises(self):
        with self.assertRaises(NotImplementedError):
            bytes_to_fp32(b"\x00", "F8_E4M3", (1,))
        with self.assertRaises(NotImplementedError):
            bytes_to_fp32(b"\x00", "I8", (1,))

    def test_bf16_reshape(self):
        raw = np.array([0x3F80, 0x3F80, 0x3F80, 0x3F80], dtype="<u2").tobytes()
        out = bytes_to_fp32(raw, "BF16", (2, 2))
        self.assertEqual(out.shape, (2, 2))
        np.testing.assert_allclose(out, np.ones((2, 2)))


if __name__ == "__main__":
    unittest.main()
