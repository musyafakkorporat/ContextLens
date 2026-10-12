import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from text_quality import is_repetitive_text


class TestRepetitiveText(unittest.TestCase):

    def test_5000_repeated_characters(self):
        self.assertTrue(is_repetitive_text("a" * 5000))

    def test_repeated_word(self):
        self.assertTrue(is_repetitive_text("hoaks " * 20))

    def test_normal_beras_sentence(self):
        text = (
            "Harga beras di pasar tradisional meningkat "
            "karena pasokan dari petani berkurang."
        )
        self.assertFalse(is_repetitive_text(text))

    def test_short_normal_text(self):
        self.assertFalse(is_repetitive_text("Halo, apa kabar?"))

    def test_empty_text(self):
        self.assertFalse(is_repetitive_text(""))

    def test_whitespace_only(self):
        self.assertFalse(is_repetitive_text("   "))

    def test_punctuation_only(self):
        self.assertFalse(is_repetitive_text("!!!! ?????"))


if __name__ == "__main__":
    unittest.main()
