import unittest

from core.naming import safe_filename, safe_stem


class TestSafeFilename(unittest.TestCase):
    def test_illegal_chars_replaced(self):
        self.assertEqual(safe_filename("a/b:c*d?e"), "a_b_c_d_e")

    def test_backslash_and_pipe(self):
        self.assertEqual(safe_filename("x\\y|z"), "x_y_z")

    def test_empty_falls_back(self):
        self.assertEqual(safe_filename(""), "output")
        self.assertEqual(safe_filename("   "), "output")
        self.assertEqual(safe_filename(None), "output")

    def test_reserved_name(self):
        self.assertTrue(safe_filename("CON").startswith("_"))

    def test_trailing_dots_stripped(self):
        self.assertEqual(safe_filename("name..."), "name")

    def test_length_capped(self):
        self.assertLessEqual(len(safe_filename("あ" * 300)), 80)

    def test_safe_stem(self):
        self.assertEqual(safe_stem(r"C:\dir\re0_1_08.mp4"), "re0_1_08")

    def test_safe_stem_with_illegal(self):
        self.assertEqual(safe_stem("a:b.mp4"), "a_b")


if __name__ == "__main__":
    unittest.main()
