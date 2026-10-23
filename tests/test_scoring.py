import unittest

import numpy as np

from core.scoring import keyword_seeds, robust_template, template_scores


def unit(v):
    v = np.array(v, dtype=np.float64)
    return v / np.linalg.norm(v)


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.emb = np.stack([unit([1, 0, 0]), unit([0.98, 0.05, 0]), unit([0, 1, 0])])
        self.ids = [0, 1, 2]

    def test_empty_seeds_returns_zero(self):
        s = template_scores(self.emb, self.ids, [])
        self.assertTrue(np.allclose(s, 0))

    def test_single_seed(self):
        s = template_scores(self.emb, self.ids, [0])
        self.assertAlmostEqual(float(s[0]), 1.0, places=5)
        self.assertAlmostEqual(float(s[2]), 0.0, places=5)

    def test_outlier_seed_rejected(self):
        t = robust_template(self.emb, [0, 1, 2])
        self.assertIsNotNone(t)
        self.assertGreater(float(t @ unit([1, 0, 0])), 0.97)

    def test_negative_ids_suppress(self):
        plain = template_scores(self.emb, self.ids, [0, 1])
        suppressed = template_scores(self.emb, self.ids, [0, 1], neg_ids=[2], margin=0.5)
        self.assertLess(float(suppressed[2]), float(plain[2]))

    def test_keyword_seeds(self):
        segs = [{"id": 0, "text": "そうかしら"}, {"id": 1, "text": "違う"}]
        self.assertEqual(keyword_seeds(segs, ["かしら"]), [0])
        self.assertEqual(keyword_seeds(segs, []), [])
        self.assertEqual(keyword_seeds(segs, ["", " "]), [])


if __name__ == "__main__":
    unittest.main()
