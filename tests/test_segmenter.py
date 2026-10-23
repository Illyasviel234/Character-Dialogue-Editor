import unittest

from core.segmenter import build_segments


def seg(i, start, end, text="台词"):
    return {"id": i, "start": start, "end": end, "text": text}


class TestBuildSegments(unittest.TestCase):
    def test_adjacent_merge(self):
        segs = [seg(0, 10.0, 12.0), seg(1, 12.5, 14.0)]
        out = build_segments([0, 1], segs, 100, merge_gap=2.0)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["ids"], [0, 1])
        self.assertAlmostEqual(out[0]["start"], 9.75, places=2)
        self.assertAlmostEqual(out[0]["end"], 14.30, places=2)

    def test_gap_too_large_splits(self):
        segs = [seg(0, 10.0, 12.0), seg(1, 16.0, 18.0)]
        out = build_segments([0, 1], segs, 100, merge_gap=2.0)
        self.assertEqual(len(out), 2)

    def test_other_speaker_blocks_merge(self):
        segs = [seg(0, 10.0, 12.0), seg(9, 12.2, 12.4), seg(1, 12.5, 14.0)]
        out = build_segments([0, 1], segs, 100, merge_gap=2.0)
        self.assertEqual(len(out), 2)

    def test_padding_avoids_neighbour(self):
        segs = [seg(0, 10.0, 12.0), seg(9, 12.05, 13.0)]
        out = build_segments([0], segs, 100, pad_tail=0.30)
        self.assertLessEqual(out[0]["end"], 12.05 - 0.05)

    def test_missing_id_ignored(self):
        segs = [seg(0, 10.0, 12.0)]
        out = build_segments([0, 99], segs, 100)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["ids"], [0])

    def test_duplicate_id_deduped(self):
        segs = [seg(0, 10.0, 12.0)]
        out = build_segments([0, 0, 0], segs, 100)
        self.assertEqual(len(out), 1)

    def test_empty_inputs(self):
        self.assertEqual(build_segments([], [], 100), [])
        self.assertEqual(build_segments([0], [], 100), [])
        self.assertEqual(build_segments([], [seg(0, 1, 2)], 100), [])

    def test_clamped_to_duration(self):
        segs = [seg(0, 98.0, 100.0)]
        out = build_segments([0], segs, 100, pad_tail=5.0)
        self.assertLessEqual(out[0]["end"], 100)

    def test_negative_duration_filtered(self):
        segs = [seg(0, 5.0, 3.0), seg(1, 10.0, 12.0)]
        out = build_segments([0, 1], segs, 100)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["ids"], [1])

    def test_zero_duration_falls_back_to_total(self):
        segs = [seg(0, 10.0, 12.0)]
        out = build_segments([0], segs, 0)
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
