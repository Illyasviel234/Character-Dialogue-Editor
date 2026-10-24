"""媒体探测的失败路径。

无效文件曾经返回一堆零值字段，调用方以为探测成功，
直到后面才以"无音轨"之类的次级错误失败，排查方向被带偏。
"""

import os
import subprocess
import tempfile
import unittest

from core import media, paths


class TestProbeInvalid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def path(self, name, data=None):
        p = os.path.join(self.tmp.name, name)
        if data is not None:
            with open(p, "wb") as f:
                f.write(data)
        return p

    def test_missing_file(self):
        self.assertEqual(media.probe(self.path("nope.mp4")), {})

    def test_empty_path(self):
        self.assertEqual(media.probe(""), {})

    def test_not_media(self):
        p = self.path("fake.mp4", b"not media")
        self.assertEqual(media.probe(p), {})

    def test_audio_only(self):
        p = self.path("audio.m4a")
        r = subprocess.run([
            paths.FFMPEG, "-y", "-v", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-c:a", "aac", p,
        ], capture_output=True)
        if r.returncode != 0 or not os.path.exists(p):
            self.skipTest("ffmpeg 不可用")
        self.assertEqual(media.probe(p), {}, "纯音频文件不应被当成可用视频")

    def test_describe_failure(self):
        self.assertEqual(media.describe({}), "探测失败")


class TestProbeValid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.video = os.path.join(cls.tmp.name, "ok.mp4")
        r = subprocess.run([
            paths.FFMPEG, "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30:duration=3",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-shortest", cls.video,
        ], capture_output=True)
        cls.ok = r.returncode == 0

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        if not self.ok:
            self.skipTest("ffmpeg 不可用")

    def test_core_fields(self):
        info = media.probe(self.video)
        self.assertTrue(info)
        self.assertEqual((info["width"], info["height"]), (160, 120))
        self.assertAlmostEqual(info["fps"], 30.0, places=2)
        self.assertGreater(info["duration"], 2.5)
        self.assertEqual(info["audio_streams"], 1)
        self.assertEqual(len(info["audio_tracks"]), 1)
        self.assertFalse(info["is_vfr"])
        self.assertEqual(info["chapters"], 0)

    def test_describe_has_size_and_fps(self):
        text = media.describe(media.probe(self.video))
        self.assertIn("160x120", text)
        self.assertIn("30.000 fps", text)


if __name__ == "__main__":
    unittest.main()
