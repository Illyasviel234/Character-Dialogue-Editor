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


class TestVfr(unittest.TestCase):
    """帧率判定：恒定帧率的时基舍入不能被当成可变帧率。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cfr = os.path.join(cls.tmp.name, "cfr.mp4")
        r = subprocess.run([
            paths.FFMPEG, "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=size=160x120:rate=24000/1001:duration=3",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-video_track_timescale", "1000",
            cls.cfr,
        ], capture_output=True)
        cls.ok = r.returncode == 0
        cls.vfr = cls._make_vfr()

    @classmethod
    def _make_vfr(cls):
        """把 30 fps 和 15 fps 两段直接拼接，得到真正的不均匀时间戳。"""
        if not cls.ok:
            return None
        a = os.path.join(cls.tmp.name, "a.mp4")
        b = os.path.join(cls.tmp.name, "b.mp4")
        for path, rate in ((a, "30"), (b, "15")):
            r = subprocess.run([
                paths.FFMPEG, "-y", "-v", "error",
                "-f", "lavfi", "-i", f"testsrc=size=160x120:rate={rate}:duration=1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-video_track_timescale", "1000", path,
            ], capture_output=True)
            if r.returncode != 0:
                return None
        lst = os.path.join(cls.tmp.name, "list.txt")
        # concat 列表里的 Windows 路径要转正斜杠，否则冒号和反斜杠会被当成转义
        with open(lst, "w", encoding="utf-8") as f:
            f.write(f"file '{a.replace(chr(92), '/')}'\n"
                    f"file '{b.replace(chr(92), '/')}'\n")
        out = os.path.join(cls.tmp.name, "vfr.mp4")
        # -safe 0：concat 默认只允许相对路径
        r = subprocess.run([paths.FFMPEG, "-y", "-v", "error",
                            "-f", "concat", "-safe", "0", "-i", lst,
                            "-c", "copy", out], capture_output=True)
        return out if r.returncode == 0 else None

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        if not self.ok:
            self.skipTest("ffmpeg 不可用")

    def test_cfr_not_flagged(self):
        verdict, off, n = media.vfr_check(self.cfr)
        self.assertEqual(verdict, "cfr", f"恒定帧率被误判（异常占比 {off:.3f}）")
        self.assertGreater(n, 10)
        self.assertFalse(media.probe(self.cfr, deep=True)["is_vfr"])

    def test_vfr_detected(self):
        if not self.vfr:
            self.skipTest("无法合成 VFR 素材")
        verdict, off, n = media.vfr_check(self.vfr)
        self.assertEqual(verdict, "vfr")
        self.assertGreater(off, 0.02)
        self.assertTrue(media.probe(self.vfr, deep=True)["is_vfr"])

    def test_unknown_when_too_few_frames(self):
        short = os.path.join(self.tmp.name, "short.mp4")
        r = subprocess.run([
            paths.FFMPEG, "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=0.2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", short,
        ], capture_output=True)
        if r.returncode != 0:
            self.skipTest("ffmpeg 不可用")
        verdict, _off, n = media.vfr_check(short)
        self.assertEqual(verdict, "unknown")
        self.assertLess(n, 10)

    def test_shallow_probe_has_no_verdict(self):
        info = media.probe(self.cfr)
        self.assertEqual(info["vfr_verdict"], "unknown", "浅探测不应读帧")


if __name__ == "__main__":
    unittest.main()
