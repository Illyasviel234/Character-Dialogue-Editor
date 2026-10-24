"""试听用的音频切片与取帧。

这两步在界面里每次选行都会触发，出错就会让试听功能整个失效，
所以单独测：区间越界要夹住、非法区间要报错、取帧要真的生成文件。
"""

import os
import subprocess
import tempfile
import unittest
import wave

import numpy as np

from core import audio, paths


def make_wav(path, seconds=3.0, rate=16000):
    n = int(seconds * rate)
    t = np.arange(n) / rate
    data = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data.tobytes())
    return path


def wav_seconds(path):
    with wave.open(path, "rb") as r:
        return r.getnframes() / r.getframerate()


class TestSliceWav(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src = make_wav(os.path.join(self.tmp.name, "a.wav"), 3.0)

    def dst(self, name="o.wav"):
        return os.path.join(self.tmp.name, name)

    def test_basic(self):
        out, dur = audio.slice_wav(self.src, 1.0, 2.0, self.dst())
        self.assertTrue(os.path.exists(out))
        self.assertAlmostEqual(dur, 1.0, delta=0.01)
        self.assertAlmostEqual(wav_seconds(out), 1.0, delta=0.01)

    def test_padding(self):
        out, dur = audio.slice_wav(self.src, 1.0, 2.0, self.dst(), head=0.2, tail=0.3)
        self.assertAlmostEqual(dur, 1.5, delta=0.02)

    def test_clamped_at_edges(self):
        """头部留白越过 0、尾部留白越过文件结尾都要夹住，不能读到越界。"""
        out, dur = audio.slice_wav(self.src, 0.05, 2.95, self.dst(), head=0.5, tail=0.5)
        self.assertAlmostEqual(dur, 3.0, delta=0.02)
        self.assertLessEqual(wav_seconds(out), 3.0)

    def test_empty_range_rejected(self):
        with self.assertRaises(ValueError):
            audio.slice_wav(self.src, 2.0, 2.0, self.dst())

    def test_missing_source(self):
        with self.assertRaises(Exception):
            audio.slice_wav(os.path.join(self.tmp.name, "nope.wav"), 0, 1, self.dst())


class TestVideoHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.video = os.path.join(cls.tmp.name, "v.mp4")
        r = subprocess.run([
            paths.FFMPEG, "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=3",
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

    def test_grab_frame(self):
        out = audio.grab_frame(self.video, 1.0, os.path.join(self.tmp.name, "f.png"), 160)
        self.assertIsNotNone(out)
        self.assertGreater(os.path.getsize(out), 500)

    def test_grab_frame_beyond_end(self):
        out = audio.grab_frame(self.video, 999.0, os.path.join(self.tmp.name, "g.png"), 160)
        self.assertIsNone(out, "越界取帧应返回 None 而不是半个文件")

    def test_extract_wav(self):
        out = audio.extract_wav(self.video, 0.5, 1.5, os.path.join(self.tmp.name, "s.wav"))
        self.assertTrue(os.path.exists(out))
        self.assertAlmostEqual(wav_seconds(out), 1.0, delta=0.15)


if __name__ == "__main__":
    unittest.main()
