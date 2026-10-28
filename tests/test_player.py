"""试听播放器的状态同步。

winsound 没有完成回调，只能按时长挂定时器。这里用假的 winsound 验证：
非循环播完状态归零、循环模式不会自己停、stop() 能让状态立刻归零。
"""

import os
import sys
import tempfile
import unittest
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gui"))

from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer  # noqa: E402

import audition  # noqa: E402


def make_wav(path, seconds=0.2, rate=16000):
    n = int(seconds * rate)
    data = (np.sin(np.arange(n) / 20.0) * 4000).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data.tobytes())
    return path


class FakeWinSound:
    SND_FILENAME = 1
    SND_ASYNC = 2
    SND_LOOP = 4

    def __init__(self):
        self.calls = []

    def PlaySound(self, path, flags):
        self.calls.append((path, flags))


def spin(ms):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


class TestWavSeconds(unittest.TestCase):
    def test_reads_duration(self):
        with tempfile.TemporaryDirectory() as d:
            p = make_wav(os.path.join(d, "a.wav"), 0.5)
            self.assertAlmostEqual(audition.wav_seconds(p), 0.5, delta=0.02)

    def test_bad_file(self):
        self.assertEqual(audition.wav_seconds("no-such.wav"), 0.0)


class TestPlayerState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])
        cls.tmp = tempfile.TemporaryDirectory()
        cls.wav = make_wav(os.path.join(cls.tmp.name, "a.wav"), 0.2)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self._orig = audition.winsound
        audition.winsound = FakeWinSound()
        self.addCleanup(self._restore)

    def _restore(self):
        audition.winsound = self._orig

    def test_natural_end_clears_state(self):
        p = audition.Player()
        done = []
        p.finished.connect(lambda: done.append(True))
        self.assertTrue(p.play(self.wav, loop=False))
        self.assertTrue(p.playing)
        spin(600)
        self.assertFalse(p.playing, "自然播完后 playing 必须归零")
        self.assertEqual(len(done), 1)

    def test_loop_keeps_playing(self):
        p = audition.Player()
        done = []
        p.finished.connect(lambda: done.append(True))
        self.assertTrue(p.play(self.wav, loop=True))
        spin(500)
        self.assertTrue(p.playing, "循环模式不该自己停")
        self.assertEqual(done, [])
        p.stop()
        self.assertFalse(p.playing)

    def test_stop_clears_immediately(self):
        p = audition.Player()
        p.play(self.wav, loop=False)
        p.stop()
        self.assertFalse(p.playing)
        spin(500)
        self.assertFalse(p.playing)

    def test_replay_after_natural_end(self):
        """播完再按一次要能直接重播，而不是先被陈旧状态吃掉一次。"""
        p = audition.Player()
        p.play(self.wav, loop=False)
        spin(600)
        self.assertFalse(p.playing)
        self.assertTrue(p.play(self.wav, loop=False))
        self.assertTrue(p.playing)
        p.stop()

    def test_missing_winsound_reports(self):
        audition.winsound = None
        p = audition.Player()
        errors = []
        p.failed.connect(errors.append)
        self.assertFalse(p.play(self.wav))
        self.assertEqual(len(errors), 1)


if __name__ == "__main__":
    unittest.main()
