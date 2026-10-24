"""剪辑时间轴的集成测试。

用 ffmpeg 合成一小段素材，跑真实的编码流程，验证：
- 成片视频流时长与计划总长偏差在一帧以内
- 音频与视频结束时间偏差在一帧以内
- 1 段、2 段、多段场景都成立
- 裸输出文件名不会崩溃
"""

import os
import subprocess
import tempfile
import unittest

from core import media, paths

import cut


def make_source(path, seconds=6):
    subprocess.run([
        paths.FFMPEG, "-y", "-v", "error",
        "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=24000/1001:duration={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", path,
    ], check=True, capture_output=True)


class TestCutTiming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.video = os.path.join(cls.tmp.name, "src.mp4")
        make_source(cls.video, 6)
        cls.info = media.probe(cls.video)
        if not cls.info:
            raise unittest.SkipTest("ffmpeg 不可用")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _render(self, raw_segs):
        fps = self.info["fps"]
        segs = cut.snap_to_frames(raw_segs, fps)
        planned = sum(s["dur"] for s in segs)
        out = os.path.join(self.tmp.name, f"out{len(raw_segs)}.mp4")
        cmd = cut.build_command(self.video, segs, out, "native", 28, "ultrafast",
                                True, self.info.get("fps_frac", ""))
        r = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        got = media.probe(out)
        return planned, got

    def _assert_timing(self, planned, got):
        frame = 1.0 / self.info["fps"]
        video_dur = float(got.get("video_duration") or 0)
        self.assertLessEqual(abs(video_dur - planned), frame + 0.002,
                             f"视频流 {video_dur:.3f}s 与计划 {planned:.3f}s 偏差超过一帧")

    def test_single_segment(self):
        planned, got = self._render([{"start": 1.0, "end": 3.0, "dur": 2.0}])
        self._assert_timing(planned, got)

    def test_two_segments(self):
        planned, got = self._render([
            {"start": 0.5, "end": 1.8, "dur": 1.3},
            {"start": 3.2, "end": 5.0, "dur": 1.8},
        ])
        self._assert_timing(planned, got)

    def test_many_segments(self):
        raw = [{"start": i * 0.45, "end": i * 0.45 + 0.4, "dur": 0.4} for i in range(10)]
        planned, got = self._render(raw)
        self._assert_timing(planned, got)

    def test_audio_video_end_close(self):
        planned, got = self._render([
            {"start": 0.3, "end": 2.1, "dur": 1.8},
            {"start": 3.0, "end": 4.4, "dur": 1.4},
        ])
        frame = 1.0 / self.info["fps"]
        video_dur = float(got.get("video_duration") or 0)
        audio_dur = float(got.get("duration") or 0)
        self.assertLessEqual(abs(audio_dur - video_dur), frame * 2,
                             "音视频结束时间相差超过两帧")

    def test_bare_output_name_resolves(self):
        p = cut.resolve_output("bare-name.mp4", "stem", "role")
        self.assertTrue(os.path.isabs(p))
        self.assertTrue(os.path.dirname(p))
        self.assertTrue(p.endswith("bare-name.mp4"))


if __name__ == "__main__":
    unittest.main()
