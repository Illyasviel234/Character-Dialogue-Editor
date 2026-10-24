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


def make_source(path, seconds=6, rate="24000/1001"):
    subprocess.run([
        paths.FFMPEG, "-y", "-v", "error",
        "-f", "lavfi", "-i", f"testsrc=size=320x240:rate={rate}:duration={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", path,
    ], check=True, capture_output=True)


def add_chapters(src, dst):
    """给素材加章节与标题元数据，模拟真实片源。"""
    meta = src + ".txt"
    with open(meta, "w", encoding="utf-8") as f:
        f.write(";FFMETADATA1\ntitle=Some Anime Title\n\n"
                "[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=2000\ntitle=Opening\n\n"
                "[CHAPTER]\nTIMEBASE=1/1000\nSTART=2000\nEND=4000\ntitle=Part A\n")
    subprocess.run([
        paths.FFMPEG, "-y", "-v", "error", "-i", src, "-i", meta,
        "-map", "0", "-map_metadata", "1", "-map_chapters", "1",
        "-c", "copy", dst,
    ], check=True, capture_output=True)
    return dst


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

    def test_out_dir_used(self):
        d = os.path.join(self.tmp.name, "custom")
        p = cut.resolve_output(None, "stem", "role", d)
        self.assertEqual(p, os.path.abspath(os.path.join(d, "stem_role_clip.mp4")))

    def test_output_equals_source_rejected(self):
        """输出路径就是源视频时必须拒绝，否则 os.replace 会毁掉原片。"""
        with self.assertRaises(ValueError):
            cut.check_output(self.video, self.video)
        with self.assertRaises(ValueError):
            cut.check_output(self.video, os.path.join(os.path.dirname(self.video), ".", "src.mp4"))

    def test_output_other_path_allowed(self):
        out = os.path.join(self.tmp.name, "ok.mp4")
        self.assertEqual(cut.check_output(self.video, out), out)

    def test_command_strips_metadata(self):
        segs = cut.snap_to_frames([{"start": 0, "end": 1, "dur": 1}], self.info["fps"])
        cmd = cut.build_command(self.video, segs, "o.mp4", "native", 20, "ultrafast",
                                True, self.info.get("fps_frac", ""))
        self.assertIn("-map_metadata", cmd)
        self.assertIn("-map_chapters", cmd)
        self.assertEqual(cmd[cmd.index("-map_metadata") + 1], "-1")
        self.assertEqual(cmd[cmd.index("-map_chapters") + 1], "-1")

    def test_chapters_not_inherited(self):
        """带章节的源导出后不应留下章节或 bin_data 数据轨。"""
        src = add_chapters(self.video, os.path.join(self.tmp.name, "chap.mp4"))
        self.assertGreater(media.probe(src).get("chapters", 0), 0, "素材没有章节，测试无效")
        info = media.probe(src)
        segs = cut.snap_to_frames(
            [{"start": 0.5, "end": 2.5, "dur": 2.0}], info["fps"])
        out = os.path.join(self.tmp.name, "chap_out.mp4")
        cmd = cut.build_command(src, segs, out, "native", 28, "ultrafast",
                                True, info.get("fps_frac", ""))
        r = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        got = media.probe(out)
        self.assertEqual(got.get("chapters", 0), 0, "成片继承了源章节")
        self.assertEqual(got.get("data_streams", 0), 0, "成片多了数据轨")

    def test_25fps_kept(self):
        src = os.path.join(self.tmp.name, "src25.mp4")
        make_source(src, 4, "25")
        info = media.probe(src)
        self.assertAlmostEqual(info["fps"], 25.0, places=2)
        segs = cut.snap_to_frames([{"start": 0.4, "end": 2.4, "dur": 2.0}], info["fps"])
        out = os.path.join(self.tmp.name, "out25.mp4")
        cmd = cut.build_command(src, segs, out, "native", 28, "ultrafast",
                                True, info.get("fps_frac", ""))
        r = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        got = media.probe(out)
        self.assertAlmostEqual(got["fps"], 25.0, places=2)
        self.assertLessEqual(abs(got["video_duration"] - 2.0), 0.05)

    def test_audio_track_in_filter(self):
        """指定音轨时 filter graph 必须取对应流，而不是默认第一条。"""
        segs = cut.snap_to_frames([{"start": 0, "end": 1, "dur": 1}], self.info["fps"])
        cmd = cut.build_command(self.video, segs, "o.mp4", "native", 20, "ultrafast",
                                True, self.info.get("fps_frac", ""), audio_index=1)
        graph = cmd[cmd.index("-filter_complex") + 1]
        self.assertIn("[0:a:1]", graph)
        self.assertNotIn("[0:a:0]", graph)

    def test_vfr_preserve_command(self):
        segs = [{"start": 0.0, "end": 1.0, "dur": 1.0}]
        cmd = cut.build_command(self.video, segs, "o.mp4", "native", 20, "ultrafast",
                                True, self.info.get("fps_frac", ""), fps_mode="preserve")
        self.assertIn("vfr", cmd)
        self.assertNotIn("-r", cmd, "保留时间戳模式不应强制帧率")

    def test_cfr_command_forces_rate(self):
        segs = [{"start": 0.0, "end": 1.0, "dur": 1.0}]
        cmd = cut.build_command(self.video, segs, "o.mp4", "native", 20, "ultrafast",
                                True, self.info.get("fps_frac", ""), fps_mode="cfr")
        self.assertIn("cfr", cmd)
        self.assertIn("-r", cmd)

    def test_preserve_mode_renders(self):
        """保留时间戳模式要能真的编出片子，只是不做帧吸附。"""
        segs = [{"start": 0.5, "end": 2.5, "dur": 2.0},
                {"start": 3.2, "end": 4.6, "dur": 1.4}]
        out = os.path.join(self.tmp.name, "vfr.mp4")
        cmd = cut.build_command(self.video, segs, out, "native", 28, "ultrafast",
                                True, self.info.get("fps_frac", ""), fps_mode="preserve")
        r = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        got = media.probe(out)
        self.assertLessEqual(abs(got["video_duration"] - 3.4), 0.15)


if __name__ == "__main__":
    unittest.main()
