"""批量队列：条目参数、输出路径、步骤展开与持久化。

队列的步骤展开决定了批量跑会不会用错音轨或漏掉分析，
所以按「有分析结果」和「没有分析结果」两种情况分别验证。
"""

import json
import os
import tempfile
import unittest

import numpy as np

from core import manifest, paths, queue


def make_video(directory, name="v.mp4", data=b"v" * 300):
    path = os.path.join(directory, name)
    with open(path, "wb") as f:
        f.write(data)
    return path


def complete_run(video, track=0):
    """造一个完整的分析运行，返回 sid。"""
    src = {"size": os.path.getsize(video), "duration": 10.0,
           "fingerprint": manifest.fingerprint(video)}
    sid = manifest.source_id(video)
    os.makedirs(manifest.job_dir(sid), exist_ok=True)
    rid = manifest.new_run(sid, src, audio_track=track)
    segs = [{"id": i, "start": i, "end": i + 1, "text": f"t{i}"} for i in (0, 1)]
    manifest.atomic_write_json(
        manifest.run_artifact(sid, rid, manifest.TRANSCRIPT),
        {"source_id": sid, "run_id": rid, "segments": segs, "duration": 2})
    np.savez(manifest.run_artifact(sid, rid, manifest.EMBEDDINGS),
             emb=np.ones((2, 192), dtype=np.float32), ids=np.array([0, 1]))
    wav = manifest.run_artifact(sid, rid, manifest.AUDIO_WAV)
    with open(wav, "wb") as f:
        f.write(b"RIFF" + b"\0" * 64)
    manifest.save_audio_info(sid, rid, sid, track, 2.0, 1)
    ok, why = manifest.finalize_run(sid, rid)
    assert ok, why
    return sid


class QueueCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._orig = (paths.JOBS, paths.CACHE, paths.OUTPUT, paths.WORK)
        paths.JOBS = os.path.join(self.tmp.name, "jobs")
        paths.CACHE = os.path.join(self.tmp.name, "cache")
        paths.OUTPUT = os.path.join(self.tmp.name, "out")
        paths.WORK = self.tmp.name
        for d in (paths.JOBS, paths.OUTPUT):
            os.makedirs(d, exist_ok=True)
        self.addCleanup(self._restore)

    def _restore(self):
        paths.JOBS, paths.CACHE, paths.OUTPUT, paths.WORK = self._orig


class TestItem(QueueCase):
    def test_defaults(self):
        item = queue.make_item(make_video(self.tmp.name))
        self.assertEqual(item["status"], queue.WAITING)
        self.assertEqual(item["role"], "目标角色")
        self.assertAlmostEqual(item["threshold"], 0.66)
        self.assertEqual(item["audio_track"], 0)

    def test_video_made_absolute(self):
        rel = os.path.relpath(make_video(self.tmp.name))
        self.assertTrue(os.path.isabs(queue.make_item(rel)["video"]))

    def test_output_path_uses_role_and_dir(self):
        item = queue.make_item(make_video(self.tmp.name, "re0_08.mp4"),
                               role="碧翠丝", out_dir=os.path.join(self.tmp.name, "x"))
        path = queue.output_path(item)
        self.assertEqual(os.path.basename(path), "re0_08_碧翠丝_clip.mp4")
        self.assertTrue(path.startswith(os.path.join(self.tmp.name, "x")))

    def test_output_path_sanitizes_role(self):
        item = queue.make_item(make_video(self.tmp.name), role='a:b/c')
        self.assertEqual(os.path.basename(queue.output_path(item)), "v_a_b_c_clip.mp4")

    def test_output_path_defaults_to_output_dir(self):
        item = queue.make_item(make_video(self.tmp.name), role="R", out_dir="")
        self.assertTrue(queue.output_path(item).startswith(paths.OUTPUT))


class TestSteps(QueueCase):
    def test_without_analysis_runs_full_pipeline(self):
        item = queue.make_item(make_video(self.tmp.name), role="R")
        scripts = [s[0] for s in queue.build_steps(item)]
        self.assertEqual(scripts, ["extract_audio.py", "transcribe.py", "voiceprint.py",
                                   "select.py", "cut.py", "gen_report.py"])
        need, _why = queue.analysis_state(item)
        self.assertTrue(need)

    def test_with_analysis_skips_pipeline(self):
        video = make_video(self.tmp.name)
        complete_run(video)
        item = queue.make_item(video, role="R", audio_track=0)
        scripts = [s[0] for s in queue.build_steps(item)]
        self.assertEqual(scripts, ["select.py", "cut.py", "gen_report.py"])
        need, why = queue.analysis_state(item)
        self.assertFalse(need)
        self.assertIn("复用", why)

    def test_track_change_forces_analysis(self):
        video = make_video(self.tmp.name)
        complete_run(video, track=0)
        item = queue.make_item(video, role="R", audio_track=1)
        need, why = queue.analysis_state(item)
        self.assertTrue(need)
        self.assertIn("音轨", why)
        scripts = [s[0] for s in queue.build_steps(item)]
        self.assertEqual(scripts[0], "extract_audio.py")
        extract = next(s for s in queue.build_steps(item) if s[0] == "extract_audio.py")
        self.assertIn("1", extract[1])

    def test_missing_video_needs_analysis(self):
        item = queue.make_item(os.path.join(self.tmp.name, "nope.mp4"))
        need, why = queue.analysis_state(item)
        self.assertTrue(need)
        self.assertIn("不存在", why)

    def test_weights_sum_to_100(self):
        for item in (queue.make_item(make_video(self.tmp.name)),
                     queue.make_item(make_video(self.tmp.name), role="R")):
            total = sum(s[2] for s in queue.build_steps(item))
            self.assertEqual(total, 100)

    def test_cut_gets_explicit_output(self):
        video = make_video(self.tmp.name)
        item = queue.make_item(video, role="R")
        cut = next(s for s in queue.build_steps(item) if s[0] == "cut.py")
        self.assertIn("--output", cut[1])
        self.assertEqual(cut[1][cut[1].index("--output") + 1], queue.output_path(item))


class TestPersistence(QueueCase):
    def test_roundtrip(self):
        items = [queue.make_item(make_video(self.tmp.name, "a.mp4"), role="甲"),
                 queue.make_item(make_video(self.tmp.name, "b.mp4"), role="乙",
                                 threshold=0.7, audio_track=1)]
        items[0]["status"] = queue.DONE
        queue.save(items)
        back = queue.load()
        self.assertEqual(len(back), 2)
        self.assertEqual(back[0]["role"], "甲")
        self.assertEqual(back[0]["status"], queue.DONE)
        self.assertAlmostEqual(back[1]["threshold"], 0.7)
        self.assertEqual(back[1]["audio_track"], 1)

    def test_running_becomes_waiting(self):
        """上次被强杀时留下的「进行中」不能一直卡着。"""
        items = [queue.make_item(make_video(self.tmp.name))]
        items[0]["status"] = queue.RUNNING
        queue.save(items)
        self.assertEqual(queue.load()[0]["status"], queue.WAITING)

    def test_broken_file_gives_empty(self):
        with open(queue.queue_path(), "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertEqual(queue.load(), [])

    def test_missing_file_gives_empty(self):
        self.assertEqual(queue.load(), [])

    def test_skips_items_without_video(self):
        with open(queue.queue_path(), "w", encoding="utf-8") as f:
            json.dump({"items": [{"role": "x"}, {"video": "a.mp4"}]}, f)
        self.assertEqual(len(queue.load()), 1)


if __name__ == "__main__":
    unittest.main()
