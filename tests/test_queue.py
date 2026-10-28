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

from tests.helpers import write_silence, write_text, write_tiny_video


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
    write_silence(wav, seconds=0.2)
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

    def test_output_path_uses_role_dir_and_hash(self):
        item = queue.make_item(make_video(self.tmp.name, "re0_08.mp4"),
                               role="碧翠丝", out_dir=os.path.join(self.tmp.name, "x"))
        path = queue.output_path(item)
        name = os.path.basename(path)
        self.assertTrue(name.startswith("re0_08_碧翠丝_"), name)
        self.assertTrue(name.endswith("_clip.mp4"), name)
        self.assertEqual(len(name.split("_")[-2]), 8, "应有 8 位源指纹摘要")
        self.assertTrue(path.startswith(os.path.join(self.tmp.name, "x")))

    def test_output_path_sanitizes_role(self):
        item = queue.make_item(make_video(self.tmp.name), role='a:b/c')
        self.assertTrue(os.path.basename(queue.output_path(item)).startswith("v_a_b_c_"))

    def test_same_name_different_dirs_do_not_collide(self):
        """两个不同目录下的 same.mp4，角色相同时不能算出同一个输出。"""
        a = os.path.join(self.tmp.name, "s1", "same.mp4")
        b = os.path.join(self.tmp.name, "s2", "same.mp4")
        for path, data in ((a, b"aaa" * 100), (b, b"bbb" * 100)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
        one = queue.make_item(a, role="R", out_dir=self.tmp.name)
        two = queue.make_item(b, role="R", out_dir=self.tmp.name)
        self.assertNotEqual(queue.output_path(one), queue.output_path(two))
        self.assertEqual(queue.conflicts([one, two]), [])

    def test_conflicts_detected(self):
        video = make_video(self.tmp.name)
        one = queue.make_item(video, role="R", out_dir=self.tmp.name)
        two = queue.make_item(video, role="R", out_dir=self.tmp.name)
        bad = queue.conflicts([one, two])
        self.assertTrue(bad)
        path, owners = bad[0]
        self.assertTrue(path.endswith("_clip.mp4"))
        self.assertEqual(len(owners), 2)

    def test_conflicts_cover_reports(self):
        """清单文件名也要参与冲突检测，否则 HTML/CSV 会互相覆盖。"""
        video = make_video(self.tmp.name)
        one = queue.make_item(video, role="R", out_dir=self.tmp.name)
        paths = queue.targets(one)
        self.assertTrue(paths["csv"].endswith("_清单.csv"))
        self.assertTrue(paths["html"].endswith("_清单.html"))
        self.assertEqual(len({os.path.normcase(p) for p in paths.values()}), 3)

    def test_apply_out_dir_skips_done(self):
        a = queue.make_item(make_video(self.tmp.name, "a.mp4"), role="甲")
        b = queue.make_item(make_video(self.tmp.name, "b.mp4"), role="乙")
        a["status"] = queue.DONE
        new_dir = os.path.join(self.tmp.name, "new")
        changed = queue.apply_out_dir([a, b], new_dir)
        self.assertEqual(changed, 1)
        self.assertEqual(b["out_dir"], os.path.abspath(new_dir))
        self.assertNotEqual(a["out_dir"], os.path.abspath(new_dir))

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

    def test_ids_are_stable(self):
        items = [queue.make_item(make_video(self.tmp.name, "a.mp4")),
                 queue.make_item(make_video(self.tmp.name, "b.mp4"))]
        ids = [i["id"] for i in items]
        self.assertEqual(len(set(ids)), 2, "id 必须唯一")
        queue.save(items)
        back = queue.load()
        self.assertEqual([i["id"] for i in back], ids, "保存后 id 不能变")

    def test_legacy_item_without_id_gets_one(self):
        with open(queue.queue_path(), "w", encoding="utf-8") as f:
            json.dump({"items": [{"video": "a.mp4", "role": "x"}]}, f)
        back = queue.load()
        self.assertTrue(back[0].get("id"))

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


class TestSourceIdentity(QueueCase):
    """入队时冻结源身份，替换文件后不能静默处理新内容。"""

    def _item(self, name="ep.mp4", data=b"v" * 300, role="R"):
        return queue.make_item(make_video(self.tmp.name, name, data), role=role)

    def test_identity_frozen_at_enqueue(self):
        item = self._item()
        src = item["source"]
        self.assertTrue(src.get("id"))
        self.assertEqual(len(src["fingerprint"]), 16)
        self.assertEqual(src["size"], 300)
        self.assertTrue(src["mtime"])

    def test_output_name_uses_frozen_fingerprint(self):
        item = self._item()
        before = queue.base_name(item)
        # 同路径换成不同内容
        with open(item["video"], "wb") as f:
            f.write(b"w" * 300)
        self.assertEqual(queue.base_name(item), before,
                         "输出名不能跟着当前文件变，否则用户看不出处理的是哪份素材")

    def test_detects_replacement(self):
        """同大小、同秒级 mtime 的内容替换只能靠深度检查发现。"""
        item = self._item()
        with open(item["video"], "wb") as f:
            f.write(b"w" * 300)
        state, why = queue.source_check(item, deep=True)
        self.assertEqual(state, "changed")
        self.assertIn("内容", why)

    def test_fast_check_detects_size_change(self):
        item = self._item()
        with open(item["video"], "wb") as f:
            f.write(b"w" * 400)
        self.assertEqual(queue.source_check(item)[0], "changed")

    def test_detects_missing(self):
        item = self._item()
        os.remove(item["video"])
        self.assertEqual(queue.source_check(item)[0], "missing")
        self.assertEqual(queue.source_check(item, deep=True)[0], "missing")

    def test_touch_only_passes_deep_check(self):
        """只改 mtime、内容没变：快速检查报警，深度检查放行。"""
        item = self._item()
        os.utime(item["video"], (1700000000, 1700000000))
        self.assertEqual(queue.source_check(item)[0], "changed")
        self.assertEqual(queue.source_check(item, deep=True)[0], "ok")

    def test_refresh_source_updates_identity(self):
        item = self._item()
        old = item["source"]["id"]
        with open(item["video"], "wb") as f:
            f.write(b"w" * 400)
        before, after = queue.refresh_source(item)
        self.assertEqual(before, old)
        self.assertNotEqual(after, old)
        self.assertEqual(queue.source_check(item, deep=True)[0], "ok")

    def test_old_item_without_identity(self):
        item = self._item()
        item["source"] = {}
        self.assertEqual(queue.source_check(item)[0], "unknown")


class TestArtifacts(QueueCase):
    """已完成条目的产物要能复核，缺了就退回等待。"""

    def _done_item(self):
        item = queue.make_item(make_video(self.tmp.name, "a.mp4"), role="R",
                               out_dir=self.tmp.name)
        paths = queue.targets(item)
        if not write_tiny_video(paths["video"]):
            self.skipTest("ffmpeg 不可用")
        write_text(paths["csv"],
                   "序号,原片起点,原片终点,时长(秒),相似度,台词\n"
                   "1,00:10.00,00:13.50,3.5,0.842,テスト\n")
        write_text(paths["html"], "<html>" + "x" * 200 + "</html>")
        item["status"] = queue.DONE
        return item, paths

    def test_artifact_state_complete(self):
        item, _ = self._done_item()
        self.assertEqual(queue.artifact_state(item), "complete")

    def test_artifact_state_partial(self):
        item, paths = self._done_item()
        os.remove(paths["html"])
        self.assertEqual(queue.artifact_state(item), "partial")

    def test_health_ok(self):
        item, _ = self._done_item()
        ok, why = queue.target_health(item)
        self.assertTrue(ok, why)

    def test_health_rejects_corrupt_clip(self):
        item, paths = self._done_item()
        with open(paths["video"], "wb") as f:
            f.write(b"not a video")
        ok, why = queue.target_health(item)
        self.assertFalse(ok)
        self.assertIn("播放", why)

    def test_audit_resets_missing_clip(self):
        item, paths = self._done_item()
        os.remove(paths["video"])
        fixed = queue.audit_done([item])
        self.assertEqual(len(fixed), 1)
        self.assertEqual(item["status"], queue.WAITING)
        self.assertIn("成片", item["message"])

    def test_audit_keeps_healthy(self):
        item, _ = self._done_item()
        self.assertEqual(queue.audit_done([item]), [])
        self.assertEqual(item["status"], queue.DONE)

    def test_existing_targets_lists_hits(self):
        item, paths = self._done_item()
        hits = queue.existing_targets([item])
        self.assertEqual(len(hits), 1)
        self.assertEqual(set(hits[0][1]), set(paths.values()))


class TestSettingsAndIds(QueueCase):
    def test_out_dir_persisted(self):
        items = [queue.make_item(make_video(self.tmp.name), role="R")]
        target = os.path.join(self.tmp.name, "out2")
        queue.save(items, out_dir=target)
        settings, back = queue.load_state()
        self.assertEqual(settings["out_dir"], target)
        self.assertEqual(len(back), 1)

    def test_duplicate_ids_regenerated(self):
        with open(queue.queue_path(), "w", encoding="utf-8") as f:
            json.dump({"items": [{"video": "a.mp4", "id": "same"},
                                 {"video": "b.mp4", "id": "same"}]}, f)
        items = queue.load()
        ids = [i["id"] for i in items]
        self.assertEqual(len(set(ids)), 2)
        self.assertNotIn("same", ids, "长度不对的 id 也要换掉")

    def test_save_fixes_duplicates(self):
        a = queue.make_item(make_video(self.tmp.name, "a.mp4"))
        b = queue.make_item(make_video(self.tmp.name, "b.mp4"))
        b["id"] = a["id"]
        queue.save([a, b])
        back = queue.load()
        self.assertEqual(len({i["id"] for i in back}), 2)

    def test_missing_source_record_survives_roundtrip(self):
        item = queue.make_item(make_video(self.tmp.name), role="R")
        queue.save([item])
        back = queue.load()[0]
        self.assertEqual(back["source"]["id"], item["source"]["id"])
