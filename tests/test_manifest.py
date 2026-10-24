import json
import os
import tempfile
import unittest

import numpy as np

from core import manifest, paths


def write(path, data=b"x" * 100):
    with open(path, "wb") as f:
        f.write(data)
    return path


def src_with_fp(path):
    st = os.stat(path)
    return {"size": st.st_size, "duration": 10.0,
            "fingerprint": manifest.fingerprint(path)}


class ManifestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._orig_jobs = paths.JOBS
        self._orig_cache = paths.CACHE
        paths.JOBS = os.path.join(self.tmp.name, "jobs")
        paths.CACHE = os.path.join(self.tmp.name, "cache")
        os.makedirs(paths.JOBS, exist_ok=True)
        self.addCleanup(self._restore)

    def _restore(self):
        paths.JOBS = self._orig_jobs
        paths.CACHE = self._orig_cache

    def file(self, name, data):
        path = os.path.join(self.tmp.name, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return write(path, data)


class TestFingerprint(ManifestCase):
    def test_stable(self):
        p = self.file("a.mp4", b"abc" * 100)
        self.assertEqual(manifest.fingerprint(p), manifest.fingerprint(p))

    def test_differs_by_content(self):
        a = self.file("a.mp4", b"a" * 500)
        b = self.file("b.mp4", b"b" * 500)
        self.assertNotEqual(manifest.fingerprint(a), manifest.fingerprint(b))

    def test_same_meta_different_content_gives_different_id(self):
        """同名、同大小、同修改时间，内容不同必须得到不同任务 ID。"""
        a = self.file("same.mp4", b"a" * 400)
        b = self.file("other/same.mp4", b"b" * 400)
        os.utime(a, (1700000000, 1700000000))
        os.utime(b, (1700000000, 1700000000))
        self.assertEqual(os.path.getsize(a), os.path.getsize(b))
        self.assertNotEqual(manifest.source_id(a), manifest.source_id(b))

    def test_same_content_same_id(self):
        a = self.file("a.mp4", b"z" * 300)
        b = self.file("copy/a.mp4", b"z" * 300)
        self.assertEqual(manifest.source_id(a), manifest.source_id(b))

    def test_middle_change_detected(self):
        """头尾完全相同、只有中间 1 MiB 不同：采样指纹会撞，完整指纹不会。"""
        mib = 1024 * 1024
        head, tail = b"H" * mib, b"T" * mib
        a = self.file("a/same.mp4", head + b"M" * mib + tail)
        b = self.file("b/same.mp4", head + b"N" * mib + tail)
        os.utime(a, (1700000000, 1700000000))
        os.utime(b, (1700000000, 1700000000))
        self.assertEqual(os.path.getsize(a), os.path.getsize(b))
        self.assertNotEqual(manifest.fingerprint(a), manifest.fingerprint(b))
        self.assertNotEqual(manifest.source_id(a), manifest.source_id(b))

    def test_cache_invalidated_by_rewrite(self):
        p = self.file("c.mp4", b"1" * 500)
        first = manifest.fingerprint(p)
        write(p, b"2" * 500)
        self.assertNotEqual(manifest.fingerprint(p), first)

    def test_cache_bypass_matches(self):
        p = self.file("d.mp4", b"1" * 500)
        self.assertEqual(manifest.fingerprint(p), manifest.fingerprint(p, cache=False))


class TestMatches(ManifestCase):
    def test_ok(self):
        p = self.file("v.mp4", b"v" * 200)
        src = src_with_fp(p)
        ok, _ = manifest.matches({"version": manifest.VERSION, "source": src}, src)
        self.assertTrue(ok)

    def test_rejects_size(self):
        p = self.file("v.mp4", b"v" * 200)
        q = self.file("q.mp4", b"v" * 300)
        ok, why = manifest.matches(
            {"version": manifest.VERSION, "source": src_with_fp(p)}, src_with_fp(q))
        self.assertFalse(ok)
        self.assertIn("指纹", why)

    def test_rejects_duration(self):
        p = self.file("v.mp4", b"v" * 200)
        other = self.file("w.mp4", b"w" * 200)
        bad = dict(src_with_fp(p), fingerprint=manifest.fingerprint(other))
        ok, why = manifest.matches(
            {"version": manifest.VERSION, "source": src_with_fp(p)}, bad)
        self.assertFalse(ok)
        self.assertIn("指纹", why)

    def test_rejects_missing_fingerprint(self):
        """旧清单没有指纹，一律不认。"""
        src = {"size": 100, "duration": 10.0}
        ok, why = manifest.matches({"version": manifest.VERSION, "source": src}, src)
        self.assertFalse(ok)
        self.assertIn("指纹", why)

    def test_rejects_missing_manifest(self):
        ok, _ = manifest.matches(None, {"size": 1, "duration": 1, "fingerprint": "x"})
        self.assertFalse(ok)

    def test_rejects_old_version(self):
        p = self.file("v.mp4", b"v" * 200)
        src = src_with_fp(p)
        ok, why = manifest.matches({"version": 2, "source": src}, src)
        self.assertFalse(ok)
        self.assertIn("版本", why)


class TestRunLifecycle(ManifestCase):
    def _prepare(self, seg_ids, emb_ids, emb=None, run_id=None, source_id=None):
        p = self.file("v.mp4", b"v" * 200)
        src = src_with_fp(p)
        sid = manifest.source_id(p)
        os.makedirs(manifest.job_dir(sid), exist_ok=True)
        rid = manifest.new_run(sid, src, model="large-v3", device="cpu")

        segs = [{"id": i, "start": i, "end": i + 1, "text": f"t{i}"} for i in seg_ids]
        manifest.atomic_write_json(
            manifest.run_artifact(sid, rid, manifest.TRANSCRIPT),
            {"source_id": source_id or sid, "run_id": run_id or rid,
             "segments": segs, "duration": len(segs)})
        arr = np.ones((len(emb_ids), 192), dtype=np.float32) if emb is None else emb
        np.savez(manifest.run_artifact(sid, rid, manifest.EMBEDDINGS),
                 emb=arr, ids=np.array(emb_ids))
        return sid, rid

    def test_complete_after_validation(self):
        sid, rid = self._prepare([0, 1], [0, 1])
        ok, detail = manifest.finalize_run(sid, rid)
        self.assertTrue(ok, detail)
        self.assertEqual(manifest.active_run(sid), rid)
        self.assertIsNone(manifest.pending_run(sid))

    def test_active_run_ignores_unfinished(self):
        p = self.file("v.mp4", b"v" * 200)
        src = src_with_fp(p)
        sid = manifest.source_id(p)
        os.makedirs(manifest.job_dir(sid), exist_ok=True)
        rid = manifest.new_run(sid, src)
        self.assertIsNone(manifest.active_run(sid))
        self.assertEqual(manifest.pending_run(sid), rid)

    def test_mixed_artifacts_rejected(self):
        """新转写配旧声纹（多出不属于本次转写的 ID）必须被拒绝。"""
        sid, rid = self._prepare([0, 1, 2], [0, 1, 2, 99])
        ok, why = manifest.finalize_run(sid, rid)
        self.assertFalse(ok)
        self.assertIn("声纹", why)
        self.assertIsNone(manifest.active_run(sid))

    def test_missing_embeddings_rejected(self):
        sid, rid = self._prepare([0], [0])
        os.remove(manifest.run_artifact(sid, rid, manifest.EMBEDDINGS))
        ok, why = manifest.finalize_run(sid, rid)
        self.assertFalse(ok)
        self.assertIn("声纹", why)

    def test_empty_transcript_rejected(self):
        sid, rid = self._prepare([], [])
        ok, why = manifest.finalize_run(sid, rid)
        self.assertFalse(ok)
        self.assertIn("为空", why)

    def test_row_count_mismatch_rejected(self):
        """emb 两行、ids 一个：zip() 会静默截断，必须提前拒绝。"""
        sid, rid = self._prepare([0, 1], [0], emb=np.ones((2, 192), dtype=np.float32))
        ok, why = manifest.finalize_run(sid, rid)
        self.assertFalse(ok)
        self.assertIn("不一致", why)

    def test_duplicate_ids_rejected(self):
        sid, rid = self._prepare([0, 1], [0, 0])
        ok, why = manifest.finalize_run(sid, rid)
        self.assertFalse(ok)
        self.assertIn("重复", why)

    def test_nan_rejected(self):
        emb = np.ones((2, 192), dtype=np.float32)
        emb[1, 5] = np.nan
        sid, rid = self._prepare([0, 1], [0, 1], emb=emb)
        ok, why = manifest.finalize_run(sid, rid)
        self.assertFalse(ok)
        self.assertIn("NaN", why)

    def test_inf_rejected(self):
        emb = np.ones((2, 192), dtype=np.float32)
        emb[0, 0] = np.inf
        sid, rid = self._prepare([0, 1], [0, 1], emb=emb)
        ok, why = manifest.finalize_run(sid, rid)
        self.assertFalse(ok)
        self.assertIn("NaN", why)

    def test_zero_vector_rejected(self):
        emb = np.ones((2, 192), dtype=np.float32)
        emb[1, :] = 0.0
        sid, rid = self._prepare([0, 1], [0, 1], emb=emb)
        ok, why = manifest.finalize_run(sid, rid)
        self.assertFalse(ok)
        self.assertIn("零向量", why)

    def test_wrong_run_id_rejected(self):
        sid, rid = self._prepare([0, 1], [0, 1], run_id="20200101-000000-deadbe")
        ok, why = manifest.finalize_run(sid, rid)
        self.assertFalse(ok)
        self.assertIn("运行", why)

    def test_wrong_source_id_rejected(self):
        sid, rid = self._prepare([0, 1], [0, 1], source_id="other-1234")
        ok, why = manifest.finalize_run(sid, rid)
        self.assertFalse(ok)
        self.assertIn("源文件", why)

    def test_new_run_abandons_previous(self):
        """连续两次 new_run：旧的那个不能一直挂在 running 上。"""
        p = self.file("v.mp4", b"v" * 200)
        src = src_with_fp(p)
        sid = manifest.source_id(p)
        os.makedirs(manifest.job_dir(sid), exist_ok=True)
        r1 = manifest.new_run(sid, src)
        r2 = manifest.new_run(sid, src)

        man = manifest.load_manifest(sid)
        self.assertEqual(man["runs"][r1]["status"], "abandoned")
        self.assertEqual(man["runs"][r2]["status"], "running")
        self.assertEqual(manifest.pending_run(sid), r2)

        manifest.cleanup_runs(sid)
        man = manifest.load_manifest(sid)
        self.assertNotIn(r1, man["runs"])
        self.assertIn(r2, man["runs"])
        self.assertFalse(os.path.exists(manifest.run_dir(sid, r1)))

    def test_cleanup_keeps_pending(self):
        p = self.file("v.mp4", b"v" * 200)
        src = src_with_fp(p)
        sid = manifest.source_id(p)
        os.makedirs(manifest.job_dir(sid), exist_ok=True)
        rid = manifest.new_run(sid, src)
        manifest.cleanup_runs(sid)
        self.assertIn(rid, (manifest.load_manifest(sid) or {}).get("runs", {}))

    def test_new_run_records_track(self):
        p = self.file("v.mp4", b"v" * 200)
        src = src_with_fp(p)
        sid = manifest.source_id(p)
        os.makedirs(manifest.job_dir(sid), exist_ok=True)
        rid = manifest.new_run(sid, src, audio_track=1)
        self.assertEqual(manifest.load_manifest(sid)["runs"][rid]["audio_track"], 1)

    def test_audio_track_match_ok(self):
        sid, rid = self._prepare([0, 1], [0, 1])
        manifest.save_audio_info(sid, sid, 0, 2.0, 2)
        ok, why = manifest.finalize_run(sid, rid)
        self.assertTrue(ok, why)

    def test_audio_track_mismatch_rejected(self):
        """音频换成第 2 条音轨后，用第 1 条音频做的运行不能再提交。"""
        sid, rid = self._prepare([0, 1], [0, 1])
        manifest.save_audio_info(sid, sid, 1, 2.0, 2)
        ok, why = manifest.finalize_run(sid, rid)
        self.assertFalse(ok)
        self.assertIn("音轨", why)
        self.assertIsNone(manifest.active_run(sid))

    def test_audio_from_other_source_rejected(self):
        sid, rid = self._prepare([0, 1], [0, 1])
        manifest.save_audio_info(sid, "other-1234", 0, 2.0, 1)
        ok, why = manifest.finalize_run(sid, rid)
        self.assertFalse(ok)
        self.assertIn("音频", why)


class TestLabels(ManifestCase):
    def test_roundtrip(self):
        p = self.file("v.mp4", b"v" * 200)
        sid = manifest.source_id(p)
        os.makedirs(manifest.job_dir(sid), exist_ok=True)
        manifest.save_labels(sid, "碧翠丝", "run-1", [3, 1, 2, 2], [9, 8], 0.66)
        d = manifest.load_labels(sid)
        self.assertEqual(d["positive"], [1, 2, 3])
        self.assertEqual(d["negative"], [8, 9])
        self.assertEqual(d["run_id"], "run-1")
        self.assertAlmostEqual(d["threshold"], 0.66)

    def test_missing_returns_empty(self):
        p = self.file("v.mp4", b"v" * 200)
        sid = manifest.source_id(p)
        self.assertEqual(manifest.load_labels(sid), {})

    def test_audio_info_roundtrip(self):
        p = self.file("v.mp4", b"v" * 200)
        sid = manifest.source_id(p)
        os.makedirs(manifest.job_dir(sid), exist_ok=True)
        self.assertEqual(manifest.audio_track(sid), 0)
        manifest.save_audio_info(sid, sid, 1, 12.5, 2)
        self.assertEqual(manifest.audio_track(sid), 1)


class TestSelection(ManifestCase):
    """切点方案必须绑定当前活动运行。"""

    def _ready(self):
        p = self.file("v.mp4", b"v" * 200)
        src = src_with_fp(p)
        sid = manifest.source_id(p)
        os.makedirs(manifest.job_dir(sid), exist_ok=True)
        rid = manifest.new_run(sid, src)
        segs = [{"id": i, "start": i, "end": i + 1, "text": f"t{i}"} for i in (0, 1, 2)]
        manifest.atomic_write_json(
            manifest.run_artifact(sid, rid, manifest.TRANSCRIPT),
            {"source_id": sid, "run_id": rid, "segments": segs, "duration": 3})
        np.savez(manifest.run_artifact(sid, rid, manifest.EMBEDDINGS),
                 emb=np.ones((3, 192), dtype=np.float32), ids=np.array([0, 1, 2]))
        ok, why = manifest.finalize_run(sid, rid)
        self.assertTrue(ok, why)
        sel = {
            "source_id": sid,
            "run_id": rid,
            "source": dict(src),
            "segments": [{"start": 0.0, "end": 1.5, "dur": 1.5, "ids": [0], "lines": ["a"]}],
        }
        return sid, rid, src, sel

    def test_ok(self):
        sid, _rid, src, sel = self._ready()
        ok, why = manifest.validate_selection(sid, sel, src)
        self.assertTrue(ok, why)

    def test_stale_run_rejected(self):
        sid, _rid, src, sel = self._ready()
        sel["run_id"] = "20200101-000000-deadbe"
        ok, why = manifest.validate_selection(sid, sel, src)
        self.assertFalse(ok)
        self.assertIn("旧运行", why)

    def test_missing_run_id_rejected(self):
        sid, _rid, src, sel = self._ready()
        sel.pop("run_id")
        ok, why = manifest.validate_selection(sid, sel, src)
        self.assertFalse(ok)
        self.assertIn("运行 ID", why)

    def test_source_mismatch_rejected(self):
        sid, _rid, src, sel = self._ready()
        sel["source"] = dict(src, fingerprint="0" * 16)
        ok, why = manifest.validate_selection(sid, sel, src)
        self.assertFalse(ok)
        self.assertIn("指纹", why)

    def test_source_id_mismatch_rejected(self):
        sid, _rid, src, sel = self._ready()
        sel["source_id"] = "other-999"
        ok, why = manifest.validate_selection(sid, sel, src)
        self.assertFalse(ok)
        self.assertIn("其他源文件", why)

    def test_out_of_range_rejected(self):
        sid, _rid, src, sel = self._ready()
        sel["segments"][0]["end"] = 999.0
        ok, why = manifest.validate_selection(sid, sel, src)
        self.assertFalse(ok)
        self.assertIn("超出", why)

    def test_unknown_line_id_rejected(self):
        sid, _rid, src, sel = self._ready()
        sel["segments"][0]["ids"] = [0, 42]
        ok, why = manifest.validate_selection(sid, sel, src)
        self.assertFalse(ok)
        self.assertIn("台词编号", why)

    def test_no_active_run_rejected(self):
        p = self.file("w.mp4", b"w" * 200)
        src = src_with_fp(p)
        sid = manifest.source_id(p)
        os.makedirs(manifest.job_dir(sid), exist_ok=True)
        sel = {"source_id": sid, "run_id": "x",
               "segments": [{"start": 0, "end": 1, "dur": 1, "ids": [0]}]}
        ok, why = manifest.validate_selection(sid, sel, src)
        self.assertFalse(ok)
        self.assertIn("分析结果", why)


class TestAtomicWrite(ManifestCase):
    def test_write_and_no_tmp_left(self):
        p = os.path.join(self.tmp.name, "sub", "m.json")
        manifest.atomic_write_json(p, {"a": 1})
        with open(p, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["a"], 1)
        self.assertFalse(os.path.exists(p + ".tmp"))


if __name__ == "__main__":
    unittest.main()
