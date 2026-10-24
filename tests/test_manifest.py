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
        paths.JOBS = os.path.join(self.tmp.name, "jobs")
        os.makedirs(paths.JOBS, exist_ok=True)
        self.addCleanup(self._restore)

    def _restore(self):
        paths.JOBS = self._orig_jobs

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
    def _prepare(self, seg_ids, emb_ids):
        p = self.file("v.mp4", b"v" * 200)
        src = src_with_fp(p)
        sid = manifest.source_id(p)
        os.makedirs(manifest.job_dir(sid), exist_ok=True)
        rid = manifest.new_run(sid, src, model="large-v3", device="cpu")

        segs = [{"id": i, "start": i, "end": i + 1, "text": f"t{i}"} for i in seg_ids]
        manifest.atomic_write_json(
            manifest.run_artifact(sid, rid, manifest.TRANSCRIPT),
            {"source_id": sid, "run_id": rid, "segments": segs, "duration": len(segs)})
        np.savez(manifest.run_artifact(sid, rid, manifest.EMBEDDINGS),
                 emb=np.ones((len(emb_ids), 192), dtype=np.float32),
                 ids=np.array(emb_ids))
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


class TestAtomicWrite(ManifestCase):
    def test_write_and_no_tmp_left(self):
        p = os.path.join(self.tmp.name, "sub", "m.json")
        manifest.atomic_write_json(p, {"a": 1})
        with open(p, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["a"], 1)
        self.assertFalse(os.path.exists(p + ".tmp"))


if __name__ == "__main__":
    unittest.main()
