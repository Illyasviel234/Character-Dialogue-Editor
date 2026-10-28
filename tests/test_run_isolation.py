"""按 run 隔离分析音频。

新分析一旦开始就会抽音频。如果音频还放在任务目录，抽到一半失败或被取消，
旧活动 run 的试听就会变成另一条音轨的内容——转写和声纹却还是旧的。
这里验证新布局：每次抽取只写自己的 run 目录。
"""

import hashlib
import os
import subprocess
import sys
import tempfile
import unittest

from core import manifest, paths


def make_two_track_video(path, seconds=2):
    r = subprocess.run([
        paths.FFMPEG, "-y", "-v", "error",
        "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=25:duration={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=880:duration={seconds}",
        "-map", "0:v", "-map", "1:a", "-map", "2:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-shortest", path,
    ], capture_output=True)
    return r.returncode == 0


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class TestAudioIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.video = os.path.join(cls.tmp.name, "two_tracks.mp4")
        cls.jobs = os.path.join(cls.tmp.name, "jobs")
        os.makedirs(cls.jobs, exist_ok=True)
        cls._orig = (paths.JOBS, paths.CACHE)
        paths.JOBS = cls.jobs
        paths.CACHE = os.path.join(cls.tmp.name, "cache")
        cls.ok = make_two_track_video(cls.video)
        if not cls.ok:
            raise unittest.SkipTest("ffmpeg 不可用")

    @classmethod
    def tearDownClass(cls):
        paths.JOBS, paths.CACHE = cls._orig
        cls.tmp.cleanup()

    def _env(self):
        """把任务目录指向临时目录，避免测试污染项目的 work/jobs。"""
        return {**os.environ,
                "PYTHONIOENCODING": "utf-8",
                "CLIP_JOBS_DIR": self.jobs,
                "CLIP_CACHE_DIR": os.path.join(self.tmp.name, "cache")}

    def _extract(self, track):
        r = subprocess.run(
            [sys.executable, "-u", "extract_audio.py", "--video", self.video,
             "--audio-track", str(track)],
            cwd=paths.WORK, capture_output=True, text=True,
            encoding="utf-8", errors="replace", env=self._env())
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r.stdout

    def test_extract_goes_into_run_dir(self):
        out = self._extract(0)
        sid = manifest.source_id(self.video)
        self.assertEqual(manifest.job_dir(sid), os.path.join(self.jobs, sid))
        rid = manifest.pending_run(sid) or manifest.active_run(sid)
        self.assertTrue(rid, out)
        wav = manifest.run_artifact(sid, rid, manifest.AUDIO_WAV)
        self.assertTrue(os.path.exists(wav), "音频应落在 run 目录")
        self.assertFalse(os.path.exists(manifest.artifact(sid, manifest.AUDIO_WAV)),
                         "任务目录不该再有音频")
        self.assertEqual(manifest.load_audio_info(sid, rid)["track"], 0)

    def test_new_extract_does_not_touch_old_run(self):
        self._extract(0)
        sid = manifest.source_id(self.video)
        first = manifest.pending_run(sid) or manifest.active_run(sid)
        first_wav = manifest.run_artifact(sid, first, manifest.AUDIO_WAV)
        before = digest(first_wav)

        self._extract(1)
        second = manifest.pending_run(sid) or manifest.active_run(sid)
        self.assertNotEqual(first, second, "第二次抽取应新建运行")
        self.assertTrue(os.path.exists(first_wav), "旧运行的音频不能消失")
        self.assertEqual(digest(first_wav), before, "旧运行的音频内容被改写了")
        self.assertEqual(manifest.load_audio_info(sid, first)["track"], 0)
        self.assertEqual(manifest.load_audio_info(sid, second)["track"], 1)

    def test_bad_track_rejected(self):
        r = subprocess.run(
            [sys.executable, "-u", "extract_audio.py", "--video", self.video,
             "--audio-track", "5"],
            cwd=paths.WORK, capture_output=True, text=True,
            encoding="utf-8", errors="replace", env=self._env())
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("音轨", r.stdout + r.stderr)


class TestActiveRunSurvivesFailure(unittest.TestCase):
    """运行失败时活动运行不能变，旧产物也不能被动过。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._orig = (paths.JOBS, paths.CACHE)
        paths.JOBS = os.path.join(self.tmp.name, "jobs")
        paths.CACHE = os.path.join(self.tmp.name, "cache")
        os.makedirs(paths.JOBS, exist_ok=True)
        self.addCleanup(self._restore)
        self.video = os.path.join(self.tmp.name, "v.mp4")
        with open(self.video, "wb") as f:
            f.write(b"v" * 300)

    def _restore(self):
        paths.JOBS, paths.CACHE = self._orig

    def _complete(self, track=0):
        import numpy as np
        src = {"size": 300, "duration": 10.0, "fingerprint": manifest.fingerprint(self.video)}
        sid = manifest.source_id(self.video)
        os.makedirs(manifest.job_dir(sid), exist_ok=True)
        rid = manifest.new_run(sid, src, audio_track=track)
        manifest.atomic_write_json(
            manifest.run_artifact(sid, rid, manifest.TRANSCRIPT),
            {"source_id": sid, "run_id": rid,
             "segments": [{"id": 0, "start": 0, "end": 1, "text": "t"}], "duration": 1})
        np.savez(manifest.run_artifact(sid, rid, manifest.EMBEDDINGS),
                 emb=np.ones((1, 192), dtype=np.float32), ids=np.array([0]))
        wav = manifest.run_artifact(sid, rid, manifest.AUDIO_WAV)
        with open(wav, "wb") as f:
            f.write(b"RIFF" + b"\0" * 64)
        manifest.save_audio_info(sid, rid, sid, track, 1.0, 1)
        ok, why = manifest.finalize_run(sid, rid)
        self.assertTrue(ok, why)
        return sid, rid

    def test_failed_run_keeps_active(self):
        sid, first = self._complete(0)
        second = manifest.new_run(sid, {"size": 300, "duration": 10.0,
                                        "fingerprint": manifest.fingerprint(self.video)},
                                  audio_track=1)
        manifest.fail_run(sid, second, "转写失败")
        self.assertEqual(manifest.active_run(sid), first)
        self.assertTrue(os.path.exists(
            manifest.run_artifact(sid, first, manifest.AUDIO_WAV)))
        self.assertEqual(manifest.audio_track(sid), 0)

    def test_cleanup_removes_failed_run_audio_only(self):
        sid, first = self._complete(0)
        second = manifest.new_run(sid, {"size": 300, "duration": 10.0,
                                        "fingerprint": manifest.fingerprint(self.video)},
                                  audio_track=1)
        wav2 = manifest.run_artifact(sid, second, manifest.AUDIO_WAV)
        with open(wav2, "wb") as f:
            f.write(b"RIFF" + b"\0" * 64)
        manifest.fail_run(sid, second, "取消")
        manifest.cleanup_runs(sid)
        self.assertFalse(os.path.exists(wav2), "失败运行的音频应被回收")
        self.assertTrue(os.path.exists(
            manifest.run_artifact(sid, first, manifest.AUDIO_WAV)),
            "活动运行的音频不能被清掉")


if __name__ == "__main__":
    unittest.main()
