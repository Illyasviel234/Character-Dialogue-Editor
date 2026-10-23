import json
import os
import tempfile
import unittest

from core import manifest


class TestManifest(unittest.TestCase):
    def _tmpfile(self, size=100):
        f = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        f.write(b"x" * size)
        f.close()
        self.addCleanup(lambda: os.path.exists(f.name) and os.unlink(f.name))
        return f.name

    def test_source_id_stable(self):
        p = self._tmpfile()
        self.assertEqual(manifest.source_id(p), manifest.source_id(p))

    def test_source_id_differs_by_size(self):
        a = self._tmpfile(100)
        b = self._tmpfile(200)
        self.assertNotEqual(manifest.source_id(a), manifest.source_id(b))

    def test_matches_ok(self):
        src = {"size": 100, "duration": 10.0}
        ok, _ = manifest.matches({"version": manifest.VERSION, "source": src}, src)
        self.assertTrue(ok)

    def test_matches_rejects_size(self):
        src = {"size": 100, "duration": 10.0}
        ok, why = manifest.matches({"version": manifest.VERSION, "source": src},
                                   {"size": 101, "duration": 10.0})
        self.assertFalse(ok)
        self.assertIn("大小", why)

    def test_matches_rejects_duration(self):
        src = {"size": 100, "duration": 10.0}
        ok, why = manifest.matches({"version": manifest.VERSION, "source": src},
                                   {"size": 100, "duration": 12.0})
        self.assertFalse(ok)
        self.assertIn("时长", why)

    def test_matches_rejects_missing(self):
        ok, _ = manifest.matches(None, {"size": 1, "duration": 1})
        self.assertFalse(ok)

    def test_matches_rejects_old_version(self):
        src = {"size": 100, "duration": 10.0}
        ok, why = manifest.matches({"version": 1, "source": src}, src)
        self.assertFalse(ok)
        self.assertIn("版本", why)

    def test_atomic_write_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "sub", "m.json")
            manifest.atomic_write_json(p, {"a": 1})
            with open(p, encoding="utf-8") as f:
                self.assertEqual(json.load(f)["a"], 1)
            self.assertFalse(os.path.exists(p + ".tmp"))


if __name__ == "__main__":
    unittest.main()
