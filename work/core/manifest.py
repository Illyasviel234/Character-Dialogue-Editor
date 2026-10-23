"""任务清单与缓存绑定。

每个源视频对应 work/jobs/<source_id>/ 一个独立目录，目录内的 manifest.json
记录源文件身份。加载缓存前必须校验清单，避免把上一部片子的时间码用到新片子上。
"""

import hashlib
import json
import os
import shutil

from . import paths

VERSION = 2
TRANSCRIPT = "transcript.json"
EMBEDDINGS = "embeddings.npz"
SELECTED = "selected.json"
MANIFEST = "manifest.json"


def source_id(path):
    """由文件名、大小和修改时间生成稳定 ID，同名不同文件不会撞车。"""
    st = os.stat(path)
    key = f"{os.path.basename(path)}|{st.st_size}|{int(st.st_mtime)}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    stem = os.path.splitext(os.path.basename(path))[0]
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in stem)[:40]
    return f"{safe}-{digest}"


def make_source(path, media):
    st = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "name": os.path.basename(path),
        "size": st.st_size,
        "mtime": int(st.st_mtime),
        "duration": round(float(media.get("duration", 0.0)), 3),
        "width": int(media.get("width", 0)),
        "height": int(media.get("height", 0)),
        "fps": round(float(media.get("fps", 0.0)), 3),
        "audio_streams": int(media.get("audio_streams", 0)),
    }


def job_dir(sid):
    return os.path.join(paths.JOBS, sid)


def artifact(sid, name):
    return os.path.join(job_dir(sid), name)


def atomic_write_json(path, obj):
    """先写临时文件再替换，失败时不会留下半截数据。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_manifest(sid, source, model="", device="", language="ja"):
    data = {
        "version": VERSION,
        "created": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "model": model,
        "device": device,
        "language": language,
    }
    atomic_write_json(artifact(sid, MANIFEST), data)
    return data


def load_manifest(sid):
    path = artifact(sid, MANIFEST)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def has_analysis(sid):
    return (os.path.exists(artifact(sid, TRANSCRIPT))
            and os.path.exists(artifact(sid, EMBEDDINGS)))


def matches(manifest, source):
    """判断清单是否对应当前源文件。返回 (是否匹配, 原因)。"""
    if not manifest:
        return False, "缺少任务清单"
    if manifest.get("version") != VERSION:
        return False, "清单版本过旧"
    old = manifest.get("source") or {}
    if old.get("size") != source.get("size"):
        return False, "文件大小不一致"
    d_old = float(old.get("duration", 0) or 0)
    d_new = float(source.get("duration", 0) or 0)
    if d_old > 0 and d_new > 0 and abs(d_old - d_new) > 0.5:
        return False, "时长不一致"
    return True, "匹配"


def migrate_legacy(source_path, sid, source):
    """把旧版扁平缓存（work/transcript.json）迁入任务目录，避免重复转写。"""
    jd = job_dir(sid)
    if os.path.exists(os.path.join(jd, TRANSCRIPT)):
        return False
    legacy_t = os.path.join(paths.WORK, TRANSCRIPT)
    legacy_e = os.path.join(paths.WORK, EMBEDDINGS)
    if not (os.path.exists(legacy_t) and os.path.exists(legacy_e)):
        return False
    try:
        with open(legacy_t, encoding="utf-8") as f:
            tr = json.load(f)
        dur = float(tr.get("duration") or 0)
        if dur and source.get("duration") and abs(dur - source["duration"]) > 0.5:
            return False
    except Exception:
        return False
    os.makedirs(jd, exist_ok=True)
    shutil.copy2(legacy_t, os.path.join(jd, TRANSCRIPT))
    shutil.copy2(legacy_e, os.path.join(jd, EMBEDDINGS))
    save_manifest(sid, source, model=tr.get("model", ""), device=tr.get("device", ""))
    return True
