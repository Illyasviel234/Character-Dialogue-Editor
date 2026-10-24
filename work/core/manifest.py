"""任务清单与缓存绑定。

每个源视频对应 work/jobs/<source_id>/ 一个独立目录：
    manifest.json          源文件身份、分析运行记录
    runs/<run_id>/         单次分析的产物（transcript / embeddings）

身份用内容指纹（大小 + 首尾各 1 MiB 的 SHA-256）而不是修改时间，
同名同大小但内容不同的文件不会互相污染。

分析过程分三个阶段提交：running → transcribed → complete。
只有 complete 且被标记为 active_run 的结果才会被加载，
因此"新转写 + 旧声纹"这种混合状态永远不会被当成完整结果。
"""

import hashlib
import json
import os
import secrets

from . import paths

VERSION = 3
MANIFEST = "manifest.json"
TRANSCRIPT = "transcript.json"
EMBEDDINGS = "embeddings.npz"
SELECTED = "selected.json"
RUNS = "runs"

CHUNK = 1024 * 1024


def fingerprint(path, chunk=CHUNK):
    """大小 + 首尾各 1 MiB 的 SHA-256。788 MB 的视频只需读 2 MB。"""
    st = os.stat(path)
    h = hashlib.sha256()
    h.update(str(st.st_size).encode("utf-8"))
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if st.st_size > 2 * chunk:
            f.seek(-chunk, os.SEEK_END)
            h.update(f.read(chunk))
    return h.hexdigest()[:16]


def source_id(path):
    """由文件名 + 内容指纹生成，同内容同 ID，不同内容必不同 ID。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in stem)[:40]
    return f"{safe}-{fingerprint(path)}"


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
        "fingerprint": fingerprint(path),
    }


# ---------- 路径 ----------

def job_dir(sid):
    return os.path.join(paths.JOBS, sid)


def artifact(sid, name):
    return os.path.join(job_dir(sid), name)


def run_dir(sid, run_id):
    return os.path.join(job_dir(sid), RUNS, run_id)


def run_artifact(sid, run_id, name):
    return os.path.join(run_dir(sid, run_id), name)


# ---------- 读写 ----------

def atomic_write_json(path, obj):
    """先写临时文件再替换，失败时不会留下半截数据。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_manifest(sid):
    path = artifact(sid, MANIFEST)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _blank(source):
    return {
        "version": VERSION,
        "source": source,
        "active_run": None,
        "pending_run": None,
        "runs": {},
    }


def save_manifest(sid, data):
    atomic_write_json(artifact(sid, MANIFEST), data)


def matches(manifest, source):
    """判断清单是否对应当前源文件。指纹是硬条件，缺指纹一律不认。"""
    if not manifest:
        return False, "缺少任务清单"
    if manifest.get("version") != VERSION:
        return False, "清单版本过旧"
    old = manifest.get("source") or {}
    if not old.get("fingerprint"):
        return False, "清单缺少内容指纹"
    if not source.get("fingerprint"):
        return False, "无法计算源文件指纹"
    if old["fingerprint"] != source["fingerprint"]:
        return False, "内容指纹不一致"
    return True, "匹配"


# ---------- 运行生命周期 ----------

def new_run(sid, source, model="", device="", language="ja"):
    """开始一次分析，状态 running。"""
    man = load_manifest(sid)
    if not man or man.get("version") != VERSION:
        man = _blank(source)
    run_id = f"{__import__('datetime').datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
    os.makedirs(run_dir(sid, run_id), exist_ok=True)
    man["runs"][run_id] = {
        "status": "running",
        "created": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "device": device,
        "language": language,
    }
    man["pending_run"] = run_id
    save_manifest(sid, man)
    return run_id


def update_run(sid, run_id, **fields):
    man = load_manifest(sid)
    if not man or run_id not in man.get("runs", {}):
        return None
    man["runs"][run_id].update(fields)
    man["runs"][run_id]["updated"] = __import__("datetime").datetime.now().isoformat(
        timespec="seconds")
    save_manifest(sid, man)
    return man["runs"][run_id]


def pending_run(sid):
    man = load_manifest(sid) or {}
    rid = man.get("pending_run")
    if rid and man.get("runs", {}).get(rid, {}).get("status") in ("running", "transcribed"):
        return rid
    return None


def validate_run(sid, run_id):
    """校验一次运行的产物是否自洽。返回 (是否通过, 原因)。"""
    tpath = run_artifact(sid, run_id, TRANSCRIPT)
    epath = run_artifact(sid, run_id, EMBEDDINGS)
    if not os.path.exists(tpath):
        return False, "缺少转写结果"
    if not os.path.exists(epath):
        return False, "缺少声纹向量"
    try:
        with open(tpath, encoding="utf-8") as f:
            tr = json.load(f)
        segs = tr.get("segments") or []
        if not segs:
            return False, "转写结果为空"
        if tr.get("source_id") and tr["source_id"] != sid:
            return False, "转写属于其他源文件"
        import numpy as np
        d = np.load(epath)
        emb, ids = d["emb"], [int(x) for x in d["ids"]]
        if emb.ndim != 2 or emb.shape[1] != 192:
            return False, f"声纹维度异常 {emb.shape}"
        if len(ids) == 0:
            return False, "声纹为空"
        seg_ids = {s["id"] for s in segs}
        stray = [i for i in ids if i not in seg_ids]
        if stray:
            return False, f"{len(stray)} 条声纹找不到对应台词"
        return True, f"{len(segs)} 段 / {len(ids)} 条声纹"
    except Exception as e:
        return False, f"校验异常：{type(e).__name__}: {str(e)[:80]}"


def finalize_run(sid, run_id):
    """校验通过后标记 complete 并设为活动运行。失败则标记 failed。"""
    ok, detail = validate_run(sid, run_id)
    man = load_manifest(sid) or {}
    if not ok:
        if run_id in man.get("runs", {}):
            man["runs"][run_id].update({"status": "failed", "error": detail})
            if man.get("pending_run") == run_id:
                man["pending_run"] = None
            save_manifest(sid, man)
        return False, detail

    man["runs"][run_id]["status"] = "complete"
    man["runs"][run_id]["detail"] = detail
    man["runs"][run_id]["error"] = None
    man["active_run"] = run_id
    if man.get("pending_run") == run_id:
        man["pending_run"] = None
    save_manifest(sid, man)
    return True, detail


def fail_run(sid, run_id, error):
    man = load_manifest(sid) or {}
    if run_id in man.get("runs", {}):
        man["runs"][run_id].update({"status": "failed", "error": str(error)[:200]})
        if man.get("pending_run") == run_id:
            man["pending_run"] = None
        save_manifest(sid, man)


def active_run(sid):
    """只有状态为 complete 的活动运行才算可用。"""
    man = load_manifest(sid) or {}
    rid = man.get("active_run")
    if rid and man.get("runs", {}).get(rid, {}).get("status") == "complete":
        return rid
    return None


def active_run_dir(sid):
    rid = active_run(sid)
    return run_dir(sid, rid) if rid else None


def has_analysis(sid):
    return active_run(sid) is not None


def cleanup_runs(sid, keep=2):
    """只保留最近若干个已完成的运行，失败和过期的清掉。"""
    man = load_manifest(sid)
    if not man:
        return
    runs = man.get("runs", {})
    finished = [(rid, r.get("created", "")) for rid, r in runs.items()
                if r.get("status") == "complete" and rid != man.get("active_run")]
    finished.sort(key=lambda x: x[1], reverse=True)
    for rid, _ in finished[keep:]:
        runs.pop(rid, None)
        _rmtree(run_dir(sid, rid))
    for rid, r in list(runs.items()):
        if r.get("status") == "failed":
            runs.pop(rid, None)
            _rmtree(run_dir(sid, rid))
    save_manifest(sid, man)


def _rmtree(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)
