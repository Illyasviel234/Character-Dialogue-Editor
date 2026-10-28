"""任务清单与缓存绑定。

每个源视频对应 work/jobs/<source_id>/ 一个独立目录：
    manifest.json          源文件身份、分析运行记录
    runs/<run_id>/         单次分析的产物（transcript / embeddings）

身份用完整文件的内容指纹（流式 BLAKE2b），而不是修改时间或头尾采样：
两个头尾相同、中间不同的文件曾经会撞成同一个 ID，从而复用错误的转写与声纹。
首次计算后按（路径, 大小, 纳秒修改时间）缓存，后续切换视频不必重复读盘。

分析过程分三个阶段提交：running → transcribed → complete。
只有 complete 且被标记为 active_run 的结果才会被加载，
因此"新转写 + 旧声纹"这种混合状态永远不会被当成完整结果。
"""

import hashlib
import json
import math
import os
import secrets

from . import paths

VERSION = 4
MANIFEST = "manifest.json"
TRANSCRIPT = "transcript.json"
EMBEDDINGS = "embeddings.npz"
SELECTED = "selected.json"
AUDIO_WAV = "audio16k.wav"
AUDIO_INFO = "audio.json"
LABELS = "labels.json"
RUNS = "runs"

# 切点方案的 schema。v2 起要求 schema/source_id/run_id/audio_track/source 必填，
# 并把 dur 纳入校验；v1（无 schema 字段）会被明确拒绝，要求重新生成。
SELECTION_SCHEMA = 2

CHUNK = 4 * 1024 * 1024
UNFINISHED = ("running", "transcribed")
DEAD = ("failed", "abandoned")


def _cache_file():
    return os.path.join(paths.CACHE, "fingerprints.json")


def _read_cache():
    try:
        with open(_cache_file(), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_cache(cache):
    path = _cache_file()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        os.replace(tmp, path)
    except Exception:
        pass  # 缓存只是加速手段，写不进去也不影响正确性


def clear_cache():
    """丢弃指纹缓存，测试和排障用。"""
    try:
        os.remove(_cache_file())
    except OSError:
        pass


def fingerprint(path, cache=True):
    """整个文件内容的 BLAKE2b 指纹（16 位十六进制）。

    早期版本只读首尾各 1 MiB，头尾相同、中间不同的两个文件会得到同一个 ID。
    现在流式读完整个文件：788 MB 约 1 秒，且只在元数据变化时重算。
    """
    st = os.stat(path)
    key = os.path.normcase(os.path.abspath(path))
    size, mtime_ns = int(st.st_size), int(st.st_mtime_ns)
    if cache:
        hit = _read_cache().get(key)
        if isinstance(hit, list) and len(hit) == 3 and hit[0] == size and hit[1] == mtime_ns:
            return str(hit[2])
    h = hashlib.blake2b(digest_size=8)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    digest = h.hexdigest()
    if cache:
        data = _read_cache()
        data[key] = [size, mtime_ns, digest]
        _write_cache(data)
    return digest


def source_id(path, cache=True):
    """由文件名 + 内容指纹生成，同内容同 ID，不同内容必不同 ID。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in stem)[:40]
    return f"{safe}-{fingerprint(path, cache=cache)}"


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


# ---------- 音频音轨 ----------

def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_audio_info(sid, run_id=None):
    """这份 16k 音频来自第几条音轨。

    新布局把音频放进 runs/<run_id>/，与转写声纹同生命周期；旧任务只有
    任务级的一份，找不到 run 内的就回退到任务级（只读，不再写入）。
    """
    for path in ((run_artifact(sid, run_id, AUDIO_INFO) if run_id else None),
                 artifact(sid, AUDIO_INFO)):
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
    return {}


def audio_path(sid, run_id=None):
    """16k 音频的实际路径，找不到返回 None。"""
    for path in ((run_artifact(sid, run_id, AUDIO_WAV) if run_id else None),
                 artifact(sid, AUDIO_WAV)):
        if path and os.path.exists(path):
            return path
    return None


def save_audio_info(sid, run_id, source_id, track, duration, streams):
    atomic_write_json(run_artifact(sid, run_id, AUDIO_INFO), {
        "version": VERSION,
        "source_id": source_id,
        "track": int(track),
        "duration": round(float(duration), 3),
        "audio_streams": int(streams),
    })


def run_audio_track(sid, run_id=None):
    """某次运行实际使用的音轨号。优先运行记录，其次 audio.json，缺省 0。"""
    rid = run_id or active_run(sid)
    rec = ((load_manifest(sid) or {}).get("runs") or {}).get(rid) or {}
    if "audio_track" in rec:
        return int(rec["audio_track"])
    return int(load_audio_info(sid, rid).get("track", 0))


def audio_track(sid):
    """当前活动运行所用的音轨号，供界面与脚本对齐。"""
    return run_audio_track(sid, active_run(sid))


# ---------- 人工标注 ----------

def load_labels(sid):
    path = artifact(sid, LABELS)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_labels(sid, role, run_id, positive, negative, threshold=0.0):
    payload = {
        "version": VERSION,
        "source_id": sid,
        "run_id": run_id,
        "role": role,
        "updated": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "threshold": float(threshold),
        "positive": sorted({int(i) for i in positive}),
        "negative": sorted({int(i) for i in negative}),
    }
    atomic_write_json(artifact(sid, LABELS), payload)
    return payload


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

def new_run(sid, source, model="", device="", language="ja", audio_track=0):
    """开始一次分析，状态 running。

    之前没跑完的运行一律标记为 abandoned：进程被强杀时来不及执行清理，
    留着它们只会在 jobs/ 下累积孤立目录，也让"当前 pending 是哪一个"变得含糊。
    """
    man = load_manifest(sid)
    if not man or man.get("version") != VERSION:
        man = _blank(source)
    else:
        man["source"] = source
    now = __import__("datetime").datetime.now()
    for rid, rec in man.get("runs", {}).items():
        if rec.get("status") in UNFINISHED:
            rec["status"] = "abandoned"
            rec["error"] = f"被新的分析运行取代（{now.isoformat(timespec='seconds')}）"
    run_id = f"{now.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
    os.makedirs(run_dir(sid, run_id), exist_ok=True)
    man["runs"][run_id] = {
        "status": "running",
        "created": now.isoformat(timespec="seconds"),
        "model": model,
        "device": device,
        "language": language,
        "audio_track": int(audio_track),
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
    if rid and man.get("runs", {}).get(rid, {}).get("status") in UNFINISHED:
        return rid
    return None


def validate_run(sid, run_id):
    """校验一次运行的产物是否自洽。返回 (是否通过, 原因)。

    只检查"有没有文件"是不够的：行数与编号数不一致时 zip() 会静默截断，
    NaN/Inf 会让相似度全部变成 nan，零向量会让余弦相似度失去意义。
    """
    tpath = run_artifact(sid, run_id, TRANSCRIPT)
    epath = run_artifact(sid, run_id, EMBEDDINGS)
    if not os.path.exists(tpath):
        return False, "缺少转写结果"
    if not os.path.exists(epath):
        return False, "缺少声纹向量"
    try:
        rec = (load_manifest(sid) or {}).get("runs", {}).get(run_id) or {}
        with open(tpath, encoding="utf-8") as f:
            tr = json.load(f)
        segs = tr.get("segments") or []
        if not segs:
            return False, "转写结果为空"
        if not tr.get("source_id"):
            return False, "转写缺少源文件 ID"
        if tr["source_id"] != sid:
            return False, "转写属于其他源文件"
        if tr.get("run_id") != run_id:
            return False, f"转写属于运行 {tr.get('run_id') or '未知'}，不是本次运行"

        # 音频必须能追溯到源文件与音轨。只有 WAV、没有 audio.json 时无法证明
        # 它来自哪条音轨，宁可拒绝也不要放行。
        run_wav = run_artifact(sid, run_id, AUDIO_WAV)
        run_info = run_artifact(sid, run_id, AUDIO_INFO)
        legacy_wav = artifact(sid, AUDIO_WAV)
        legacy_info = artifact(sid, AUDIO_INFO)
        if os.path.exists(run_wav):
            layout, info_path = "本次运行", run_info
        elif os.path.exists(legacy_wav):
            layout, info_path = "旧布局", legacy_info
        else:
            return False, "缺少 16k 音频，请重新抽取"
        if not os.path.exists(info_path):
            return False, f"{layout}缺少 audio.json，无法确认音频来源，请重新抽取"
        side = _read_json(info_path)
        if not isinstance(side, dict) or not side:
            return False, "audio.json 无法解析，请重新抽取音频"
        missing = [k for k in ("source_id", "track", "duration", "audio_streams")
                   if k not in side]
        if missing:
            return False, "audio.json 缺少字段：" + "、".join(missing)
        if side["source_id"] != sid:
            return False, "音频属于其他源文件"
        used = int(rec.get("audio_track", 0))
        have = int(side["track"])
        if used != have:
            return False, (f"音频来自第 {have + 1} 条音轨，本次运行用的是第 {used + 1} 条，"
                           f"请重新抽取音频后再分析")

        import numpy as np
        with np.load(epath) as d:
            emb = np.asarray(d["emb"], dtype=np.float32)
            raw_ids = d["ids"]
        if emb.ndim != 2 or emb.shape[1] != 192:
            return False, f"声纹维度异常 {emb.shape}"
        try:
            ids = [int(x) for x in raw_ids]
        except (TypeError, ValueError):
            return False, "声纹编号不是整数"
        if not ids:
            return False, "声纹为空"
        if emb.shape[0] != len(ids):
            return False, f"声纹行数 {emb.shape[0]} 与编号数 {len(ids)} 不一致"
        if len(set(ids)) != len(ids):
            return False, "声纹编号有重复"
        if not np.isfinite(emb).all():
            return False, "声纹含 NaN 或 Inf"
        if (np.linalg.norm(emb, axis=1) <= 1e-8).any():
            return False, "声纹含零向量"
        seg_ids = {s["id"] for s in segs}
        stray = [i for i in ids if i not in seg_ids]
        if stray:
            return False, f"{len(stray)} 条声纹找不到对应台词"
        return True, f"{len(segs)} 段 / {len(ids)} 条声纹"
    except Exception as e:
        return False, f"校验异常：{type(e).__name__}: {str(e)[:80]}"


def validate_selection(sid, selected, source=None, active=None):
    """导出与出报告前的统一校验：切点方案必须属于当前活动运行。

    重新分析成功后若没有重新筛选，selected.json 里还是旧 run 的台词和时间码。
    静默用它切片会得到"画面是新的、片段是旧的"这种最难发现的问题，
    因此这里直接拒绝，要求重新生成切点方案。

    音轨也必须一致：时间码与声纹来自第 N 条音轨，导出却用第 M 条，
    对白语言或混音不同时成片会与判断依据对不上。
    """
    if not selected:
        return False, "切点方案为空"
    if int(selected.get("schema") or 0) != SELECTION_SCHEMA:
        return False, (f"切点方案版本过旧（schema {selected.get('schema')}，"
                       f"需要 {SELECTION_SCHEMA}），请重新生成")
    if not selected.get("source_id"):
        return False, "切点方案缺少源文件 ID，请重新生成"
    if selected["source_id"] != sid:
        return False, "切点方案属于其他源文件"

    rid = active if active is not None else active_run(sid)
    if not rid:
        return False, "没有可用的分析结果，请先完成分析"
    srid = selected.get("run_id")
    if not srid:
        return False, "切点方案没有记录运行 ID，请重新生成"
    if srid != rid:
        return False, f"切点方案来自旧运行 {srid}（当前 {rid}），请重新生成切点方案"

    if "audio_track" not in selected:
        return False, "切点方案没有记录音轨，请重新生成"
    want = run_audio_track(sid, rid)
    have = int(selected["audio_track"])
    if have != want:
        return False, (f"切点方案用的是第 {have + 1} 条音轨，当前分析用的是第 {want + 1} 条，"
                       f"请重新生成切点方案")

    if source:
        old = (selected.get("source") or {}).get("fingerprint")
        new = source.get("fingerprint")
        if not old or not new:
            return False, "切点方案缺少源文件指纹"
        if old != new:
            return False, "切点方案对应的源文件指纹与当前视频不一致"

    segs = selected.get("segments") or []
    if not segs:
        return False, "切点方案为空"
    duration = float((source or selected.get("source") or {}).get("duration") or 0)
    for i, s in enumerate(segs, 1):
        try:
            start = float(s.get("start"))
            end = float(s.get("end"))
            dur = float(s.get("dur"))
        except (TypeError, ValueError):
            return False, f"第 {i} 段缺少时间字段"
        if not all(map(math.isfinite, (start, end, dur))):
            return False, f"第 {i} 段时间不是有限数值"
        if start < 0 or end <= start:
            return False, f"第 {i} 段时间非法（{start:.2f}–{end:.2f}）"
        if dur <= 0 or abs(dur - (end - start)) > 1e-3:
            return False, (f"第 {i} 段时长字段与区间不一致"
                           f"（dur {dur:.3f} vs {end - start:.3f}）")
        if duration and end > duration + 0.5:
            return False, f"第 {i} 段超出源视频时长（{end:.2f} > {duration:.2f}）"

    try:
        with open(run_artifact(sid, rid, TRANSCRIPT), encoding="utf-8") as f:
            tr = json.load(f)
    except Exception as e:
        return False, f"无法读取转写结果：{e}"
    known = {s["id"] for s in (tr.get("segments") or [])}
    missing = sorted({i for s in segs for i in s.get("ids", []) if i not in known})
    if missing:
        return False, f"{len(missing)} 个台词编号不在当前转写中，请重新生成切点方案"
    return True, f"运行 {rid} · {len(segs)} 段"


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
    """清掉没跑完的、失败的，以及除活动运行外多余的已完成运行。

    被强杀或崩溃的运行会停在 running/transcribed，永远等不到 finalize，
    只能在这里按"不是当前 pending"来回收。
    """
    man = load_manifest(sid)
    if not man:
        return
    runs = man.get("runs", {})
    pending = man.get("pending_run")
    active = man.get("active_run")

    doomed = []
    for rid, rec in runs.items():
        status = rec.get("status")
        if rid == pending and status in UNFINISHED:
            continue
        if status in UNFINISHED or status in DEAD:
            doomed.append(rid)

    finished = [(rid, r.get("created", "")) for rid, r in runs.items()
                if r.get("status") == "complete" and rid != active and rid not in doomed]
    finished.sort(key=lambda x: x[1], reverse=True)
    doomed += [rid for rid, _ in finished[keep:]]

    for rid in doomed:
        runs.pop(rid, None)
        _rmtree(run_dir(sid, rid))
    if man.get("pending_run") not in runs:
        man["pending_run"] = None
    save_manifest(sid, man)


def _rmtree(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)
