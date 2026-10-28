"""批量任务队列：把「分析 → 筛选 → 导出 → 清单」串成一条流水线。

这里只描述"要做什么"，不负责执行。执行由界面里的 Runner 或命令行驱动，
所以队列本身可以存成 JSON，关掉程序下次接着跑。

三条不变量：

1. **条目自带全部参数**，不依赖界面上当前的值——否则改一次阈值会把
   队列里已排好的任务也改掉。
2. **条目带稳定 id**，完成回调按 id 回写，不会因为顺序变化记到别的任务上。
3. **源文件身份在入队时冻结**。同一路径的文件可能被替换、同步盘可能
   换成另一集，执行前必须复核；不一致就停下来等用户确认，不能静默
   处理"看起来还是那个文件名"的另一份内容。
"""

import json
import os
import uuid

from . import manifest, media, naming, paths

QUEUE_FILE = "queue.json"
SCHEMA = 2

WAITING = "等待"
RUNNING = "进行中"
DONE = "完成"
FAILED = "失败"
CANCELLED = "已取消"
SKIPPED = "已跳过"

# 已有目标文件的处理策略
ASK = "ask"
OVERWRITE = "overwrite"
SKIP = "skip"

FIELDS = {
    "role": "目标角色",
    "keyword": "かしら,ベティ",
    "threshold": 0.66,
    "merge_gap": 2.0,
    "pad_head": 0.25,
    "pad_tail": 0.30,
    "audio_track": 0,
    "scale": "native",
    "crf": 20,
    "vfr": "cfr",
    "model": "large-v3",
    "device": "cuda",
    "out_dir": "",
}

REPORT_MIN_BYTES = 32


def queue_path():
    return os.path.join(paths.WORK, QUEUE_FILE)


def new_id():
    return uuid.uuid4().hex[:8]


def source_identity(video):
    """入队时冻结源文件身份：内容指纹 + 大小 + 修改时间。"""
    try:
        st = os.stat(video)
        fp = manifest.fingerprint(video)
        sid = manifest.source_id(video)
    except Exception:
        return {}
    return {
        "id": sid,
        "fingerprint": fp,
        "size": int(st.st_size),
        "mtime": int(st.st_mtime),
        "name": os.path.basename(video),
    }


def make_item(video, **kw):
    item = dict(FIELDS)
    item.update({k: v for k, v in kw.items() if k in FIELDS})
    item["video"] = os.path.abspath(video) if video else ""
    item["id"] = kw.get("id") or new_id()
    item["source"] = kw.get("source") or source_identity(item["video"])
    item["status"] = WAITING
    item["message"] = ""
    return item


# ---------- 持久化 ----------

def _read_raw(path=None):
    path = path or queue_path()
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_state(path=None):
    """读回 (队列设置, 条目列表)。坏文件按空队列处理。"""
    data = _read_raw(path)
    if data is None:
        return {"out_dir": ""}, []
    settings = {"out_dir": ""}
    items = data
    if isinstance(data, dict):
        items = data.get("items") or []
        settings["out_dir"] = data.get("out_dir") or ""
    if not isinstance(items, list):
        return settings, []

    out = []
    seen = set()
    for raw in items:
        if not isinstance(raw, dict) or not raw.get("video"):
            continue
        item = make_item(raw["video"], source=raw.get("source") or None)
        item.update({k: raw[k] for k in list(FIELDS) + ["video"] if k in raw})
        # id 缺失、格式不对或重复都要换新的，否则完成回调可能写错条目
        rid = raw.get("id")
        if not isinstance(rid, str) or len(rid) != 8 or rid in seen:
            rid = new_id()
        seen.add(rid)
        item["id"] = rid
        item["status"] = (WAITING if raw.get("status") in (RUNNING, CANCELLED)
                          else raw.get("status") or WAITING)
        item["message"] = raw.get("message", "")
        out.append(item)
    return settings, out


def load(path=None):
    """只要条目列表（保持旧接口）。"""
    return load_state(path)[1]


def save(items, path=None, out_dir=None):
    """写回队列。重复 id 在写之前再修一次，防止外部拼出来的列表带坏数据。"""
    path = path or queue_path()
    seen = set()
    for item in items:
        rid = item.get("id")
        if not isinstance(rid, str) or len(rid) != 8 or rid in seen:
            rid = new_id()
            item["id"] = rid
        seen.add(rid)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"schema": SCHEMA, "version": manifest.VERSION,
                   "out_dir": out_dir or "", "items": items}, f,
                  ensure_ascii=False, indent=1)
    os.replace(tmp, path)


# ---------- 源文件身份 ----------

def source_check(item, deep=False):
    """检查条目记录的源文件是否还是当前这个文件。

    返回 (状态, 说明)，状态为 ok / changed / missing / unknown。
    deep=False 只比大小和修改时间（表格刷新用，不读盘）；
    deep=True 重算内容指纹（启动前用，能识别"改了内容又改回时间"）。
    """
    src = item.get("source") or {}
    if not src.get("id"):
        return "unknown", "条目没有记录源文件身份"
    video = item.get("video") or ""
    if not os.path.exists(video):
        return "missing", "源文件不存在"
    if not deep:
        try:
            st = os.stat(video)
        except OSError:
            return "missing", "源文件无法读取"
        if int(st.st_size) != int(src.get("size", -1)):
            return "changed", "文件大小已变化"
        if int(st.st_mtime) != int(src.get("mtime", -1)):
            return "changed", "文件修改时间已变化"
        return "ok", ""
    try:
        now = manifest.source_id(video)
    except Exception as e:
        return "unknown", f"无法读取源文件：{e}"
    if now != src["id"]:
        return "changed", (f"源文件内容已变化（{src.get('fingerprint', '')[:8]} → "
                           f"{now.rsplit('-', 1)[-1][:8]}）")
    return "ok", ""


def refresh_source(item):
    """把条目更新为当前文件的身份，返回 (旧 id, 新 id)。"""
    old = (item.get("source") or {}).get("id") or ""
    item["source"] = source_identity(item["video"])
    return old, (item.get("source") or {}).get("id") or ""


# ---------- 路径 ----------

_SUFFIX_CACHE = {}


def source_suffix(video):
    """源文件内容指纹的后 8 位，用来区分不同目录下的同名视频。

    表格每刷新一次都会算一次文件名，所以这里按（路径, 大小, mtime）记一层，
    避免反复读指纹缓存文件。
    """
    try:
        st = os.stat(video)
    except OSError:
        return ""
    key = (os.path.normcase(os.path.abspath(video)), st.st_size, st.st_mtime_ns)
    if key in _SUFFIX_CACHE:
        return _SUFFIX_CACHE[key]
    try:
        sid = manifest.source_id(video)
        suffix = sid.rsplit("-", 1)[-1][:8] if "-" in sid else ""
    except Exception:
        suffix = ""
    _SUFFIX_CACHE[key] = suffix
    return suffix


def base_name(item):
    """成片与清单共用的文件名主干。

    摘要取自**入队时冻结的指纹**，不是当前文件：同一路径被换掉时，
    输出名不会无声变化，用户能看出这条处理的是哪份素材。
    """
    stem = naming.safe_stem(item["video"])
    role = naming.safe_filename(item.get("role") or "目标角色", fallback="role")
    saved = (item.get("source") or {}).get("fingerprint") or ""
    suffix = saved[:8] or source_suffix(item["video"])
    return f"{stem}_{role}_{suffix}" if suffix else f"{stem}_{role}"


def output_path(item):
    """队列条目的成片路径：默认 output/，也可指定目录。"""
    out_dir = item.get("out_dir") or paths.OUTPUT
    return os.path.abspath(os.path.join(out_dir, f"{base_name(item)}_clip.mp4"))


def targets(item):
    """这一条会写出的全部文件：成片 + CSV + HTML 清单。"""
    clip = output_path(item)
    folder = os.path.dirname(clip)
    base = base_name(item)
    return {
        "video": clip,
        "csv": os.path.join(folder, f"{base}_清单.csv"),
        "html": os.path.join(folder, f"{base}_清单.html"),
    }


def conflicts(items):
    """找出会写入同一文件的条目。返回 [(路径, [(id, 视频名), ...]), ...]。"""
    seen = {}
    for item in items:
        for path in targets(item).values():
            seen.setdefault(os.path.normcase(path), []).append(
                (item.get("id") or "?", os.path.basename(item.get("video") or "")))
    return [(path, owners) for path, owners in seen.items() if len(owners) > 1]


def apply_out_dir(items, out_dir):
    """把队列级输出目录应用到所有未完成的条目，返回改动的条数。"""
    target = os.path.abspath(out_dir)
    changed = 0
    for item in items:
        if item.get("status") != DONE and item.get("out_dir") != target:
            item["out_dir"] = target
            changed += 1
    return changed


# ---------- 产物状态 ----------

def artifact_state(item):
    """三个目标的存在情况：complete / partial / missing。"""
    have = [os.path.exists(p) for p in targets(item).values()]
    if all(have):
        return "complete"
    return "partial" if any(have) else "missing"


def target_health(item):
    """检查已存在产物是否还可用。返回 (是否健康, 说明)。"""
    paths = targets(item)
    clip = paths["video"]
    if not os.path.exists(clip):
        return False, "成片不存在"
    info = media.probe(clip)
    if not info or float(info.get("duration") or 0) <= 0:
        return False, "成片无法播放"
    for key, label in (("csv", "CSV"), ("html", "HTML")):
        path = paths[key]
        if not os.path.exists(path):
            return False, f"{label} 清单不存在"
        if os.path.getsize(path) < REPORT_MIN_BYTES:
            return False, f"{label} 清单为空"
    return True, ""


def audit_done(items):
    """把"标记完成但产物没了"的条目退回等待。返回 [(条目, 原因)]。"""
    fixed = []
    for item in items:
        if item.get("status") != DONE:
            continue
        ok, why = target_health(item)
        if not ok:
            item["status"] = WAITING
            item["message"] = f"{why}，将重新生成"
            fixed.append((item, why))
    return fixed


def existing_targets(items):
    """列出会被覆盖的已有文件：[(条目, [路径...]), ...]。"""
    out = []
    for item in items:
        hit = [p for p in targets(item).values() if os.path.exists(p)]
        if hit:
            out.append((item, hit))
    return out


# ---------- 执行计划 ----------

def analysis_state(item):
    """返回 (是否需要分析, 原因)。音轨变了也算需要重新分析。"""
    video = item["video"]
    if not os.path.exists(video):
        return True, "源视频不存在"
    sid = (item.get("source") or {}).get("id") or manifest.source_id(video)
    rid = manifest.active_run(sid)
    if not rid:
        return True, "尚无分析结果"
    used = manifest.run_audio_track(sid, rid)
    want = int(item.get("audio_track", 0))
    if used != want:
        return True, f"音轨从第 {used + 1} 条改为第 {want + 1} 条"
    return False, f"复用运行 {rid}"


def _select_args(item, video):
    return ["--video", video,
            "--role", item.get("role") or "目标角色",
            "--keyword", item.get("keyword") or "",
            "--threshold", f"{float(item.get('threshold', 0.66)):.2f}",
            "--merge-gap", f"{float(item.get('merge_gap', 2.0)):.1f}",
            "--pad-head", f"{float(item.get('pad_head', 0.25)):.2f}",
            "--pad-tail", f"{float(item.get('pad_tail', 0.30)):.2f}"]


def _cut_args(item, video):
    return ["--video", video,
            "--scale", item.get("scale") or "native",
            "--crf", str(int(item.get("crf", 20))),
            "--vfr", item.get("vfr") or "cfr",
            "--output", output_path(item)]


def _report_args(item, video):
    return ["--video", video,
            "--out-dir", os.path.dirname(output_path(item)),
            "--base-name", base_name(item)]


def build_steps(item):
    """把条目展开成命令行步骤：[(脚本, 参数, 权重, 阶段名)]。"""
    video = item["video"]
    need, _why = analysis_state(item)
    steps = []
    if need:
        steps += [
            ("extract_audio.py", ["--video", video,
                                  "--audio-track", str(int(item.get("audio_track", 0)))],
             4, "抽取音频"),
            ("transcribe.py", ["--video", video,
                               "--model", item.get("model") or "large-v3",
                               "--device", item.get("device") or "cuda"],
             56, "日语转写"),
            ("voiceprint.py", ["--video", video], 14, "声纹提取"),
            ("select.py", _select_args(item, video), 1, "生成切点"),
            ("cut.py", _cut_args(item, video), 22, "切割拼接"),
            ("gen_report.py", _report_args(item, video), 3, "生成清单"),
        ]
    else:
        steps += [
            ("select.py", _select_args(item, video), 3, "生成切点"),
            ("cut.py", _cut_args(item, video), 90, "切割拼接"),
            ("gen_report.py", _report_args(item, video), 7, "生成清单"),
        ]
    return steps
