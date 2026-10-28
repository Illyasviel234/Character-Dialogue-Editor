"""批量任务队列：把「分析 → 筛选 → 导出 → 清单」串成一条流水线。

这里只描述"要做什么"，不负责执行。执行由界面里的 Runner 或命令行驱动，
所以队列本身可以存成 JSON，关掉程序下次接着跑。

每个条目自带全部参数（角色、特征词、阈值、留白、音轨、输出规格），
不依赖界面上当前的值——否则改一次阈值会把队列里已排好的任务也改掉。

条目带一个稳定的 id：运行中调整顺序时，完成回调按 id 回写，
不会因为数组下标变化把结果记到别的任务上。
"""

import json
import os
import uuid

from . import manifest, naming, paths

QUEUE_FILE = "queue.json"

WAITING = "等待"
RUNNING = "进行中"
DONE = "完成"
FAILED = "失败"
CANCELLED = "已取消"

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


def queue_path():
    return os.path.join(paths.WORK, QUEUE_FILE)


def new_id():
    return uuid.uuid4().hex[:8]


def make_item(video, **kw):
    item = dict(FIELDS)
    item.update(kw)
    item["video"] = os.path.abspath(video) if video else ""
    item["id"] = kw.get("id") or new_id()
    item["status"] = WAITING
    item["message"] = ""
    return item


def load(path=None):
    """读回队列，坏文件按空队列处理（队列丢了不影响已有产物）。"""
    path = path or queue_path()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    out = []
    for raw in items:
        if not isinstance(raw, dict) or not raw.get("video"):
            continue
        item = make_item(raw["video"])
        item.update({k: raw[k] for k in list(FIELDS) + ["video", "id"] if k in raw})
        if not item.get("id"):
            item["id"] = new_id()
        item["status"] = WAITING if raw.get("status") in (RUNNING, CANCELLED) else raw.get("status", WAITING)
        item["message"] = raw.get("message", "")
        out.append(item)
    return out


def save(items, path=None):
    path = path or queue_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"version": manifest.VERSION, "items": items}, f,
                  ensure_ascii=False, indent=1)
    os.replace(tmp, path)


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

    带上源指纹短摘要：`D:/s1/same.mp4` 和 `E:/s2/same.mp4` 角色相同时，
    只靠 stem + 角色名会算出同一个输出路径，后跑的会静默覆盖前一个。
    """
    stem = naming.safe_stem(item["video"])
    role = naming.safe_filename(item.get("role") or "目标角色", fallback="role")
    suffix = source_suffix(item["video"])
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


def analysis_state(item):
    """返回 (是否需要分析, 原因)。音轨变了也算需要重新分析。"""
    video = item["video"]
    if not os.path.exists(video):
        return True, "源视频不存在"
    sid = manifest.source_id(video)
    rid = manifest.active_run(sid)
    if not rid:
        return True, "尚无分析结果"
    used = manifest.run_audio_track(sid, rid)
    want = int(item.get("audio_track", 0))
    if used != want:
        return True, f"音轨从第 {used + 1} 条改为第 {want + 1} 条"
    return False, f"复用运行 {rid}"


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
        ]
        steps += [
            ("select.py", ["--video", video,
                           "--role", item.get("role") or "目标角色",
                           "--keyword", item.get("keyword") or "",
                           "--threshold", f"{float(item.get('threshold', 0.66)):.2f}",
                           "--merge-gap", f"{float(item.get('merge_gap', 2.0)):.1f}",
                           "--pad-head", f"{float(item.get('pad_head', 0.25)):.2f}",
                           "--pad-tail", f"{float(item.get('pad_tail', 0.30)):.2f}"],
             1, "生成切点"),
            ("cut.py", ["--video", video,
                        "--scale", item.get("scale") or "native",
                        "--crf", str(int(item.get("crf", 20))),
                        "--vfr", item.get("vfr") or "cfr",
                        "--output", output_path(item)],
             22, "切割拼接"),
            ("gen_report.py", ["--video", video,
                               "--out-dir", os.path.dirname(output_path(item)),
                               "--base-name", base_name(item)],
             3, "生成清单"),
        ]
    else:
        steps += [
            ("select.py", ["--video", video,
                           "--role", item.get("role") or "目标角色",
                           "--keyword", item.get("keyword") or "",
                           "--threshold", f"{float(item.get('threshold', 0.66)):.2f}",
                           "--merge-gap", f"{float(item.get('merge_gap', 2.0)):.1f}",
                           "--pad-head", f"{float(item.get('pad_head', 0.25)):.2f}",
                           "--pad-tail", f"{float(item.get('pad_tail', 0.30)):.2f}"],
             3, "生成切点"),
            ("cut.py", ["--video", video,
                        "--scale", item.get("scale") or "native",
                        "--crf", str(int(item.get("crf", 20))),
                        "--vfr", item.get("vfr") or "cfr",
                        "--output", output_path(item)],
             90, "切割拼接"),
            ("gen_report.py", ["--video", video,
                               "--out-dir", os.path.dirname(output_path(item)),
                               "--base-name", base_name(item)],
             7, "生成清单"),
        ]
    return steps
