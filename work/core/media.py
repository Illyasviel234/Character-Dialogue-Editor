"""媒体探测。导出前用它决定缩放、帧率和流映射，避免写死参数改变素材。

探测失败一律返回空字典。早期版本对内容为 "not media" 的伪 MP4 也返回
一堆零值字段，调用方因此以为探测成功，直到后面以"无音轨"之类的次级错误失败。
"""

import json
import os
import subprocess

from .paths import FFPROBE


def _ratio(text, default=0.0):
    try:
        if "/" in str(text):
            a, b = str(text).split("/")
            b = float(b)
            return float(a) / b if b else default
        return float(text)
    except (TypeError, ValueError):
        return default


def _rotation(video):
    rot = (video.get("tags") or {}).get("rotate")
    if rot is None:
        for sd in video.get("side_data_list") or []:
            if "rotation" in sd:
                rot = sd["rotation"]
                break
    try:
        return int(float(rot)) if rot is not None else 0
    except (TypeError, ValueError):
        return 0


def vfr_check(path, limit=120):
    """抽样帧时间戳，看帧间隔是否真的不均。

    r_frame_rate 与 avg_frame_rate 相等并不代表恒定帧率（两者都是估算），
    所以再读一批真实帧的 pts。判据是"偏离中位间隔超过 10% 的帧占比"：
    恒定帧率素材受时基舍入影响，间隔只会在 ±1 个 tick 内抖动（约 2%），
    真正可变帧率的素材会有成片的不同间隔。
    结果只作诊断，不做绝对判定。
    """
    cmd = [FFPROBE, "-v", "error", "-select_streams", "v:0",
           "-show_entries", "frame=pts_time", "-of", "csv=p=0",
           "-read_intervals", f"%+#{int(limit)}", path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return "unknown", 0.0, 0
        pts = []
        for line in r.stdout.splitlines():
            line = line.strip().rstrip(",")
            try:
                pts.append(float(line))
            except ValueError:
                continue
    except Exception:
        return "unknown", 0.0, 0

    deltas = [b - a for a, b in zip(pts, pts[1:]) if b > a]
    if len(deltas) < 10:
        return "unknown", 0.0, len(deltas)
    ordered = sorted(deltas)
    med = ordered[len(ordered) // 2]
    if med <= 0:
        return "unknown", 0.0, len(deltas)
    off = sum(1 for d in deltas if abs(d - med) / med > 0.10)
    ratio = off / len(deltas)
    return ("vfr" if ratio > 0.02 else "cfr"), ratio, len(deltas)


def probe(path, deep=False):
    """返回时长、宽高、帧率、音轨与章节信息。不是有效视频时返回空字典。

    deep=True 时额外抽样帧时间戳判断 VFR，约多花几十毫秒。
    """
    if not path or not os.path.exists(path):
        return {}
    cmd = [FFPROBE, "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", "-show_chapters", path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return {}
        data = json.loads(r.stdout or "{}")
    except Exception:
        return {}

    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        return {}
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    fmt = data.get("format") or {}

    duration = _ratio(fmt.get("duration")) or _ratio(video.get("duration"))
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    if duration <= 0 or width <= 0 or height <= 0:
        return {}

    fps = _ratio(video.get("r_frame_rate"))
    avg_fps = _ratio(video.get("avg_frame_rate"))
    is_vfr = bool(fps and avg_fps and abs(avg_fps - fps) / fps > 0.005)
    verdict, cv, samples = ("unknown", 0.0, 0)
    if deep:
        verdict, cv, samples = vfr_check(path)
        if verdict == "vfr":
            is_vfr = True
        elif verdict == "cfr":
            is_vfr = False

    info = {
        "duration": round(duration, 3),
        "width": width,
        "height": height,
        "fps": round(fps, 6),
        "fps_frac": video.get("r_frame_rate") or "",
        "avg_fps": round(avg_fps, 6),
        "is_vfr": is_vfr,
        "vfr_verdict": verdict,
        "vfr_off": round(cv, 4),
        "vfr_samples": samples,
        "sar": video.get("sample_aspect_ratio") or "1:1",
        "rotation": _rotation(video),
        "video_duration": round(_ratio(video.get("duration")) or duration, 3),
        "nb_frames": int(video.get("nb_frames") or 0),
        "audio_streams": len(audio),
        "audio_tracks": [
            {
                "index": i,
                "codec": a.get("codec_name") or "",
                "channels": int(a.get("channels") or 0),
                "language": (a.get("tags") or {}).get("language") or "",
            }
            for i, a in enumerate(audio)
        ],
        "chapters": len(data.get("chapters") or []),
        "data_streams": sum(1 for s in streams if s.get("codec_type") == "data"),
        "format": fmt.get("format_name") or "",
    }
    return info


def describe(info):
    if not info:
        return "探测失败"
    bits = [f"{info.get('width', 0)}x{info.get('height', 0)}",
            f"{info.get('fps', 0):.3f} fps"]
    if info.get("is_vfr"):
        bits.append(f"VFR(avg {info.get('avg_fps', 0):.3f})")
    bits.append(f"{info.get('audio_streams', 0)} 条音轨")
    if info.get("chapters"):
        bits.append(f"{info['chapters']} 章节")
    return " · ".join(bits)
