"""媒体探测。导出前用它决定缩放、帧率和流映射，避免写死参数改变素材。"""

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


def probe(path):
    """返回时长、宽高、帧率、像素比、旋转角和音轨数量。探测失败返回空字典。"""
    if not path or not os.path.exists(path):
        return {}
    cmd = [FFPROBE, "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        data = json.loads(r.stdout or "{}")
    except Exception:
        return {}

    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    fmt = data.get("format", {})

    info = {
        "duration": _ratio(fmt.get("duration"), 0.0),
        "audio_streams": len(audio),
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "sar": "1:1",
        "rotation": 0,
    }
    if video:
        info["width"] = int(video.get("width") or 0)
        info["height"] = int(video.get("height") or 0)
        info["fps"] = round(_ratio(video.get("r_frame_rate")), 3)
        info["sar"] = video.get("sample_aspect_ratio") or "1:1"
        rot = video.get("tags", {}).get("rotate")
        if rot is None:
            for sd in video.get("side_data_list", []) or []:
                if "rotation" in sd:
                    rot = sd["rotation"]
                    break
        try:
            info["rotation"] = int(float(rot)) if rot is not None else 0
        except (TypeError, ValueError):
            info["rotation"] = 0
    return info


def describe(info):
    if not info:
        return "探测失败"
    w, h = info.get("width", 0), info.get("height", 0)
    return f"{w}x{h} · {info.get('fps', 0):.3f} fps · {info.get('audio_streams', 0)} 条音轨"
