"""音频切片与取帧，供界面试听使用。

不依赖 QtMultimedia：PySide6-Essentials 里没有这个模块，装 Addons 要多占
300 MB 以上。从任务目录的 16 kHz 音频里切一句用标准库 wave 就够了，
画面另取一帧确认场景。
"""

import os
import subprocess
import wave

from .paths import FFMPEG


def slice_wav(src, start, end, dst, head=0.0, tail=0.0):
    """把 [start, end] 从 16 kHz WAV 里切出来，返回 (路径, 实际时长)。"""
    start, end = float(start), float(end)
    with wave.open(src, "rb") as r:
        rate = r.getframerate()
        nch = r.getnchannels()
        width = r.getsampwidth()
        if width != 2:
            raise ValueError(f"只支持 16 位 WAV，当前 {width * 8} 位")
        total = r.getnframes()
        dur = total / rate if rate else 0.0
        a = max(0.0, start - head)
        b = min(dur, end + tail)
        if b <= a:
            raise ValueError("片段区间为空")
        r.setpos(int(a * rate))
        data = r.readframes(max(1, int(round((b - a) * rate))))

    parent = os.path.dirname(dst)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with wave.open(dst, "wb") as w:
        w.setnchannels(nch)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(data)
    return dst, len(data) / (nch * width * rate)


def extract_wav(video, start, end, dst, audio_track=0):
    """源视频没有预抽音频时，直接从视频截一段。"""
    parent = os.path.dirname(dst)
    if parent:
        os.makedirs(parent, exist_ok=True)
    cmd = [FFMPEG, "-y", "-v", "error", "-ss", f"{float(start):.3f}",
           "-t", f"{max(0.05, float(end) - float(start)):.3f}", "-i", video,
           "-map", f"0:a:{int(audio_track)}", "-ac", "1", "-ar", "16000",
           "-c:a", "pcm_s16le", dst]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError("抽取试听音频失败")
    return dst


def grab_frame(video, ts, dst, width=320):
    """取指定时间点的一帧，缩到 width 宽。失败返回 None。"""
    parent = os.path.dirname(dst)
    if parent:
        os.makedirs(parent, exist_ok=True)
    cmd = [FFMPEG, "-y", "-v", "error", "-ss", f"{max(0.0, float(ts)):.3f}",
           "-i", video, "-frames:v", "1", "-vf", f"scale={int(width)}:-2", dst]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) < 100:
        return None
    return dst
