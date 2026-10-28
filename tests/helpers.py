"""测试共用的小工具。

run 校验会读 WAV 头，所以测试里不能再拿几个字节冒充音频，
统一在这里生成合法的 16 kHz 单声道 WAV。
"""

import os
import subprocess
import wave

import numpy as np

from core import paths


def write_silence(path, seconds=0.2, rate=16000):
    """写一段合法的 16 kHz / 单声道 / 16 位 WAV。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = max(1, int(seconds * rate))
    data = (np.sin(np.arange(n) / 20.0) * 4000).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data.tobytes())
    return path


def write_bad_wav(path):
    """只有 RIFF 头、没有合法格式块的坏文件。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"RIFF" + b"\0" * 64)
    return path


def write_tiny_video(path, seconds=0.3):
    """用 ffmpeg 生成一个能通过探测的极小视频。失败返回 None。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = subprocess.run([
        paths.FFMPEG, "-y", "-v", "error",
        "-f", "lavfi", "-i", f"testsrc=size=64x48:rate=10:duration={seconds}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", path,
    ], capture_output=True)
    return path if r.returncode == 0 else None


def write_text(path, text="x" * 200):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path
