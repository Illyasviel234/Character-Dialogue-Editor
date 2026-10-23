"""环境自检。

每项检查独立捕获异常，最后输出完整清单并返回退出码。
探针音频从源视频临时截取，不依赖仓库里的固定文件。
"""

import json
import os
import subprocess
import sys
import tempfile

from core import cli, manifest, media, paths

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HOME"] = paths.MODELS
os.environ["HF_HUB_DISABLE_XET"] = "1"

RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append(("PASS", name, detail or ""))
    except Exception as e:
        RESULTS.append(("FAIL", name, f"{type(e).__name__}: {str(e)[:150]}"))


def skip(name, why):
    RESULTS.append(("SKIP", name, why))


def make_probe(video):
    out = os.path.join(tempfile.gettempdir(), "clip_probe.wav")
    info = media.probe(video)
    start = min(60, max(0, info.get("duration", 0) / 3))
    cmd = [paths.FFMPEG, "-y", "-v", "error", "-ss", f"{start:.1f}", "-t", "5",
           "-i", video, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out]
    subprocess.run(cmd, capture_output=True)
    if not os.path.exists(out) or os.path.getsize(out) < 1000:
        raise RuntimeError("探针音频生成失败")
    return out


def main():
    video = cli.default_video()
    if video is None:
        video = None

    def c_paths():
        need = [paths.WORK, paths.OUTPUT, paths.INPUT, paths.MODELS]
        missing = [p for p in need if not os.path.exists(p)]
        if missing:
            raise RuntimeError("缺少目录：" + "、".join(os.path.basename(m) for m in missing))
        return f"根目录 {os.path.basename(paths.ROOT)}"

    def c_ffmpeg():
        r = subprocess.run([paths.FFMPEG, "-version"], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("ffmpeg 无法执行")
        return r.stdout.splitlines()[0].split(" Copyright")[0]

    def c_model_files():
        w = os.path.join(paths.MODELS, "models--Systran--faster-whisper-large-v3")
        e = os.path.join(paths.MODELS, "spkrec-ecapa-voxceleb")
        miss = [p for p in (w, e) if not os.path.isdir(p)]
        if miss:
            raise RuntimeError("缺少模型：" + "、".join(os.path.basename(m) for m in miss))
        return "whisper + ECAPA 均已就位"

    def c_probe_media():
        if not video:
            raise RuntimeError("input/ 下没有视频")
        info = media.probe(video)
        if not info:
            raise RuntimeError("探测失败")
        return media.describe(info)

    def c_whisper():
        from transcribe import load_model
        probe = make_probe(video)
        model, device = load_model("large-v3", "cuda")
        from scipy.io import wavfile
        import numpy as np
        sr, raw = wavfile.read(probe)
        a = raw.astype(np.float32) / 32768.0
        segs, _ = model.transcribe(a, language="ja", beam_size=1)
        txt = "".join(s.text for s in segs)
        os.remove(probe)
        return f"{device} · {len(txt)} 字符"

    def c_ecapa():
        from voiceprint import load_encoder
        import torch
        from scipy.io import wavfile
        import numpy as np
        probe = make_probe(video)
        enc = load_encoder()
        sr, raw = wavfile.read(probe)
        a = raw.astype(np.float32) / 32768.0
        with torch.no_grad():
            v = enc.encode_batch(torch.from_numpy(a).unsqueeze(0))
        os.remove(probe)
        return f"向量维度 {tuple(v.shape)[-1]}"

    def c_plan():
        if not video:
            raise RuntimeError("没有视频")
        sid, jdir = cli.job_of(video)
        sel = os.path.join(jdir, manifest.SELECTED)
        if not os.path.exists(sel):
            return "尚未生成切点方案（可忽略）"
        with open(sel, encoding="utf-8") as f:
            d = json.load(f)
        segs = d.get("segments") or []
        bad = [s for s in segs if s["end"] <= s["start"] or s["start"] < 0]
        if bad:
            raise RuntimeError(f"{len(bad)} 个片段时间非法")
        return f"{len(segs)} 段，{d.get('total_sec')} 秒"

    check("目录结构", c_paths)
    check("ffmpeg", c_ffmpeg)
    check("模型文件", c_model_files)
    check("源视频探测", c_probe_media)
    if video:
        check("whisper 转写", c_whisper)
        check("ECAPA 声纹", c_ecapa)
        check("切点方案", c_plan)
    else:
        skip("whisper 转写", "没有源视频")
        skip("ECAPA 声纹", "没有源视频")
        skip("切点方案", "没有源视频")

    print("=== 环境自检 ===")
    for status, name, detail in RESULTS:
        print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))

    fails = sum(1 for s, _, _ in RESULTS if s == "FAIL")
    print(f"\n{len(RESULTS)} 项检查，{fails} 项失败")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
