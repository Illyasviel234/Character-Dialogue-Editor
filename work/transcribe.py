"""日语语音转写。

每次分析都会新建一个 run，产物写入 runs/<run_id>/。
转写成功后只把状态标为 transcribed，必须等声纹阶段校验通过才算 complete。
"""

import os
import sys
import time

import numpy as np
from scipy.io import wavfile

from core import cli, manifest, paths

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HOME"] = paths.MODELS
os.environ["HF_HUB_DISABLE_XET"] = "1"

from faster_whisper import WhisperModel  # noqa: E402


def load_model(name, device):
    """优先用指定设备，失败自动回退 CPU。只读本地模型，不联网下载。"""
    if device == "cuda":
        try:
            m = WhisperModel(name, device="cuda", compute_type="float16",
                             download_root=paths.MODELS, local_files_only=True)
            return m, "cuda"
        except Exception as e:
            print(f"[warn] cuda 初始化失败，回退 CPU：{str(e)[:160]}", flush=True)
    m = WhisperModel(name, device="cpu", compute_type="int8",
                     download_root=paths.MODELS, local_files_only=True)
    return m, "cpu"


def transcribe_audio(model, audio):
    segments, info = model.transcribe(
        audio,
        language="ja",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=400, speech_pad_ms=120),
        condition_on_previous_text=False,
        word_timestamps=True,
        initial_prompt="これは日本語のアニメの会話です。",
    )

    rows = []
    for s in segments:
        words = [{"w": w.word, "s": round(w.start, 3), "e": round(w.end, 3),
                  "p": round(w.probability, 3)} for w in (s.words or [])]
        rows.append({
            "id": len(rows),
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "text": s.text.strip(),
            "avg_logprob": round(s.avg_logprob, 4),
            "no_speech_prob": round(s.no_speech_prob, 4),
            "words": words,
        })
        if len(rows) % 50 == 0:
            print(f"  ...{len(rows)} segments, t={s.end:.0f}s", flush=True)
    return rows, info


def main():
    p = cli.new_parser("日语语音转写（faster-whisper）")
    cli.add_video(p)
    p.add_argument("--model", default="large-v3", help="本地已安装的模型名")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--audio", default=None, help="16k 音频路径，默认取任务目录内的")
    args = p.parse_args()

    video = cli.resolve_video(args.video)
    sid, source, jdir, info = cli.prepare(video)

    audio_path = args.audio or os.path.join(jdir, "audio16k.wav")
    if not os.path.exists(audio_path):
        sys.exit("缺少 16k 音频，请先运行 extract_audio.py")

    sr, raw = wavfile.read(audio_path)
    if raw.ndim > 1:
        raw = raw.mean(axis=1)
    audio = raw.astype(np.float32) / 32768.0
    duration = len(audio) / float(sr or 16000)
    print(f"[ok] 音频 {duration:.1f}s", flush=True)

    run_id = manifest.new_run(sid, source, model=args.model, device=args.device)
    print(f"[run] {run_id}", flush=True)

    try:
        t0 = time.time()
        model, used = load_model(args.model, args.device)
        print(f"[ok] 模型 {args.model} 运行于 {used}", flush=True)

        rows, winfo = transcribe_audio(model, audio)
        if not rows:
            raise RuntimeError("没有识别到任何语段，请检查音频或换用更大的模型")

        payload = {
            "version": manifest.VERSION,
            "run_id": run_id,
            "source_id": sid,
            "source_name": source["name"],
            "model": args.model,
            "device": used,
            "duration": round(duration, 3),
            "language": winfo.language,
            "language_probability": round(winfo.language_probability, 4),
            "elapsed_sec": round(time.time() - t0, 1),
            "segments": rows,
        }
        manifest.atomic_write_json(
            manifest.run_artifact(sid, run_id, manifest.TRANSCRIPT), payload)
        manifest.update_run(sid, run_id, status="transcribed", device=used,
                            segments=len(rows))
    except Exception as e:
        manifest.fail_run(sid, run_id, e)
        sys.exit(f"转写失败：{e}")

    print(f"[done] {len(rows)} 段，耗时 {time.time() - t0:.1f}s", flush=True)
    print(f"[next] 继续运行 voiceprint.py 完成本次分析", flush=True)


if __name__ == "__main__":
    main()
