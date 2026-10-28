"""抽取 16 kHz 单声道音频到本次运行目录。

多音轨素材可以指定 --audio-track。音频放进 runs/<run_id>/ 而不是任务目录，
这样新分析失败或取消时，旧活动 run 的音频、转写、声纹、试听全部保持不变。

抽音频是分析的第一个阶段，所以由它创建运行；转写和声纹接着这个运行做。
"""

import os
import subprocess
import sys

from core import cli, manifest, paths


def probe_wav(path):
    """返回 (采样率, 时长)。用 scipy 读一遍，顺便确认文件没被写坏。"""
    from scipy.io import wavfile
    sr, raw = wavfile.read(path)
    if not len(raw):
        raise RuntimeError("音频为空")
    return int(sr), len(raw) / float(sr)


def main():
    p = cli.new_parser("抽取 16 kHz 单声道音频，供声纹与转写使用")
    cli.add_video(p)
    p.add_argument("--audio-track", type=int, default=0,
                   help="第几条音轨（从 0 开始），默认第 1 条")
    p.add_argument("--run", default=None, help="接续已有运行，默认新建一个")
    args = p.parse_args()

    video = cli.resolve_video(args.video)
    sid, source, jdir, info = cli.prepare(video)

    streams = int(info.get("audio_streams") or 0)
    if not streams:
        sys.exit(f"该视频没有音轨：{video}")
    track = int(args.audio_track)
    if not 0 <= track < streams:
        sys.exit(f"音轨 {track} 不存在，该视频共 {streams} 条（0–{streams - 1}）")

    run_id = args.run or manifest.new_run(sid, source, audio_track=track)
    out = manifest.run_artifact(sid, run_id, manifest.AUDIO_WAV)
    tmp = out + ".tmp.wav"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    cmd = [paths.FFMPEG, "-y", "-v", "error", "-i", video,
           "-map", f"0:a:{track}", "-ac", "1", "-ar", "16000",
           "-c:a", "pcm_s16le", tmp]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(tmp):
        manifest.fail_run(sid, run_id, "抽音频失败")
        sys.exit(f"抽音频失败：{r.stderr[-400:]}")

    try:
        sr, duration = probe_wav(tmp)
    except Exception as e:
        manifest.fail_run(sid, run_id, f"音频无效：{e}")
        sys.exit(f"音频无效：{e}")

    # 探测通过才替换正式文件，中途失败不会留下半截 WAV
    os.replace(tmp, out)
    manifest.save_audio_info(sid, run_id, sid, track, duration, streams)
    manifest.update_run(sid, run_id, audio_ready=True)

    label = ""
    tracks = info.get("audio_tracks") or []
    if track < len(tracks):
        t = tracks[track]
        label = f" · {t.get('codec') or '?'} {t.get('channels')}ch"
        if t.get("language"):
            label += f" {t['language']}"
    print(f"[ok] {out}  {os.path.getsize(out) / 1048576:.1f} MB  {sr} Hz  {duration:.1f}s")
    print(f"[track] 第 {track + 1}/{streams} 条音轨{label}")
    print(f"[run] {run_id}")
    print("[next] 继续运行 transcribe.py 完成本次分析")


if __name__ == "__main__":
    main()
