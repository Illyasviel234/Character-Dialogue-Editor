"""抽取 16 kHz 单声道音频到任务目录。

多音轨素材可以指定 --audio-track。用了哪一条会写进 audio.json，
转写时再记录到本次运行里，避免"换了音轨但沿用旧转写"这种隐性不一致。
"""

import os
import subprocess
import sys

from core import cli, manifest, paths


def main():
    p = cli.new_parser("抽取 16 kHz 单声道音频，供声纹与转写使用")
    cli.add_video(p)
    p.add_argument("--audio-track", type=int, default=0,
                   help="第几条音轨（从 0 开始），默认第 1 条")
    args = p.parse_args()

    video = cli.resolve_video(args.video)
    sid, source, jdir, info = cli.prepare(video)

    streams = int(info.get("audio_streams") or 0)
    if not streams:
        sys.exit(f"该视频没有音轨：{video}")
    track = int(args.audio_track)
    if not 0 <= track < streams:
        sys.exit(f"音轨 {track} 不存在，该视频共 {streams} 条（0–{streams - 1}）")

    out = os.path.join(jdir, "audio16k.wav")
    cmd = [paths.FFMPEG, "-y", "-v", "error", "-i", video,
           "-map", f"0:a:{track}", "-ac", "1", "-ar", "16000",
           "-c:a", "pcm_s16le", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"抽音频失败：{r.stderr[-400:]}")

    duration = 0.0
    try:
        from scipy.io import wavfile
        sr, raw = wavfile.read(out)
        duration = len(raw) / float(sr or 16000)
    except Exception:
        pass

    manifest.save_audio_info(sid, sid, track, duration or source["duration"], streams)

    label = ""
    tracks = info.get("audio_tracks") or []
    if track < len(tracks):
        t = tracks[track]
        label = f" · {t.get('codec') or '?'} {t.get('channels')}ch"
        if t.get("language"):
            label += f" {t['language']}"
    print(f"[ok] {out}  {os.path.getsize(out) / 1048576:.1f} MB")
    print(f"[track] 第 {track + 1}/{streams} 条音轨{label}")
    print(f"[job] {sid}")


if __name__ == "__main__":
    main()
