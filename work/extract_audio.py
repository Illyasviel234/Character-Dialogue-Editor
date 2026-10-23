"""抽取 16 kHz 单声道音频到任务目录。"""

import os
import subprocess
import sys

from core import cli, paths


def main():
    p = cli.new_parser("抽取 16 kHz 单声道音频，供声纹与转写使用")
    cli.add_video(p)
    args = p.parse_args()

    video = cli.resolve_video(args.video)
    sid, source, jdir, info = cli.prepare(video)

    if not info.get("audio_streams"):
        sys.exit(f"该视频没有音轨：{video}")

    out = os.path.join(jdir, "audio16k.wav")
    cmd = [paths.FFMPEG, "-y", "-v", "error", "-i", video,
           "-map", "0:a:0", "-ac", "1", "-ar", "16000",
           "-c:a", "pcm_s16le", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"抽音频失败：{r.stderr[-400:]}")

    print(f"[ok] {out}  {os.path.getsize(out) / 1048576:.1f} MB")
    print(f"[job] {sid}")


if __name__ == "__main__":
    main()
