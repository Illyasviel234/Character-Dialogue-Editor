"""按切点方案切割拼接。

默认保持源视频的分辨率、帧率和宽高比，只在显式指定 --scale 时才缩放，
且缩放过程保持画面比例、用黑边补齐，不会拉伸。
"""

import json
import os
import shutil
import subprocess
import sys

from core import cli, manifest, media, paths

PRESETS = ("ultrafast", "superfast", "veryfast", "faster", "fast",
           "medium", "slow", "slower")


def scale_filter(target):
    """target 为 native 时不做任何缩放。"""
    if not target or target == "native":
        return []
    if ":" not in target:
        sys.exit("--scale 格式应为 宽:高，例如 1920:1080，或写 native")
    w, h = target.split(":")
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1")
    return ["-vf", vf]


def encode_part(video, seg, out, vf, crf, preset):
    dur = seg["end"] - seg["start"]
    cmd = [paths.FFMPEG, "-y", "-v", "error",
           "-ss", f"{seg['start']:.3f}", "-i", video,
           "-t", f"{dur:.3f}",
           "-map", "0:v:0", "-map", "0:a:0?"] + vf + [
           "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
           "-profile:v", "high", "-pix_fmt", "yuv420p",
           "-fps_mode", "cfr", "-avoid_negative_ts", "make_zero",
           "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
           out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stderr


def main():
    p = cli.new_parser("按切点方案切割拼接成片")
    cli.add_video(p)
    p.add_argument("--scale", default="native",
                   help="输出尺寸，默认 native 保持源规格；也可写 1920:1080")
    p.add_argument("--crf", type=int, default=20)
    p.add_argument("--preset", default="medium", choices=PRESETS)
    p.add_argument("--output", default=None, help="输出文件路径")
    p.add_argument("--keep-parts", action="store_true", help="保留中间片段")
    args = p.parse_args()

    video = cli.resolve_video(args.video)
    sid, source, jdir, info = cli.prepare(video)

    sel_path = os.path.join(jdir, manifest.SELECTED)
    if not os.path.exists(sel_path):
        sys.exit("缺少切点方案，请先运行 select.py")
    with open(sel_path, encoding="utf-8") as f:
        sel = json.load(f)
    segs = sel.get("segments") or []
    if not segs:
        sys.exit("切点方案为空")

    role = sel.get("role") or "目标角色"
    stem = os.path.splitext(source["name"])[0]
    out_path = args.output or os.path.join(paths.OUTPUT, f"{stem}_{role}_clip.mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    vf = scale_filter(args.scale)
    parts_dir = os.path.join(paths.WORK, "parts", sid)
    if os.path.isdir(parts_dir):
        shutil.rmtree(parts_dir, ignore_errors=True)
    os.makedirs(parts_dir, exist_ok=True)

    src = media.probe(video)
    print(f"[src] {media.describe(src)}")
    print(f"[out] {out_path}")

    parts = []
    try:
        for i, s in enumerate(segs):
            part = os.path.join(parts_dir, f"p{i:03d}.mp4")
            code, err = encode_part(video, s, part, vf, args.crf, args.preset)
            if code != 0 or not os.path.exists(part):
                print(f"[fail] 第 {i} 段：{err[-400:]}", flush=True)
                sys.exit(1)
            parts.append(part)
            print(f"[ok] {i:03d} {s['start']:.2f}-{s['end']:.2f} "
                  f"({s['end'] - s['start']:.2f}s)", flush=True)

        listfile = os.path.join(parts_dir, "list.txt")
        with open(listfile, "w", encoding="utf-8") as f:
            for part in parts:
                f.write(f"file '{part.replace(os.sep, '/')}'\n")

        tmp_out = out_path + ".tmp.mp4"
        cmd = [paths.FFMPEG, "-y", "-v", "error",
               "-f", "concat", "-safe", "0", "-i", listfile,
               "-c", "copy", "-movflags", "+faststart", tmp_out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(tmp_out):
            print(f"[fail] 拼接失败：{r.stderr[-400:]}", flush=True)
            sys.exit(1)

        os.replace(tmp_out, out_path)
    finally:
        if not args.keep_parts:
            shutil.rmtree(parts_dir, ignore_errors=True)

    size = os.path.getsize(out_path) / 1048576
    print(f"[done] {out_path}  {size:.1f} MB")


if __name__ == "__main__":
    main()
