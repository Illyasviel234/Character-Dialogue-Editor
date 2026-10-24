"""按切点方案切割拼接。

采用单次 filter graph 编码：所有片段作为独立输入 seek 到各自位置，
经 setpts/asetpts 归零后在 concat filter 里拼接，只做一次 H.264/AAC 编码。
这样不会出现「每段分别编码再流复制拼接」时的 AAC 延迟与时间戳舍入累积。

默认保持源视频的分辨率、帧率和宽高比，只在显式指定 --scale 时才缩放，
且缩放过程保持画面比例、用黑边补齐。
"""

import json
import os
import subprocess
import sys

from core import cli, manifest, media, naming, paths

PRESETS = ("ultrafast", "superfast", "veryfast", "faster", "fast",
           "medium", "slow", "slower")


def scale_filter(target):
    if not target or target == "native":
        return ""
    if ":" not in target:
        sys.exit("--scale 格式应为 宽:高，例如 1920:1080，或写 native")
    w, h = target.split(":")
    return (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,")


def resolve_output(output, stem, role):
    """裸文件名也要落到绝对路径，否则 os.makedirs("") 会抛 FileNotFoundError。"""
    if output:
        return os.path.abspath(output)
    return os.path.join(paths.OUTPUT, f"{stem}_{role}_clip.mp4")


def snap_to_frames(segs, fps):
    """把切点吸附到帧网格。

    concat filter 拼接视频流时每段必须是整数帧，任意秒数的切点会让每段各自
    凑整，误差随片段数累积。提前对齐到帧，总时长误差就能压在一帧以内。
    """
    if not fps or fps <= 0:
        return segs
    out = []
    for s in segs:
        sf = int(round(s["start"] * fps))
        ef = int(round(s["end"] * fps))
        if ef <= sf:
            ef = sf + 1
        out.append({**s, "start": sf / fps, "end": ef / fps,
                    "dur": (ef - sf) / fps, "frames": ef - sf})
    return out


def build_command(video, segs, out, scale, crf, preset, has_audio, fps_frac=""):
    inputs = []
    for s in segs:
        inputs += ["-ss", f"{s['start']:.3f}", "-t", f"{s['dur']:.3f}", "-i", video]

    vf = scale_filter(scale)
    chains = []
    for i in range(len(segs)):
        chains.append(f"[{i}:v]{vf}setpts=PTS-STARTPTS[v{i}]")
        if has_audio:
            chains.append(f"[{i}:a]asetpts=PTS-STARTPTS[a{i}]")

    if has_audio:
        joined = "".join(f"[v{i}][a{i}]" for i in range(len(segs)))
        chains.append(f"{joined}concat=n={len(segs)}:v=1:a=1[vout][aout]")
    else:
        joined = "".join(f"[v{i}]" for i in range(len(segs)))
        chains.append(f"{joined}concat=n={len(segs)}:v=1:a=0[vout]")

    cmd = [paths.FFMPEG, "-y", "-v", "error"] + inputs + [
        "-filter_complex", ";".join(chains),
        "-map", "[vout]",
    ]
    if has_audio:
        cmd += ["-map", "[aout]"]
    cmd += [
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
        "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-fps_mode", "cfr", "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
    ]
    if fps_frac:
        cmd += ["-r", fps_frac]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
    cmd.append(out)
    return cmd


def main():
    p = cli.new_parser("按切点方案切割拼接成片")
    cli.add_video(p)
    p.add_argument("--scale", default="native",
                   help="输出尺寸，默认 native 保持源规格；也可写 1920:1080")
    p.add_argument("--crf", type=int, default=20)
    p.add_argument("--preset", default="medium", choices=PRESETS)
    p.add_argument("--output", default=None, help="输出文件路径")
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

    role = naming.safe_filename(sel.get("role") or "目标角色", fallback="clip")
    stem = naming.safe_stem(source["name"])
    out_path = resolve_output(args.output, stem, role)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    src = media.probe(video)
    has_audio = bool(src.get("audio_streams"))
    segs = snap_to_frames(segs, src.get("fps", 0))
    planned = round(sum(s["dur"] for s in segs), 3)
    frames = sum(s.get("frames", 0) for s in segs)

    print(f"[src] {media.describe(src)}")
    print(f"[plan] {len(segs)} 段 / {frames} 帧，计划 {planned:.3f}s")
    print(f"[out] {out_path}")

    tmp_out = out_path + ".tmp.mp4"
    cmd = build_command(video, segs, tmp_out, args.scale, args.crf, args.preset,
                        has_audio, src.get("fps_frac") or "")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(tmp_out):
        print(f"[fail] 编码失败：{r.stderr[-600:]}", flush=True)
        sys.exit(1)

    os.replace(tmp_out, out_path)

    got = media.probe(out_path)
    # 用视频流时长而不是容器时长：MP4 的 B 帧编辑列表会让容器时长多出起始偏移
    actual = float(got.get("video_duration") or got.get("duration") or 0)
    drift = actual - planned
    size = os.path.getsize(out_path) / 1048576
    print(f"[done] {out_path}  {size:.1f} MB")
    print(f"[check] 视频流 {actual:.3f}s / 计划 {planned:.3f}s，"
          f"偏差 {drift * 1000:+.0f} ms，{got.get('nb_frames', 0)} 帧")
    if abs(drift) > 0.05:
        print("[warn] 时长偏差超过 50 ms，请检查素材是否为可变帧率")


if __name__ == "__main__":
    main()
