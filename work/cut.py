"""按切点方案切割拼接。

采用单次 filter graph 编码：所有片段作为独立输入 seek 到各自位置，
经 setpts/asetpts 归零后在 concat filter 里拼接，只做一次 H.264/AAC 编码。
这样不会出现「每段分别编码再流复制拼接」时的 AAC 延迟与时间戳舍入累积。

默认保持源视频的分辨率、帧率和宽高比，只在显式指定 --scale 时才缩放，
且缩放过程保持画面比例、用黑边补齐。

导出前会校验切点方案是否属于当前活动运行，并拒绝把结果写回源视频路径。
源文件的章节与元数据一律不继承，避免输出里混入无关的数据轨。
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


def resolve_output(output, stem, role, out_dir=None):
    """裸文件名也要落到绝对路径，否则 os.makedirs("") 会抛 FileNotFoundError。"""
    if output:
        return os.path.abspath(output)
    return os.path.abspath(os.path.join(out_dir or paths.OUTPUT,
                                      f"{stem}_{role}_clip.mp4"))


def check_output(video, out_path):
    """禁止输出覆盖源视频。

    编码结果先写 .tmp.mp4 再 os.replace，一旦目标路径就是源视频，
    用户的原片会被静默替换成剪辑版，且无法恢复。
    """
    src = os.path.normcase(os.path.abspath(video))
    dst = os.path.normcase(os.path.abspath(out_path))
    if src == dst:
        raise ValueError("输出路径不能与源视频相同，否则会覆盖原片")
    return out_path


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


def build_command(video, segs, out, scale, crf, preset, has_audio, fps_frac="",
                  audio_index=0, fps_mode="cfr"):
    """fps_mode=cfr 时强制恒定帧率；preserve 时保留源时间戳（VFR 素材用）。"""
    inputs = []
    for s in segs:
        inputs += ["-ss", f"{s['start']:.3f}", "-t", f"{s['dur']:.3f}", "-i", video]

    vf = scale_filter(scale)
    chains = []
    for i in range(len(segs)):
        chains.append(f"[{i}:v]{vf}setpts=PTS-STARTPTS[v{i}]")
        if has_audio:
            chains.append(f"[{i}:a:{int(audio_index)}]asetpts=PTS-STARTPTS[a{i}]")

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
        # 源视频的章节在 MP4 里会变成一条 bin_data 数据轨，与剪辑内容无关，
        # 还可能带出原标题与章节名，因此全部丢弃。
        "-map_metadata", "-1",
        "-map_chapters", "-1",
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
        "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
    ]
    if fps_mode == "preserve":
        cmd += ["-fps_mode", "vfr"]
    else:
        cmd += ["-fps_mode", "cfr"]
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
    p.add_argument("--output", default=None, help="成片输出路径（文件）")
    p.add_argument("--out-dir", default=None,
                   help="成片输出目录，默认 output/；指定 --output 时忽略")
    p.add_argument("--audio-track", type=int, default=None,
                   help="必须与分析所用音轨一致；换音轨请重新分析")
    p.add_argument("--vfr", default="cfr", choices=("cfr", "preserve"),
                   help="cfr 转恒定帧率（默认，时长可控）；preserve 保留源时间戳")
    args = p.parse_args()

    video = cli.resolve_video(args.video)
    sid, source, jdir, info = cli.prepare(video)

    sel_path = os.path.join(jdir, manifest.SELECTED)
    if not os.path.exists(sel_path):
        sys.exit("缺少切点方案，请先运行 select.py")
    with open(sel_path, encoding="utf-8") as f:
        sel = json.load(f)
    ok, why = manifest.validate_selection(sid, sel, source)
    if not ok:
        sys.exit(f"切点方案不可用：{why}")
    segs = sel.get("segments") or []

    role = naming.safe_filename(sel.get("role") or "目标角色", fallback="clip")
    stem = naming.safe_stem(source["name"])
    out_path = resolve_output(args.output, stem, role, args.out_dir)
    try:
        check_output(video, out_path)
    except ValueError as e:
        sys.exit(str(e))
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    src = media.probe(video)
    has_audio = bool(src.get("audio_streams"))
    streams = int(src.get("audio_streams") or 0)
    # 音轨由活动运行决定：时间码和声纹都来自那条音轨，导出必须跟着它
    track = manifest.run_audio_track(sid)
    if args.audio_track is not None and int(args.audio_track) != track:
        sys.exit(f"--audio-track {args.audio_track} 与分析所用音轨 {track} 不一致。\n"
                 f"换音轨必须重新分析（extract_audio.py --audio-track {args.audio_track}），"
                 f"否则时间码与声纹来自另一条音轨。")
    if has_audio and not 0 <= track < streams:
        sys.exit(f"音轨 {track} 不存在，该视频共 {streams} 条（0–{streams - 1}）")

    if args.vfr == "cfr":
        segs = snap_to_frames(segs, src.get("fps", 0))
    planned = round(sum(s["dur"] for s in segs), 3)
    frames = sum(s.get("frames", 0) for s in segs)

    print(f"[src] {media.describe(src)}")
    if src.get("is_vfr"):
        note = ("已按 r_frame_rate 转恒定帧率" if args.vfr == "cfr"
                else "保留源时间戳，不做帧率转换")
        print(f"[warn] 源为可变帧率（r={src.get('fps')} / avg={src.get('avg_fps')}），{note}")
    if args.vfr == "preserve":
        print("[note] 保留时间戳模式不吸附帧网格，时长偏差可能超过一帧")
    if has_audio:
        print(f"[audio] 使用第 {track + 1}/{streams} 条音轨")
    if frames:
        print(f"[plan] {len(segs)} 段 / {frames} 帧，计划 {planned:.3f}s")
    else:
        print(f"[plan] {len(segs)} 段，计划 {planned:.3f}s")
    print(f"[out] {out_path}")

    tmp_out = out_path + ".tmp.mp4"
    cmd = build_command(video, segs, tmp_out, args.scale, args.crf, args.preset,
                        has_audio, src.get("fps_frac") or "",
                        audio_index=track, fps_mode=args.vfr)
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
        hint = "（保留时间戳模式属预期）" if args.vfr == "preserve" else ""
        print(f"[warn] 时长偏差超过 50 ms{hint}，请检查素材是否为可变帧率")


if __name__ == "__main__":
    main()
