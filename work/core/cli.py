"""命令行公共部分：视频参数、任务目录解析。"""

import argparse
import os
import sys

from . import manifest, media, paths

VIDEO_EXT = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v")


def default_video():
    if not os.path.isdir(paths.INPUT):
        return None
    for name in sorted(os.listdir(paths.INPUT)):
        if name.lower().endswith(VIDEO_EXT):
            return os.path.join(paths.INPUT, name)
    return None


def add_video(parser):
    parser.add_argument("--video", default=None,
                        help="源视频路径，省略时取 input/ 下第一个视频")


def resolve_video(arg):
    video = arg or default_video()
    if not video:
        sys.exit("找不到视频文件，请用 --video 指定")
    if not os.path.exists(video):
        sys.exit(f"视频不存在：{video}")
    return os.path.abspath(video)


def prepare(video):
    """解析任务身份并创建目录，返回 (sid, source, jobdir, media_info)。"""
    info = media.probe(video)
    if not info:
        sys.exit(f"无法探测视频：{video}")
    source = manifest.make_source(video, info)
    sid = manifest.source_id(video)
    jdir = manifest.job_dir(sid)
    os.makedirs(jdir, exist_ok=True)
    return sid, source, jdir, info


def job_of(video):
    """只读地拿到任务目录，不创建。"""
    sid = manifest.source_id(video)
    return sid, manifest.job_dir(sid)


def new_parser(description):
    return argparse.ArgumentParser(description=description)
