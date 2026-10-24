"""把旧版任务目录升级到 v4（完整文件指纹）。

v3 的任务 ID 来自"大小 + 首尾各 1 MiB"的采样哈希，v4 改成整个文件的
BLAKE2b。算法一变，同一部片子算出的 ID 就不同，旧目录会变成孤儿。

本脚本按清单里记录的源路径重新计算指纹，确认文件大小与时长都没变之后，
把任务目录改名、清单升版，并把 runs/*/transcript.json 和 selected.json
里的 source_id 一并改写。任何一项对不上就跳过，不做猜测。

用法：
    python migrate_jobs.py --dry-run     # 只看会改什么
    python migrate_jobs.py               # 实际执行
"""

import json
import os
import shutil
import sys

from core import cli, manifest, media, paths


def load_raw(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def plan_one(jdir):
    """返回 (old_sid, new_sid, data, info) 或 (old_sid, None, 原因, None)。"""
    old_sid = os.path.basename(jdir)
    data = load_raw(os.path.join(jdir, manifest.MANIFEST))
    if not data:
        return old_sid, None, "清单无法读取", None
    src = data.get("source") or {}
    path = src.get("path")
    if not path or not os.path.exists(path):
        return old_sid, None, "源文件不存在", None

    size = os.path.getsize(path)
    if src.get("size") and size != src["size"]:
        return old_sid, None, f"源文件大小已变（{src['size']} → {size}）", None

    info = media.probe(path)
    if not info:
        return old_sid, None, "源文件无法探测", None
    if src.get("duration") and abs(float(src["duration"]) - info["duration"]) > 0.5:
        return old_sid, None, f"源文件时长已变（{src['duration']} → {info['duration']}）", None

    new_sid = manifest.source_id(path)
    if new_sid == old_sid and data.get("version") == manifest.VERSION:
        return old_sid, None, "已是最新", info
    return old_sid, new_sid, "待迁移", info


def apply(jdir, new_sid, info):
    data = load_raw(os.path.join(jdir, manifest.MANIFEST))
    src_path = data["source"]["path"]
    data["version"] = manifest.VERSION
    data["source"] = manifest.make_source(src_path, info)

    for rid in (data.get("runs") or {}):
        tpath = os.path.join(jdir, manifest.RUNS, rid, manifest.TRANSCRIPT)
        if not os.path.exists(tpath):
            continue
        tr = load_raw(tpath)
        if not tr:
            continue
        tr["source_id"] = new_sid
        manifest.atomic_write_json(tpath, tr)

    sel_path = os.path.join(jdir, manifest.SELECTED)
    if os.path.exists(sel_path):
        sel = load_raw(sel_path)
        if sel:
            sel["source_id"] = new_sid
            sel["source"] = data["source"]
            manifest.atomic_write_json(sel_path, sel)

    manifest.atomic_write_json(os.path.join(jdir, manifest.MANIFEST), data)

    dst = manifest.job_dir(new_sid)
    if os.path.exists(dst):
        raise RuntimeError(f"目标目录已存在：{dst}")
    shutil.move(jdir, dst)
    return dst


def main():
    p = cli.new_parser("升级旧版任务目录到 v4 完整指纹")
    p.add_argument("--dry-run", action="store_true", help="只打印计划，不修改文件")
    args = p.parse_args()

    if not os.path.isdir(paths.JOBS):
        sys.exit("没有 jobs 目录，无需迁移")

    todo, skipped = [], []
    for name in sorted(os.listdir(paths.JOBS)):
        jdir = os.path.join(paths.JOBS, name)
        if not os.path.isdir(jdir):
            continue
        old_sid, new_sid, why, info = plan_one(jdir)
        if new_sid:
            todo.append((jdir, old_sid, new_sid, info))
        else:
            skipped.append((old_sid, why))

    for sid, why in skipped:
        print(f"[skip] {sid} — {why}")
    for jdir, old_sid, new_sid, info in todo:
        print(f"[plan] {old_sid} → {new_sid}  ({media.describe(info)})")
        if args.dry_run:
            continue
        try:
            dst = apply(jdir, new_sid, info)
            print(f"[done] {os.path.basename(dst)}")
        except Exception as e:
            print(f"[fail] {old_sid}: {e}")

    print(f"\n{len(todo)} 个待迁移，{len(skipped)} 个跳过"
          + ("（dry-run，未改动任何文件）" if args.dry_run else ""))


if __name__ == "__main__":
    main()
