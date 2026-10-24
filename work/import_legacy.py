"""显式导入旧版扁平缓存。

旧版把 transcript.json 和 embeddings.npz 直接放在 work/ 下，没有记录来源。
这类文件无法自动判断属于哪个视频——同名同大小同修改时间的不同文件都可能撞上，
所以不再自动迁移，只提供这个需要手动确认的导入入口。

用法：
    python import_legacy.py                      # 默认 input/ 第一个视频
    python import_legacy.py --video X.mp4 --yes
"""

import json
import os
import shutil
import sys

from core import cli, manifest, paths

LEGACY_TRANSCRIPT = os.path.join(paths.WORK, "transcript.json")
LEGACY_EMBEDDINGS = os.path.join(paths.WORK, "embeddings.npz")


def main():
    p = cli.new_parser("导入旧版扁平缓存（需手动确认）")
    cli.add_video(p)
    p.add_argument("--yes", action="store_true", help="确认这些旧产物确实属于该视频")
    args = p.parse_args()

    video = cli.resolve_video(args.video)
    sid, source, jdir, info = cli.prepare(video)

    if not os.path.exists(LEGACY_TRANSCRIPT) or not os.path.exists(LEGACY_EMBEDDINGS):
        sys.exit("没有找到旧版缓存（work/transcript.json 与 work/embeddings.npz）")

    with open(LEGACY_TRANSCRIPT, encoding="utf-8") as f:
        tr = json.load(f)
    old_dur = float(tr.get("duration") or 0)
    new_dur = float(source.get("duration") or 0)

    print(f"源视频：{source['name']}  {new_dur:.2f}s  指纹 {source['fingerprint']}")
    print(f"旧转写：{len(tr.get('segments') or [])} 段  {old_dur:.2f}s")
    if old_dur and new_dur and abs(old_dur - new_dur) > 0.5:
        sys.exit(f"时长不匹配（相差 {abs(old_dur - new_dur):.2f}s），拒绝导入")

    if manifest.active_run(sid):
        sys.exit("该视频已有完整的分析结果，无需导入")

    if not args.yes:
        sys.exit("旧缓存没有来源指纹，无法自动确认归属。\n"
                 "请核对上面的时长后加 --yes 重新执行。")

    run_id = manifest.new_run(sid, source, model=tr.get("model", ""),
                              device=tr.get("device", ""))
    shutil.copy2(LEGACY_TRANSCRIPT, manifest.run_artifact(sid, run_id, manifest.TRANSCRIPT))
    shutil.copy2(LEGACY_EMBEDDINGS, manifest.run_artifact(sid, run_id, manifest.EMBEDDINGS))

    # 补写 source_id，便于后续校验
    tpath = manifest.run_artifact(sid, run_id, manifest.TRANSCRIPT)
    with open(tpath, encoding="utf-8") as f:
        payload = json.load(f)
    payload["source_id"] = sid
    payload["run_id"] = run_id
    manifest.atomic_write_json(tpath, payload)

    ok, detail = manifest.finalize_run(sid, run_id)
    if not ok:
        sys.exit(f"导入失败：{detail}")

    print(f"[ok] 已导入到运行 {run_id}")
    print(f"[ok] 校验：{detail}")


if __name__ == "__main__":
    main()
