"""按台词编号生成切点方案，写入任务目录。

编号可以显式给出，也可以由特征词自动挑选。
"要保留哪些台词"（selected）与"用什么建立声纹模板"（seed）是两个概念，
显式指定台词时不再偷偷套用默认特征词，避免用错角色的模板去打分。
"""

import json
import os
import sys
from datetime import datetime

from core import cli, manifest, scoring, segmenter


def parse_ids(text):
    out = []
    for part in (text or "").replace("，", ",").split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return out


def main():
    p = cli.new_parser("按台词编号生成切点方案")
    cli.add_video(p)
    p.add_argument("--role", default="目标角色", help="角色名，写入报告与输出文件名")
    p.add_argument("--selected-ids", default=None, help="要保留的台词编号，逗号分隔")
    p.add_argument("--ids", default=None, help="--selected-ids 的旧名，保留兼容")
    p.add_argument("--seed-ids", default=None,
                   help="建立声纹模板用的台词编号；不给则用特征词命中结果")
    p.add_argument("--keyword", default="かしら,ベティ",
                   help="自动挑选台词与种子时使用的特征词")
    p.add_argument("--threshold", type=float, default=0.66)
    p.add_argument("--merge-gap", type=float, default=2.0)
    p.add_argument("--pad-head", type=float, default=0.25)
    p.add_argument("--pad-tail", type=float, default=0.30)
    args = p.parse_args()

    video = cli.resolve_video(args.video)
    sid, source, jdir, info = cli.prepare(video)

    rdir = manifest.active_run_dir(sid)
    if not rdir:
        sys.exit("没有可用的分析结果，请先运行 transcribe.py 与 voiceprint.py")
    tpath = os.path.join(rdir, manifest.TRANSCRIPT)
    epath = os.path.join(rdir, manifest.EMBEDDINGS)
    with open(tpath, encoding="utf-8") as f:
        tr = json.load(f)
    segments = tr.get("segments") or []
    if not segments:
        sys.exit("转写结果为空")

    keywords = (args.keyword or "").replace("，", ",").split(",")
    explicit = args.selected_ids or args.ids

    if explicit:
        ids = parse_ids(explicit)
        if not ids:
            sys.exit("--selected-ids 里没有可用的编号")
    else:
        seeds = scoring.keyword_seeds(segments, keywords)
        if not seeds:
            sys.exit(f"特征词 {args.keyword} 没有命中任何台词，请改用 --selected-ids")
        ids = seeds
        try:
            emb, emb_ids = scoring.load_embeddings(epath)
            arr = scoring.template_scores(emb, emb_ids, seeds)
            smap = {i: float(s) for i, s in zip(emb_ids, arr)}
            picked = [s["id"] for s in segments if smap.get(s["id"], -1) >= args.threshold]
            if picked:
                ids = picked
        except Exception as e:
            print(f"[warn] 声纹筛选失败，退回仅用特征词命中的台词：{e}")

    duration = float(tr.get("duration") or info.get("duration") or 0)
    segs = segmenter.build_segments(ids, segments, duration,
                                    args.merge_gap, args.pad_head, args.pad_tail)
    if not segs:
        sys.exit("这些编号无法组成有效片段")

    # 模板种子：显式给了就用；否则只有自动模式才用特征词命中结果
    seed_ids = []
    if args.seed_ids:
        seed_ids = parse_ids(args.seed_ids)
    elif not explicit:
        seed_ids = scoring.keyword_seeds(segments, keywords)

    scores = {}
    if seed_ids:
        try:
            emb, emb_ids = scoring.load_embeddings(epath)
            arr = scoring.template_scores(emb, emb_ids, seed_ids)
            scores = {i: round(float(s), 4) for i, s in zip(emb_ids, arr)}
        except Exception as e:
            print(f"[warn] 声纹打分失败，报告将不含相似度：{e}")
    elif explicit:
        print("[note] 显式指定台词且未提供 --seed-ids，本次不生成相似度分数")

    for s in segs:
        vals = [scores[i] for i in s["ids"] if i in scores]
        s["score"] = round(sum(vals) / len(vals), 4) if vals else None

    total = round(sum(s["dur"] for s in segs), 2)
    rid = manifest.active_run(sid)
    run_rec = ((manifest.load_manifest(sid) or {}).get("runs") or {}).get(rid) or {}
    track = int(run_rec.get("audio_track", manifest.audio_track(sid)))
    payload = {
        "schema": manifest.SELECTION_SCHEMA,
        "version": manifest.VERSION,
        "source_id": sid,
        "run_id": tr.get("run_id"),
        "source": source,
        "role": args.role,
        "created": datetime.now().isoformat(timespec="seconds"),
        "threshold": args.threshold,
        "model": tr.get("model", ""),
        "audio_track": track,
        "seed_ids": seed_ids,
        "merge_gap": args.merge_gap,
        "pad_head": args.pad_head,
        "pad_tail": args.pad_tail,
        "total_sec": total,
        "segments": segs,
    }
    manifest.atomic_write_json(os.path.join(jdir, manifest.SELECTED), payload)

    print(f"{len(ids)} 句 -> {len(segs)} 个片段，合计 {total:.1f}s = {total / 60:.2f} 分钟\n")
    for i, s in enumerate(segs, 1):
        sc = f"{s['score']:.3f}" if s["score"] is not None else "  —  "
        print(f"{i:02d} {s['start']:7.2f}-{s['end']:7.2f} ({s['dur']:5.2f}s) {sc}  "
              f"{' / '.join(s['lines'])[:56]}")
    print(f"\n[out] {os.path.join(jdir, manifest.SELECTED)}")


if __name__ == "__main__":
    main()
