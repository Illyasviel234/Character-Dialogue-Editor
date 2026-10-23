"""按台词编号生成切点方案，写入任务目录。

编号可以显式给出，也可以由特征词自动挑选，不再依赖某一集的固定编号。
"""

import json
import os
import sys
from datetime import datetime

from core import cli, manifest, scoring, segmenter


def parse_ids(text):
    out = []
    for part in text.replace("，", ",").split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return out


def main():
    p = cli.new_parser("按台词编号生成切点方案")
    cli.add_video(p)
    p.add_argument("--role", default="目标角色", help="角色名，写入报告与输出文件名")
    p.add_argument("--ids", default=None, help="逗号分隔的台词编号")
    p.add_argument("--keyword", default="かしら,ベティ",
                   help="未给 --ids 时用这些特征词自动挑选台词")
    p.add_argument("--threshold", type=float, default=0.66)
    p.add_argument("--merge-gap", type=float, default=2.0)
    p.add_argument("--pad-head", type=float, default=0.25)
    p.add_argument("--pad-tail", type=float, default=0.30)
    args = p.parse_args()

    video = cli.resolve_video(args.video)
    sid, source, jdir, info = cli.prepare(video)

    tpath = os.path.join(jdir, manifest.TRANSCRIPT)
    if not os.path.exists(tpath):
        sys.exit("缺少转写结果，请先运行 transcribe.py")
    with open(tpath, encoding="utf-8") as f:
        tr = json.load(f)
    segments = tr.get("segments") or []
    if not segments:
        sys.exit("转写结果为空")

    if args.ids:
        ids = parse_ids(args.ids)
        if not ids:
            sys.exit("--ids 里没有可用的编号")
    else:
        seeds = scoring.keyword_seeds(segments, args.keyword.replace("，", ",").split(","))
        if not seeds:
            sys.exit(f"特征词 {args.keyword} 没有命中任何台词，请改用 --ids 指定编号")
        # 用种子模板给全片打分，再按阈值筛选，与图形界面保持一致
        epath = os.path.join(jdir, manifest.EMBEDDINGS)
        ids = seeds
        if os.path.exists(epath):
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

    scores = {}
    epath = os.path.join(jdir, manifest.EMBEDDINGS)
    if os.path.exists(epath):
        try:
            emb, emb_ids = scoring.load_embeddings(epath)
            seed_ids = scoring.keyword_seeds(segments, args.keyword.split(",")) or ids
            arr = scoring.template_scores(emb, emb_ids, seed_ids)
            scores = {i: round(float(s), 4) for i, s in zip(emb_ids, arr)}
        except Exception as e:
            print(f"[warn] 声纹打分失败，报告将不含相似度：{e}")

    for s in segs:
        vals = [scores[i] for i in s["ids"] if i in scores]
        s["score"] = round(sum(vals) / len(vals), 4) if vals else None

    total = round(sum(s["dur"] for s in segs), 2)
    payload = {
        "version": manifest.VERSION,
        "source_id": sid,
        "source": source,
        "role": args.role,
        "created": datetime.now().isoformat(timespec="seconds"),
        "threshold": args.threshold,
        "model": tr.get("model", ""),
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
