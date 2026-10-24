"""阈值评估。

用人工确认过的标注样本，比较不同阈值下的 precision / recall / F1，
并对比「简单平均模板」与「稳健模板」的效果，为默认阈值提供依据。

标注优先读任务目录下的 labels.json（界面上点「保存人工样本」生成），
没有时退回内置的单集标注，仅用于评估，不参与生产逻辑。
"""

import argparse
import json
import os
import sys

import numpy as np

from core import cli, manifest, paths, scoring

# 人工确认属于目标角色的台词编号（Re:Zero 第 8 话 · 碧翠丝）
LABELS = [132, 133, 134, 136, 137, 138, 142, 143, 145, 146, 147, 148,
          149, 150, 151, 152, 153, 154, 155, 159, 160, 164,
          323, 325, 328, 332, 334, 335, 337, 339, 347, 348]


def simple_scores(emb, ids, seeds):
    pos = {i: k for k, i in enumerate(ids)}
    idx = [pos[s] for s in seeds if s in pos]
    if not idx:
        return np.zeros(len(ids))
    t = emb[idx].mean(axis=0)
    t = t / np.linalg.norm(t)
    return emb @ t


def evaluate(scores, labels, thresholds):
    rows = []
    for thr in thresholds:
        pred = {i for i, s in scores.items() if s >= thr}
        tp = len(pred & labels)
        p = tp / len(pred) if pred else 0.0
        r = tp / len(labels) if labels else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        rows.append((thr, p, r, f1, len(pred)))
    return rows


def main():
    p = cli.new_parser("阈值评估（需要人工标注样本）")
    cli.add_video(p)
    p.add_argument("--keyword", default="かしら,ベティ")
    p.add_argument("--labels", default="auto", choices=("auto", "builtin"),
                   help="auto 优先用任务目录里的 labels.json，builtin 用内置标注")
    args = p.parse_args()

    video = cli.resolve_video(args.video)
    sid, jdir = cli.job_of(video)

    rdir = manifest.active_run_dir(sid)
    if not rdir:
        sys.exit("没有可用的分析结果，请先运行 transcribe.py 与 voiceprint.py")
    tpath = os.path.join(rdir, manifest.TRANSCRIPT)
    epath = os.path.join(rdir, manifest.EMBEDDINGS)
    if not (os.path.exists(tpath) and os.path.exists(epath)):
        sys.exit("需要先完成转写与声纹提取")

    with open(tpath, encoding="utf-8") as f:
        segments = json.load(f)["segments"]
    emb, ids = scoring.load_embeddings(epath)

    seeds = scoring.keyword_seeds(segments, args.keyword.replace("，", ",").split(","))
    if not seeds:
        sys.exit("特征词没有命中任何台词")

    labels = set(LABELS)
    negatives = []
    saved = manifest.load_labels(sid) if args.labels == "auto" else {}
    if saved.get("positive"):
        labels = {int(i) for i in saved["positive"]}
        negatives = [int(i) for i in saved.get("negative") or []]
        stale = saved.get("run_id") and saved["run_id"] != manifest.active_run(sid)
        print(f"[labels] 使用 {manifest.LABELS}"
              f"（{saved.get('updated') or '未知时间'}，{saved.get('role') or '未命名角色'}）")
        if stale:
            print(f"[warn] 标注来自运行 {saved['run_id']}，"
                  f"当前活动运行是 {manifest.active_run(sid)}，编号可能对不上")
        print("[note] 这份标注来自界面勾选。若没有逐条人工核对，"
              "阈值附近的样本与打分自洽，对应阈值的指标会偏乐观；"
              "想看独立评估请加 --labels builtin。")
    else:
        print("[labels] 使用内置标注集（人工逐条核对过的 32 句）")

    robust = scoring.template_scores(emb, ids, seeds)
    plain = simple_scores(emb, ids, seeds)
    rmap = {i: float(s) for i, s in zip(ids, robust)}
    pmap = {i: float(s) for i, s in zip(ids, plain)}

    thresholds = [0.55, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.75, 0.80]

    print(f"种子 {len(seeds)} 句，标注 {len(labels)} 句，"
          f"负样本 {len(negatives)} 句，候选 {len(ids)} 条\n")
    print("=== 稳健模板（默认） ===")
    print("阈值   精确率  召回率   F1     命中数")
    for thr, pr, rc, f1, n in evaluate(rmap, labels, thresholds):
        print(f"{thr:.2f}   {pr:6.3f}  {rc:6.3f}  {f1:6.3f}   {n}")

    print("\n=== 简单平均模板（对照） ===")
    print("阈值   精确率  召回率   F1     命中数")
    for thr, pr, rc, f1, n in evaluate(pmap, labels, thresholds):
        print(f"{thr:.2f}   {pr:6.3f}  {rc:6.3f}  {f1:6.3f}   {n}")

    best_r = max(evaluate(rmap, labels, thresholds), key=lambda x: x[3])
    best_p = max(evaluate(pmap, labels, thresholds), key=lambda x: x[3])
    print(f"\n最佳 F1：稳健 {best_r[3]:.3f} @ {best_r[0]:.2f}　"
          f"简单平均 {best_p[3]:.3f} @ {best_p[0]:.2f}")

    if negatives:
        print("\n=== 负样本抑制（margin 扫描） ===")
        print("margin  最佳F1  对应阈值  0.66处F1  0.66命中数")
        best = None
        for margin in (0.05, 0.10, 0.20, 0.30, 0.50):
            mixed = scoring.template_scores(emb, ids, seeds,
                                            neg_ids=negatives, margin=margin)
            mmap = {i: float(s) for i, s in zip(ids, mixed)}
            rows = evaluate(mmap, labels, thresholds)
            top = max(rows, key=lambda x: x[3])
            at66 = next(r for r in rows if abs(r[0] - 0.66) < 1e-9)
            print(f"{margin:5.2f}   {top[3]:6.3f}   {top[0]:.2f}     "
                  f"{at66[3]:6.3f}   {at66[4]}")
            if best is None or top[3] > best[1][3]:
                best = (margin, top)
        print(f"\n最优 margin {best[0]:.2f}：F1 {best[1][3]:.3f} @ {best[1][0]:.2f}"
              f"（默认模板 {best_r[3]:.3f} @ {best_r[0]:.2f}）")
        if best[1][3] <= best_r[3]:
            print("结论：负样本抑制没有带来提升，当前默认不启用。")


if __name__ == "__main__":
    main()
