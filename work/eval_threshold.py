"""阈值评估。

用人工确认过的标注样本，比较不同阈值下的 precision / recall / F1，
并对比「简单平均模板」与「稳健模板」的效果，为默认阈值提供依据。

标注集来自单集人工核对结果，仅用于评估，不参与生产逻辑。
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
    args = p.parse_args()

    video = cli.resolve_video(args.video)
    sid, jdir = cli.job_of(video)

    tpath = os.path.join(jdir, manifest.TRANSCRIPT)
    epath = os.path.join(jdir, manifest.EMBEDDINGS)
    if not (os.path.exists(tpath) and os.path.exists(epath)):
        sys.exit("需要先完成转写与声纹提取")

    with open(tpath, encoding="utf-8") as f:
        segments = json.load(f)["segments"]
    emb, ids = scoring.load_embeddings(epath)

    seeds = scoring.keyword_seeds(segments, args.keyword.replace("，", ",").split(","))
    if not seeds:
        sys.exit("特征词没有命中任何台词")

    labels = set(LABELS)
    robust = scoring.template_scores(emb, ids, seeds)
    plain = simple_scores(emb, ids, seeds)
    rmap = {i: float(s) for i, s in zip(ids, robust)}
    pmap = {i: float(s) for i, s in zip(ids, plain)}

    thresholds = [0.55, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.75, 0.80]

    print(f"种子 {len(seeds)} 句，标注 {len(labels)} 句，候选 {len(ids)} 条\n")
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


if __name__ == "__main__":
    main()
