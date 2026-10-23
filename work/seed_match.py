import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "work")

SEEDS = [133, 134, 137, 138, 149, 160, 161, 324, 326, 328, 332, 339, 347, 349]

d = np.load(f"{WORK}\\embeddings.npz")
E, ids = d["emb"], [int(x) for x in d["ids"]]
tr = {s["id"]: s for s in json.load(open(f"{WORK}\\transcript.json", encoding="utf-8"))["segments"]}

pos = {i: k for k, i in enumerate(ids)}
seed_idx = [pos[s] for s in SEEDS if s in pos]
tmpl = E[seed_idx].mean(axis=0)
tmpl /= np.linalg.norm(tmpl)
score = E @ tmpl

order = np.argsort(-score)
print(f"seeds={len(seed_idx)}")
print("score distribution:", " ".join(
    f"p{p}={np.percentile(score, p):.3f}" for p in [10, 25, 50, 75, 90, 95, 99]))
print()
for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
    sel = [ids[i] for i in order if score[i] >= thr]
    tot = sum(tr[i]["end"] - tr[i]["start"] for i in sel)
    print(f"thr={thr:.2f} -> {len(sel):3d} segs, {tot/60:5.2f} min")

print("\n--- top 40 ---")
for i in order[:40]:
    s = tr[ids[i]]
    mark = "*" if ids[i] in SEEDS else " "
    print(f"{mark} {ids[i]:3d} {s['start']/60:05.2f}m {score[i]:.3f} {s['text'][:52]}")

np.savez(f"{WORK}\\scores.npz", ids=np.array(ids), score=score)
