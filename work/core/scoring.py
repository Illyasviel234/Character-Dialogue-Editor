"""声纹打分。

模板不是简单平均：先算种子与中心的相似度，用中位数和 MAD 剔除离群种子，
再求均值。混入的误识别、强音效或情绪极端的台词不会把模板拉偏。
"""

import numpy as np


def load_embeddings(path):
    d = np.load(path)
    return d["emb"], [int(x) for x in d["ids"]]


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def robust_template(emb, idx, trim=1.5):
    """中位中心 + MAD 离群剔除后的均值模板。"""
    V = emb[idx]
    if len(V) == 0:
        return None
    if len(V) == 1:
        return _normalize(V[0].astype(np.float64))

    center = _normalize(V.mean(axis=0).astype(np.float64))
    sims = V @ center
    med = float(np.median(sims))
    mad = float(np.median(np.abs(sims - med)))
    if mad <= 1e-6:
        keep = sims >= med
    else:
        keep = sims >= med - trim * mad * 1.4826
    if not keep.any():
        keep = np.ones(len(V), dtype=bool)
    return _normalize(V[keep].mean(axis=0).astype(np.float64))


def template_scores(emb, ids, seed_ids, neg_ids=None, margin=0.5):
    """返回每个语段与目标角色模板的相似度。

    neg_ids 给出易混淆角色的台词时，会从正向分数里扣掉一部分负向相似度，
    用来压制声线接近的配角。
    """
    pos = {i: k for k, i in enumerate(ids)}
    idx = [pos[s] for s in seed_ids if s in pos]
    if not idx:
        return np.zeros(len(ids), dtype=np.float32)

    t = robust_template(emb, idx)
    if t is None:
        return np.zeros(len(ids), dtype=np.float32)
    score = emb @ t

    if neg_ids:
        nidx = [pos[s] for s in neg_ids if s in pos]
        if nidx:
            nt = robust_template(emb, nidx)
            if nt is not None:
                score = score - margin * np.maximum(0.0, emb @ nt)

    return score.astype(np.float32)


def keyword_seeds(segments, keywords):
    ks = [k.strip() for k in keywords if k and k.strip()]
    if not ks:
        return []
    return [s["id"] for s in segments if any(k in s.get("text", "") for k in ks)]


def threshold_report(scores, thr):
    """给界面和命令行用的简要统计。"""
    vals = np.asarray(list(scores.values()), dtype=np.float32)
    if vals.size == 0:
        return {"total": 0, "hit": 0, "near": 0}
    return {
        "total": int(vals.size),
        "hit": int((vals >= thr).sum()),
        "near": int(((vals >= thr) & (vals < thr + 0.08)).sum()),
    }
