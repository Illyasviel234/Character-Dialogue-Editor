"""逐句提取声纹向量。

主流程只负责产出 embeddings.npz；聚类容易失败且对结果无影响，
因此移到 --cluster 的可选诊断里。
"""

import os
import sys

import numpy as np
from scipy.io import wavfile

from core import cli, manifest, paths

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HOME"] = paths.MODELS

import torch  # noqa: E402
import speechbrain.inference.interfaces as _iface  # noqa: E402
from speechbrain.utils.fetching import LocalStrategy, fetch as _fetch  # noqa: E402
from speechbrain.inference.speaker import EncoderClassifier  # noqa: E402

ECAPA_DIR = os.path.join(paths.MODELS, "spkrec-ecapa-voxceleb")
MIN_DUR = 0.40
PAD = 0.05


def load_encoder():
    # Windows 下 speechbrain 默认的符号链接策略会产出 0 字节文件，强制复制
    _iface.fetch = lambda *a, **kw: _fetch(*a, **{**kw, "local_strategy": LocalStrategy.COPY})
    if not os.path.isdir(ECAPA_DIR):
        sys.exit(f"缺少声纹模型：{ECAPA_DIR}")
    return EncoderClassifier.from_hparams(source=ECAPA_DIR, savedir=ECAPA_DIR,
                                          run_opts={"device": "cpu"})


def extract(encoder, data, sr, segments):
    embs, ids, skipped = [], [], 0
    for s in segments:
        if s["end"] - s["start"] < MIN_DUR:
            skipped += 1
            continue
        a = max(0, int((s["start"] - PAD) * sr))
        b = min(len(data), int((s["end"] + PAD) * sr))
        clip = data[a:b]
        if len(clip) < int(0.3 * sr):
            skipped += 1
            continue
        with torch.no_grad():
            v = encoder.encode_batch(torch.from_numpy(clip).unsqueeze(0))
        embs.append(v.squeeze().cpu().numpy())
        ids.append(s["id"])
        if len(ids) % 100 == 0:
            print(f"  ...{len(ids)} 条", flush=True)
    if not embs:
        return np.zeros((0, 192), dtype=np.float32), [], skipped
    E = np.vstack(embs)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    return E.astype(np.float32), ids, skipped


def cluster_diagnose(E, ids, segments):
    """可选的聚类诊断。样本太少或全独立成簇时只提示，不抛异常。"""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    if len(E) < 8:
        print("[skip] 样本不足 8 条，跳过聚类诊断")
        return
    by_id = {s["id"]: s for s in segments}
    for thr in (0.40, 0.50, 0.60):
        try:
            lab = AgglomerativeClustering(n_clusters=None, metric="cosine",
                                          linkage="average",
                                          distance_threshold=thr).fit_predict(E)
            k = len(set(lab))
            if k < 2:
                print(f"  thr={thr} -> 只有 1 簇")
                continue
            sil = silhouette_score(E, lab, metric="cosine")
            sizes = sorted((int((lab == c).sum()) for c in set(lab)), reverse=True)[:6]
            print(f"  thr={thr} -> k={k} silhouette={sil:.3f} 前几簇={sizes}")
        except Exception as e:
            print(f"  thr={thr} -> 诊断失败：{type(e).__name__}: {str(e)[:80]}")
    print("  注：聚类结果仅供观察，角色判定始终使用种子模板分数。")


def main():
    p = cli.new_parser("逐句提取声纹向量（ECAPA-TDNN）")
    cli.add_video(p)
    p.add_argument("--cluster", action="store_true", help="额外运行聚类诊断")
    args = p.parse_args()

    video = cli.resolve_video(args.video)
    sid, source, jdir, info = cli.prepare(video)

    tpath = os.path.join(jdir, manifest.TRANSCRIPT)
    apath = os.path.join(jdir, "audio16k.wav")
    if not os.path.exists(tpath):
        sys.exit("缺少转写结果，请先运行 transcribe.py")
    if not os.path.exists(apath):
        sys.exit("缺少 16k 音频，请先运行 extract_audio.py")

    import json
    with open(tpath, encoding="utf-8") as f:
        tr = json.load(f)
    segments = tr.get("segments") or []
    if not segments:
        sys.exit("转写结果为空，无法提取声纹")

    sr, data = wavfile.read(apath)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32) / 32768.0

    encoder = load_encoder()
    E, ids, skipped = extract(encoder, data, sr, segments)
    if not ids:
        sys.exit(f"所有语段都短于 {MIN_DUR}s，没有可用声纹")

    out = os.path.join(jdir, manifest.EMBEDDINGS)
    tmp = out + ".tmp.npz"
    np.savez(tmp, emb=E, ids=np.array(ids))
    os.replace(tmp, out)

    print(f"[ok] {E.shape[0]} 条声纹（跳过 {skipped} 条过短语段）")
    print(f"[out] {out}")

    if args.cluster:
        print("[cluster] 聚类诊断：")
        cluster_diagnose(E, ids, segments)


if __name__ == "__main__":
    main()
