"""逐句提取声纹向量。

承接 transcribe.py 建立的 run，产物写入同一 run 目录。
完成后由 manifest.finalize_run 校验两份产物是否自洽，
通过才把这次运行标记为 complete 并设为活动结果。
"""

import json
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
    p.add_argument("--run", default=None, help="指定运行 ID，默认接续待完成的运行")
    p.add_argument("--cluster", action="store_true", help="额外运行聚类诊断")
    args = p.parse_args()

    video = cli.resolve_video(args.video)
    sid, source, jdir, info = cli.prepare(video)

    run_id = args.run or manifest.pending_run(sid)
    if not run_id:
        if manifest.active_run(sid):
            sys.exit("已有完整的分析结果。如需重新分析，请先运行 transcribe.py")
        sys.exit("没有待完成的分析运行，请先运行 transcribe.py")

    tpath = manifest.run_artifact(sid, run_id, manifest.TRANSCRIPT)
    apath = os.path.join(jdir, "audio16k.wav")
    if not os.path.exists(tpath):
        sys.exit(f"运行 {run_id} 缺少转写结果")
    if not os.path.exists(apath):
        sys.exit("缺少 16k 音频，请先运行 extract_audio.py")

    with open(tpath, encoding="utf-8") as f:
        tr = json.load(f)
    segments = tr.get("segments") or []
    if not segments:
        manifest.fail_run(sid, run_id, "转写为空")
        sys.exit("转写结果为空，无法提取声纹")

    sr, data = wavfile.read(apath)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32) / 32768.0

    try:
        encoder = load_encoder()
        E, ids, skipped = extract(encoder, data, sr, segments)
        if not ids:
            raise RuntimeError(f"所有语段都短于 {MIN_DUR}s，没有可用声纹")

        out = manifest.run_artifact(sid, run_id, manifest.EMBEDDINGS)
        tmp = out + ".tmp.npz"
        np.savez(tmp, emb=E, ids=np.array(ids))
        os.replace(tmp, out)
    except Exception as e:
        manifest.fail_run(sid, run_id, e)
        sys.exit(f"声纹提取失败：{e}")

    ok, detail = manifest.finalize_run(sid, run_id)
    if not ok:
        sys.exit(f"分析结果未通过校验：{detail}")

    print(f"[ok] {E.shape[0]} 条声纹（跳过 {skipped} 条过短语段）")
    print(f"[ok] 本次运行已提交：{run_id}")
    print(f"[ok] 校验：{detail}")

    if args.cluster:
        print("[cluster] 聚类诊断：")
        cluster_diagnose(E, ids, segments)


if __name__ == "__main__":
    main()
