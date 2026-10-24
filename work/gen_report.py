"""生成片段核对清单（HTML + CSV）。

标题、时长、保留比例、文件名全部来自任务清单与切点方案，不再写死作品和角色。
"""

import csv
import html
import json
import os
import sys

from core import cli, manifest, naming, paths


def ts(t):
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def human(sec):
    sec = float(sec or 0)
    return f"{int(sec // 60)}分{sec % 60:04.1f}秒"


def main():
    p = cli.new_parser("生成片段核对清单")
    cli.add_video(p)
    p.add_argument("--out-dir", default=None,
                   help="清单输出目录，默认 output/")
    args = p.parse_args()

    video = cli.resolve_video(args.video)
    sid, jdir = cli.job_of(video)

    sel_path = os.path.join(jdir, manifest.SELECTED)
    if not os.path.exists(sel_path):
        sys.exit("缺少切点方案，请先运行 select.py")
    with open(sel_path, encoding="utf-8") as f:
        sel = json.load(f)

    ok, why = manifest.validate_selection(sid, sel)
    if not ok:
        sys.exit(f"切点方案不可用：{why}")
    segs = sel.get("segments") or []

    man = manifest.load_manifest(sid) or {}
    src = sel.get("source") or man.get("source") or {}
    role = naming.safe_filename(sel.get("role") or "目标角色", fallback="role")
    stem = naming.safe_stem(src.get("name") or os.path.basename(video))
    base = f"{stem}_{role}"
    duration = float(src.get("duration") or 0)
    total = float(sel.get("total_sec") or sum(s["dur"] for s in segs))
    pct = (total / duration * 100) if duration else 0

    out_dir = os.path.abspath(args.out_dir) if args.out_dir else paths.OUTPUT
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{base}_清单.csv")
    html_path = os.path.join(out_dir, f"{base}_清单.html")

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["序号", "原片起点", "原片终点", "时长(秒)", "相似度", "台词"])
        for i, s in enumerate(segs, 1):
            sc = "" if s.get("score") is None else f"{s['score']:.3f}"
            w.writerow([i, ts(s["start"]), ts(s["end"]), s["dur"], sc,
                        " / ".join(s["lines"])])

    rows = []
    for i, s in enumerate(segs, 1):
        lines = "<br>".join(html.escape(x) for x in s["lines"])
        sc = s.get("score")
        if sc is None:
            badge = '<span class="tag">—</span>'
        elif sc >= 0.7:
            badge = f'<span class="tag ok">{sc:.3f}</span>'
        elif sc >= 0.6:
            badge = f'<span class="tag warn">{sc:.3f}</span>'
        else:
            badge = f'<span class="tag low">{sc:.3f}</span>'
        rows.append(
            f'<tr><td class="num">{i:02d}</td>'
            f'<td class="tc">{ts(s["start"])} – {ts(s["end"])}</td>'
            f'<td class="dur">{s["dur"]:.1f}s</td>'
            f'<td>{badge}</td>'
            f'<td class="txt">{lines}</td></tr>')

    meta_bits = [f"原片 {human(duration)}"]
    if src.get("width"):
        meta_bits.append(f'{src["width"]}×{src["height"]}')
    if src.get("fps"):
        meta_bits.append(f'{src["fps"]:.3f} fps')
    if sel.get("model"):
        meta_bits.append(f'模型 {sel["model"]}')
    if sel.get("threshold") is not None:
        meta_bits.append(f'阈值 {sel["threshold"]:.2f}')

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{html.escape(src.get('name') or stem)} · {html.escape(role)} 台词片段清单</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;padding:32px 28px 48px;background:#fafaf8;color:#1f1f1d;
font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;font-size:14px;line-height:1.65}}
h1{{font-size:19px;font-weight:500;margin:0 0 6px}}
.sub{{color:#6b6a65;font-size:13px;margin-bottom:22px}}
.stats{{display:flex;gap:12px;margin-bottom:24px;flex-wrap:wrap}}
.card{{background:#fff;border:1px solid #e3e2dc;border-radius:10px;padding:12px 16px;min-width:118px}}
.card .k{{font-size:12px;color:#6b6a65;margin-bottom:2px}}
.card .v{{font-size:17px;font-weight:500}}
table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e3e2dc;
border-radius:10px;overflow:hidden}}
th{{text-align:left;font-size:12px;font-weight:500;color:#6b6a65;padding:10px 12px;
background:#f4f3ee;border-bottom:1px solid #e3e2dc}}
td{{padding:10px 12px;border-bottom:1px solid #eeede8;vertical-align:top}}
tr:last-child td{{border-bottom:none}}
.num{{color:#9b9a94;font-variant-numeric:tabular-nums;width:38px}}
.tc{{font-variant-numeric:tabular-nums;white-space:nowrap;color:#0f6e56;width:152px}}
.dur{{color:#6b6a65;white-space:nowrap;width:66px}}
.txt{{line-height:1.75}}
.tag{{display:inline-block;font-size:11.5px;padding:1px 8px;border-radius:20px;
font-variant-numeric:tabular-nums;white-space:nowrap}}
.ok{{background:#e1f5ee;color:#0f6e56}}
.warn{{background:#faeeda;color:#854f0b}}
.low{{background:#f1efe8;color:#5f5e5a}}
.note{{margin-top:22px;font-size:12.5px;color:#6b6a65;line-height:1.8;background:#fff;
border:1px solid #e3e2dc;border-radius:10px;padding:14px 16px}}
.note b{{color:#1f1f1d;font-weight:500}}
</style></head><body>
<h1>{html.escape(role)} 台词片段清单</h1>
<div class="sub">{html.escape(src.get('name') or stem)} · {' · '.join(meta_bits)}</div>
<div class="stats">
<div class="card"><div class="k">片段数</div><div class="v">{len(segs)}</div></div>
<div class="card"><div class="k">成片时长</div><div class="v">{human(total)}</div></div>
<div class="card"><div class="k">原片时长</div><div class="v">{human(duration)}</div></div>
<div class="card"><div class="k">保留比例</div><div class="v">{pct:.1f}%</div></div>
</div>
<table><thead><tr><th>#</th><th>原片时间码</th><th>时长</th><th>相似度</th>
<th>台词（日语识别原文）</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody></table>
<div class="note">
<b>相似度</b>　该段与目标角色声纹模板的余弦相似度。0.70 以上为高置信，0.60–0.70 建议复核，
低于 0.60 多为其他角色或音效干扰。<br>
<b>时间码</b>　均为原片绝对时间，可直接在播放器中定位核对。<br>
<b>生成时间</b>　{html.escape(sel.get('created') or '')}
</div>
</body></html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(doc)

    print(f"[ok] {len(segs)} 段，{human(total)}，占原片 {pct:.1f}%")
    print(f"[out] {html_path}")
    print(f"[out] {csv_path}")


if __name__ == "__main__":
    main()
