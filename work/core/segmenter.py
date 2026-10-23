"""把选中的台词编号合并成可剪辑的时间段。命令行与图形界面共用。"""


def build_segments(seg_ids, segments, duration,
                   merge_gap=2.0, pad_head=0.25, pad_tail=0.30, min_dur=0.6):
    """合并规则：

    - 相邻台词间隔不超过 merge_gap、且中间没有其他角色插话，合并为一段
    - 每段前后留出呼吸空间，但会避让相邻的其他角色语段
    - 过短的碎片按原始边界回退，避免裁掉音节

    对空输入、缺失 ID、重复 ID、负时长和越界时间都做了处理。
    """
    if not seg_ids or not segments:
        return []

    by_id = {s["id"]: s for s in segments if "id" in s}
    seen, ordered = set(), []
    for i in seg_ids:
        if i in by_id and i not in seen:
            seen.add(i)
            ordered.append(i)

    mine = sorted((by_id[i] for i in ordered), key=lambda s: s["start"])
    mine = [s for s in mine if s["end"] > s["start"] >= 0]
    if not mine:
        return []

    total = float(duration or 0)
    if total <= 0:
        total = max(s["end"] for s in segments)

    picked = {s["id"] for s in mine}
    others = [s for s in segments
              if s.get("id") not in picked and s["end"] > s["start"] >= 0]

    groups = []
    for s in mine:
        if groups:
            prev = groups[-1][-1]
            gap = s["start"] - prev["end"]
            intruded = any(o["end"] > prev["end"] + 0.05 and o["start"] < s["start"] - 0.05
                           for o in others)
            if gap <= merge_gap and not intruded:
                groups[-1].append(s)
                continue
        groups.append([s])

    def prev_other(t):
        c = [o["end"] for o in others if o["end"] <= t + 0.001]
        return max(c) if c else 0.0

    def next_other(t):
        c = [o["start"] for o in others if o["start"] >= t - 0.001]
        return min(c) if c else total

    out = []
    for g in groups:
        s0 = max(g[0]["start"] - pad_head, prev_other(g[0]["start"]) + 0.06, 0.0)
        e0 = min(g[-1]["end"] + pad_tail, next_other(g[-1]["end"]) - 0.06, total)
        if e0 - s0 < min_dur:
            s0, e0 = g[0]["start"], min(g[-1]["end"], total)
        if e0 <= s0:
            continue
        out.append({"start": round(s0, 3), "end": round(e0, 3),
                    "dur": round(e0 - s0, 2),
                    "ids": [x["id"] for x in g],
                    "lines": [x["text"] for x in g]})
    return out
