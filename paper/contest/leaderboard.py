#!/usr/bin/env python3
"""TRIZ-Bench 擂台总表构建 (leaderboard)。

输入 (paper/contest/):
  gen_<arm>.jsonl                  选手回答 (contestant_gen_v5.py / 既有存档)
  contest_cache_<judge>_<arm>.json 评委逐题分 (contest_judge.py / 终审存档)
  v5_gold.jsonl                    金标 (paper/external_review/)

输出:
  leaderboard.json / leaderboard.md
  - 每臂: 关键词轨命中率 (含别名表口径), 逐评委 overall 均值 (家族回避)
  - 每臂 vs base: 逐评委 paired bootstrap 差值 + CI95 (seed=42, n=10000)
  - 子集分解; 评委间 Spearman; 披露评委-选手家族回避矩阵

用法: python3 paper/contest/leaderboard.py
"""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT / "pipeline_v5" / "eval"))

from keyword_scorer import load_alias_map, keyword_score  # noqa: E402

GOLD = PROJECT / "paper" / "external_review" / "v5_gold.jsonl"
AMAP = PROJECT / "pipeline_v5" / "eval" / "keyword_map_v5.json"
N_BOOT = 10000
SEED = 42

ARMS = ["base", "v5a", "gpt-5.4", "claude-sonnet-4-6",
        "claude-opus-4-8", "gemini-3.5-flash"]
ANCHOR = "base"
FAMILY = {"claude": "anthropic", "gpt": "openai", "gemini": "google",
          "moonshot": "moonshot", "base": "local", "v5a": "local",
          "v5b": "local"}


def family_of(m):
    return FAMILY.get(m.split("-")[0], m.split("-")[0])


def paired_boot(a, b):
    n = len(a)
    if n == 0:
        return None
    rng = random.Random(SEED)
    boots = []
    for _ in range(N_BOOT):
        s = 0.0
        for _ in range(n):
            i = rng.randrange(n)   # 同一下标配对抽样 (与 harness 口径一致)
            s += b[i] - a[i]
        boots.append(s / n)
    boots.sort()
    return {"diff": sum(y - x for x, y in zip(a, b)) / n,
            "ci95": [boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT)]],
            "n": n}


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j < len(v) and v[order[j]] == v[order[i]]:
                j += 1
            for k in range(i, j):
                r[order[k]] = (i + j - 1) / 2 + 1
            i = j
        return r
    n = len(xs)
    if n < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    return cov / (vx * vy) ** 0.5 if vx and vy else None


def main():
    gold = {}
    for line in open(GOLD, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            gold[r["id"]] = r
    ids = sorted(gold)
    amap = load_alias_map(AMAP)

    # 关键词轨
    kw = {}
    for arm in ARMS:
        p = HERE / f"gen_{arm}.jsonl"
        if not p.is_file():
            continue
        resps = {}
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("response"):
                    resps[r["id"]] = r["response"]
        kw[arm] = {}
        for qid in ids:
            if qid in resps:
                s = keyword_score(resps[qid], gold[qid].get("keywords", []), amap)
                if s["kw_hit_rate"] is not None:
                    kw[arm][qid] = s["kw_hit_rate"]

    # 评委轨
    judges_by_arm = {}
    scores = defaultdict(dict)   # (arm, judge) -> {qid: overall}
    for arm in ARMS:
        for p in sorted(HERE.glob(f"contest_cache_*_{arm}.json")):
            judge = p.name[len("contest_cache_"):-len(f"_{arm}.json")]
            if family_of(judge) == family_of(arm):
                continue  # 家族回避
            cache = json.load(open(p, encoding="utf-8"))
            per = {}
            for qid, v in cache.items():
                if isinstance(v, dict) and isinstance(v.get("overall"), (int, float)):
                    per[qid] = v["overall"]
            if len(per) >= 100:
                scores[(arm, judge)] = per
                judges_by_arm.setdefault(arm, []).append(judge)

    # 汇总
    table = {}
    for arm in ARMS:
        if arm not in kw and (arm, "moonshot") not in scores:
            continue
        row = {"kw_mean": (sum(kw[arm].values()) / len(kw[arm])
                           if kw.get(arm) else None),
               "kw_n": len(kw.get(arm, {}))}
        per_judge = {}
        for j in judges_by_arm.get(arm, []):
            sc = scores[(arm, j)]
            vals = [sc[q] for q in ids if q in sc]
            entry = {"mean": sum(vals) / len(vals), "n": len(vals)}
            # vs base 配对差值
            if (ANCHOR, j) in scores and arm != ANCHOR:
                anch = scores[(ANCHOR, j)]
                common = [q for q in ids if q in sc and q in anch]
                pb = paired_boot([anch[q] for q in common], [sc[q] for q in common])
                if pb:
                    entry["vs_base"] = pb
            per_judge[j] = entry
        row["judges"] = per_judge
        # 子集分解 (kw + moonshot 主轨)
        subsets = defaultdict(list)
        for qid, v in kw.get(arm, {}).items():
            subsets[gold[qid]["subset"]].append(v)
        row["kw_per_subset"] = {s: sum(v) / len(v) for s, v in subsets.items()}
        if (arm, "moonshot") in scores:
            sc = scores[(arm, "moonshot")]
            ss = defaultdict(list)
            for qid, v in sc.items():
                if qid in gold:
                    ss[gold[qid]["subset"]].append(v)
            row["moonshot_per_subset"] = {s: sum(v) / len(v)
                                          for s, v in ss.items()}
        table[arm] = row

    # 评委间一致性 (对 base+v5a 都有缓存的评委)
    spear = {}
    ext = [j for j in ("claude-sonnet-4-6", "gpt-5.4", "gemini-3.5-flash")
           if ("base", j) in scores]
    for i, j1 in enumerate(ext):
        for j2 in ext[i + 1:]:
            common = [q for q in ids if q in scores[("base", j1)]
                      and q in scores[("base", j2)]]
            r = spearman([scores[("base", j1)][q] for q in common],
                         [scores[("base", j2)][q] for q in common])
            spear[f"{j1}~{j2}"] = round(r, 3) if r is not None else None

    out = {"arms": table, "anchor": ANCHOR, "judge_spearman_base": spear,
           "family_rule": "评委与选手同家族回避",
           "protocol": "臂A rubric, T=0, paired bootstrap n=10000 seed=42"}
    with open(HERE / "leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # markdown
    L = ["# TRIZ-Bench 擂台总表 (v5 金标 300 题)", "",
         f"锚点: {ANCHOR} | 评委-选手同家族回避 | "
         f"paired bootstrap n={N_BOOT} seed={SEED}", "",
         "## 关键词轨 (别名表口径, 命中率)", "",
         "| 选手 | kw 均值 | n |", "|---|---|---|"]
    for arm, row in table.items():
        if row["kw_mean"] is not None:
            L.append(f"| {arm} | {row['kw_mean']:.4f} | {row['kw_n']} |")
    L += ["", "## judge 轨 (逐评委 overall 均值 0-4; vs base 配对差值 [CI95])", ""]
    all_judges = sorted({j for row in table.values() for j in row["judges"]})
    L.append("| 选手 | " + " | ".join(all_judges) + " |")
    L.append("|---|" + "---|" * len(all_judges))
    for arm, row in table.items():
        cells = []
        for j in all_judges:
            e = row["judges"].get(j)
            if not e:
                cells.append("—(回避)")
            else:
                s = f"{e['mean']:.3f}"
                if "vs_base" in e:
                    v = e["vs_base"]
                    s += f" ({v['diff']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}])"
                cells.append(s)
        L.append(f"| {arm} | " + " | ".join(cells) + " |")
    L += ["", f"## 评委间 Spearman (base 臂): {spear}", ""]
    with open(HERE / "leaderboard.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
