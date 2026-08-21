#!/usr/bin/env python
"""
E6 通用探针分析: 关键词命中率 + 配对 bootstrap + G6 裁决。

输入 (需先从远端取回):
  gen_base.jsonl / gen_v2.jsonl / gen_v5a.jsonl   (results/e6_probe/)
  general_probe_v5.json                          (120 题, 含 expected_keywords)

口径:
  - 评分 = 纯子串关键词命中率 (无别名表; 别名表是 TRIZ 专用的);
  - 差值 = 逐题配对差 (命中率差 ×100 = pp), paired bootstrap
    n=10000, stdlib Random(42), 与 v5 harness / 异源终审同口径;
  - G6: v5a−v2 overall > −5pp 判 PASS (方案 §7.1, MDE≈4.7pp);
  - 类级 n=20 一律描述性 (§6.1 风险②)。

用法: python e6_probe_analyze.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from random import Random

HERE = Path(__file__).resolve().parent
ARMS = ["base", "v2", "v5a"]
BOOT_N = 10000
G6_FLOOR_PP = -5.0


def kw_hit_rate(response: str, keywords: list) -> float:
    if not keywords:
        return None
    low = response.lower()
    return sum(1 for k in keywords if k.lower() in low) / len(keywords)


def paired_boot(diff, n=BOOT_N, seed=42):
    """逐题配对差值的 bootstrap 95% CI (有放回重抽样题目)。"""
    rng = Random(seed)
    m = len(diff)
    stats = []
    for _ in range(n):
        s = sum(diff[rng.randrange(m)] for _ in range(m)) / m
        stats.append(s)
    stats.sort()
    return stats[int(0.025 * n)], stats[int(0.975 * n)]


def main():
    probe = {p["id"]: p for p in json.loads(
        (HERE / "general_probe_v5.json").read_text(encoding="utf-8"))}
    arms = {}
    for arm in ARMS:
        recs = {}
        with open(HERE / f"gen_{arm}.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    recs[r["id"]] = r["response"]
        arms[arm] = recs
        missing = [i for i in probe if i not in recs]
        print(f"{arm}: {len(recs)}/120 条, 缺失 {len(missing)}")
        if missing:
            print(f"  缺失 id: {missing[:10]}")

    common = [i for i in probe if all(i in arms[a] for a in ARMS)]
    print(f"\n三臂共同题数: {len(common)}")

    # 每臂逐题命中率
    rates = {a: {} for a in ARMS}
    residue = Counter()
    for i in common:
        kws = probe[i]["expected_keywords"]
        for a in ARMS:
            resp = arms[a][i]
            if "<think>" in resp or "</think>" in resp:
                residue[a] += 1
            rates[a][i] = kw_hit_rate(resp, kws)

    def mean(a, ids=None):
        xs = [rates[a][i] for i in (ids or common) if rates[a][i] is not None]
        return sum(xs) / len(xs) if xs else None

    print("\n== overall 关键词命中率 ==")
    for a in ARMS:
        print(f"  {a}: {mean(a)*100:.2f}pp  (think 残留 {residue[a]} 条)")

    print("\n== 配对差值 (pp) ==")
    results = {}
    for pair in [("v5a", "base"), ("v5a", "v2")]:
        x, y = pair
        diff = [(rates[x][i] - rates[y][i]) * 100
                for i in common
                if rates[x][i] is not None and rates[y][i] is not None]
        lo, hi = paired_boot(diff)
        d = sum(diff) / len(diff)
        sig = "显著" if (lo > 0 or hi < 0) else "不显著"
        results[f"{x}-{y}"] = {"diff_pp": round(d, 2), "ci95": [round(lo, 2), round(hi, 2)], "n": len(diff), "sig": sig}
        print(f"  {x}−{y}: {d:+.2f}pp [{lo:+.2f}, {hi:+.2f}] n={len(diff)} {sig}")

    # G6 裁决
    g6 = results["v5a-v2"]
    g6_pass = g6["diff_pp"] > G6_FLOOR_PP
    print(f"\n== G6 裁决: v5a−v2 探针 overall {g6['diff_pp']:+.2f}pp "
          f"(要求 > {G6_FLOOR_PP:.0f}pp) -> {'PASS' if g6_pass else 'FAIL'} ==")

    print("\n== 类级命中率 (描述性, n=20/类) ==")
    cats = defaultdict(list)
    for i in common:
        cats[probe[i]["subcategory"]].append(i)
    header = "category".ljust(24) + "".join(a.rjust(8) for a in ARMS) + "v5a-base".rjust(11) + "v5a-v2".rjust(9)
    print(header)
    per_cat = {}
    for c, ids in sorted(cats.items()):
        row = c.ljust(24)
        for a in ARMS:
            row += f"{mean(a, ids)*100:8.1f}"
        row += f"{(mean('v5a', ids)-mean('base', ids))*100:11.1f}"
        row += f"{(mean('v5a', ids)-mean('v2', ids))*100:9.1f}"
        per_cat[c] = {a: round(mean(a, ids)*100, 2) for a in ARMS}
        print(row)

    out = {
        "protocol": "关键词命中率(纯子串); 逐题配对 bootstrap n=10000 Random(42); 类级描述性",
        "n_common": len(common),
        "think_residue": dict(residue),
        "overall_pp": {a: round(mean(a)*100, 2) for a in ARMS},
        "paired": results,
        "G6": {"floor_pp": G6_FLOOR_PP, "value_pp": g6["diff_pp"],
               "pass": g6_pass},
        "per_category_pp": per_cat,
    }
    (HERE / "e6_probe_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {HERE/'e6_probe_result.json'}")


if __name__ == "__main__":
    main()
