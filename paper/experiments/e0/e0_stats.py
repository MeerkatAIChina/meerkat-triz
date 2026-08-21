#!/usr/bin/env python
"""E0: 干净 base 锚点下的配对统计 + 决策门重判。

输入: 各模型 eval_v4_<tag>_*.json (records 内含 kw_hit_rate / judge_overall / subset)。
方法: 与 eval_harness.py 完全一致的 stdlib 实现 —
  paired bootstrap 10000 次 (random.Random(42)) 95% 百分位 CI,
  McNemar 精确双侧, Wilson 95% CI。
对比: v4 vs base_goldfix, v2 vs base_goldfix, v4 vs v2 (双轨, overall + 子集)。
决策门 (与 chain_v4/README 一致):
  v4 judge 轨 overall 显著 > base 且 judge 轨所有子集无显著退化 → "建议替代 v2",
  否则 "保留 v2"。另附 W3 关注的 concept_explanation kw 轨子项。
"""
import argparse
import json
import math
import random
from collections import defaultdict


def wilson_ci(k, n, z=1.959963984540054):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def bootstrap_diff(arr_a, arr_b, n_boot=10000, seed=42):
    n = len(arr_a)
    if n == 0:
        return None
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            s += arr_b[i] - arr_a[i]
        diffs.append(s / n)
    diffs.sort()
    return {"diff": sum(arr_b[i] - arr_a[i] for i in range(n)) / n,
            "ci95": [diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]],
            "n": n}


def load_records(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["records"]


KW_THR, JD_THR = 0.5, 3


def paired(base_records, this_records):
    base_by_id = {r["id"]: r for r in base_records}
    common = [r for r in this_records if r["id"] in base_by_id]
    out = {"n_common": len(common), "tracks": {}}

    def run(vfn, pfn):
        pairs = [(vfn(base_by_id[r["id"]]), vfn(r), r["subset"],
                  pfn(base_by_id[r["id"]]), pfn(r)) for r in common]
        pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
        a = [p[0] for p in pairs]
        b = [p[1] for p in pairs]
        res = {"overall": bootstrap_diff(a, b)}
        per = {}
        for s in sorted({p[2] for p in pairs}):
            sa = [p[0] for p in pairs if p[2] == s]
            sb = [p[1] for p in pairs if p[2] == s]
            per[s] = bootstrap_diff(sa, sb)
        res["per_subset"] = per
        b10 = sum(1 for p in pairs if not p[3] and p[4])
        b01 = sum(1 for p in pairs if p[3] and not p[4])
        res["mcnemar"] = {"base_fail_this_pass": b10, "base_pass_this_fail": b01,
                          "p": mcnemar_exact_p(b10, b01)}
        return res

    out["tracks"]["keyword"] = run(
        lambda r: r["kw_hit_rate"], lambda r: (r["kw_hit_rate"] or 0) >= KW_THR)
    out["tracks"]["judge"] = run(
        lambda r: r["judge_overall"], lambda r: (r["judge_overall"] or 0) >= JD_THR)
    return out


def track_summary(records, vfn, pfn):
    vals = [vfn(r) for r in records if vfn(r) is not None]
    passes = [1 if pfn(r) else 0 for r in records if vfn(r) is not None]
    p, lo, hi = wilson_ci(sum(passes), len(passes))
    subsets = defaultdict(list)
    for r in records:
        v = vfn(r)
        if v is not None:
            subsets[r["subset"]].append(v)
    return {"n": len(vals), "mean": sum(vals) / len(vals) if vals else None,
            "per_subset": {s: {"n": len(v), "mean": sum(v) / len(v)}
                           for s, v in sorted(subsets.items())},
            "pass_rate": {"p": p, "wilson_ci95": [lo, hi],
                          "k": sum(passes), "n": len(passes)}}


def fmt(d, nd=4):
    if d is None:
        return "N/A"
    return f"{d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"


def sig(d):
    return d and (d["ci95"][0] > 0 or d["ci95"][1] < 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="干净 base (base_goldfix) eval json")
    ap.add_argument("--v2", required=True)
    ap.add_argument("--v4", required=True)
    ap.add_argument("--base-polluted", default=None, help="原污染 base eval json (前后对比)")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    base = load_records(args.base)
    v2 = load_records(args.v2)
    v4 = load_records(args.v4)
    models = {"base_goldfix": base, "v2": v2, "v4": v4}

    summaries = {}
    for name, recs in models.items():
        summaries[name] = {
            "keyword": track_summary(recs, lambda r: r["kw_hit_rate"],
                                     lambda r: (r["kw_hit_rate"] or 0) >= KW_THR),
            "judge": track_summary(recs, lambda r: r["judge_overall"],
                                   lambda r: (r["judge_overall"] or 0) >= JD_THR)}

    comparisons = {
        "v4_vs_base_goldfix": paired(base, v4),
        "v2_vs_base_goldfix": paired(base, v2),
        "v4_vs_v2": paired(v2, v4),
    }
    if args.base_polluted:
        pol = load_records(args.base_polluted)
        summaries["base_polluted"] = {
            "keyword": track_summary(pol, lambda r: r["kw_hit_rate"],
                                     lambda r: (r["kw_hit_rate"] or 0) >= KW_THR),
            "judge": track_summary(pol, lambda r: r["judge_overall"],
                                   lambda r: (r["judge_overall"] or 0) >= JD_THR)}
        comparisons["v4_vs_base_polluted"] = paired(pol, v4)

    # ---- 决策门 ----
    c = comparisons["v4_vs_base_goldfix"]
    judge_overall_ok = sig(c["tracks"]["judge"]["overall"]) and \
        c["tracks"]["judge"]["overall"]["diff"] > 0
    judge_regressions = {s: d for s, d in c["tracks"]["judge"]["per_subset"].items()
                         if sig(d) and d["diff"] < 0}
    kw_regressions = {s: d for s, d in c["tracks"]["keyword"]["per_subset"].items()
                      if sig(d) and d["diff"] < 0}
    gate = "replace_v2" if (judge_overall_ok and not judge_regressions
                            and not kw_regressions) else "keep_v2"
    decision_gate = {
        "rule": "v4 judge overall 显著>base 且两轨所有子集无显著退化 → 建议替代 v2, 否则保留 v2",
        "judge_overall_significant_positive": judge_overall_ok,
        "judge_subset_significant_regressions": judge_regressions,
        "kw_subset_significant_regressions": kw_regressions,
        "decision": gate,
    }

    result = {"anchor": "base_goldfix (干净 base 锚点)",
              "stats_method": "paired bootstrap n=10000 seed=42 (stdlib, 与 eval_harness 一致)"
                              " + McNemar 精确 + Wilson CI",
              "summaries": summaries, "comparisons": comparisons,
              "decision_gate": decision_gate}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ---- markdown ----
    L = ["# E0 干净 base 锚点统计报告", ""]
    L.append("## 各模型双轨总览 (n=100)")
    L.append("")
    L.append("| 模型 | kw 均值 | kw pass (Wilson) | judge 均值 | judge pass (Wilson) |")
    L.append("|---|---|---|---|---|")
    for name, s in summaries.items():
        kw, jd = s["keyword"], s["judge"]
        L.append(f"| {name} | {kw['mean']:.4f} | "
                 f"{kw['pass_rate']['p']:.3f} [{kw['pass_rate']['wilson_ci95'][0]:.3f}, "
                 f"{kw['pass_rate']['wilson_ci95'][1]:.3f}] | "
                 f"{jd['mean']:.4f} | "
                 f"{jd['pass_rate']['p']:.3f} [{jd['pass_rate']['wilson_ci95'][0]:.3f}, "
                 f"{jd['pass_rate']['wilson_ci95'][1]:.3f}] |")
    L.append("")
    for cname, comp in comparisons.items():
        L.append(f"## {cname} (共同题 n={comp['n_common']})")
        L.append("")
        for tname, t in comp["tracks"].items():
            o = t["overall"]
            mc = t["mcnemar"]
            L.append(f"### {tname} 轨")
            L.append(f"- overall: **{fmt(o)}** "
                     f"({'显著' if sig(o) else '不显著'})")
            L.append(f"- McNemar: 翻正 {mc['base_fail_this_pass']} / "
                     f"翻负 {mc['base_pass_this_fail']}, p={mc['p']:.4g}")
            L.append("")
            L.append("| 子集 | 差值 | 95% CI | 显著 |")
            L.append("|---|---|---|---|")
            for s, d in t["per_subset"].items():
                if d:
                    L.append(f"| {s} | {d['diff']:+.4f} | "
                             f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] | "
                             f"{'✅' if sig(d) else '—'} |")
            L.append("")
    L.append("## 决策门 (干净 base 锚点)")
    L.append("")
    L.append(f"- 规则: {decision_gate['rule']}")
    L.append(f"- judge overall 显著为正: {judge_overall_ok}")
    L.append(f"- judge 子集显著退化: {list(judge_regressions) or '无'}")
    L.append(f"- kw 子集显著退化: {list(kw_regressions) or '无'}")
    L.append(f"- **判定: {'建议替代 v2' if gate == 'replace_v2' else '保留 v2'}**")
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(json.dumps({"status": "OK", "decision": gate,
                      "v4_vs_basefix_judge": comparisons["v4_vs_base_goldfix"]["tracks"]["judge"]["overall"],
                      "v4_vs_basefix_kw": comparisons["v4_vs_base_goldfix"]["tracks"]["keyword"]["overall"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
