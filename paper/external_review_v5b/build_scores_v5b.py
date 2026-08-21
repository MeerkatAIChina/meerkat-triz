#!/usr/bin/env python3
"""构建 v5b 决策门 scores.json（schema 对齐 v5a 的 results/v5/v5_scores.json）。

- v5_vs_base：直接取 v5b 评测 json 的 paired_vs_baseline.tracks（harness 原生）
- v5_vs_v2 ：用 stats_utils.bootstrap_diff（stdlib Random(42), n=10000）从
             v5b 与 v2 两个评测 json 的 records 逐题配对计算（v5b − v2）
- 自检：用同法复算 v5b−base，与 harness 原生数字逐位比对，不一致即报错退出

用法：python3 build_scores_v5b.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "pipeline_v5" / "eval"))
from stats_utils import bootstrap_diff  # noqa: E402

WD = HERE
V5B = WD / "eval_v5_v5b_gold_20260731_195036.json"
BASE = WD / "eval_v5_base_goldfix_v5_20260726_234434.json"
V2 = WD / "eval_v5_v2_gold_v5_20260727_081912.json"
OUT = WD / "v5b_scores.json"

SUBSETS = ["ariz_guidance", "case_generation", "concept_explanation",
           "contradiction_analysis", "innovation_assessment",
           "principle_recommendation"]


def per_item(eval_json):
    """id 排序后的逐题 (kw_hit_rate, judge_overall)。"""
    d = json.load(open(eval_json, encoding="utf-8"))
    rows = sorted(d["records"], key=lambda r: r["id"])
    kw = [float(r["kw_hit_rate"]) for r in rows]
    ju = [float(r["judge_overall"]) for r in rows]
    subs = [r["subset"] for r in rows]
    return kw, ju, subs


def paired_block(a_json, b_json):
    """b − a 的 overall + per_subset 配对差值（两轨）。"""
    ka, ja, sa = per_item(a_json)
    kb, jb, sb = per_item(b_json)
    assert sa == sb, "两评测题序/子集不一致"
    out = {}
    for name, aa, bb in (("keyword", ka, kb), ("judge_armA", ja, jb)):
        blk = {"overall": bootstrap_diff(aa, bb), "per_subset": {}}
        for s in SUBSETS:
            idx = [i for i, x in enumerate(sb) if x == s]
            blk["per_subset"][s] = bootstrap_diff([aa[i] for i in idx],
                                                  [bb[i] for i in idx])
        out[name] = blk
    return out


def main():
    v5b = json.load(open(V5B, encoding="utf-8"))

    # ---- 自检：复算 v5b−base 应与 harness 原生 paired_vs_baseline 一致 ----
    recomputed = paired_block(BASE, V5B)
    native = v5b["paired_vs_baseline"]["tracks"]
    for track in ("keyword", "judge_armA"):
        rd = recomputed[track]["overall"]
        nd = native[track]["overall"]
        assert abs(rd["diff"] - nd["diff"]) < 1e-9, \
            f"{track} diff 不一致: 复算 {rd['diff']} vs 原生 {nd['diff']}"
        for s in SUBSETS:
            r = recomputed[track]["per_subset"][s]
            n = native[track]["per_subset"][s]
            assert abs(r["diff"] - n["diff"]) < 1e-9, f"{track}/{s} diff 不一致"
    print("自检 PASSED: 复算 v5b−base 与 harness 原生数字逐位一致")

    scores = {
        "meta": {"candidate": "v5b", "anchor": "base_goldfix_v5",
                 "eval_file": "data/processed/v5_data/v5_gold.jsonl",
                 "n_items": 300,
                 "note": "v5b scores 由 build_scores_v5b.py 构建; "
                         "v5_vs_base=harness 原生, v5_vs_v2=同法复算"},
        "v5_vs_base": native,
        "v5_vs_v2": paired_block(V2, V5B),
        "subset_n": {s: sum(1 for r in v5b["records"] if r["subset"] == s)
                     for s in SUBSETS},
        "attributions": {},
        "probe": None,
        "overrefusal": {"hit_rate": v5b["overrefusal"]["hit_rate"],
                        "pass": v5b["overrefusal"]["pass"]},
        "pairwise_final": None,
    }
    json.dump(scores, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"已写出 {OUT}")
    # 速览
    for cmp_name in ("v5_vs_base", "v5_vs_v2"):
        c = scores[cmp_name]
        j, k = c["judge_armA"]["overall"], c["keyword"]["overall"]
        print(f"{cmp_name}: judge {j['diff']:+.4f} {j['ci95']} | "
              f"kw {k['diff']:+.4f} {k['ci95']}")
        for s in SUBSETS:
            jj, kk = c["judge_armA"]["per_subset"][s], c["keyword"]["per_subset"][s]
            print(f"  {s}: judge {jj['diff']:+.4f} {jj['ci95']} | "
                  f"kw {kk['diff']:+.4f} {kk['ci95']}")


if __name__ == "__main__":
    main()
