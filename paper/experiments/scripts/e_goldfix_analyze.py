# -*- coding: utf-8 -*-
"""干净 base 锚点 (base_goldfix) 补跑分析: E1a' / E1b补臂 / E3'。
追加写入 e1_report.md 与 e3_report.md, 并输出 e_goldfix_report.json。"""
import sys, json, math, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from e1_common import (load_jsonl, load_gold, load_judge_cache, load_kw_cache,
                       log, RESULTS)
from e1_analyze import wilson, spearman, pos_to_v4

E1 = RESULTS / "e1"
E3 = RESULTS / "e3"


def paired_bootstrap(a, b, n_boot=10000, seed=42):
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    return [sum(diffs) / n, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]]


def analyze_e1a_file(path):
    recs = load_jsonl(path)
    by_q = {}
    for r in recs:
        by_q.setdefault(r["qid"], {})[r["order"]] = pos_to_v4(r)
    out = {"n_verdicts": len(recs), "n_questions": len(by_q)}
    for order in ("AB", "BA"):
        vs = [d[order] for d in by_q.values() if order in d]
        out[f"v4_winrate_{order}"] = wilson(sum(1 for v in vs if v == "v4_win"), len(vs))
        out[f"tie_{order}"] = sum(1 for v in vs if v == "tie")
    both = [(d["AB"], d["BA"]) for d in by_q.values() if "AB" in d and "BA" in d]
    incon = sum(1 for a, b in both if a != b)
    out["position_inconsistency"] = {"n_pairs": len(both), "n_inconsistent": incon,
                                     "rate": incon / len(both) if both else None,
                                     "wilson": wilson(incon, len(both))}
    out["hard_flip_rate"] = sum(1 for a, b in both if {a, b} == {"v4_win", "base_win"}) / len(both) if both else None
    allv = [v for d in by_q.values() for v in d.values()]
    out["v4_winrate_pooled"] = wilson(sum(1 for v in allv if v == "v4_win"), len(allv))
    out["tie_pooled"] = sum(1 for v in allv if v == "tie")
    out["b_pos_winrate"] = sum(1 for r in recs if r["winner_pos"] == "B") / len(recs)
    return out


def main():
    report = {}
    md1 = ["\n---\n# 干净 base 锚点 (base_goldfix) 补跑 — 旧 base (think 污染) 相关数字以此为准\n"]

    # E1a'
    old = analyze_e1a_file(E1 / "e1a_position_swap.jsonl")
    new = analyze_e1a_file(E1 / "e1a_position_swap_goldfix.jsonl")
    report["e1a_goldfix"] = {"clean": new, "polluted_ref": old}
    md1.append(f"""## E1a' 位置交换 (v4 vs base_goldfix, judge=moonshot-v1-32k)
- AB 序 v4 胜率: {new['v4_winrate_AB'][0]:.3f} [{new['v4_winrate_AB'][1]:.3f},{new['v4_winrate_AB'][2]:.3f}] | BA 序: {new['v4_winrate_BA'][0]:.3f} [{new['v4_winrate_BA'][1]:.3f},{new['v4_winrate_BA'][2]:.3f}]
- **双序合并 v4 胜率: {new['v4_winrate_pooled'][0]:.3f} [{new['v4_winrate_pooled'][1]:.3f},{new['v4_winrate_pooled'][2]:.3f}]** (tie={new['tie_pooled']})
- 位置不一致率: {new['position_inconsistency']['rate']:.3f} (硬翻转 {new['hard_flip_rate']:.3f}, B 位胜率 {new['b_pos_winrate']:.3f})
- 对照旧污染 base: 合并胜率 {old['v4_winrate_pooled'][0]:.3f}, 位置不一致率 {old['position_inconsistency']['rate']:.3f}
- 结论: 干净锚点下第二位锚定依然存在 ({new['b_pos_winrate']:.2f} vs 旧 {old['b_pos_winrate']:.2f}), 双序合并后 v4 {'仍显著优于' if new['v4_winrate_pooled'][1] > 0.5 else '不显著优于' if new['v4_winrate_pooled'][2] < 0.5 else '与'} 干净 base{' (CI 跨过 0.5)' if new['v4_winrate_pooled'][1] <= 0.5 <= new['v4_winrate_pooled'][2] else ''}
""")

    # E1b 补臂
    j32 = {m: load_judge_cache(m) for m in ("base_goldfix", "v2", "v4")}
    re8 = {(r["qid"], r["model"]): r for r in load_jsonl(E1 / "e1b_rejudge_moonshot_v1_8k.jsonl")}
    dims = ["accuracy", "completeness", "triz_correctness", "structure", "overall"]
    pairs = [(q, m, j32[m][q], re8[(q, m)])
             for m in ("base_goldfix", "v2", "v4") for q in j32[m]
             if (q, m) in re8]
    rho, _ = spearman([float(p[2]["overall"]) for p in pairs],
                      [float(p[3]["overall"]) for p in pairs])
    means = {}
    for m in ("base_goldfix", "v2", "v4"):
        xs = [float(j32[m][q]["overall"]) for q in j32[m]]
        ys = [float(re8[(q, m)]["overall"]) for q in j32[m] if (q, m) in re8]
        means[m] = {"j32": sum(xs) / len(xs), "j8": sum(ys) / len(ys)}
    sign = {}
    for a in ("v4", "v2"):
        diffs = [(float(j32[a][q]["overall"]) - float(j32["base_goldfix"][q]["overall"]),
                  float(re8[(q, a)]["overall"]) - float(re8[(q, "base_goldfix")]["overall"]))
                 for q in j32[a] if (q, a) in re8 and (q, "base_goldfix") in re8]
        agree = sum(1 for x, y in diffs if (x > 0) == (y > 0))
        sign[f"{a}_vs_base_goldfix"] = {
            "sign_agree": agree / len(diffs), "n": len(diffs),
            "j32_meandiff": sum(x for x, _ in diffs) / len(diffs),
            "j8_meandiff": sum(y for _, y in diffs) / len(diffs),
            "j32_bootstrap": paired_bootstrap(
                [float(j32[a][q]["overall"]) for q in j32[a]],
                [float(j32["base_goldfix"][q]["overall"]) for q in j32[a]])}
    report["e1b_goldfix"] = {"spearman_overall": rho, "n_pairs": len(pairs),
                             "model_means": means, "sign": sign}
    md1.append(f"""## E1b 补臂 干净锚点跨评委 (32k vs 8k, base_goldfix/v2/v4, n={len(pairs)})
- **逐题 Spearman ρ = {rho:.3f}**
- 模型均值 (32k/8k): base_goldfix {means['base_goldfix']['j32']:.2f}/{means['base_goldfix']['j8']:.2f}, v2 {means['v2']['j32']:.2f}/{means['v2']['j8']:.2f}, v4 {means['v4']['j32']:.2f}/{means['v4']['j8']:.2f}
- v4 vs base_goldfix: 32k 均差 {sign['v4_vs_base_goldfix']['j32_meandiff']:+.3f} {sign['v4_vs_base_goldfix']['j32_bootstrap']}, 8k 均差 {sign['v4_vs_base_goldfix']['j8_meandiff']:+.3f}, 符号一致率 {sign['v4_vs_base_goldfix']['sign_agree']:.3f}
- v2 vs base_goldfix: 32k 均差 {sign['v2_vs_base_goldfix']['j32_meandiff']:+.3f} {sign['v2_vs_base_goldfix']['j32_bootstrap']}, 8k 均差 {sign['v2_vs_base_goldfix']['j8_meandiff']:+.3f}, 符号一致率 {sign['v2_vs_base_goldfix']['sign_agree']:.3f}
- ⚠️ 注意: 干净 base judge 2.87 已高于 v2 2.58/v4 2.57 — "v4 大幅优于 base" 的旧结论 (+1.00) 是 think 污染伪影, 干净锚点下 v4/v2 反而低于 base_goldfix
""")

    # E3'
    recs = {(r["qid"], r["model"]): r for r in load_jsonl(E3 / "e3_ariz_rubric.jsonl")}
    gold = [it for it in load_gold() if it["subset"] == "ariz_guidance"]
    kw = {m: load_kw_cache(m) for m in ("base_goldfix", "v2", "v4")}
    ids = [it["id"] for it in gold]
    steps = [f"step{i}" for i in range(1, 7)]
    rub = {k: sum(r[s] for s in steps) / 6 for k, r in recs.items()}
    models3 = {}
    for m in ("base_goldfix", "v2", "v4"):
        rv = [rub[(i, m)] for i in ids if (i, m) in rub]
        kv = [kw[m][i]["kw_hit_rate"] for i in ids]
        models3[m] = {"rubric": sum(rv) / len(rv), "kw": sum(kv) / len(kv),
                      "per_step": {s: sum(recs[(i, m)][s] for i in ids) / len(rv) for s in steps}}
    pairs3 = {}
    for a in ("v4", "v2"):
        ra = [rub[(i, a)] for i in ids]
        rb = [rub[(i, "base_goldfix")] for i in ids]
        pairs3[f"{a}_vs_base_goldfix_rubric"] = paired_bootstrap(ra, rb)
    leak = sum(1 for i in ids
               if kw["base_goldfix"][i]["kw_hit_rate"] < 0.5 and rub[(i, "base_goldfix")] >= 0.5)
    report["e3_goldfix"] = {"models": models3, "pairs": pairs3,
                            "base_goldfix_leak": f"{leak}/20"}
    md3 = [f"""
---
# 干净 base 锚点 (base_goldfix) 补跑 — 上文 base rubric=0.675 (草稿分) 作废

## E3' rubric 轨三方 (干净锚点)
- base_goldfix: rubric={models3['base_goldfix']['rubric']:.3f} kw={models3['base_goldfix']['kw']:.3f} per_step={json.dumps({k: round(v,2) for k,v in models3['base_goldfix']['per_step'].items()})}
- v2: rubric={models3['v2']['rubric']:.3f} | v4: rubric={models3['v4']['rubric']:.3f} (不变)
- **v4 vs base_goldfix rubric: {[round(x,4) for x in pairs3['v4_vs_base_goldfix_rubric']]}**
- **v2 vs base_goldfix rubric: {[round(x,4) for x in pairs3['v2_vs_base_goldfix_rubric']]}**
- base_goldfix 关键词漏判 (kw<0.5 且 rubric>=0.5): {leak}/20
"""]
    (E1 / "e_goldfix_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    with open(E1 / "e1_report.md", "a") as f:
        f.write("\n".join(md1))
    with open(E3 / "e3_report.md", "a") as f:
        f.write("\n".join(md3))
    log("goldfix 分析完成")
    print("\n".join(md1 + md3))


if __name__ == "__main__":
    main()
