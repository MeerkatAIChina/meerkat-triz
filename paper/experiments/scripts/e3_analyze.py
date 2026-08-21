# -*- coding: utf-8 -*-
"""E3 分析: ARIZ rubric 轨 vs 关键词轨对照, 漏判率, 配对 bootstrap。
输出 results/e3/e3_report.json + e3_report.md"""
import sys, json, math, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "e1"))
from e1_common import load_jsonl, load_gold, load_kw_cache, log, RESULTS

E3 = RESULTS / "e3"


def paired_bootstrap(a, b, n_boot=10000, seed=42):
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    return [sum(diffs) / n, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]]


def main():
    recs = {(r["qid"], r["model"]): r for r in load_jsonl(E3 / "e3_ariz_rubric.jsonl")}
    gold = [it for it in load_gold() if it["subset"] == "ariz_guidance"]
    kw = {m: load_kw_cache(m) for m in ("base", "v2", "v4")}
    ids = [it["id"] for it in gold]
    kws = {it["id"]: it["keywords"] for it in gold}

    steps = [f"step{i}" for i in range(1, 7)]
    # rubric 覆盖 = 6 步均值
    rub = {}
    for (qid, m), r in recs.items():
        rub[(qid, m)] = sum(r[s] for s in steps) / 6

    report = {"n_q": len(ids), "models": {}, "leakage": {}, "pairs": {},
              "miss_examples": []}
    for m in ("base", "v2", "v4"):
        rv = [rub[(i, m)] for i in ids if (i, m) in rub]
        kv = [kw[m][i]["kw_hit_rate"] for i in ids]
        report["models"][m] = {
            "rubric_coverage_mean": sum(rv) / len(rv) if rv else None,
            "kw_mean": sum(kv) / len(kv),
            "per_step": {s: sum(recs[(i, m)][s] for i in ids if (i, m) in recs) / len(rv)
                         for s in steps} if rv else {}}
    # 漏判率: rubric 步=1 但题目 kw_hit_rate < 0.5 的 (qid,model); 更细:
    # 漏判 = rubric 判该步覆盖(1)而对应期望关键词未字面命中 — 用题级近似:
    # kw 低(<0.5) 且 rubric 高(>=0.5) = 关键词低估
    n_leak, n_tot = 0, 0
    for i in ids:
        for m in ("base", "v2", "v4"):
            if (i, m) not in rub:
                continue
            n_tot += 1
            if kw[m][i]["kw_hit_rate"] < 0.5 and rub[(i, m)] >= 0.5:
                n_leak += 1
                report["miss_examples"].append({
                    "qid": i, "model": m,
                    "kw_hit_rate": kw[m][i]["kw_hit_rate"],
                    "rubric": rub[(i, m)],
                    "keywords": kws[i],
                    "evidence": {s: recs[(i, m)]["evidence"][s] for s in steps
                                 if recs[(i, m)][s] == 1}})
    report["leakage"] = {"kw_low_rubric_high": n_leak, "total": n_tot,
                         "rate": n_leak / n_tot if n_tot else None}
    # 反向: kw 高 rubric 低
    n_rev = sum(1 for i in ids for m in ("base", "v2", "v4")
                if (i, m) in rub and kw[m][i]["kw_hit_rate"] >= 0.5 and rub[(i, m)] < 0.5)
    report["leakage"]["kw_high_rubric_low"] = n_rev
    # 配对 bootstrap: rubric 轨 v4 vs v2, v4 vs base, v2 vs base; kw 轨同
    for a, b in (("v4", "v2"), ("v4", "base"), ("v2", "base")):
        ra = [rub.get((i, a)) for i in ids]
        rb = [rub.get((i, b)) for i in ids]
        valid = [(x, y) for x, y in zip(ra, rb) if x is not None and y is not None]
        ka = [kw[a][i]["kw_hit_rate"] for i in ids]
        kb = [kw[b][i]["kw_hit_rate"] for i in ids]
        report["pairs"][f"{a}_vs_{b}"] = {
            "rubric_bootstrap": paired_bootstrap([x for x, _ in valid], [y for _, y in valid]) if valid else None,
            "kw_bootstrap": paired_bootstrap(ka, kb)}
    (E3 / "e3_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    md = ["# E3 ARIZ rubric 重判报告\n",
          f"- 题数 {len(ids)}, judge 裁决 {len(recs)}/60",
          "## 模型均值 (rubric 覆盖 vs 关键词轨)"]
    for m, d in report["models"].items():
        md.append(f"- {m}: rubric={d['rubric_coverage_mean']:.3f} kw={d['kw_mean']:.3f} "
                  f"per_step={json.dumps({k: round(v,2) for k,v in d['per_step'].items()})}")
    md.append(f"\n## 漏判率\n- kw<0.5 且 rubric>=0.5 (关键词低估): **{n_leak}/{n_tot} = {report['leakage']['rate']:.3f}**")
    md.append(f"- kw>=0.5 且 rubric<0.5 (关键词高估): {n_rev}/{n_tot}")
    md.append("\n## 配对 bootstrap 95%CI")
    for k, v in report["pairs"].items():
        md.append(f"- {k}: rubric {v['rubric_bootstrap'] and [round(x,4) for x in v['rubric_bootstrap']]} | kw {[round(x,4) for x in v['kw_bootstrap']]}")
    md.append(f"\n## 漏判表述清单 ({len(report['miss_examples'])} 条)")
    for e in report["miss_examples"]:
        md.append(f"- {e['qid']}/{e['model']} kw={e['kw_hit_rate']:.2f} rubric={e['rubric']:.2f} "
                  f"期望词={e['keywords']} 证据={json.dumps(e['evidence'], ensure_ascii=False)}")
    (E3 / "e3_report.md").write_text("\n".join(md))
    log("e3_report 已写")
    print("\n".join(md))


if __name__ == "__main__":
    main()
