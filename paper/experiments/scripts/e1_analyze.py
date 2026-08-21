# -*- coding: utf-8 -*-
"""E1 包分析: E1a 位置不一致率 / E1b 评委间 Spearman / E1c 翻转率。
按各 .done 标志存在与否分别分析。输出 results/e1/e1_report.json + e1_report.md"""
import sys, json, math, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from e1_common import load_jsonl, load_gold, load_judge_cache, log, RESULTS

E1 = RESULTS / "e1"


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [0, 0, 0]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [p, max(0, (c - w) / d), min(1, (c + w) / d)]


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None, None

    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and v[order[j]] == v[order[i]]:
                j += 1
            for k in range(i, j):
                r[order[k]] = (i + j - 1) / 2 + 1
            i = j
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None, None
    r = cov / math.sqrt(vx * vy)
    t = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
    # t 近似 p (大样本近似用正态)
    p = math.erfc(abs(t) / math.sqrt(2)) if n >= 40 else None
    return r, p


def pos_to_v4(rec):
    """把位置裁决映射为 v4 视角: v4_win / base_win / tie"""
    w = rec["winner_pos"]
    if w == "tie":
        return "tie"
    if rec["order"] == "AB":  # A=v4
        return "v4_win" if w == "A" else "base_win"
    else:  # BA: A=base
        return "base_win" if w == "A" else "v4_win"


def analyze_e1a():
    recs = load_jsonl(E1 / "e1a_position_swap.jsonl")
    by_q = {}
    for r in recs:
        by_q.setdefault(r["qid"], {})[r["order"]] = pos_to_v4(r)
    out = {"n_verdicts": len(recs), "n_questions": len(by_q),
           "judge": recs[0]["judge"] if recs else None,
           "base_src": recs[0].get("base_src") if recs else None}
    for order in ("AB", "BA"):
        vs = [d[order] for d in by_q.values() if order in d]
        k = sum(1 for v in vs if v == "v4_win")
        kt = sum(1 for v in vs if v == "tie")
        out[f"v4_winrate_{order}"] = wilson(k, len(vs))
        out[f"tie_{order}"] = kt
    both = [(d["AB"], d["BA"]) for d in by_q.values() if "AB" in d and "BA" in d]
    incon = sum(1 for a, b in both if a != b)
    # 位置不一致 = 两序映射到 v4 视角后裁决不同 (含 tie 差异)
    out["position_inconsistency"] = {"n_pairs": len(both), "n_inconsistent": incon,
                                     "rate": incon / len(both) if both else None,
                                     "wilson": wilson(incon, len(both))}
    hard = sum(1 for a, b in both
               if {a, b} == {"v4_win", "base_win"})
    out["hard_flip_rate"] = hard / len(both) if both else None
    # 双序合并
    allv = [v for d in by_q.values() for v in d.values()]
    out["v4_winrate_pooled"] = wilson(sum(1 for v in allv if v == "v4_win"), len(allv))
    out["tie_pooled"] = sum(1 for v in allv if v == "tie")
    return out


def analyze_e1b():
    meta = json.load(open(E1 / "e1b_meta.json"))
    judge2 = meta["judge"]
    tag = judge2.replace("-", "_")
    re = {(r["qid"], r["model"]): r for r in load_jsonl(E1 / f"e1b_rejudge_{tag}.jsonl")}
    j32 = {m: load_judge_cache(m) for m in ("base", "v2", "v4")}
    dims = ["accuracy", "completeness", "triz_correctness", "structure", "overall"]
    pairs = []
    for m in ("base", "v2", "v4"):
        for qid, jd in j32[m].items():
            r2 = re.get((qid, m))
            if r2 is not None:
                pairs.append((qid, m, jd, r2))
    out = {"judge2": judge2, "judge1": "moonshot-v1-32k", "n_pairs": len(pairs),
           "caveat": meta.get("note")}
    for dim in dims:
        xs = [float(p[2][dim]) for p in pairs if p[2].get(dim) is not None and p[3].get(dim) is not None]
        ys = [float(p[3][dim]) for p in pairs if p[2].get(dim) is not None and p[3].get(dim) is not None]
        r, p = spearman(xs, ys)
        out[f"spearman_{dim}"] = r
        out[f"meandiff_{dim}"] = (sum(y - x for x, y in zip(xs, ys)) / len(xs)) if xs else None
    # 模型均值与排序
    means = {}
    for m in ("base", "v2", "v4"):
        xs = [float(j32[m][q]["overall"]) for q in j32[m]]
        ys = [float(re[(q, m)]["overall"]) for q in j32[m] if (q, m) in re]
        means[m] = {"j32": sum(xs) / len(xs), "j2": sum(ys) / len(ys) if ys else None}
    out["model_means"] = means
    out["rank_j32"] = sorted(means, key=lambda m: -means[m]["j32"])
    out["rank_j2"] = sorted(means, key=lambda m: -(means[m]["j2"] or 0))
    # v4-base 逐题差值符号一致率
    diffs = []
    for qid in j32["v4"]:
        if (qid, "v4") in re and (qid, "base") in re:
            d1 = float(j32["v4"][qid]["overall"]) - float(j32["base"][qid]["overall"])
            d2 = float(re[(qid, "v4")]["overall"]) - float(re[(qid, "base")]["overall"])
            diffs.append((d1, d2))
    agree = sum(1 for a, b in diffs if (a > 0) == (b > 0))
    out["v4base_diff_sign_agree"] = {"n": len(diffs), "agree": agree,
                                     "rate": agree / len(diffs) if diffs else None}
    r, p = spearman([a for a, _ in diffs], [b for _, b in diffs])
    out["v4base_diff_spearman"] = r
    # 子集级: 每子集两评委的 v4>base 方向
    gold = {it["id"]: it["subset"] for it in load_gold()}
    subs = {}
    for qid in j32["v4"]:
        if (qid, "v4") in re and (qid, "base") in re:
            subs.setdefault(gold[qid], []).append(
                (float(j32["v4"][qid]["overall"]) - float(j32["base"][qid]["overall"]),
                 float(re[(qid, "v4")]["overall"]) - float(re[(qid, "base")]["overall"])))
    out["subset_diff"] = {s: {"n": len(v),
                              "j32_meandiff": sum(a for a, _ in v) / len(v),
                              "j2_meandiff": sum(b for _, b in v) / len(v)}
                          for s, v in subs.items()}
    return out


def majority(verdicts):
    from collections import Counter
    c = Counter(verdicts)
    top = c.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "tie"  # 平票归 tie
    return top[0][0]


def analyze_e1c():
    recs = load_jsonl(E1 / "e1c_flip.jsonl")
    out = {"n_verdicts": len(recs), "judge": recs[0]["judge"] if recs else None}
    for temp in (0.0, 0.7):
        rs = [r for r in recs if r["temp"] == temp]
        if not rs:
            continue
        # 映射到 v4 视角
        for r in rs:
            r["v"] = pos_to_v4(r)
        cells = {}
        for r in rs:
            cells.setdefault((r["qid"], r["order"]), []).append(r["v"])
        # 翻转率: 5 次重复非全一致的 cell 比例; 平均不一致比例
        n_unanimous = sum(1 for v in cells.values() if len(set(v)) == 1)
        flips = []
        for vs in cells.values():
            from collections import Counter
            c = Counter(vs)
            flips.append(1 - c.most_common(1)[0][1] / len(vs))
        cell_flip = {f"{k[0]}|{k[1]}": v for k, v in cells.items()}
        # 最终裁决: 每 (qid,order) 5 次多数; 每题最终 = 双序 10 次多数
        final_cell = {k: majority(v) for k, v in cells.items()}
        by_q = {}
        for (q, o), v in final_cell.items():
            by_q.setdefault(q, []).append(v)
        final_q = {q: majority(vs) for q, vs in by_q.items()}
        # 多数表决收敛: k=1,3,5 (每 cell 前 k 次多数 vs 5 次多数)
        conv = {}
        for k in (1, 3, 5):
            agree = sum(1 for key, vs in cells.items()
                        if len(vs) >= 5 and majority(vs[:k]) == final_cell[key])
            tot = sum(1 for vs in cells.values() if len(vs) >= 5)
            conv[f"k{k}"] = {"agree": agree, "n": tot,
                             "rate": agree / tot if tot else None}
        # 单次裁决 vs 最终题裁决一致率
        single_agree = sum(1 for r in rs if final_q.get(r["qid"]) == r["v"])
        out[f"T{temp}"] = {
            "n_verdicts": len(rs),
            "n_cells": len(cells),
            "unanimous_cell_rate": n_unanimous / len(cells) if cells else None,
            "flip_rate_mean": sum(flips) / len(flips) if flips else None,
            "convergence": conv,
            "single_vs_final_agree": single_agree / len(rs) if rs else None,
            "final_verdicts": final_q,
            "v4_win_final": sum(1 for v in final_q.values() if v == "v4_win"),
            "tie_final": sum(1 for v in final_q.values() if v == "tie"),
        }
    out["literature_flip_rate"] = 0.136
    return out


def main():
    report = {}
    md = ["# E1 包报告 (judge 方法学)\n"]
    if (E1 / "e1a.done").exists():
        a = analyze_e1a()
        report["e1a"] = a
        md.append(f"""## E1a 位置交换双跑 (judge={a['judge']}, base_src={a['base_src']})
- 裁决数 {a['n_verdicts']} / 题数 {a['n_questions']}
- AB 序 v4 胜率: {a['v4_winrate_AB'][0]:.3f} [{a['v4_winrate_AB'][1]:.3f}, {a['v4_winrate_AB'][2]:.3f}] (tie={a['tie_AB']})
- BA 序 v4 胜率: {a['v4_winrate_BA'][0]:.3f} [{a['v4_winrate_BA'][1]:.3f}, {a['v4_winrate_BA'][2]:.3f}] (tie={a['tie_BA']})
- 合并 v4 胜率: {a['v4_winrate_pooled'][0]:.3f} [{a['v4_winrate_pooled'][1]:.3f}, {a['v4_winrate_pooled'][2]:.3f}]
- **位置不一致率: {a['position_inconsistency']['rate']:.3f}** [{a['position_inconsistency']['wilson'][1]:.3f}, {a['position_inconsistency']['wilson'][2]:.3f}] (含 tie 差异)
- 硬翻转 (v4_win<->base_win) 率: {a['hard_flip_rate']:.3f}
- 判定: 位置不一致率 {'<10%, judge 轨位置稳健' if a['position_inconsistency']['rate'] < 0.10 else '>10%, judge 轨数字须以双序平均重报'}
""")
    if (E1 / "e1b.done").exists():
        b = analyze_e1b()
        report["e1b"] = b
        md.append(f"""## E1b 多评委交叉 ({b['judge1']} vs {b['judge2']}, n={b['n_pairs']})
- 注意: {b['caveat']}
- 逐题 Spearman: overall={b['spearman_overall']:.3f}, accuracy={b.get('spearman_accuracy') and round(b['spearman_accuracy'],3)}, triz={b.get('spearman_triz_correctness') and round(b['spearman_triz_correctness'],3)}
- 模型均值: {json.dumps(b['model_means'], ensure_ascii=False)}
- 排序一致性: 32k={b['rank_j32']} vs {b['judge2']}={b['rank_j2']}
- v4-base 逐题差值符号一致率: {b['v4base_diff_sign_agree']['rate']:.3f} (n={b['v4base_diff_sign_agree']['n']})
- 子集差值: {json.dumps(b['subset_diff'], ensure_ascii=False)}
""")
    if (E1 / "e1c.done").exists():
        c = analyze_e1c()
        report["e1c"] = c
        for tkey in ("T0.0", "T0.7"):
            if tkey in c:
                t = c[tkey]
                md.append(f"""## E1c 翻转率 {tkey} (n_verdicts={t['n_verdicts']}, cells={t['n_cells']})
- 全一致 cell 比例: {t['unanimous_cell_rate']:.3f}
- **平均翻转率: {t['flip_rate_mean']:.4f}** (文献 13.6%)
- 多数表决收敛 (与 5 次多数一致率): k=1 {t['convergence']['k1']['rate']:.3f}, k=3 {t['convergence']['k3']['rate']:.3f}, k=5 {t['convergence']['k5']['rate']:.3f}
- 单次裁决 vs 最终题裁决一致率: {t['single_vs_final_agree']:.3f}
- 最终裁决: v4 胜 {t['v4_win_final']}/40, tie {t['tie_final']}/40
""")
    (E1 / "e1_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    (E1 / "e1_report.md").write_text("\n".join(md))
    log("e1_report 已写")
    print("\n".join(md))


if __name__ == "__main__":
    main()
