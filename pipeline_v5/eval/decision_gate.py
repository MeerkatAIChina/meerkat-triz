#!/usr/bin/env python
"""
pipeline_v5 决策门 2.0 (§6.7): 七道判据 G1–G7 + G0 overrefusal 附加门,
全部可机检, 无自由裁量。

输入: scores JSON (schema 见 backtest/gate_scores_v4_backtest.json 实例):
{
  "meta": {"candidate": "v5", "anchor": "base_goldfix 谱系", ...},
  "v5_vs_base": {
    "judge_armA": {"overall": {"diff":…,"ci95":[…]}, "per_subset": {…}},
    "judge_armB": {"overall": {…} | null, "note": "…"},
    "keyword":    {"overall": {…}, "per_subset": {…}}
  },
  "v5_vs_v2": {同上结构},
  "subset_n": {"ariz_guidance": 20, ...},
  "attributions": {"<subset>": "true_missing"|"artifact"|null},
  "probe": {"v5_minus_v2_overall_pp": float | null},
  "overrefusal": {"hit_rate": float, "pass": bool},
  "pairwise_final": {"position_inconsistency_rate": float,
                      "single_order_void": bool} | null,
  "external_review": {                      # G8 异源评委复核 (可缺省=SKIP)
    "judges": ["claude-sonnet-4-6", ...],
    "v5_vs_base": {"<judge>": {"diff":…, "ci95":[…], "n":…}},
    "v5_vs_v2":  {"<judge>": {"diff":…, "ci95":[…], "n":…}} | null
  } | null
}

判定: 全部 PASS → "ship_v5" (v5 替代 v2); 任一 FAIL → "keep_v2"
(拒绝替代); 无 FAIL 但有 FREEZE/SKIP → "freeze" (补数据/人工)。

回溯验证 (§6.7 门设计验收测试): v4 代入必须复现
  G1 FAIL (CI 上限 −0.14 已低于 −0.15)、G2 FAIL (−0.01 [−0.13,+0.11]
  无增量)、G3 FAIL (CE kw 真退化, E2 归因 8 词次) → 综合 keep_v2,
  与 E0 实际结论一致。
"""

import json
import sys

# 阈值 (§6.7 阈值对账, 不得偏离)
G1_CI_LOWER = -0.15   # v5−base 臂 A overall CI 下限 > −0.15
G2_CI_LOWER = -0.05   # v5−v2 overall CI 下限 > −0.05
G6_PROBE_FLOOR_PP = -5.0  # v5−v2 探针 overall > −5pp
SUBSET_DESCRIPTIVE_N = 30
OVERREFUSAL_MAX = 0.02

PASS, FAIL, SKIP, FREEZE = "PASS", "FAIL", "SKIP", "FREEZE"


def _sig(ci95):
    return ci95[0] > 0 or ci95[1] < 0


def _ov(track):
    return (track or {}).get("overall")


def gate_g0_overrefusal(scores):
    o = scores.get("overrefusal")
    if o is None:
        return {"gate": "G0", "status": SKIP,
                "reason": "overrefusal 数据缺失 (仅 base 评测时可缺省)"}
    ok = o["hit_rate"] <= OVERREFUSAL_MAX and o.get("pass", True)
    return {"gate": "G0", "status": PASS if ok else FAIL,
            "reason": f"拒答模板命中率 {o['hit_rate']:.2%} "
                      f"(阈值 ≤{OVERREFUSAL_MAX:.0%})",
            "value": o["hit_rate"]}


def gate_g1(scores):
    a = _ov((scores.get("v5_vs_base") or {}).get("judge_armA"))
    if a is None:
        return {"gate": "G1", "status": SKIP, "reason": "臂 A v5−base overall 缺失"}
    reasons = [f"臂A v5−base {a['diff']:+.4f} "
               f"[{a['ci95'][0]:+.4f}, {a['ci95'][1]:+.4f}] "
               f"(要求 CI 下限 > {G1_CI_LOWER})"]
    ok = a["ci95"][0] > G1_CI_LOWER
    b = _ov((scores.get("v5_vs_base") or {}).get("judge_armB"))
    if b is None:
        reasons.append("臂 B 无同桶配对/未跑, 以臂 A 为准 (§6.4)")
    else:
        # 臂 B 不矛盾: 臂 B 显著为负且低于 G1 阈值 → 矛盾
        contradict = b["ci95"][1] < G1_CI_LOWER
        reasons.append(f"臂B v5−base {b['diff']:+.4f} "
                       f"[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] "
                       f"{'矛盾' if contradict else '不矛盾'}")
        ok = ok and not contradict
    return {"gate": "G1", "status": PASS if ok else FAIL,
            "reason": "; ".join(reasons), "value": a["diff"]}


def gate_g2(scores):
    cmpv = scores.get("v5_vs_v2") or {}
    j = _ov(cmpv.get("judge_armA"))
    if j is None:
        return {"gate": "G2", "status": SKIP, "reason": "v5−v2 judge overall 缺失"}
    ci_ok = j["ci95"][0] > G2_CI_LOWER
    j_pos = _sig(j["ci95"]) and j["diff"] > 0
    k = _ov(cmpv.get("keyword"))
    k_pos = bool(k) and _sig(k["ci95"]) and k["diff"] > 0
    ok = ci_ok and (j_pos or k_pos)
    reason = (f"judge v5−v2 {j['diff']:+.4f} [{j['ci95'][0]:+.4f}, "
              f"{j['ci95'][1]:+.4f}] (CI 下限须 > {G2_CI_LOWER}); "
              f"judge 显著为正={j_pos}, kw 显著为正={k_pos} "
              "(至少一轨显著为正)")
    return {"gate": "G2", "status": PASS if ok else FAIL,
            "reason": reason, "value": j["diff"]}


def gate_g3(scores):
    """关键词轨退化禁令: 任一子集 v5−v2 kw CI 上限 <0 → 须归因;
    真缺失 → FAIL; 伪影 → 语义轨裁决后可放行; 未归因 → FREEZE。"""
    kw = ((scores.get("v5_vs_v2") or {}).get("keyword") or {}).get("per_subset") or {}
    attr = scores.get("attributions") or {}
    bad, frozen, notes = [], [], []
    for subset, d in sorted(kw.items()):
        if d["ci95"][1] < 0:
            a = attr.get(subset)
            if a == "true_missing":
                bad.append(subset)
                notes.append(f"{subset} kw {d['diff']:+.4f} "
                             f"[{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] "
                             "归因=真缺失 → 退化成立")
            elif a == "artifact":
                j = (((scores.get("v5_vs_v2") or {}).get("judge_armA") or {})
                     .get("per_subset") or {}).get(subset)
                if j and _sig(j["ci95"]) and j["diff"] < 0:
                    bad.append(subset)
                    notes.append(f"{subset} kw 退化归因=伪影, 但语义轨同向显著为负 "
                                 "→ 与 G4 合并视为真退化")
                else:
                    notes.append(f"{subset} kw 退化归因=伪影, 语义轨未显著为负 → 放行")
            else:
                frozen.append(subset)
                notes.append(f"{subset} kw CI 上限<0 但未做 E2 式归因 → 冻结")
    if bad:
        return {"gate": "G3", "status": FAIL, "reason": "; ".join(notes),
                "value": bad}
    if frozen:
        return {"gate": "G3", "status": FREEZE, "reason": "; ".join(notes),
                "value": frozen}
    return {"gate": "G3", "status": PASS,
            "reason": "; ".join(notes) if notes else "无子集 kw CI 上限<0"}


def gate_g4(scores):
    """子集退化容忍度: n<30 子集标描述性, 不独立否决 (信息门, 恒 PASS,
    但与 G3 的合并逻辑在 gate_g3 内执行)。"""
    ns = scores.get("subset_n") or {}
    desc = [f"{s}(n={n})" for s, n in sorted(ns.items()) if n < SUBSET_DESCRIPTIVE_N]
    return {"gate": "G4", "status": PASS,
            "reason": ("描述性子集: " + ", ".join(desc) if desc
                       else "全部子集 n≥30") + " (n<30 不独立否决)"}


def gate_g5(scores):
    """ARIZ 语义轨优先: ariz 两轨分歧时以 rubric 轨为准 (仅适用 ariz)。"""
    cmpv = scores.get("v5_vs_v2") or {}
    j = ((cmpv.get("judge_armA") or {}).get("per_subset") or {}).get("ariz_guidance")
    k = ((cmpv.get("keyword") or {}).get("per_subset") or {}).get("ariz_guidance")
    if not j or not k:
        return {"gate": "G5", "status": SKIP, "reason": "ariz 子集数据缺失"}
    j_sig, k_sig = _sig(j["ci95"]), _sig(k["ci95"])
    diverge = (j_sig and k_sig and (j["diff"] > 0) != (k["diff"] > 0)) or \
              (j_sig and not k_sig) or (k_sig and not j_sig)
    if diverge:
        verdict = "以 rubric 轨为准: " + (
            f"ariz judge {j['diff']:+.4f} [{j['ci95'][0]:+.4f},{j['ci95'][1]:+.4f}]")
        return {"gate": "G5", "status": PASS, "reason": "两轨分歧, " + verdict}
    return {"gate": "G5", "status": PASS, "reason": "ariz 两轨无分歧"}


def gate_g6(scores):
    p = (scores.get("probe") or {}).get("v5_minus_v2_overall_pp")
    if p is None:
        return {"gate": "G6", "status": SKIP,
                "reason": "探针数据缺失 (120 题探针未跑; 回溯场景可缺省)"}
    ok = p > G6_PROBE_FLOOR_PP
    return {"gate": "G6", "status": PASS if ok else FAIL,
            "reason": f"v5−v2 探针 overall {p:+.2f}pp (要求 > "
                      f"{G6_PROBE_FLOOR_PP:.0f}pp)", "value": p}


def gate_g7(scores):
    """双轨反向熔断: 任一维度 (overall 或子集) 两轨显著反向 → FAIL。"""
    flips = []
    for cmp_name in ("v5_vs_base", "v5_vs_v2"):
        cmpv = scores.get(cmp_name) or {}
        j = cmpv.get("judge_armA") or {}
        k = cmpv.get("keyword") or {}
        dims = [("overall", _ov(j), _ov(k))]
        for s in (j.get("per_subset") or {}):
            dims.append((s, (j["per_subset"] or {}).get(s),
                         (k.get("per_subset") or {}).get(s)))
        for dim, jd, kd in dims:
            if not jd or not kd:
                continue
            if _sig(jd["ci95"]) and _sig(kd["ci95"]) and \
               (jd["diff"] > 0) != (kd["diff"] > 0):
                flips.append(f"{cmp_name}/{dim}: judge {jd['diff']:+.4f} vs "
                             f"kw {kd['diff']:+.4f} 显著反向")
    if flips:
        return {"gate": "G7", "status": FAIL,
                "reason": "; ".join(flips) + " → 默认回滚 + 归因"}
    return {"gate": "G7", "status": PASS, "reason": "无两轨显著反向维度"}


def gate_g8(scores):
    """异源评委复核门: 主轨 (moonshot 同族) 结论在外部评委下不得矛盾。

    规则 (2026-07-29 终审标定: 外部评委互相 ρ=0.63–0.75, vs 同族仅 0.27–0.31;
    同族绝对读数约 4× 膨胀, 不可外推 — EXTERNAL_JUDGE_REVIEW.md):
    - external_review 缺失 → SKIP (发货前须补终审; 回溯场景可缺省);
    - 主轨 overall 显著为正 且 全部外部评委显著为负 → FAIL (家族矛盾熔断);
    - 过半数外部评委显著为负 → FAIL (外部复核确认退化);
    - 其余 → PASS。外部读数仅作方向/矛盾校验, 不替代主轨效应量。
    """
    er = scores.get("external_review")
    if not er:
        return {"gate": "G8", "status": SKIP,
                "reason": "异源评委终审数据缺失 (发货前须补; 回溯场景可缺省)"}
    notes, fail = [], False
    for cmp_name in ("v5_vs_base", "v5_vs_v2"):
        judges = er.get(cmp_name) or {}
        if not judges:
            continue
        main_ov = _ov((scores.get(cmp_name) or {}).get("judge_armA"))
        sig_neg = sig_pos = n = 0
        for j, d in sorted(judges.items()):
            n += 1
            lo, hi = d["ci95"]
            if hi < 0:
                sig_neg += 1
            if lo > 0:
                sig_pos += 1
            notes.append(f"{cmp_name}/{j} {d['diff']:+.4f} "
                         f"[{lo:+.4f}, {hi:+.4f}]")
        if sig_neg == n and main_ov and _sig(main_ov["ci95"]) \
                and main_ov["diff"] > 0:
            fail = True
            notes.append(f"{cmp_name}: 主轨显著为正但全部 {n} 个外部评委 "
                         "显著为负 → 家族矛盾熔断")
        elif sig_neg > n / 2:
            fail = True
            notes.append(f"{cmp_name}: 过半外部评委显著为负 "
                         f"({sig_neg}/{n}) → 外部复核确认退化")
    if not notes:
        return {"gate": "G8", "status": SKIP,
                "reason": "external_review 无 v5_vs_base/v5_vs_v2 评委数据"}
    return {"gate": "G8", "status": FAIL if fail else PASS,
            "reason": "; ".join(notes)}


ALL_GATES = [gate_g0_overrefusal, gate_g1, gate_g2, gate_g3,
             gate_g4, gate_g5, gate_g6, gate_g7, gate_g8]


def run_decision_gate(scores):
    results = [g(scores) for g in ALL_GATES]
    statuses = [r["status"] for r in results]
    if FAIL in statuses:
        verdict, label = "keep_v2", "拒绝: 保留 v2"
    elif FREEZE in statuses:
        verdict, label = "freeze", "冻结: 有判据需归因/人工"
    elif SKIP in statuses:
        verdict, label = "freeze", "冻结: 有判据缺数据 (SKIP)"
    else:
        verdict, label = "ship_v5", "通过: v5 可替代 v2"
    return {"verdict": verdict, "verdict_label": label,
            "gates": results,
            "rule": "G1–G8 同时通过才放行 (§6.7 + G8 异源复核); G0 overrefusal 为附加门"}


def format_report(decision):
    L = ["# 决策门 2.0 机检报告", "",
         f"## 总判定: **{decision['verdict_label']}** (`{decision['verdict']}`)", ""]
    L.append("| 门 | 判定 | 理由 |")
    L.append("|---|---|---|")
    for g in decision["gates"]:
        L.append(f"| {g['gate']} | **{g['status']}** | {g['reason']} |")
    L.append("")
    L.append(f"- 规则: {decision['rule']}")
    return "\n".join(L)


def main():
    if len(sys.argv) != 2:
        print("用法: decision_gate.py <scores.json>", file=sys.stderr)
        sys.exit(2)
    with open(sys.argv[1], encoding="utf-8") as f:
        scores = json.load(f)
    decision = run_decision_gate(scores)
    print(format_report(decision))
    sys.exit(0 if decision["verdict"] == "ship_v5" else 1)


if __name__ == "__main__":
    main()
