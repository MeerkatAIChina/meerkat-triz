#!/usr/bin/env python
"""
pipeline_v5 pairwise 双序强制 (§6.3, §11.3)。

纪律 (E1a 已证实: AB 序 v4 胜率 0.180 vs BA 序 0.800, 摆动 62pp,
文献均值 25pp 的两倍多):
  - 一切 pairwise 必须 AB/BA 双序; 单序一律无效;
  - 报告 = 合并胜率 [Wilson 95%CI] + 位置不一致率 + B 位胜率,
    三数齐全才有效;
  - 不一致率 >10% 时任何单序数字作废 (机检断言);
  - ≤10% 仍须双序 (防选择性执行);
  - pairwise 仅用于决策门终审与关键对比 (≤400 裁决/轮);
    rubric 点值轨不双序。
"""

from stats_utils import wilson_ci

INCONSISTENCY_VOID_THRESHOLD = 0.10

_VERDICTS = ("A", "B", "tie")


def merge_double_order(verdicts_ab, verdicts_ba, strict=True):
    """合并 AB/BA 双序裁决。

    verdicts_ab / verdicts_ba: {item_id: "A"|"B"|"tie"}
      AB 序 = 候选模型在 A 位; BA 序 = 候选模型在 B 位。
      "A" 表示 A 位模型胜。
    返回报告 dict:
      candidate_winrate_merged (+Wilson CI), b_position_winrate,
      position_inconsistency_rate, single_order_void (bool),
      n_items, n_judgments。
    strict=True 且不一致率 >10% 时抛 AssertionError (单序作废断言)。
    """
    common = sorted(set(verdicts_ab) & set(verdicts_ba))
    if not common:
        raise ValueError("AB/BA 无共同题, 无法合并")
    for vid, v in list(verdicts_ab.items()) + list(verdicts_ba.items()):
        if v not in _VERDICTS:
            raise ValueError(f"非法裁决 {vid}: {v!r} (须为 A/B/tie)")

    cand_wins, n_judge = 0.0, 0
    b_pos_wins, b_pos_n = 0.0, 0
    inconsistent = []
    for qid in common:
        v_ab, v_ba = verdicts_ab[qid], verdicts_ba[qid]
        # 候选模型视角: AB 序中候选=A 位; BA 序中候选=B 位
        cand_ab = 1.0 if v_ab == "A" else (0.5 if v_ab == "tie" else 0.0)
        cand_ba = 1.0 if v_ba == "B" else (0.5 if v_ba == "tie" else 0.0)
        cand_wins += cand_ab + cand_ba
        n_judge += 2
        # B 位胜率 (位置效应观测)
        b_pos_wins += 1.0 if v_ab == "B" else (0.5 if v_ab == "tie" else 0.0)
        b_pos_wins += 1.0 if v_ba == "B" else (0.5 if v_ba == "tie" else 0.0)
        b_pos_n += 2
        # 位置不一致: 同一题两序结论不一致 (候选视角胜负不同)
        if cand_ab != cand_ba:
            inconsistent.append(qid)

    n_items = len(common)
    # Wilson 需要整数 k: 用胜场×2 (tie 半胜→整数) 在 2*n_judge 口径
    k2 = int(round(cand_wins * 2))
    p2, lo, hi = wilson_ci(k2, 2 * n_judge)
    inc_rate = len(inconsistent) / n_items
    void = inc_rate > INCONSISTENCY_VOID_THRESHOLD
    report = {
        "n_items": n_items,
        "n_judgments": n_judge,
        "candidate_winrate_merged": round(p2, 4),
        "candidate_winrate_wilson_ci95": [round(lo, 4), round(hi, 4)],
        "b_position_winrate": round(b_pos_wins / b_pos_n, 4),
        "position_inconsistency_rate": round(inc_rate, 4),
        "position_inconsistent_items": inconsistent,
        "single_order_void": void,
        "discipline": ("一切 pairwise 必须 AB/BA 双序; 不一致率 >"
                       f"{INCONSISTENCY_VOID_THRESHOLD:.0%} 时任何单序数字作废 (§6.3)"),
    }
    if void and strict:
        raise AssertionError(
            f"位置不一致率 {inc_rate:.1%} > {INCONSISTENCY_VOID_THRESHOLD:.0%}: "
            "任何单序数字作废 (§6.3), 禁止据单序下结论")
    return report


PAIRWISE_SYSTEM = (
    "你是 TRIZ 领域资深评审专家。给定一道 TRIZ 评测题、参考答案与两个 AI 回答 "
    "(A 与 B), 判断哪个回答更好 (准确性/TRIZ 正确性优先, 长度本身不构成加分项, "
    "冗余重复扣分)。只输出一个 JSON 对象, 不要输出任何其他文字: "
    '{"verdict": "A"|"B"|"tie", "reason": "一句话理由"}')


def build_pairwise_user(item, resp_a, resp_b):
    return (f"问题: {item['question']}\n"
            f"参考答案: {item['reference_answer']}\n"
            f"【回答 A】\n{resp_a}\n\n【回答 B】\n{resp_b}\n"
            '请输出 {"verdict": "A"|"B"|"tie", "reason": "..."}')
