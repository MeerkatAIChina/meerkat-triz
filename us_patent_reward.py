#!/usr/bin/env python3
"""
美国专利 RLVR 的 verl 自定义 reward function (英文, 三任务)。

  cpc_class : CPC 部分类 (A-H/Y), 精确匹配 1/0
  grant     : 授权预测 (yes/no), 二分类 1/0
  cited     : 高被引预测 (yes/no), 二分类 1/0

verl 配置: --custom_reward_function.path=us_patent_reward.py --custom_reward_function.name=compute_score
"""
import re

CPC_SECTION = set("ABCDEFGHY")


def _reward_cpc(solution_str, ground_truth):
    s = solution_str
    # 优先: "section X" / "belongs to X" 后面的 A-H/Y 字母 (避开 THE/CPC 等词里的字母)
    m = re.search(r"(?:section|belongs to|belongs)\s*[:：]?\s*[^A-HY]*?([A-HY])\b", s, re.IGNORECASE)
    if not m:
        # 回退: 独立的单个大写 A-H/Y 字母 (单词边界)
        m = re.search(r"\b([A-HY])\b", s)
    if not m:
        return 0.0
    return 1.0 if m.group(1).upper() == str(ground_truth).upper() else 0.0


def _reward_yesno(solution_str, ground_truth):
    """提取 yes/no, 二分类。"""
    s = solution_str.lower()
    yes = re.search(r"\byes\b|\b授权\b|\b是\b", s)
    no = re.search(r"\bno\b|\bnot\b|\b未授权\b|\b否\b", s)
    if yes and not no:
        pred = 1
    elif no and not yes:
        pred = 0
    else:
        return 0.0
    try:
        gt = int(float(ground_truth))
    except (TypeError, ValueError):
        return 0.0
    return 1.0 if pred == gt else 0.0


_REWARDS = {
    "cpc_class": _reward_cpc,
    "grant": _reward_yesno,
    "cited": _reward_yesno,
}


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    fn = _REWARDS.get(data_source)
    if fn is None:
        return 0.0
    return fn(solution_str or "", ground_truth)


if __name__ == "__main__":
    print("=== us_patent_reward 自测 ===")
    cases = [
        ("cpc_class", "The CPC section is G", "G", "CPC 精确"),
        ("cpc_class", "Belongs to section H", "H", "CPC 精确2"),
        ("cpc_class", "unknown", "G", "CPC 无答案"),
        ("grant", "Yes, it is granted", "1", "授权=yes"),
        ("grant", "No, not granted", "0", "授权=no"),
        ("cited", "Yes highly cited", "1", "被引=yes"),
        ("cited", "No", "0", "被引=no"),
    ]
    for ds, sol, gt, desc in cases:
        r = compute_score(ds, sol, gt)
        print(f"  {desc:12s} ds={ds:10s} '{sol}' gt={gt} → {r}")
