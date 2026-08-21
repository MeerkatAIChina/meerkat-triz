#!/usr/bin/env python3
"""盲评回收计分：把评审人的 30 题选择转成 win/tie/loss + 二项检验。

输入答卷格式（ab_answers.json 或 .csv）：
  JSON: [{"n": 1, "choice": "A"}, {"n": 2, "choice": "tie"}, ...]   # choice ∈ A|B|tie
  CSV : n,choice  （一行一题）

用法：
  python3 ab_score.py --key ab_key.json --answers ab_answers.json
"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def binom_p_two_sided(k, n, p=0.5):
    """P(X >= k) 的双侧近似：2 * P(X >= max(k, n-k))，n 为有效（非平局）题数。"""
    if n == 0:
        return 1.0
    k = max(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) * (p ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--answers", required=True)
    args = ap.parse_args()

    key = json.load(open(args.key, encoding="utf-8"))
    x_name, y_name = key["arms"]["x"], key["arms"]["y"]
    kmap = {it["n"]: it for it in key["items"]}

    path = Path(args.answers)
    if path.suffix == ".json":
        answers = {int(a["n"]): a["choice"] for a in json.load(open(path, encoding="utf-8"))}
    else:
        with open(path, encoding="utf-8") as f:
            answers = {int(r["n"]): r["choice"].strip() for r in csv.DictReader(f)}

    assert len(answers) == len(kmap), f"答卷 {len(answers)} 题 vs 试卷 {len(kmap)} 题"
    score = Counter()
    per_family = {}
    for n, choice in sorted(answers.items()):
        it = kmap[n]
        c = choice.lower()
        assert c in ("a", "b", "tie"), f"第 {n} 题选择无效: {choice}"
        if c == "tie":
            score["tie"] += 1
            per_family.setdefault(it["subset"], Counter())["tie"] += 1
            continue
        winner = it[c.upper()]          # "base" 或候选名
        loser = y_name if winner == x_name else x_name
        score[f"{winner}_win"] += 1
        per_family.setdefault(it["subset"], Counter())[f"{winner}_win"] += 1

    xw = score[f"{x_name}_win"]; yw = score[f"{y_name}_win"]; t = score["tie"]
    n_eff = xw + yw
    p = binom_p_two_sided(xw, n_eff)
    print(f"== 盲评结果（{x_name} vs {y_name}）==")
    print(f"{x_name} 胜 {xw} / {y_name} 胜 {yw} / 平局 {t}  (有效对决 {n_eff})")
    print(f"双侧二项检验 p = {p:.4f}  ({'显著' if p < 0.05 else '不显著'})")
    print("\n分任务族:")
    for fam, c in sorted(per_family.items()):
        print(f"  {fam}: {x_name} 胜 {c[f'{x_name}_win']} / {y_name} 胜 {c[f'{y_name}_win']} / 平 {c['tie']}")


if __name__ == "__main__":
    main()
