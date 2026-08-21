#!/usr/bin/env python3
"""30 题人类盲评 AB 试卷生成器（base vs 候选模型）。

分层抽样：6 个任务族各 5 题（seed 42）；逐题掷币决定 A/B 归属（seed 4200+i）；
题目顺序洗牌（seed 777）。评审人只看匿名 A/B，映射关系只写入 ab_key.json。

用法（v5b 生成出炉后正式跑）：
  python3 make_ab.py \
    --gold ../external_review/v5_gold.jsonl \
    --arm-x ../contest/gen_base.jsonl --arm-x-name base \
    --arm-y gen_v5b.jsonl          --arm-y-name v5b \
    --out .

校验跑（用 v5a 占位）：
  python3 make_ab.py --arm-y ../contest/gen_v5a.jsonl --arm-y-name v5a --out dryrun
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

FAMILIES = [
    "principle_recommendation",
    "contradiction_analysis",
    "ariz_guidance",
    "concept_explanation",
    "case_generation",
    "innovation_assessment",
]
PER_FAMILY = 5


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(Path(__file__).parent.parent / "external_review" / "v5_gold.jsonl"))
    ap.add_argument("--arm-x", default=str(Path(__file__).parent.parent / "contest" / "gen_base.jsonl"))
    ap.add_argument("--arm-x-name", default="base")
    ap.add_argument("--arm-y", required=True)
    ap.add_argument("--arm-y-name", required=True)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    gold = {r["id"]: r for r in load_jsonl(args.gold)}
    gx = {r["id"]: r["response"] for r in load_jsonl(args.arm_x)}
    gy = {r["id"]: r["response"] for r in load_jsonl(args.arm_y)}

    # ---- 分层抽样：每族 5 题，seed 42 ----
    rng = random.Random(42)
    by_family = defaultdict(list)
    for qid, row in gold.items():
        by_family[row["subset"]].append(qid)
    picked = []
    for fam in FAMILIES:
        pool = sorted(by_family[fam])
        assert len(pool) >= PER_FAMILY, f"{fam} 只有 {len(pool)} 题"
        picked += [(qid, fam) for qid in rng.sample(pool, PER_FAMILY)]

    # ---- 洗牌题目顺序，seed 777 ----
    rng_order = random.Random(777)
    rng_order.shuffle(picked)

    # ---- 逐题掷币决定 A/B，seed 4200+i；组装 ----
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    key = {"arms": {"x": args.arm_x_name, "y": args.arm_y_name},
           "seeds": {"sample": 42, "order": 777, "coin_base": 4200},
           "items": []}
    md = []
    md.append("# TRIZ 模型盲评试卷（30 题）\n")
    md.append("每题给出两个匿名回答 **回答 A** 与 **回答 B**。请只根据回答质量判断：")
    md.append("对这道题而言，哪个回答**对 TRIZ 实践者更有用**。\n")
    md.append("评价维度（重要性递减）：① 是否真正回答了题目所问（紧扣任务）；")
    md.append("② TRIZ 方法运用的正确性与承诺度（敢于给出明确原理/步骤并说明理由，而非泛泛而谈）；")
    md.append("③ 论证的可辩护性；④ 表达清晰。**不要**单纯奖励篇幅长短。\n")
    md.append("每题三选一：**A 更好 / B 更好 / 平局**，可附一句备注。预计用时 60–90 分钟。\n")
    md.append("---\n")
    n_missing = 0
    for i, (qid, fam) in enumerate(picked):
        coin = random.Random(4200 + i).random() < 0.5
        a_arm, b_arm = ("x", "y") if coin else ("y", "x")
        resp = {"x": gx.get(qid), "y": gy.get(qid)}
        if resp["x"] is None or resp["y"] is None:
            n_missing += 1
            continue
        key["items"].append({"n": i + 1, "id": qid, "subset": fam,
                             "A": key["arms"][a_arm], "B": key["arms"][b_arm]})
        q = gold[qid]["question"].strip()
        md.append(f"## 第 {i + 1} 题\n")
        md.append(f"**题目**：{q}\n")
        md.append(f"**回答 A**：\n\n{resp[a_arm].strip()}\n")
        md.append(f"**回答 B**：\n\n{resp[b_arm].strip()}\n")
        md.append("**你的选择**：☐ A 更好　☐ B 更好　☐ 平局　　备注：____________\n")
        md.append("---\n")

    assert n_missing == 0, f"{n_missing} 题缺少某一臂的生成"
    (out / "ab_questionnaire.md").write_text("\n".join(md), encoding="utf-8")
    (out / "ab_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")

    dist = Counter(it["subset"] for it in key["items"])
    ab = Counter((it["A"], it["B"]) for it in key["items"])
    print(f"试卷 {len(key['items'])} 题 -> {out/'ab_questionnaire.md'}")
    print(f"映射(保密) -> {out/'ab_key.json'}")
    print("分层分布:", dict(dist))
    print(f"A/B 归属: A={args.arm_x_name}&B={args.arm_y_name} 出现 {ab.get((args.arm_x_name, args.arm_y_name), 0)} 次, "
          f"反向 {ab.get((args.arm_y_name, args.arm_x_name), 0)} 次")
    lens = [len((gx if it['A'] == args.arm_x_name else gy)[it['id']]) for it in key["items"]]
    print(f"回答长度中位: {sorted(lens)[len(lens)//2]} 字符")


if __name__ == "__main__":
    main()
