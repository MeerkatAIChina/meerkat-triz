#!/usr/bin/env python3
"""PR 子集原理集合 F1 度量 (benchmark v2 候选, v5b 教训 2026-07-31)。

keyword 逐词命中对 principle_recommendation 是失效度量: 它编码某一个
参考答案的选理与措辞, 而 PR 题天然有多个可辩护答案。本度量改为:
从回答中抽取提及的 40 发明原理集合, 与从金标参考答案抽取的原理集合
算 set F1——度量"选对原理"而非"复现措辞"。

三方对比: base / v5a / v5b (60 题 PR 子集)。
"""
import json
import re
from pathlib import Path

W = Path("/Volumes/2nd-HD/claude/Meerkat-AI/paper")

# 40 发明原理: 编号 -> 别名列表 (取并集匹配; 含常见教材译法)
PRINCIPLES = {
    1: ["分割"], 2: ["抽取", "提取"], 3: ["局部质量"], 4: ["不对称"],
    5: ["合并", "组合"], 6: ["通用", "普遍", "多用"], 7: ["嵌套", "套娃"],
    8: ["配重", "反重力", "补偿重量"], 9: ["预先反作用", "预反作用"],
    10: ["预先作用", "预作用"], 11: ["预先防范", "事先防范", "预防"],
    12: ["等势"], 13: ["反向", "反过来", "逆向"],
    14: ["曲面", "球面", "曲线化"], 15: ["动态", "柔性化"],
    16: ["部分或过度", "过度作用", "过量"], 17: ["维数变化", "另一维", "多维"],
    18: ["机械振动", "振动", "震动"], 19: ["周期"],
    20: ["连续", "持续有效"], 21: ["急速", "紧急", "快速通过"],
    22: ["变害为利", "化害为利", "有害因素"], 23: ["反馈"],
    24: ["中介", "中间物"], 25: ["自服务", "自维护"],
    26: ["复制", "替代品代替昂"], 27: ["廉价", "一次性"],
    28: ["机械系统替代", "替代机械"], 29: ["气动", "液压"],
    30: ["柔性壳体", "薄膜"], 31: ["多孔"], 32: ["颜色", "变色"],
    33: ["同质", "均匀材料"], 34: ["抛弃", "再生", "废弃回收"],
    35: ["参数变化", "改变参数"], 36: ["相变"], 37: ["热膨胀"],
    38: ["氧化"], 39: ["惰性"], 40: ["复合材料"],
}
NUM_RE = re.compile(r"原理\s*[No\.#]*\s*(\d{1,2})")


def extract_principles(text):
    found = set()
    for num, aliases in PRINCIPLES.items():
        for a in aliases:
            if a in text:
                found.add(num)
                break
    for m in NUM_RE.finditer(text):
        n = int(m.group(1))
        if 1 <= n <= 40:
            found.add(n)
    return found


def f1(pred, ref):
    if not ref:
        return None
    if not pred:
        return 0.0
    tp = len(pred & ref)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(ref)
    return 2 * p * r / (p + r)


def load_eval_records(path):
    d = json.load(open(path, encoding="utf-8"))
    return {r["id"]: r["response"] for r in d["records"]}


def load_gen_jsonl(path):
    return {json.loads(l)["id"]: json.loads(l)["response"]
            for l in open(path, encoding="utf-8") if l.strip()}


def main():
    gold = {}
    for l in open(W / "external_review_v5b" / "v5_gold.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["subset"] == "principle_recommendation":
            gold[r["id"]] = r
    ref_sets = {qid: extract_principles(g["reference_answer"])
                for qid, g in gold.items()}
    n_ref = sum(1 for s in ref_sets.values() if s)
    print(f"PR 子集 {len(gold)} 题, 参考答案可抽取原理集合的 {n_ref} 题")

    arms = {
        "base": load_eval_records(W / "external_review_v5b" / "eval_v5_base_goldfix_v5_20260726_234434.json"),
        "v5a": load_gen_jsonl(W / "contest" / "gen_v5a.jsonl"),
        "v5b": load_eval_records(W / "external_review_v5b" / "eval_v5_v5b_gold_20260731_195036.json"),
    }
    print(f"{'arm':<6} {'mean F1':>8} {'mean |pred|':>11} {'mean |ref|':>10}")
    summary = {}
    for name, resps in arms.items():
        scores, npred, nref = [], [], []
        for qid, ref in ref_sets.items():
            if not ref or qid not in resps:
                continue
            pred = extract_principles(resps[qid])
            s = f1(pred, ref)
            scores.append(s)
            npred.append(len(pred))
            nref.append(len(ref))
        m = sum(scores) / len(scores)
        summary[name] = scores
        print(f"{name:<6} {m:8.3f} {sum(npred)/len(npred):11.1f} {sum(nref)/len(nref):10.1f}")

    # 配对差值 v5b−base, v5a−base
    import random
    def paired(a, b, n_boot=10000, seed=42):
        ids = [q for q, r in ref_sets.items() if r and q in arms["base"]]
        da, db = summary[a], summary[b]
        n = len(da)
        rng = random.Random(seed)
        boots = []
        for _ in range(n_boot):
            s = sum(db[rng.randrange(n)] - da[rng.randrange(n)] for _ in range(n)) / n
            boots.append(s)
        boots.sort()
        diff = sum(db[i] - da[i] for i in range(n)) / n
        return diff, boots[250], boots[9749]
    for pair in (("base", "v5a"), ("base", "v5b"), ("v5a", "v5b")):
        d, lo, hi = paired(*pair)
        print(f"{pair[1]}−{pair[0]}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}] "
              f"{'显著' if (lo > 0 or hi < 0) else 'n.s.'}")


if __name__ == "__main__":
    main()
