#!/usr/bin/env python
"""数据契约预检 (决策门前置数据门, v5b 教训 2026-07-31)。

任何"定向注入"在烧训练算力之前, 注入数据必须先过本检查:
把注入样本的 completion 对着金标关键词集评分, 回答"这批数据教会模型
满足评测契约吗"。v5b 的 408 条注入若先过此检, keyword −0.104 的
反噬在训练前即可预言。

用法:
  python data_contract_check.py \
      --inject data/processed/v5b_data/pr_inject_answers.jsonl \
      --gold   data/processed/v5_data/v5_gold.jsonl \
      --subset principle_recommendation \
      [--completion-field completion] [--min-framing 0.5] [--report out.md]

判定规则 ( WARN = 阻止全量训练, 回炉修数据 ):
  R1 框架词覆盖: 注入 completion 中提及该子集高频框架词
     (如 TRIZ/发明原理/技术矛盾) 的比例 < --min-framing → WARN
  R2 关键词密度: 注入 completion 对金标期望关键词的平均命中率
     显著低于现役训练语料同子集 (若有 --corpus 对照) → WARN
  R3 长度带: completion 中位长度与金标参考答案中位长度比 < 0.3 → WARN
     (keyword 轨是长度敏感契约, 过短注定掉词)
"""
import argparse
import json
import statistics
from collections import Counter

FRAMING_TERMS = ["TRIZ", "发明原理", "技术矛盾"]


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--completion-field", default=None,
                    help="缺省自动探测 completion/response/answer/text")
    ap.add_argument("--min-framing", type=float, default=0.5)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    inject = load_jsonl(args.inject)
    gold = [r for r in load_jsonl(args.gold) if r.get("subset") == args.subset]
    assert gold, f"金标中无子集 {args.subset}"

    def completion_of(row):
        if args.completion_field:
            return str(row.get(args.completion_field) or "")
        for k in ("completion", "response", "answer", "text", "output"):
            if row.get(k):
                return str(row[k])
        return ""

    comps = [completion_of(r) for r in inject]
    comps = [c for c in comps if c]
    assert comps, "注入数据中没有可用 completion"

    # ---- R1 框架词覆盖 ----
    framing_rate = {t: sum(1 for c in comps if t in c) / len(comps)
                    for t in FRAMING_TERMS}
    r1_bad = {t: r for t, r in framing_rate.items() if r < args.min_framing}

    # ---- R2 期望关键词命中密度 (按题: 金标每题 keywords 在全语料上的平均命中率) ----
    kw_pool = [k for r in gold for k in r.get("keywords", [])]
    kw_top = [k for k, _ in Counter(kw_pool).most_common(30)]  # 高频期望词
    hits = sum(1 for c in comps for k in kw_top if k in c)
    total_chars = sum(len(c) for c in comps)
    density = hits / max(total_chars, 1) * 1000  # 次/千字

    # ---- R3 长度带 ----
    ref_lens = [len(r.get("reference_answer", "")) for r in gold]
    med_comp = statistics.median(len(c) for c in comps)
    med_ref = statistics.median(ref_lens) if any(ref_lens) else None
    len_ratio = (med_comp / med_ref) if med_ref else None
    r3_bad = len_ratio is not None and len_ratio < 0.3

    verdict = "WARN" if (len(r1_bad) >= 2 or r3_bad) else "PASS"
    lines = [
        f"# 数据契约预检报告 — {args.subset}",
        "",
        f"- 注入样本: {len(comps)} 条 | 金标对照: {len(gold)} 题",
        f"- 总判定: **{verdict}**",
        "",
        "## R1 框架词覆盖 (阈值 ≥{:.0%})".format(args.min_framing),
    ]
    for t, r in framing_rate.items():
        flag = " ⚠️" if t in r1_bad else ""
        lines.append(f"- `{t}`: {r:.0%}{flag}")
    lines += [
        "",
        "## R2 期望关键词命中密度",
        f"- 高频期望词 top30 命中密度: {density:.2f} 次/千字",
        "",
        "## R3 长度带",
        f"- completion 中位 {med_comp:.0f} 字 vs 金标参考中位 "
        f"{med_ref:.0f} 字 (比值 {len_ratio:.2f})" if med_ref else "- 无参考答案长度",
        "",
        "判定规则: ≥2 个框架词低于阈值, 或长度比 <0.3 → WARN (阻止全量训练)",
    ]
    text = "\n".join(lines)
    print(text)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 1 if verdict == "WARN" else 0


if __name__ == "__main__":
    raise SystemExit(main())
