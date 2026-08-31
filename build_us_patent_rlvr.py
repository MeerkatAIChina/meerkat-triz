#!/usr/bin/env python3
"""
美国专利 → RLVR 样本 (自动 ground_truth, 英文全文)。

三个可验证任务:
  1. cpc_class : 给定标题+摘要 → 预测 CPC 主分类 (首字母 A-H/Y)
  2. grant     : 给定标题+摘要 → 预测能否授权 (A1申请公开=0, B1/B2授权=1)
  3. cited     : 给定标题+摘要 → 预测高/低被引 (引文数据)

美国专利字段是 Python-repr 列表格式, 如:
  本地化标题: [{'文本': 'Fixing device', '语言': 'en', '截断': False}]
  CPC: [{'分类代码': 'G06F40/00', '首要分类': True, ...}]

用法:
  python3 build_us_patent_rlvr.py <年份.csv> <out.jsonl> [--limit N]
"""
import argparse
import ast
import csv
import json
import sys

csv.field_size_limit(sys.maxsize)

CPC_SECTION = "A B C D E F G H Y".split()
CPC_SECTION_CN = {
    "A": "人类生活必需", "B": "作业运输", "C": "化学冶金", "D": "纺织造纸",
    "E": "固定建筑", "F": "机械工程", "G": "物理", "H": "电学", "Y": "新兴交叉技术",
}


def parse_text(raw):
    """[{'文本': '...', ...}] → 文本字符串"""
    if not raw or raw.strip() in ("", "[]", "None"):
        return ""
    try:
        items = ast.literal_eval(raw)
        if items and isinstance(items, list):
            return items[0].get("文本", "") if isinstance(items[0], dict) else ""
    except Exception:
        pass
    return ""


def parse_cpc(raw):
    """[{'分类代码': 'G06F40/00', '首要分类': True, ...}] → 首要分类代码"""
    if not raw or raw.strip() in ("", "[]", "None"):
        return ""
    try:
        items = ast.literal_eval(raw)
        for it in items:
            if isinstance(it, dict) and it.get("首要分类"):
                return it.get("分类代码", "")
        if items and isinstance(items[0], dict):
            return items[0].get("分类代码", "")
    except Exception:
        pass
    return ""


def parse_citations(raw):
    """[{'公开编号': ..., '被引': bool, ...}] → 被引次数"""
    if not raw or raw.strip() in ("", "[]", "None"):
        return 0
    try:
        items = ast.literal_eval(raw)
        return len([x for x in items if isinstance(x, dict) and x.get("被引")])
    except Exception:
        return 0


def build_samples(csv_path, limit=None):
    samples = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            title = parse_text(row.get("本地化标题", ""))
            abstract = parse_text(row.get("本地化摘要", ""))
            if not abstract or not title:
                continue
            cpc = parse_cpc(row.get("合作专利分类（CPC）", ""))
            kind = (row.get("种类代码") or "").strip()
            cited = parse_citations(row.get("专利引用", ""))

            # 任务1: CPC 分类
            if cpc and cpc[0].upper() in CPC_SECTION:
                samples.append({
                    "task": "cpc_class",
                    "prompt": f"Patent title: {title}\nAbstract: {abstract}\nWhich CPC section (A/B/C/D/E/F/G/H/Y) does this patent belong to?",
                    "ground_truth": cpc[0].upper(),
                    "metadata": {"cpc": cpc, "year": row.get("year", ""), "kind": kind},
                })
            # 任务2: 授权预测 (A1=申请公开=0, B1/B2=授权=1)
            if kind in ("A1", "B1", "B2"):
                samples.append({
                    "task": "grant",
                    "prompt": f"Patent title: {title}\nAbstract: {abstract}\nIs this a granted patent? Answer yes or no.",
                    "ground_truth": 1 if kind in ("B1", "B2") else 0,
                    "metadata": {"kind": kind, "year": row.get("year", "")},
                })
            # 任务3: 被引预测 (高被引=1, 低被引=0)
            samples.append({
                "task": "cited",
                "prompt": f"Patent title: {title}\nAbstract: {abstract}\nWill this patent be highly cited? Answer yes or no.",
                "ground_truth": 1 if cited >= 10 else 0,
                "metadata": {"cited_count": cited, "year": row.get("year", "")},
            })

            if limit and len(samples) >= limit:
                break
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_file")
    ap.add_argument("out_jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    samples = build_samples(args.csv_file, args.limit)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    from collections import Counter
    tasks = Counter(s["task"] for s in samples)
    print(f"[build] {args.csv_file} → {len(samples)} 样本")
    for t, n in sorted(tasks.items()):
        print(f"  {t}: {n}")


if __name__ == "__main__":
    main()
