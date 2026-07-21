#!/usr/bin/env python3
"""Audit processed SFT jsonl splits (train/validation/test) on the DGX box.

For each split reports:
  - record count
  - subset distribution (6 TRIZ subsets)
  - records whose output contains an empty <think></think> block
  - normalized-instruction (whitespace-stripped, lowercased) duplicate rate
  - output length distribution (min/p50/p95/max, in chars)
  - instruction length distribution (min/p50/p95/max, in chars)

Prints markdown tables to stdout; optionally dumps the full stats as JSON.

Pure stdlib (json/argparse/collections/re/math) — no numpy/torch required.

Usage:
    python3 scripts/audit_training_data.py [--dir data/processed] [--prefix v2_]
                                           [--out audit.json]
"""

import argparse
import glob
import json
import math
import os
import re
from collections import Counter, OrderedDict

SPLITS = ["train", "validation", "test"]

VALID_SUBSETS = [
    "concept_explanation",
    "contradiction_analysis",
    "principle_recommendation",
    "case_generation",
    "ariz_guidance",
    "innovation_assessment",
]

EMPTY_THINK_RE = re.compile(r"<think>\s*</think>")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_instruction(instr):
    """Whitespace-stripped, lowercased instruction for duplicate detection."""
    return WHITESPACE_RE.sub("", instr or "").lower()


def percentile(sorted_vals, p):
    """Linear-interpolation percentile (numpy 'linear' method)."""
    if not sorted_vals:
        return 0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(math.floor(k))
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def length_stats(values):
    s = sorted(values)
    return {
        "min": s[0] if s else 0,
        "p50": round(percentile(s, 50), 1),
        "p95": round(percentile(s, 95), 1),
        "max": s[-1] if s else 0,
    }


def find_split_file(directory, prefix, split):
    """Locate the jsonl for a split: exact {prefix}{split}.jsonl first,
    then fall back to any *{split}.jsonl in the directory."""
    exact = os.path.join(directory, f"{prefix}{split}.jsonl")
    if os.path.isfile(exact):
        return exact
    candidates = sorted(glob.glob(os.path.join(directory, f"*{split}.jsonl")))
    return candidates[0] if candidates else None


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: {path}:{lineno} JSON 解析失败，跳过: {e}")
    return records


def audit_split(records):
    n = len(records)
    subset_counts = Counter(r.get("subset", "<missing>") for r in records)
    output_empty_think = sum(
        1 for r in records if EMPTY_THINK_RE.search(r.get("output", "") or "")
    )
    text_empty_think = sum(
        1 for r in records if EMPTY_THINK_RE.search(r.get("text", "") or "")
    )
    norm_instrs = [normalize_instruction(r.get("instruction", "")) for r in records]
    unique_norm = len(set(norm_instrs))
    dup_count = n - unique_norm
    dup_rate = (dup_count / n * 100.0) if n else 0.0
    return {
        "count": n,
        "subset_distribution": OrderedDict(
            (sub, subset_counts.get(sub, 0)) for sub in VALID_SUBSETS
        ),
        "subset_other": {
            k: v for k, v in subset_counts.items() if k not in VALID_SUBSETS
        },
        "output_empty_think": output_empty_think,
        "text_empty_think": text_empty_think,
        "instruction_unique_normalized": unique_norm,
        "instruction_duplicates": dup_count,
        "instruction_duplicate_rate_pct": round(dup_rate, 2),
        "output_length": length_stats([len(r.get("output", "") or "") for r in records]),
        "instruction_length": length_stats(
            [len(r.get("instruction", "") or "") for r in records]
        ),
    }


def print_markdown(results):
    print("## 训练数据审计报告\n")

    print("### 总览\n")
    print("| split | 条数 | output 空 think | text 空 think | "
          "归一化 instruction 唯一数 | 重复数 | 重复率 |")
    print("|---|---|---|---|---|---|---|")
    for split, st in results.items():
        print(
            f"| {split} | {st['count']} | {st['output_empty_think']} "
            f"| {st['text_empty_think']} | {st['instruction_unique_normalized']} "
            f"| {st['instruction_duplicates']} "
            f"| {st['instruction_duplicate_rate_pct']}% |"
        )

    print("\n### 子集分布\n")
    header = "| split | " + " | ".join(VALID_SUBSETS) + " | 其他 |"
    print(header)
    print("|" + "---|" * (len(VALID_SUBSETS) + 2))
    for split, st in results.items():
        row = [str(st["subset_distribution"][s]) for s in VALID_SUBSETS]
        other = sum(st["subset_other"].values())
        print(f"| {split} | " + " | ".join(row) + f" | {other} |")

    print("\n### 长度分布（字符数）\n")
    print("| split | 字段 | min | p50 | p95 | max |")
    print("|---|---|---|---|---|---|")
    for split, st in results.items():
        for field, key in (("output", "output_length"),
                           ("instruction", "instruction_length")):
            ls = st[key]
            print(f"| {split} | {field} | {ls['min']} | {ls['p50']} "
                  f"| {ls['p95']} | {ls['max']} |")


def main():
    parser = argparse.ArgumentParser(
        description="Audit processed train/validation/test jsonl splits."
    )
    parser.add_argument("--dir", default="data/processed",
                        help="Directory containing the split jsonl files "
                             "(default: data/processed)")
    parser.add_argument("--prefix", default="",
                        help="Jsonl filename prefix, e.g. v2_ (default: none; "
                             "falls back to *{split}.jsonl glob)")
    parser.add_argument("--out", default=None,
                        help="Optional path to write full stats as JSON")
    args = parser.parse_args()

    results = OrderedDict()
    for split in SPLITS:
        path = find_split_file(args.dir, args.prefix, split)
        if not path:
            print(f"WARNING: 未找到 {split} 的 jsonl（目录 {args.dir}），跳过")
            continue
        records = load_jsonl(path)
        print(f"# {split}: {path} ({len(records)} 条)")
        results[split] = audit_split(records)

    if not results:
        print("ERROR: 未审计到任何 split，请检查 --dir/--prefix")
        raise SystemExit(1)

    print()
    print_markdown(results)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nJSON 已写入: {args.out}")


if __name__ == "__main__":
    main()
