#!/usr/bin/env python3
"""v3 数据构建：合并 v2 全量语料 + ARIZ boost 新样本（扣除评测预留），重建 jsonl；
同时把预留的 ARIZ 样本扩充进评测集（data/processed/sample_data_expanded.json）。

- 训练份: boost 样本去掉 80 条评测预留后并入 v2 语料 → v3_train/validation/test.jsonl
- 评测预留: 80 条新 ARIZ 样本加入 sample_data 的 ariz_guidance (22 → ~102)
- 三重去重: vs v2 语料 / vs 既有评测集 / 质量门内去重
- 只写 data/processed/，不触碰 data/raw 与 sample_data.json 原文件

用法: venv_v5/bin/python /tmp/build_v3_and_evalset.py
"""

import json
import random
import sys
from pathlib import Path

sys.path.append("/home/meerkat/mongoose_ai")
sys.path.append("/home/meerkat/mongoose_ai/scripts")

from run_corpus_sft_v2 import build_chatml_jsonl  # noqa: E402
from utils.corpus_to_sft import (  # noqa: E402
    VALID_SUBSETS,
    _normalize_instruction,
    apply_v2_quality_gates,
)

BASE = "/home/meerkat/mongoose_ai"
V2_CKPT = f"{BASE}/data/processed/checkpoint_corpus_sft_v2/corpus_sft_checkpoint.json"
BOOST_CKPT = f"{BASE}/data/processed/checkpoint_corpus_sft_v2_ariz_boost/corpus_sft_checkpoint.json"
SAMPLE_DATA = f"{BASE}/data/sample_data.json"
EVAL_OUT = f"{BASE}/data/processed/sample_data_expanded.json"
EVAL_RESERVE = 80
SEED = 42


def main():
    v2_samples = json.load(open(V2_CKPT, encoding="utf-8"))["samples"]
    boost_raw = json.load(open(BOOST_CKPT, encoding="utf-8"))["samples"]
    print(f"v2 语料: {len(v2_samples)} 条 | boost 原始: {len(boost_raw)} 条")

    # 1) boost 质量门
    boost, gates = apply_v2_quality_gates(boost_raw, min_output_chars=150)
    print(f"boost 质量门: {gates}")

    # 2) 交叉去重: vs v2 语料 + vs 既有评测集 (防止 train/eval 污染)
    v2_keys = {_normalize_instruction(s.get("instruction", "")) for s in v2_samples}
    sample_data = json.load(open(SAMPLE_DATA, encoding="utf-8"))
    eval_keys = {_normalize_instruction(s["instruction"]) for v in sample_data.values() for s in v}
    n0 = len(boost)
    boost = [s for s in boost if _normalize_instruction(s.get("instruction", "")) not in v2_keys]
    n1 = len(boost)
    boost = [s for s in boost if _normalize_instruction(s.get("instruction", "")) not in eval_keys]
    print(f"交叉去重: vs v2 语料剔除 {n0 - n1}, vs 评测集剔除 {n1 - len(boost)}, 余 {len(boost)}")

    assert len(boost) >= EVAL_RESERVE, f"boost 样本不足 {EVAL_RESERVE} 条，无法划分评测预留"

    # 3) 划分评测预留 / 训练份
    rng = random.Random(SEED)
    rng.shuffle(boost)
    eval_reserve = boost[:EVAL_RESERVE]
    train_boost = boost[EVAL_RESERVE:]
    print(f"评测预留: {len(eval_reserve)} 条 | 训练份: {len(train_boost)} 条")

    # 4) 扩充评测集 (ariz_guidance 22 → ~102)
    expanded = {k: list(v) for k, v in sample_data.items()}
    expanded["ariz_guidance"] = expanded.get("ariz_guidance", []) + [
        {"instruction": s["instruction"], "input": s.get("input", ""), "output": s["output"]}
        for s in eval_reserve
    ]
    with open(EVAL_OUT, "w", encoding="utf-8") as f:
        json.dump(expanded, f, ensure_ascii=False, indent=2)
    print(f"扩充评测集 → {EVAL_OUT}: "
          f"ariz_guidance {len(sample_data.get('ariz_guidance', []))} → {len(expanded['ariz_guidance'])}, "
          f"总 {sum(len(v) for v in expanded.values())} 条")

    # 5) 合并训练语料并重建 v3 jsonl
    all_samples = v2_samples + train_boost
    all_samples, gates2 = apply_v2_quality_gates(all_samples, min_output_chars=150)
    print(f"合并后质量门: {gates2}")

    grouped = {s: [] for s in VALID_SUBSETS}
    for s in all_samples:
        grouped[s["subset"]].append({
            "instruction": s["instruction"],
            "input": s.get("input", ""),
            "output": s["output"],
        })
    dist = {k: len(v) for k, v in grouped.items()}
    print(f"v3 子集分布: {dist}")

    split_counts, think_stripped = build_chatml_jsonl(grouped, "v3_")
    print(f"v3 jsonl 划分: {split_counts} | think 剥离: {think_stripped}")

    # 6) 校验
    n_train = sum(1 for _ in open(f"{BASE}/data/processed/v3_train.jsonl", encoding="utf-8"))
    first = json.loads(open(f"{BASE}/data/processed/v3_train.jsonl", encoding="utf-8").readline())
    assert n_train > 8000, f"v3_train 行数异常: {n_train}"
    assert first.get("instruction") and first.get("output"), "首行缺 instruction/output 字段"
    print(f"校验通过: v3_train.jsonl {n_train} 行, 首行含 instruction/output ✓")


if __name__ == "__main__":
    main()
