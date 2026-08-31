#!/usr/bin/env python3
"""
把 build_rlvr_dataset.py 产出的 JSONL 转成 verl 的 parquet 格式。

verl 数据契约 (RLHFDataset):
  data_source  : 数据集名
  prompt       : list[dict] (chat format)
  ability      : 任务类别
  reward_model : {"style": "rule", "ground_truth": "..."}  (RLVR rule reward)
  extra_info   : 额外 metadata (传给 reward function 的 extra_info)

输出: <out_dir>/train.parquet + test.parquet (默认 98/2 随机分割)

依赖: pip install datasets pyarrow (verl 训练环境自带)
用法: python3 convert_to_verl_parquet.py data/rlvr_sales_forecast.jsonl data/verl_parquet
"""
import argparse
import json
import os
import random


def load_jsonl(path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def to_verl_record(s):
    """JSONL 样本 → verl parquet 记录。ground_truth 转字符串 (verl reward_model 约定)。"""
    m = s["metadata"]
    return {
        "data_source": "sales_forecast",
        "prompt": [{"role": "user", "content": s["prompt"]}],
        "ability": "sales_forecast",
        "reward_model": {"style": "rule", "ground_truth": str(s["ground_truth"])},
        "extra_info": {
            "product_id": m["product_id"],
            "brand": m["brand"],
            "category": m["category"],
            "window_start": m["window_start"],
            "window_end": m["window_end"],
            "life_days": m.get("life_days"),
            "weekend_count": m.get("weekend_count"),
            "future_promo": m.get("future_promo", []),
            "price_min": m.get("price_min"),
            "price_max": m.get("price_max"),
            "price_last": m.get("price_last"),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="build_rlvr_dataset.py 输出的 JSONL")
    ap.add_argument("out_dir", help="parquet 输出目录")
    ap.add_argument("--test_ratio", type=float, default=0.02, help="测试集比例")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="只转前 N 条 (验证用)")
    args = ap.parse_args()

    try:
        from datasets import Dataset
    except ImportError:
        raise SystemExit("缺少 datasets 库, 请先: pip install datasets pyarrow")

    print(f"[load] {args.jsonl} ...")
    samples = load_jsonl(args.jsonl)
    if args.limit:
        samples = samples[:args.limit]
    print(f"[load] {len(samples)} 条样本")

    random.seed(args.seed)
    random.shuffle(samples)
    n_test = int(len(samples) * args.test_ratio)
    test, train = samples[:n_test], samples[n_test:]

    os.makedirs(args.out_dir, exist_ok=True)
    Dataset.from_list([to_verl_record(s) for s in train]).to_parquet(
        os.path.join(args.out_dir, "train.parquet"))
    Dataset.from_list([to_verl_record(s) for s in test]).to_parquet(
        os.path.join(args.out_dir, "test.parquet"))

    print(f"[convert] train={len(train)} test={len(test)} → {args.out_dir}/")
    print("  字段: data_source, prompt, ability, reward_model(style=rule, ground_truth), extra_info")


if __name__ == "__main__":
    main()
