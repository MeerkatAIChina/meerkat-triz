#!/usr/bin/env python3
"""
RLVR 样本生成器: order.csv → 商品-天销量预测样本 (JSONL)

任务 (Tier 1, 零反事实): 给定商品过去 30 天每日销量 + 静态属性, 预测未来 7 天总销量。
输出每行: {prompt, ground_truth, metadata}  (response/reward 在 RLVR 训练时由模型+verifier 产生)

用法:
  python3 build_rlvr_dataset.py <order.csv> <out.jsonl> [--limit N] [--history 30] [--forecast 7]

订单状态过滤: 只统计 Delivered/Shipped/Paid (排除 Pending/Cancelled), 即"有效成交销量"。
防泄漏: prompt 只含窗口之前的销量, ground_truth 严格是窗口之后的销量。
"""
import argparse
import csv
import json
import sys
from collections import defaultdict

csv.field_size_limit(sys.maxsize)

VALID_STATUS = {"Delivered", "Shipped", "Paid"}


def verify_sales_forecast(pred, actual):
    """销量预测奖励: -clip(|pred-actual|/actual, 0, 1)"""
    if actual <= 0:
        return 0.0 if pred <= 0 else -1.0
    return -min(abs(pred - actual) / actual, 1.0)


def load_orders(path):
    """读 order.csv → (product_day_qty, product_meta, all_days)。"""
    pday = defaultdict(lambda: defaultdict(int))
    meta = {}       # product_id -> {brand, category, price_sum, price_cnt}
    price = defaultdict(list)
    all_days = set()

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["order_status"] not in VALID_STATUS:
                continue
            pid = row["product_id"]
            day = row["order_time"][:10]
            qty = int(row["quantity"])
            pday[pid][day] += qty
            all_days.add(day)
            if pid not in meta:
                meta[pid] = {"brand": row["brand"], "category": row["category"]}
            price[pid].append(float(row["price"]))

    # 静态价格 = 平均成交价
    for pid in meta:
        meta[pid]["price"] = round(sum(price[pid]) / len(price[pid]), 2)

    days = sorted(all_days)
    return pday, meta, days


def build_samples(pday, meta, days, history_days=30, forecast_days=7, limit=None):
    """滑动窗口切分 → (prompt, ground_truth, metadata) 列表。"""
    samples = []
    day_index = {d: i for i, d in enumerate(days)}
    n_days = len(days)

    for pid, daily in pday.items():
        if pid not in meta:
            continue
        # 商品的完整日销量序列 (缺失天 = 0)
        series = [daily.get(d, 0) for d in days]

        for start in range(0, n_days - history_days - forecast_days):
            hist = series[start:start + history_days]
            fut = series[start + history_days:start + history_days + forecast_days]
            actual = sum(fut)
            m = meta[pid]

            prompt = (
                f"商品【{m['brand']}】(品类:{m['category']}, 价格:{m['price']}元)。"
                f"过去{history_days}天每日销量: [{', '.join(map(str, hist))}]。"
                f"请预测未来{forecast_days}天总销量。"
            )
            samples.append({
                "prompt": prompt,
                "ground_truth": actual,
                "metadata": {
                    "product_id": pid, "brand": m["brand"], "category": m["category"],
                    "price": m["price"],
                    "window_start": days[start + history_days],
                    "window_end": days[start + history_days + forecast_days - 1],
                    "history": hist,
                },
            })
            if limit and len(samples) >= limit:
                return samples
    return samples


def baseline_report(samples):
    """用朴素基线 (预测=过去7天销量) 评估奖励分布, 验证区分度。"""
    rewards = []
    for s in samples:
        hist = s["metadata"]["history"]
        pred = sum(hist[-7:])          # naive: 用最近7天销量预测未来7天
        rewards.append(verify_sales_forecast(pred, s["ground_truth"]))

    rewards.sort()
    n = len(rewards)
    def pct(q):
        return rewards[int(n * q)] if n else 0
    print(f"[baseline] 样本数 {n}")
    print(f"  reward 分布: min={rewards[0]:.3f} p25={pct(.25):.3f} 中位={pct(.5):.3f} p75={pct(.75):.3f} max={rewards[-1]:.3f}")
    print(f"  均值={sum(rewards)/n:.3f}  标准差={ (sum((r-sum(rewards)/n)**2 for r in rewards)/n)**0.5 :.3f}")
    return rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("order_csv")
    ap.add_argument("out_jsonl")
    ap.add_argument("--limit", type=int, default=None, help="只生成前 N 条 (验证用)")
    ap.add_argument("--history", type=int, default=30)
    ap.add_argument("--forecast", type=int, default=7)
    args = ap.parse_args()

    print(f"[load] {args.order_csv} ...")
    pday, meta, days = load_orders(args.order_csv)
    print(f"[load] {len(meta)} 商品, {len(days)} 天")

    samples = build_samples(pday, meta, days, args.history, args.forecast, args.limit)
    print(f"[build] 生成 {len(samples)} 条样本")

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[save] → {args.out_jsonl}")

    if samples:
        print("\n=== 样例 (前2条) ===")
        for s in samples[:2]:
            print(json.dumps(s, ensure_ascii=False)[:400], "\n")
        print("=== 基线奖励分布 ===")
        baseline_report(samples)


if __name__ == "__main__":
    main()
