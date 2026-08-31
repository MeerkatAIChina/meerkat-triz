#!/usr/bin/env python3
"""
RLVR 样本生成器 (增强 signal 版): order.csv → 商品-天销量预测样本 (JSONL)

任务 (Tier 1, 零反事实): 给定商品过去 30 天销量 + 促销 + 价格 + 生命周期, 预测未来 7 天总销量。

signal (进 prompt, 全部为窗口前可观测信息, 严格防泄漏):
  1. 过去 30 天每日销量序列
  2. 过去 30 天每日促销序列 (None/Coupon/FlashSale/FullDiscount)
  3. 价格: 当前价 + 近30天 [min~max]
  4. 商品生命周期: 上架后天数
  5. 未来 7 天周末数 (日历信息, 可观测)

metadata (不进 prompt, 供 oracle 分析 / 后续研究):
  - 未来 7 天促销计划 (验证"促销"作为未来 signal 的预测价值)

用法:
  python3 build_rlvr_dataset.py <order.csv> <out.jsonl> [--limit N] [--history 30] [--forecast 7]
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta

csv.field_size_limit(sys.maxsize)

VALID_STATUS = {"Delivered", "Shipped", "Paid"}
PROMO_SHORT = {"None": "无", "Coupon": "券", "FlashSale": "闪购", "FullDiscount": "满减"}
WEEKDAY_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def verify_sales_forecast(pred, actual):
    """销量预测奖励: -clip(|pred-actual|/actual, 0, 1)"""
    if actual <= 0:
        return 0.0 if pred <= 0 else -1.0
    return -min(abs(pred - actual) / actual, 1.0)


def load_orders(path):
    """读 order.csv → 各商品的每日 (销量, 促销, 价格) + 静态属性。"""
    pday = defaultdict(lambda: defaultdict(int))          # pid -> day -> qty
    ppromo = defaultdict(lambda: defaultdict(Counter))    # pid -> day -> Counter(promo)
    pprice = defaultdict(lambda: defaultdict(list))       # pid -> day -> [prices]
    meta = {}                                            # pid -> {brand, category, launch_date}

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["order_status"] not in VALID_STATUS:
                continue
            pid = row["product_id"]
            day = row["order_time"][:10]
            qty = int(row["quantity"])
            promo = row["promotion_type"] or "None"
            pday[pid][day] += qty
            ppromo[pid][day][promo] += 1
            pprice[pid][day].append(float(row["price"]))
            if pid not in meta:
                meta[pid] = {
                    "brand": row["brand"],
                    "category": row["category"],
                    "launch_date": row["launch_date"][:10] if row["launch_date"] else None,
                }
    return pday, ppromo, pprice, meta


def _promo_of_day(day_counter):
    """当天主要促销类型 (出现最多), 无促销返回 None。"""
    if not day_counter:
        return None
    promo = day_counter.most_common(1)[0][0]
    return None if promo == "None" else promo


def build_samples(pday, ppromo, pprice, meta, history_days=30, forecast_days=7, limit=None):
    """滑动窗口切分 → (prompt, ground_truth, metadata) 列表, 带增强 signal。"""
    samples = []
    all_days = sorted({d for dd in pday.values() for d in dd})
    day_index = {d: i for i, d in enumerate(all_days)}
    n_days = len(all_days)

    for pid, daily in pday.items():
        if pid not in meta:
            continue
        m = meta[pid]
        series = [daily.get(d, 0) for d in all_days]
        # 促销序列 + 价格序列 (对齐 all_days)
        promo_series = [_promo_of_day(ppromo[pid].get(d)) for d in all_days]
        price_series = []
        for d in all_days:
            ps = pprice[pid].get(d)
            price_series.append(round(sum(ps) / len(ps), 2) if ps else None)

        # 生命周期 (上架到 window_start 的天数)
        def life_days(window_start_str):
            if not m["launch_date"]:
                return None
            try:
                launch = datetime.strptime(m["launch_date"], "%Y-%m-%d")
                ws = datetime.strptime(window_start_str, "%Y-%m-%d")
                return (ws - launch).days
            except Exception:
                return None

        for start in range(0, n_days - history_days - forecast_days):
            hist = series[start:start + history_days]
            hist_promo = promo_series[start:start + history_days]
            hist_price = [p for p in price_series[start:start + history_days] if p is not None]

            ws_idx = start + history_days
            ws_str = all_days[ws_idx]
            we_idx = ws_idx + forecast_days - 1
            we_str = all_days[we_idx]
            actual = sum(series[ws_idx:ws_idx + forecast_days])

            # 未来 7 天周末数 (日历信息, 可观测)
            try:
                ws_dt = datetime.strptime(ws_str, "%Y-%m-%d")
                weekend_count = sum(
                    1 for i in range(forecast_days)
                    if (ws_dt + timedelta(days=i)).weekday() >= 5
                )
                weekday_label = WEEKDAY_CN[ws_dt.weekday()]
            except Exception:
                weekend_count = 0
                weekday_label = ""

            # 生命周期
            life = life_days(ws_str)

            # 促销序列编码
            promo_str = ", ".join(PROMO_SHORT.get(p or "None", "无") for p in hist_promo)

            # 价格信息
            if hist_price:
                cur_price = hist_price[-1]
                price_info = f"当前{cur_price}元, 近30天[{min(hist_price)}~{max(hist_price)}]元"
            else:
                price_info = "价格未知"

            # 未来 7 天促销计划 (只进 metadata, 不进 prompt)
            future_promo = [
                _promo_of_day(ppromo[pid].get(all_days[ws_idx + i])) for i in range(forecast_days)
            ]

            prompt = (
                f"商品【{m['brand']}】(品类:{m['category']}, "
                f"上架{m['launch_date'] or '未知'}"
                f"{f', 已上架{life}天' if life is not None else ''})。"
                f"价格: {price_info}。"
                f"过去{history_days}天每日销量: [{', '.join(map(str, hist))}]。"
                f"过去{history_days}天每日促销: [{promo_str}]。"
                f"未来{forecast_days}天起始于{ws_str}({weekday_label}), 含{weekend_count}个周末。"
                f"请预测未来{forecast_days}天总销量。"
            )

            samples.append({
                "prompt": prompt,
                "ground_truth": actual,
                "metadata": {
                    "product_id": pid, "brand": m["brand"], "category": m["category"],
                    "window_start": ws_str, "window_end": we_str,
                    "history": hist,
                    "history_promo": hist_promo,
                    "future_promo": future_promo,   # 仅供 oracle 分析, 不进 prompt
                    "life_days": life,
                    "weekend_count": weekend_count,
                    "price_min": min(hist_price) if hist_price else None,
                    "price_max": max(hist_price) if hist_price else None,
                    "price_last": hist_price[-1] if hist_price else None,
                },
            })
            if limit and len(samples) >= limit:
                return samples
    return samples


def baseline_report(samples):
    """朴素基线 (预测=最近7天销量) + oracle 基线 (知道未来促销则上调) 的奖励分布对比。"""
    def rewards_of(predictor):
        rs = []
        for s in samples:
            pred = predictor(s)
            rs.append(verify_sales_forecast(pred, s["ground_truth"]))
        return rs

    def naive(s):
        return sum(s["metadata"]["history"][-7:])

    def oracle_promo(s):
        """知道未来7天促销: 有促销则把naive预测翻倍 (近似促销效应)。"""
        base = sum(s["metadata"]["history"][-7:])
        has_future_promo = any(p for p in s["metadata"].get("future_promo", []))
        return base * 2 if has_future_promo else base

    def report(name, rs):
        rs = sorted(rs)
        n = len(rs)
        mean = sum(rs) / n
        sd = (sum((r - mean) ** 2 for r in rs) / n) ** 0.5
        print(f"  [{name}] 均值={mean:.3f} 标准差={sd:.3f} 中位={rs[n//2]:.3f}")
        return mean

    print(f"[baseline] 样本数 {len(samples)}")
    report("naive(最近7天)", rewards_of(naive))
    report("oracle(知未来促销翻倍)", rewards_of(oracle_promo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("order_csv")
    ap.add_argument("out_jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--history", type=int, default=30)
    ap.add_argument("--forecast", type=int, default=7)
    args = ap.parse_args()

    print(f"[load] {args.order_csv} ...")
    pday, ppromo, pprice, meta = load_orders(args.order_csv)
    print(f"[load] {len(meta)} 商品")

    samples = build_samples(pday, ppromo, pprice, meta, args.history, args.forecast, args.limit)
    print(f"[build] 生成 {len(samples)} 条样本")

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[save] → {args.out_jsonl}")

    if samples:
        print("\n=== 样例 (第1条, 含 signal) ===")
        print(json.dumps(samples[0], ensure_ascii=False)[:500], "\n")
        print("=== 基线奖励分布 ===")
        baseline_report(samples)


if __name__ == "__main__":
    main()
