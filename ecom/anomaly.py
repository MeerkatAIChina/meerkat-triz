"""异常检测层：基于指标和历史，用规则引擎检测异常。

先规则，后 AI：规则检测出异常，再触发 AI 决策。
"""

from config import THRESHOLDS

# 历史指标（内存缓存，生产可用 Redis/DB）
_HISTORY = {}


def detect_anomalies(metrics, items):
    """检测异常，返回 alert 列表。

    metrics: compute_metrics 的输出
    items: 统一 schema 商品列表
    """
    alerts = []
    th = THRESHOLDS

    # 1. 库存告急
    for it in items:
        if it["sales_24h"] > 0 and it["stock"] < it["sales_24h"] * th["stock_safety_days"]:
            hours_left = it["stock"] / it["sales_24h"] * 24 if it["sales_24h"] else float("inf")
            alerts.append({
                "type": "stockout", "sku_id": it["sku_id"], "sku_name": it["sku_name"],
                "severity": "high" if hours_left < 24 else "medium",
                "detail": f"库存 {it['stock']} 件，日均销量 {it['sales_24h']}，预计 {hours_left:.1f} 小时断货",
            })

    # 2. GMV 环比突变（需历史）
    prev_gmv = _HISTORY.get("total_gmv_24h")
    if prev_gmv and prev_gmv > 0:
        delta = (metrics["total_gmv_24h"] - prev_gmv) / prev_gmv * 100
        if delta <= -th["gmv_drop_pct"]:
            alerts.append({"type": "gmv_drop", "severity": "high",
                           "detail": f"GMV 环比下降 {abs(delta):.1f}%"})
        elif delta >= th["gmv_surge_pct"]:
            alerts.append({"type": "gmv_surge", "severity": "info",
                           "detail": f"GMV 环比上升 {delta:.1f}%"})

    # 3. 转化率突变
    prev_conv = _HISTORY.get("overall_conversion")
    if prev_conv and prev_conv > 0:
        delta = (metrics["overall_conversion"] - prev_conv) / prev_conv * 100
        if delta <= -th["conversion_drop_pct"]:
            alerts.append({"type": "conversion_drop", "severity": "medium",
                           "detail": f"转化率环比下降 {abs(delta):.1f}%"})

    # 4. 单渠道 ROI 恶化
    for p, v in metrics.get("by_platform", {}).items():
        if 0 < v["roi"] < th["roi_breakeven"]:
            alerts.append({"type": "roi_drop", "channel": p, "severity": "high",
                           "detail": f"{p} 投放 ROI {v['roi']} 低于盈亏线 {th['roi_breakeven']}"})

    # 更新历史（供下一分钟环比）
    _HISTORY["total_gmv_24h"] = metrics["total_gmv_24h"]
    _HISTORY["overall_conversion"] = metrics["overall_conversion"]

    return alerts


if __name__ == "__main__":
    from data_ingestion import fetch_all
    from metrics import compute_metrics
    items = fetch_all()
    m = compute_metrics(items)
    a = detect_anomalies(m, items)
    print(f"检测到 {len(a)} 个异常")
    for x in a:
        print(f"  [{x['severity']}] {x['type']}: {x.get('detail', '')}")
