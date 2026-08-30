"""指标计算层：从统一 schema 计算核心指标，构建运营快照。"""

from collections import defaultdict


def compute_metrics(items):
    """计算整体 + 分平台 + 分 SKU 的指标。

    items: 统一 schema 的商品列表
    返回: 指标 dict
    """
    total = defaultdict(float)
    by_platform = defaultdict(lambda: defaultdict(float))

    for it in items:
        p = it["platform"]
        for field in ["gmv_24h", "orders_24h", "visitors_24h", "ad_spend_24h", "sales_24h"]:
            total[field] += it.get(field, 0)
            by_platform[p][field] += it.get(field, 0)

    # 转化率 = 订单 / UV × 100
    def conv(o, v):
        return round(o / v * 100, 2) if v else 0.0

    # ROI = GMV / 广告花费
    def roi(g, a):
        return round(g / a, 2) if a else 0.0

    metrics = {
        "total_gmv_24h": total["gmv_24h"],
        "total_orders_24h": total["orders_24h"],
        "total_visitors_24h": total["visitors_24h"],
        "overall_conversion": conv(total["orders_24h"], total["visitors_24h"]),
        "total_ad_spend_24h": total["ad_spend_24h"],
        "overall_roi": roi(total["gmv_24h"], total["ad_spend_24h"]),
        "by_platform": {
            p: {
                "gmv_24h": v["gmv_24h"],
                "orders_24h": v["orders_24h"],
                "conversion": conv(v["orders_24h"], v["visitors_24h"]),
                "roi": roi(v["gmv_24h"], v["ad_spend_24h"]),
            }
            for p, v in by_platform.items()
        },
    }

    # 库存告急 + 滞销 SKU
    stockout = [it for it in items if it["stock"] < it["sales_24h"] * 1.5 and it["sales_24h"] > 0]
    slow = [it for it in items if it["sales_24h"] == 0 and it["stock"] > 100]

    metrics["stockout_risk_skus"] = len(stockout)
    metrics["slow_moving_skus"] = len(slow)
    metrics["top_skus"] = sorted(items, key=lambda x: -x["gmv_24h"])[:10]

    return metrics


def build_snapshot(metrics, alerts):
    """构建运营快照（喂给 AI 决策层的标准格式）。"""
    import datetime
    return {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "metrics": {k: v for k, v in metrics.items() if k not in ("top_skus",)},
        "alerts": alerts,
        "top_skus": [
            {"sku_id": s["sku_id"], "name": s["sku_name"], "gmv_24h": s["gmv_24h"],
             "conversion": round(s["orders_24h"] / s["visitors_24h"] * 100, 2) if s["visitors_24h"] else 0}
            for s in metrics.get("top_skus", [])[:5]
        ],
    }


if __name__ == "__main__":
    from data_ingestion import fetch_all
    m = compute_metrics(fetch_all())
    print("指标:", {k: round(v, 2) if isinstance(v, float) else v
                  for k, v in m.items() if k not in ("top_skus", "by_platform")})
