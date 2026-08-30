# 电商运营数据层接口规范

数据层（采集+计算）与 AI 决策层的**接口契约**。数据层产出"运营快照"，AI 决策层消费。

## 一、统一数据 schema（各平台归一化）

| 统一字段 | 淘宝 | 京东 | 拼多多 | 抖音 | 类型 |
|---|---|---|---|---|---|
| `platform` | taobao | jd | pdd | douyin | string |
| `sku_id` | num_iid | sku_id | goods_id | product_id | string |
| `sku_name` | title | name | goods_name | title | string |
| `price` | price | price | price | price | float |
| `stock` | quantity | stock_num | stock | stock | int |
| `sales_24h` | 近24h销量 | 同 | 同 | 同 | int |
| `gmv_24h` | 成交额 | 同 | 同 | 同 | float |
| `orders_24h` | 订单数 | 同 | 同 | 同 | int |
| `visitors_24h` | UV | 同 | 同 | 同 | int |
| `ad_spend_24h` | 直通车 | 京准通 | 多多搜索 | 千川 | float |

## 二、运营快照格式（每分钟产出）

```json
{
  "timestamp": "2026-08-30T10:37:00",
  "metrics": {
    "total_gmv_24h": 1280000,
    "total_orders_24h": 5320,
    "total_visitors_24h": 286000,
    "overall_conversion": 1.86,
    "total_ad_spend_24h": 86000,
    "overall_roi": 14.88,
    "stockout_risk_skus": 3,
    "slow_moving_skus": 12,
    "by_platform": {
      "douyin": {"gmv_24h": 400000, "orders_24h": 1800, "conversion": 2.1, "roi": 1.8}
    }
  },
  "alerts": [
    {"type": "stockout", "sku_id": "xxx", "sku_name": "xxx", "severity": "high",
     "detail": "库存 5 件，预计 2 小时断货"}
  ],
  "top_skus": [
    {"sku_id": "xxx", "name": "xxx", "gmv_24h": 89000, "conversion": 4.2}
  ]
}
```

## 三、异常类型 → AI skill 映射

| 异常 type | 触发 skill | severity |
|---|---|---|
| stockout | replenishment（补货管理） | high/medium |
| gmv_drop | price-optimization（调价策略） | high |
| gmv_surge | replenishment（补货） | info |
| conversion_drop | price-optimization（调价） | medium |
| roi_drop | ad-optimization（投放优化） | high |
| competitor_price | price-optimization（调价） | high |

## 四、异常检测规则（先规则后 AI）

| 异常 | 规则 | 阈值（config.py 可调） |
|---|---|---|
| 库存告急 | 库存 < 日均销量 × 1.5 | stock_safety_days=1.5 |
| GMV 突变 | 环比 ±20% | gmv_drop_pct=20 |
| 转化率突变 | 环比 ±15% | conversion_drop_pct=15 |
| ROI 恶化 | 单渠道 < 盈亏线 | roi_breakeven=3.0 |
| 竞品变价 | 竞品降价 >5% | competitor_price_drop_pct=5 |

## 五、执行分级策略

| 动作 | 策略 | 说明 |
|---|---|---|
| 补货预警 | auto | 自动通知（不花钱、可逆） |
| 小幅调价(<5%) | auto | 自动执行 |
| 大幅调价(≥5%) | notify | 人工确认 |
| 投放预算 | notify | 人工确认 |
| 选品/新品 | suggest | 仅建议 |

原则：**花真金白银的动作 AI 只建议，不直接执行；可逆/不花钱的动作全自动。**
