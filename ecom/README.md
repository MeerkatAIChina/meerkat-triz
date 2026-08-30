# 电商运营数据分析闭环框架

分钟级电商运营闭环：数据采集 → 指标计算 → 异常检测 → AI 决策 → 执行 → 反馈。

## 架构

```
数据采集(data_ingestion) → 指标计算(metrics) → 异常检测(anomaly)
    → AI 决策(ai_advisor，调 Meerkat-AI 电商 skill) → 执行(executor) → 反馈(feedback)
```

## 文件

| 文件 | 职责 |
|---|---|
| `config.py` | 平台凭证、阈值、AI 端点、执行分级（凭证占位，接入时填写） |
| `data_ingestion.py` | 4 平台 API + 数据库 + log → 统一 schema（mock 兜底） |
| `metrics.py` | 指标计算（GMV/转化/ROI/库存）+ 快照构建 |
| `anomaly.py` | 异常检测（库存/GMV/转化/ROI 规则引擎） |
| `ai_advisor.py` | 调 Meerkat-AI，按异常触发对应电商 skill |
| `executor.py` | 执行层（分级：auto/notify/suggest） |
| `feedback.py` | 反馈追踪（动作 + 指标归因） |
| `scheduler.py` | 主调度（分钟级循环） |
| `snapshot_schema.md` | 数据层接口规范（与 AI 决策层的契约） |

## 运行

```bash
# 单轮闭环（测试）
python3 scheduler.py --once

# 持续运行（每分钟一轮）
python3 scheduler.py
```

## 接入真实数据

1. 在 `config.py` 填入平台凭证（app_key/app_secret/access_token）
2. 在 `data_ingestion.py` 实现各平台的 `_api_request`（签名+请求）
3. 数据库 DSN 和 log 路径填入 `config.py`

凭证未配置时，框架用 mock 数据兜底，保证闭环可运行（供测试和演示）。

## AI 决策层（已部署）

5 个电商 skill（Web UI 上）：
- 【电商】选品分析 / 新品建议 / 调价策略 / 补货管理 / 投放优化

异常类型自动映射到对应 skill（见 snapshot_schema.md）。
