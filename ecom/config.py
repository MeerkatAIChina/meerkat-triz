"""电商运营分析框架 —— 全局配置。

凭证占位：接入真实平台时填入对应的 app_key/app_secret/session。
数据库和 log 路径按实际环境填写。
"""

# ===== 平台 API 凭证（占位，接入时填写）=====
PLATFORMS = {
    "taobao": {
        "app_key": "",       # 淘宝开放平台 app_key
        "app_secret": "",    # 淘宝开放平台 app_secret
        "session": "",       # 授权 session
        "base_url": "https://eco.taobao.com/router/rest",
    },
    "jd": {
        "app_key": "",
        "app_secret": "",
        "access_token": "",
        "base_url": "https://api.jd.com/routerjson",
    },
    "pdd": {
        "client_id": "",
        "client_secret": "",
        "access_token": "",
        "base_url": "https://gw-api.pinduoduo.com/api/router",
    },
    "douyin": {
        "app_key": "",
        "app_secret": "",
        "access_token": "",
        "base_url": "https://openapi-fxg.jinritemai.com",
    },
}

# ===== 数据库（业务库，按实际填写）=====
DATABASE = {
    "type": "postgresql",     # postgresql / mysql / sqlite
    "dsn": "",                # 如 postgresql://user:pass@host:5432/ecom
    "orders_table": "orders",
    "stock_table": "inventory",
    "ads_table": "ad_campaigns",
}

# ===== log 路径 =====
LOG_PATHS = []               # 如 ["/var/log/ecom/access.log"]

# ===== Meerkat-AI 端点（AI 决策层）=====
MEERKAT_AI = {
    "base_url": "http://127.0.0.1:8888/v1",
    "model": "Meerkat-TRIZ-v1-Qwen3.6-35B-A3B",
    "max_tokens": 2048,
    "temperature": 0.7,
}

# ===== 异常检测阈值（可调）=====
THRESHOLDS = {
    "stock_safety_days": 1.5,    # 安全库存 = 日均销量 × 1.5
    "gmv_drop_pct": 20,          # GMV 环比下降 20% 告警
    "gmv_surge_pct": 20,         # GMV 环比上升 20% 告警
    "conversion_drop_pct": 15,   # 转化率环比下降 15% 告警
    "roi_breakeven": 3.0,        # 单渠道 ROI 盈亏线
    "competitor_price_drop_pct": 5,  # 竞品降价 5% 告警
}

# ===== 分钟级调度 =====
SCHEDULE_INTERVAL = 60          # 秒（60 = 每分钟）

# ===== 执行分级 =====
# auto: 自动执行；notify: 通知人工；suggest: 仅建议
EXECUTION_POLICY = {
    "replenishment": "auto",    # 补货预警：自动通知
    "price_change_small": "auto",   # 小幅调价(<5%)：自动
    "price_change_large": "notify", # 大幅调价(>=5%)：人工确认
    "ad_budget": "notify",          # 投放预算：人工确认
    "selection": "suggest",         # 选品/新品：仅建议
}
