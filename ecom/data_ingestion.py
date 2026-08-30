"""数据采集层：4 平台 API + 数据库 + log → 统一 schema。

骨架代码：API 客户端留 TODO（填凭证后实现），归一化和 mock 数据可运行。
"""

import json
import random
import time
from config import PLATFORMS, DATABASE, LOG_PATHS

# ===== 统一 schema =====
UNIFIED_FIELDS = [
    "platform", "sku_id", "sku_name", "price", "stock",
    "sales_24h", "gmv_24h", "orders_24h", "visitors_24h", "ad_spend_24h",
]


def _api_request(platform, method, params):
    """调用平台 API 的通用骨架。

    TODO: 各平台签名算法不同，接入时在此实现签名和请求。
    返回原始 JSON。
    """
    cfg = PLATFORMS.get(platform, {})
    if not cfg.get("app_key") or not cfg.get("app_secret"):
        return None  # 凭证未配置，返回 None（调用方用 mock 兜底）
    # TODO: 实现签名 + 请求
    return None


def fetch_taobao():
    """淘宝：拉取商品/订单/投放数据。TODO 实现具体 API。"""
    raw = _api_request("taobao", "taobao.items.get", {})
    return raw


def fetch_jd():
    """京东：拉取商品/订单/投放数据。TODO 实现具体 API。"""
    return _api_request("jd", "jd.item.list", {})


def fetch_pdd():
    """拼多多：拉取商品/订单/投放数据。TODO 实现具体 API。"""
    return _api_request("pdd", "pdd.goods.list", {})


def fetch_douyin():
    """抖音：拉取商品/订单/投放数据。TODO 实现具体 API。"""
    return _api_request("douyin", "product.list", {})


def read_database():
    """从业务数据库读取订单/库存/投放数据。TODO 按 DATABASE 配置实现。"""
    if not DATABASE.get("dsn"):
        return None
    # TODO: 连数据库读 orders/stock/ads 表
    return None


def read_log():
    """从 log 读取行为数据。TODO 按 LOG_PATHS 解析。"""
    if not LOG_PATHS:
        return None
    # TODO: 解析 log
    return None


def normalize(platform, raw_items):
    """把某平台的原始数据归一化到统一 schema。

    raw_items: 平台原始商品列表
    返回: 统一 schema 的商品列表
    """
    result = []
    for item in raw_items or []:
        # 各平台字段名不同，统一映射
        result.append({
            "platform": platform,
            "sku_id": item.get("sku_id") or item.get("num_iid") or item.get("goods_id") or item.get("product_id"),
            "sku_name": item.get("name") or item.get("title") or "",
            "price": float(item.get("price", 0)),
            "stock": int(item.get("stock", 0)),
            "sales_24h": int(item.get("sales_24h", 0)),
            "gmv_24h": float(item.get("gmv_24h", 0)),
            "orders_24h": int(item.get("orders_24h", 0)),
            "visitors_24h": int(item.get("visitors_24h", 0)),
            "ad_spend_24h": float(item.get("ad_spend_24h", 0)),
        })
    return result


def _mock_data():
    """Mock 数据（凭证未配置时兜底，保证框架可运行）。"""
    mock = []
    for platform in ["taobao", "jd", "pdd", "douyin"]:
        for i in range(20):
            mock.append({
                "platform": platform,
                "sku_id": f"{platform}-sku-{i}",
                "sku_name": f"商品{i}",
                "price": round(random.uniform(30, 300), 2),
                "stock": random.randint(0, 500),
                "sales_24h": random.randint(0, 100),
                "gmv_24h": round(random.uniform(0, 30000), 2),
                "orders_24h": random.randint(0, 50),
                "visitors_24h": random.randint(0, 5000),
                "ad_spend_24h": round(random.uniform(0, 3000), 2),
            })
    return mock


def fetch_all():
    """主入口：拉取所有数据源，归一化到统一 schema。

    返回: 统一 schema 的商品列表（含所有平台）
    """
    all_items = []

    # 4 平台（凭证未配时 mock 兜底）
    for platform, fetcher in [("taobao", fetch_taobao), ("jd", fetch_jd),
                              ("pdd", fetch_pdd), ("douyin", fetch_douyin)]:
        raw = fetcher()
        if raw is None:
            # 凭证未配置，用 mock（保证框架可跑，接真实 API 后自动切换）
            all_items.extend(_mock_data() if platform == "taobao" else _mock_data())
        else:
            all_items.extend(normalize(platform, raw))

    # 数据库（TODO 实现后合并）
    db_raw = read_database()
    if db_raw is not None:
        all_items.extend(normalize("db", db_raw))

    # log（TODO 实现后合并）
    log_raw = read_log()
    if log_raw is not None:
        all_items.extend(normalize("log", log_raw))

    return all_items


if __name__ == "__main__":
    items = fetch_all()
    print(f"采集 {len(items)} 条商品数据（mock 模式）")
    print(json.dumps(items[:2], ensure_ascii=False, indent=2))
