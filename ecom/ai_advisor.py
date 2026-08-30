"""AI 决策层：把运营快照 + 异常喂给 Meerkat-AI，触发对应电商 skill。"""

import json
import requests
from config import MEERKAT_AI

# 异常类型 → 电商 skill 映射
ALERT_TO_SKILL = {
    "stockout": "replenishment",        # 补货管理
    "gmv_drop": "price-optimization",   # 调价策略
    "gmv_surge": "replenishment",       # 销量激增 → 补货
    "conversion_drop": "price-optimization",  # 转化降 → 调价
    "roi_drop": "ad-optimization",      # 投放优化
    "competitor_price": "price-optimization",
}

SKILL_NAMES = {
    "replenishment": "补货管理",
    "price-optimization": "调价策略",
    "ad-optimization": "投放优化",
    "product-selection": "选品分析",
    "new-product-suggestion": "新品建议",
}


def _decide_skill(alerts):
    """根据异常决定要触发哪些电商 skill。"""
    skills = set()
    for a in alerts:
        s = ALERT_TO_SKILL.get(a["type"])
        if s:
            skills.add(s)
    return list(skills) or ["product-selection"]  # 无异常时做选品分析兜底


def call_ai(snapshot, alerts):
    """调用 Meerkat-AI，传入快照和触发的 skill，返回建议。

    snapshot: build_snapshot 的输出
    alerts: detect_anomalies 的输出
    """
    skills = _decide_skill(alerts)
    skill_names = "、".join(SKILL_NAMES.get(s, s) for s in skills)

    # 构造用户消息：快照 + 触发的 skill
    user_msg = (
        f"以下是当前电商运营快照，请基于此给出【{skill_names}】建议。\n\n"
        f"运营快照：\n{json.dumps(snapshot, ensure_ascii=False, indent=2)}\n\n"
        f"检测到的异常：\n{json.dumps(alerts, ensure_ascii=False, indent=2)}\n\n"
        f"请输出：1) 核心洞察 2) 具体建议（可执行、带参数）3) 风险提示。"
    )

    payload = {
        "model": MEERKAT_AI["model"],
        "messages": [
            {"role": "system", "content": "你是电商运营分析专家，基于实时数据给出可执行的运营建议。"},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": MEERKAT_AI["max_tokens"],
        "temperature": MEERKAT_AI["temperature"],
    }

    try:
        r = requests.post(
            f"{MEERKAT_AI['base_url']}/chat/completions",
            json=payload, timeout=120,
        )
        r.raise_for_status()
        d = r.json()
        content = d["choices"][0]["message"].get("content") or ""
        return {"skills": skills, "advice": content, "status": "ok"}
    except Exception as e:
        return {"skills": skills, "advice": "", "status": "error", "error": str(e)}


if __name__ == "__main__":
    from data_ingestion import fetch_all
    from metrics import compute_metrics, build_snapshot
    from anomaly import detect_anomalies
    items = fetch_all()
    m = compute_metrics(items)
    a = detect_anomalies(m, items)
    snap = build_snapshot(m, a)
    result = call_ai(snap, a)
    print(f"触发 skill: {result['skills']}")
    print(f"状态: {result['status']}")
    print(f"AI 建议前 500 字:\n{result['advice'][:500]}")
