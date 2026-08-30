"""执行层：把 AI 建议转化为动作，按风险分级执行。

- auto: 自动执行（通知、低风险动作）
- notify: 通知人工确认
- suggest: 仅输出建议，不执行
"""

from config import EXECUTION_POLICY


def execute(advice_result, alerts):
    """根据 AI 建议和异常，执行动作。

    advice_result: ai_advisor.call_ai 的输出
    alerts: 异常列表
    """
    actions = []

    # 1. 补货预警（auto：自动通知）
    for a in alerts:
        if a["type"] == "stockout":
            action = {
                "type": "replenishment",
                "policy": EXECUTION_POLICY.get("replenishment", "auto"),
                "sku_id": a.get("sku_id"),
                "sku_name": a.get("sku_name"),
                "detail": a.get("detail"),
                "status": "executed" if EXECUTION_POLICY.get("replenishment") == "auto" else "pending",
            }
            actions.append(action)

    # 2. 调价（根据 AI 建议，TODO 解析 AI 输出的调价参数）
    # TODO: 从 advice 里解析调价建议，按幅度分级执行
    for a in alerts:
        if a["type"] in ("gmv_drop", "conversion_drop"):
            actions.append({
                "type": "price_change",
                "policy": EXECUTION_POLICY.get("price_change_large", "notify"),
                "detail": "建议调价（需人工确认幅度）",
                "status": "notify",
            })

    # 3. 投放（notify：人工确认）
    for a in alerts:
        if a["type"] == "roi_drop":
            actions.append({
                "type": "ad_budget",
                "policy": EXECUTION_POLICY.get("ad_budget", "notify"),
                "channel": a.get("channel"),
                "detail": "建议调整投放预算（需人工确认）",
                "status": "notify",
            })

    return actions


def _notify(action):
    """通知动作。TODO: 接入企业微信/钉钉/邮件通知。"""
    print(f"[通知] {action['type']}: {action.get('detail', '')}")


def run_execution(advice_result, alerts):
    """执行入口：遍历动作，按分级执行或通知。"""
    actions = execute(advice_result, alerts)
    for a in actions:
        if a["status"] == "executed":
            _notify(a)  # 自动执行的动作（这里简化为通知）
        elif a["status"] == "notify":
            _notify(a)  # 需人工确认的通知
    return actions


if __name__ == "__main__":
    # mock 测试
    alerts = [{"type": "stockout", "sku_id": "x1", "sku_name": "测试商品", "detail": "库存 5 件", "severity": "high"}]
    actions = run_execution({"advice": ""}, alerts)
    print(f"产生 {len(actions)} 个执行动作")
