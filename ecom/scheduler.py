"""主调度：分钟级闭环入口。

数据采集 → 指标计算 → 异常检测 → AI 决策 → 执行 → 反馈
"""

import time
from data_ingestion import fetch_all
from metrics import compute_metrics, build_snapshot
from anomaly import detect_anomalies
from ai_advisor import call_ai
from executor import run_execution
from feedback import record_action, track_effects
from config import SCHEDULE_INTERVAL


def one_cycle():
    """执行一次完整闭环。"""
    # 1. 数据采集
    items = fetch_all()

    # 2. 指标计算
    metrics = compute_metrics(items)

    # 3. 异常检测
    alerts = detect_anomalies(metrics, items)

    # 4. AI 决策（有异常时触发；无异常也定期做选品分析）
    snapshot = build_snapshot(metrics, alerts)
    advice = call_ai(snapshot, alerts)

    # 5. 执行
    actions = run_execution(advice, alerts)

    # 6. 反馈（记录动作 + 指标，供后续归因）
    for a in actions:
        record_action(a, metrics)

    # 输出本轮摘要
    print(f"[{time.strftime('%H:%M:%S')}] 闭环完成: "
          f"{len(items)} 商品 | {len(alerts)} 异常 | {len(actions)} 动作 | AI建议 {len(advice.get('advice',''))} 字")

    return {"metrics": metrics, "alerts": alerts, "actions": actions, "advice": advice}


def run_forever():
    """持续运行（分钟级循环）。"""
    print(f"电商运营分析框架启动（每 {SCHEDULE_INTERVAL} 秒一轮）")
    while True:
        try:
            one_cycle()
        except Exception as e:
            print(f"[错误] {e}")
        time.sleep(SCHEDULE_INTERVAL)


if __name__ == "__main__":
    import sys
    if "--once" in sys.argv:
        result = one_cycle()
        print(f"\n=== 本轮异常 ===")
        for a in result["alerts"]:
            print(f"  [{a['severity']}] {a['type']}: {a.get('detail','')}")
        print(f"\n=== AI 建议 ===\n{result['advice'].get('advice','')[:800]}")
    else:
        run_forever()
