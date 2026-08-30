"""反馈层：执行动作后追踪效果，归因，反馈给 AI 迭代。

核心：执行了动作 X → N 分钟后指标 Y 变化 Z% → 记录 → 下次 AI 决策时参考。
"""

import json
import os
import time

FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), "feedback_log.jsonl")


def record_action(action, metrics_before):
    """记录一个动作及其执行前的指标快照，供后续对比。"""
    entry = {
        "timestamp": time.time(),
        "action": action,
        "metrics_before": metrics_before,
    }
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def track_effects(action, metrics_after):
    """对比执行前后的指标，归因效果。"""
    # 简化版：从 feedback_log 找对应动作，对比指标
    effects = []
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                if e["action"].get("type") == action.get("type"):
                    before = e["metrics_before"]
                    delta = {
                        "gmv_delta_pct": _pct(metrics_after.get("total_gmv_24h"), before.get("total_gmv_24h")),
                        "conversion_delta_pct": _pct(metrics_after.get("overall_conversion"), before.get("overall_conversion")),
                    }
                    effects.append({"action": action, "delta": delta})
    return effects


def _pct(after, before):
    if not before:
        return None
    return round((after - before) / before * 100, 2)


if __name__ == "__main__":
    print("反馈层骨架：record_action 记录动作 + track_effects 归因效果")
