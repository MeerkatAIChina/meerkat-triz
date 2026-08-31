#!/usr/bin/env python3
"""
verl 自定义 reward function: 商品-天销量预测 (RLVR rule reward)

接口 (verl 官方约定): compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float

训练时 verl 配置:
  --custom_reward_function.path=verl_reward.py
  --custom_reward_function.name=compute_score

奖励定义: -clip(|pred - actual| / actual, 0, 1)
  预测准确(误差0%) -> 0; 误差>=100% -> -1; 实际为0且预测>0 -> -1
"""
import re


def verify_sales_forecast(pred, actual):
    """销量预测奖励: 相对误差的负值, clip 到 [-1, 0]"""
    if actual <= 0:
        return 0.0 if (pred is not None and pred <= 0) else -1.0
    if pred is None:
        return -1.0
    return -min(abs(pred - actual) / actual, 1.0)


def extract_prediction(solution_str):
    """从模型输出提取预测的销量数字。

    优先级: 「销量」后数字 > 剥离天数后「预测」后数字 > 「:」后数字 > 最后数字。
    关键: 先剥离「未来7天/过去30天」等天数短语, 避免把天数(7)当销量值。
    """
    if not solution_str:
        return None
    s = solution_str
    # 1. 「销量」后面的数字 (如 "总销量: 14 件" / "销量约为14")
    m = re.search(r"销量\s*[:：为是约大约]?\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1))
    # 2. 剥离天数短语后, 「预测」后面的数字
    s2 = re.sub(r"(?:未来|过去|近|最近|前|后)\s*\d+\s*天", " ", s)
    m = re.search(r"预测[^0-9]*(-?\d+(?:\.\d+)?)", s2)
    if m:
        return float(m.group(1))
    # 3. 冒号后面的数字 (如 "预测: 14")
    m = re.search(r"[:：]\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1))
    # 4. 回退: 最后一个数字 (覆盖纯数字 "14")
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if nums:
        return float(nums[-1])
    return None


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """verl 入口。data_source/ground_truth/extra_info 来自 parquet 的对应字段。"""
    pred = extract_prediction(solution_str)
    try:
        actual = float(ground_truth)
    except (TypeError, ValueError):
        return -1.0
    return verify_sales_forecast(pred, actual)


# ===== 自测 (不依赖 verl) =====
if __name__ == "__main__":
    cases = [
        ("预测未来7天总销量: 14 件", "12", "正常带标签"),
        ("14", "12", "纯数字"),
        ("约为14件", "12", "约数"),
        ("预测销量 0 件", "0", "实际为0预测为0"),
        ("预测销量 5 件", "0", "实际为0预测>0"),
        ("无法预测", "12", "无数字"),
        ("预测14件(过去7天平均12件)", "12", "多数字取预测后的"),
    ]
    print("=== verl_reward 自测 ===")
    for sol, gt, desc in cases:
        r = compute_score("sales_forecast", sol, gt)
        print(f"  {desc:24s} solution='{sol}' gt={gt} → reward={r:.3f}")
