#!/usr/bin/env python
"""
pipeline_v5 judge 双臂 + judge 纪律 (§6.4, §11.3)。

臂 A 长度归一化 judge:
  - rubric 加显式反冗长条款: "回答长度本身不构成加分项, 冗余重复内容在
    completeness 维度扣分";
  - judge 输入**不截断** (消除 v4 的 1500 字符截断不对称 —— 旧截断下
    base 仅 ~46% 可见);
  - 臂 A rubric 先在 v4 缓存试跑, v4−base 差值移动 >±0.15 时人工校准后
    再上决策门 (开放项 #12)。

臂 B 同长度桶对比:
  - <500 / 500–1500 / 1500–3000 / >3000 字符分桶, 只在同桶内配对;
  - v5 vs base 不同桶时报告"无同桶配对"并以臂 A 为准;
  - 臂 B 主用于 v5 vs v2 (同 ~300 字符, 同桶高功效)。

judge 纪律:
  - T=0 锁定: 温度硬编码断言;
  - 版本钉死: v5 全周期 judge = moonshot-v1-32k, 回退链
    kimi-k2-0711-preview → moonshot-v1-32k → moonshot-v1-8k,
    偏离须报告首页红色标注;
  - 评委家族谱系写入报告元数据 (数据/金标/评委同族结构风险声明)。
"""

JUDGE_PINNED = "moonshot-v1-32k"
JUDGE_FALLBACK_CHAIN = ["moonshot-v1-32k", "moonshot-v1-8k"]
# 注: kimi-k2-0711-preview 在链首但 §6.4 钉死 32k 为主; 链仅作可用性回退。
JUDGE_CHAIN_FULL = ["kimi-k2-0711-preview"] + JUDGE_FALLBACK_CHAIN
JUDGE_TEMPERATURE = 0.0  # T=0 锁定, 硬编码断言

ANTI_VERBOSITY_CLAUSE = (
    "\n- length_discipline: 回答长度本身不构成加分项; 冗余重复内容在 "
    "completeness 维度扣分 (反冗长条款, 臂 A 专用)")

JUDGE_SYSTEM_ARM_A = (
    "你是 TRIZ 领域资深评审专家, 正在评估一个 AI 助手对 TRIZ 评测题的回答质量。"
    "对每道题, 依据参考答案与期望关键词, 按以下 rubric 打 0-4 整数分 "
    "(0=完全错误/无关, 1=严重缺陷, 2=部分正确, 3=基本正确且较完整, 4=优秀):\n"
    "- accuracy: 事实与概念准确性\n"
    "- completeness: 相对参考答案的完整性\n"
    "- triz_correctness: TRIZ 方法论运用是否正确 (原理编号/矛盾分析/ARIZ步骤等)\n"
    "- structure: 回答结构与条理性\n"
    "- overall: 综合质量 (不是简单平均, 以 TRIZ 正确性为重)"
    + ANTI_VERBOSITY_CLAUSE +
    "\n只输出一个 JSON 数组, 不要输出任何其他文字或 markdown 围栏。格式:\n"
    '[{"id": "题目id", "accuracy": 0-4, "completeness": 0-4, '
    '"triz_correctness": 0-4, "structure": 0-4, "overall": 0-4}, ...]'
)

LENGTH_BUCKETS = [(0, 500, "<500"), (500, 1500, "500-1500"),
                  (1500, 3000, "1500-3000"), (3000, None, ">3000")]

# 评委家族谱系 (§6.4 异源评委终审: 同族评委必须声明"同族弱异源")
JUDGE_LINEAGE = {
    "moonshot-v1-32k": {"family": "moonshot", "vendor": "Moonshot AI",
                         "same_family_as_data": True,
                         "declaration": "同族弱异源 (数据/金标/评委同族)"},
    "moonshot-v1-8k": {"family": "moonshot", "vendor": "Moonshot AI",
                        "same_family_as_data": True,
                        "declaration": "同族弱异源 + 与金标生成器同源, 仅供参考"},
    "kimi-k2-0711-preview": {"family": "moonshot/kimi", "vendor": "Moonshot AI",
                              "same_family_as_data": True,
                              "declaration": "同族弱异源"},
    "gpt-4o": {"family": "openai", "vendor": "OpenAI",
                "same_family_as_data": False,
                "declaration": "真异源评委 (决策门终审要求 ≥1 席)"},
    "claude": {"family": "anthropic", "vendor": "Anthropic",
                "same_family_as_data": False, "declaration": "真异源评委 (备选)"},
    "deepseek": {"family": "deepseek", "vendor": "DeepSeek",
                  "same_family_as_data": False, "declaration": "真异源评委 (备选)"},
}


def judge_lineage(model):
    """评委家族谱系元数据; 未知评委如实标 unknown。"""
    info = JUDGE_LINEAGE.get(model)
    if info is None:
        return {"model": model, "family": "unknown",
                "same_family_as_data": None,
                "declaration": "未知评委, 谱系未登记, 须人工确认"}
    return {"model": model, **info}


def assert_temperature_zero(temperature):
    """T=0 锁定硬断言 (§6.4)。"""
    assert temperature == JUDGE_TEMPERATURE, \
        f"judge 温度必须锁定 T=0, 收到 {temperature}"


def assert_pinned_judge(model):
    """版本钉死检查。返回 (ok, deviation_note)。"""
    if model == JUDGE_PINNED:
        return True, ""
    if model in JUDGE_CHAIN_FULL:
        return False, (f"judge 偏离钉死版本 {JUDGE_PINNED} → {model} "
                       "(回退链内, 报告首页红色标注)")
    return False, (f"judge {model} 不在钉死版本/回退链内 "
                   f"{JUDGE_CHAIN_FULL}, 报告首页红色标注且禁止上决策门")


def build_judge_user_arm_a(batch, responses):
    """臂 A: judge 输入**不截断** (§6.4: 消除 1500 字符截断不对称)。"""
    parts = []
    for it in batch:
        resp = responses.get(it["id"], "")  # 不截断
        parts.append(
            f"【题目 {it['id']}】({it['subset']})\n"
            f"问题: {it['question']}\n"
            f"参考答案: {it['reference_answer']}\n"  # 不截断
            f"期望关键词: {'、'.join(it.get('keywords', []))}\n"
            f"AI 回答: {resp}\n")
    return "\n".join(parts) + "\n请按 rubric 对上述每题打分, 输出 JSON 数组。"


def length_bucket(n):
    for lo, hi, name in LENGTH_BUCKETS:
        if n >= lo and (hi is None or n < hi):
            return name
    raise ValueError(n)


def arm_b_pairs(responses_a, responses_b):
    """臂 B 同长度桶配对 (§6.4)。

    responses_a/responses_b: {id: response} (如 base / v5)。
    返回 {"pairs": [(id, bucket)], "no_pair": [id], "bucket_counts": {...},
           "note": ...}。仅同桶题进入配对; 不同桶题列入 no_pair 并报告
    "无同桶配对", 以臂 A 为准。
    """
    pairs, no_pair, counts = [], [], {}
    for qid, ra in responses_a.items():
        rb = responses_b.get(qid)
        if rb is None:
            no_pair.append(qid)
            continue
        ba, bb = length_bucket(len(ra.strip())), length_bucket(len(rb.strip()))
        if ba == bb:
            pairs.append((qid, ba))
            counts[ba] = counts.get(ba, 0) + 1
        else:
            no_pair.append(qid)
    note = ("同桶配对 %d 题; 无同桶配对 %d 题 (以臂 A 为准)"
            % (len(pairs), len(no_pair)))
    return {"pairs": pairs, "no_pair": no_pair,
            "bucket_counts": counts, "note": note}


def arms_conflict(arm_a_overall, arm_b_overall):
    """两臂结论冲突检测 (§6.4: 冲突 → 冻结判定 + 人工抽检 20 题)。

    arm_*_overall: {"diff": float, "ci95": [lo, hi]} 或 None (臂 B 无同桶配对)。
    冲突定义: 一臂显著为正而另一臂显著为负, 或一臂显著而另一臂异号且
    点值方向相反。
    """
    if arm_a_overall is None or arm_b_overall is None:
        return False, "臂 B 无可用配对, 以臂 A 为准"
    da, db = arm_a_overall["diff"], arm_b_overall["diff"]
    sig = lambda o: o["ci95"][0] > 0 or o["ci95"][1] < 0
    if sig(arm_a_overall) and sig(arm_b_overall) and (da > 0) != (db > 0):
        return True, "两臂显著方向相反 → 冻结判定 + 人工抽检 20 题"
    return False, "两臂不冲突"
