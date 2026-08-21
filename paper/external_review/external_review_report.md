# 异源评委终审报告 — Meerkat-TRIZ-v1 (v5a vs base, v5 金标 300 题)

- 协议: 同 v5 臂 A: 反冗长 rubric, 不截断, T=0, batch=5, paired bootstrap n=10000 seed=42
- 评委: claude-sonnet-4-6, gpt-5.4, gemini-3.5-flash (真异源: anthropic/openai/google)

| 评委 | n | base 均分 | v5a 均分 | 配对差值 [95% CI] | 显著 | 与 moonshot 逐题 ρ |
|---|---|---|---|---|---|---|
| moonshot-v1-32k (参照, 同族) | 300 | — | — | +0.3933 [+0.2967, +0.4900] | 显著 | — |
| claude-sonnet-4-6 | 299 | 2.555 | 2.649 | +0.0936 [+0.0201, +0.1672] | 显著 | 0.3054 |
| gpt-5.4 | 299 | 2.301 | 2.405 | +0.1037 [+0.0201, +0.1839] | 显著 | 0.3002 |
| gemini-3.5-flash | 299 | 2.871 | 2.823 | -0.0485 [-0.1438, +0.0452] | 不显著 | 0.2727 |

## 核心结论

### 1. 配对差值方向与量级（v5a vs base）

- **moonshot 同族参照**: +0.3933 [+0.2967, +0.4900]（高度显著）
- **claude-sonnet-4-6**: +0.0936 [+0.0201, +0.1672]（显著，但量级仅为 moonshot 的 **~24%**）
- **gpt-5.4**: +0.1037 [+0.0201, +0.1839]（显著，量级约为 moonshot 的 **~26%**）
- **gemini-3.5-flash**: -0.0485 [-0.1438, +0.0452]（**不显著**，方向甚至相反）

**综合判定**: 在真异源评委下，v5a 相较 base 的提升幅度大幅缩水。Anthropic/OpenAI 两评委虽维持正方向且统计显著，但差值仅为同族 moonshot 评委的约 1/4；Google 评委则未观测到显著优势，反而略偏负向。这表明原 harness 中 moonshot-v1-32k 作为同族评委给出的 +0.3933 增益，在异源评审下不能被简单外推。

### 2. 跨评委逐题 Spearman 一致性（base / v5a 臂）

| 评委对 | base 臂 ρ | v5a 臂 ρ |
|---|---|---|
| claude-sonnet-4-6 vs gpt-5.4 | **0.7375** | **0.7458** |
| claude-sonnet-4-6 vs gemini-3.5-flash | **0.6280** | **0.7354** |
| gpt-5.4 vs gemini-3.5-flash | **0.6918** | **0.7517** |

三评委间的逐题评分秩相关在 **0.63–0.75** 区间，属于**中高度一致**。
- v5a 臂的一致性普遍高于 base 臂（尤其 claude-gemini 从 0.628 提升至 0.735），说明 v5a 的改进方向在评委间更易达成共识。
- 三评委平均 ρ ≈ 0.69（base）/ 0.74（v5a），表明异源评委对题目难度的排序判断具有可重复性，但绝对评分尺度存在系统差异（gemini 整体打分偏高，gpt 偏低）。

### 3. 执行备注

- 仅 v4_gold_028 被 tensoris API 安全过滤（403 blocked），所有评委该题均降级为单条请求后仍被拦截，故 n=299。
- 脚本已适配 403 自动降级策略：批次 5 题遇 block 时逐条单发，仅真正触发过滤的题目被标记为 `__BLOCKED__` 并跳过，其余正常续评。

## claude-sonnet-4-6 子集差值

| 子集 | 差值 | 95% CI |
|---|---|---|
| ariz_guidance | +0.2500 | [+0.1167, +0.3833] |
| case_generation | -0.0227 | [-0.1818, +0.1591] |
| concept_explanation | +0.2444 | [+0.0667, +0.4444] |
| contradiction_analysis | +0.0167 | [-0.1833, +0.2167] |
| innovation_assessment | +0.2333 | [+0.1000, +0.4000] |
| principle_recommendation | -0.0833 | [-0.2333, +0.0667] |

## gpt-5.4 子集差值

| 子集 | 差值 | 95% CI |
|---|---|---|
| ariz_guidance | +0.2000 | [+0.0500, +0.3500] |
| case_generation | -0.0455 | [-0.2045, +0.1136] |
| concept_explanation | +0.3111 | [+0.0444, +0.5778] |
| contradiction_analysis | -0.0333 | [-0.2500, +0.1833] |
| innovation_assessment | +0.2000 | [-0.0333, +0.4333] |
| principle_recommendation | +0.0500 | [-0.1333, +0.2167] |

## gemini-3.5-flash 子集差值

| 子集 | 差值 | 95% CI |
|---|---|---|
| ariz_guidance | +0.0333 | [-0.1000, +0.1667] |
| case_generation | -0.0909 | [-0.2955, +0.1136] |
| concept_explanation | +0.2889 | [+0.0222, +0.5778] |
| contradiction_analysis | -0.0833 | [-0.3333, +0.1500] |
| innovation_assessment | -0.0667 | [-0.2667, +0.1333] |
| principle_recommendation | -0.3083 | [-0.5250, -0.1000] |
