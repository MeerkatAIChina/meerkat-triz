# E1 包报告 (judge 方法学)

## E1a 位置交换双跑 (judge=moonshot-v1-32k, base_src=gold_cache:eval_v4_base_gold_20260723_105438.json)
- 裁决数 200 / 题数 100
- AB 序 v4 胜率: 0.180 [0.117, 0.267] (tie=0)
- BA 序 v4 胜率: 0.800 [0.711, 0.867] (tie=10)
- 合并 v4 胜率: 0.490 [0.422, 0.559]
- **位置不一致率: 0.870** [0.790, 0.922] (含 tie 差异)
- 硬翻转 (v4_win<->base_win) 率: 0.770
- 判定: 位置不一致率 >10%, judge 轨数字须以双序平均重报

## E1b 多评委交叉 (moonshot-v1-32k vs moonshot-v1-8k, n=300)
- 注意: kimi-k2-0711-preview 探测 404 时兜底 moonshot-v1-8k(同族弱异源, 非完全异源)
- 逐题 Spearman: overall=0.945, accuracy=0.948, triz=0.945
- 模型均值: {"base": {"j32": 1.57, "j2": 1.5}, "v2": {"j32": 2.58, "j2": 2.57}, "v4": {"j32": 2.57, "j2": 2.57}}
- 排序一致性: 32k=['v2', 'v4', 'base'] vs moonshot-v1-8k=['v2', 'v4', 'base']
- v4-base 逐题差值符号一致率: 0.940 (n=100)
- 子集差值: {"principle_recommendation": {"n": 20, "j32_meandiff": 0.5, "j2_meandiff": 0.8}, "contradiction_analysis": {"n": 20, "j32_meandiff": 1.35, "j2_meandiff": 1.3}, "ariz_guidance": {"n": 20, "j32_meandiff": 1.15, "j2_meandiff": 1.3}, "concept_explanation": {"n": 15, "j32_meandiff": 1.0666666666666667, "j2_meandiff": 1.0}, "case_generation": {"n": 15, "j32_meandiff": 1.0666666666666667, "j2_meandiff": 1.0}, "innovation_assessment": {"n": 10, "j32_meandiff": 0.8, "j2_meandiff": 0.9}}

## E1c 翻转率 T0.0 (n_verdicts=400, cells=80)
- 全一致 cell 比例: 1.000
- **平均翻转率: 0.0000** (文献 13.6%)
- 多数表决收敛 (与 5 次多数一致率): k=1 1.000, k=3 1.000, k=5 1.000
- 单次裁决 vs 最终题裁决一致率: 0.825
- 最终裁决: v4 胜 25/40, tie 7/40

## E1c 翻转率 T0.7 (n_verdicts=400, cells=80)
- 全一致 cell 比例: 0.938
- **平均翻转率: 0.0150** (文献 13.6%)
- 多数表决收敛 (与 5 次多数一致率): k=1 0.950, k=3 0.988, k=5 1.000
- 单次裁决 vs 最终题裁决一致率: 0.810
- 最终裁决: v4 胜 27/40, tie 7/40

---
# 干净 base 锚点 (base_goldfix) 补跑 — 旧 base (think 污染) 相关数字以此为准

## E1a' 位置交换 (v4 vs base_goldfix, judge=moonshot-v1-32k)
- AB 序 v4 胜率: 0.110 [0.063,0.186] | BA 序: 0.410 [0.319,0.508]
- **双序合并 v4 胜率: 0.260 [0.204,0.325]** (tie=8)
- 位置不一致率: 0.580 (硬翻转 0.500, B 位胜率 0.650)
- 对照旧污染 base: 合并胜率 0.490, 位置不一致率 0.870
- 结论: 干净锚点下第二位锚定依然存在 (0.65 vs 旧 0.81), 双序合并后 v4 不显著优于 干净 base

## E1b 补臂 干净锚点跨评委 (32k vs 8k, base_goldfix/v2/v4, n=300)
- **逐题 Spearman ρ = 0.759**
- 模型均值 (32k/8k): base_goldfix 2.87/2.68, v2 2.58/2.57, v4 2.57/2.57
- v4 vs base_goldfix: 32k 均差 -0.300 [-0.3, -0.46, -0.14], 8k 均差 -0.110, 符号一致率 0.850
- v2 vs base_goldfix: 32k 均差 -0.290 [-0.29, -0.42, -0.16], 8k 均差 -0.110, 符号一致率 0.820
- ⚠️ 注意: 干净 base judge 2.87 已高于 v2 2.58/v4 2.57 — "v4 大幅优于 base" 的旧结论 (+1.00) 是 think 污染伪影, 干净锚点下 v4/v2 反而低于 base_goldfix
