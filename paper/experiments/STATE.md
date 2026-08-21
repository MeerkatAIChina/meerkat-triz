# P0 实验包执行状态 (实验工程师_API包) — ✅ 全部完成(含干净 base 补跑)

> 补跑完成: 2026-07-24 06:36 (远端 UTC+8)。`results/goldfix_chain.done` 已置位。

## 干净 base 补跑包 (base_goldfix) — 核心结果

### E1a' 位置交换 (干净锚点)
- AB 序 v4 胜率 0.110 / BA 序 0.410; **双序合并 v4 胜率 0.260 [0.204,0.325] — v4 显著输给干净 base** (旧污染 base 为 0.49 平局)
- 位置不一致率 0.58 (旧 0.87), B 位胜率 0.65 (旧 0.81) — 第二位锚定仍在但减弱; 污染 base 放大了位置偏差
- **论文含义: v4 vs base 的 pairwise 故事完全改写 — judge 轨 +1.00 是 think 污染伪影**

### E1b 补臂 跨评委 (32k vs 8k, 干净锚点, n=300)
- **Spearman ρ=0.759** (低于污染锚点的 0.945, 仍为中高一致)
- 均值: base_goldfix 2.87/2.68 > v2 2.58/2.57 ≈ v4 2.57/2.57
- v4 vs base_goldfix 32k **-0.30 [-0.46,-0.14] 显著为负**, 8k -0.11, 符号一致率 0.85; v2 -0.29 [-0.42,-0.16] ✅, 一致率 0.82
- 注意: base_goldfix 是"干净生成"而 v2/v4 是微调模型, judge 轨 base 反超需谨慎解读 (judge 可能偏爱 base 的长篇教科书式回答风格; 与 E3 rubric 轨 v4>base 结论张力待论文统合)

### E3' ARIZ rubric (干净锚点)
- base_goldfix rubric=0.800 (旧草稿分 0.675 作废) kw=0.605; v2 0.875 / v4 0.883 不变
- **v4 vs base_goldfix +0.083 [0.033,0.142] ✅ 显著**; v2 vs base_goldfix +0.075 [-0.025,0.15] 不显著
- base_goldfix 关键词漏判 3/20 — 干净 base 的 ARIZ 回答也有漏判但少于污染版
- 结论: **v4 的 ARIZ 优势在干净锚点 + rubric 语义轨下依然成立且显著**, 是 v4 最稳健的能力增益点

## 产物 (新增)
- 远端: `results/e1/e1a_position_swap_goldfix.jsonl`, `e_goldfix_report.json`, `results/goldfix_chain.log`; `e1b_rejudge_moonshot_v1_8k.jsonl` 与 `e3_ariz_rubric.jsonl` 已追加 base_goldfix 行; `e1_report.md`/`e3_report.md` 已追加干净锚点章节
- 本地: `paper/experiments/e1/` (e1a_position_swap_goldfix.jsonl, e_goldfix_report.json, e1_report.md 更新), `e3/` (e3_report.md 更新)
- 脚本: `paper/experiments/scripts/{e1a_goldfix,e1b_goldfix,e3_goldfix,e_goldfix_analyze}.py`, `run_goldfix_chain.sh`

## 论文数字替换清单 (旧→新)
1. v4 vs base judge 差值: +1.00 [0.80,1.19] → **-0.30 [-0.46,-0.14]** (think 污染伪影, 作废)
2. v4 vs base pairwise 胜率: 无双序旧数据 → 双序合并 0.260 (v4 显著输)
3. ARIZ rubric base: 0.675 → **0.800**; v4 vs base ARIZ rubric 显著为正 (+0.083) 保留
4. 跨评委 ρ: 0.945 (污染锚点) → **0.759** (干净锚点)
5. E1a 位置不一致率: 0.87 (污染) → 0.58 (干净), 结论 "必须双序" 不变
6. E1c 翻转率 (T=0 恒 0 / T=0.7 0.015) 不涉及 base 内容变化? ⚠️ E1c 用的是污染 base responses — 干净锚点下翻转率实验未重跑 (pairwise 文本变了), 论文引用 E1c 时须注明

---
(以下为原 P0 包完成记录)

> 完成时间: 2026-07-24 04:37 (远端 UTC+8)。`results/p0_chain.done` 已置位。tmux 会话 `p0exp` 已结束。

## 产物位置
- 远端: `/home/meerkat/mongoose_ai/results/e1|e2|e3/`, 总日志 `results/p0_chain.log`
- 本地: `paper/experiments/e1/` (e1a_position_swap.jsonl, e1b_rejudge_moonshot_v1_8k.jsonl, e1b_meta.json, e1c_flip.jsonl, e1_report.{json,md}), `e2/` (e2_data_attr.json, e2_rejudge.jsonl, e2_report.json), `e3/` (e3_ariz_rubric.jsonl, e3_report.{json,md})
- 脚本副本: `paper/experiments/scripts/`

## 关键纪律记录
- **E0 (results/e0_basefix/) 全程无 .done** → E1a/E1b/E1c/E3/E2 全部使用旧金标缓存 base (91/100 think 污染)。E0 完成后若要干净 base 锚点, E1a/E1c pairwise 与 E2/E3 中涉及 base 的对比需重跑 (脚本已内置自动切换: `e1_common.load_responses("base")` 检测 e0_basefix/*.done)
- kimi-k2-0711-preview 404 (两次探测) → E1b 兜底 moonshot-v1-8k, 跨评委结论为"同族弱异源", 论文须标注
- 中途事故: e2/e3 脚本 import 路径 bug 导致 04:25 链假完成, 已修复重跑, 04:37 真正完成

## 核心结果 (五项实验全部完成)

### E1a 位置交换 ⚠️ 重大发现
- AB 序 v4 胜率 0.180 [0.117,0.267] vs BA 序 0.800 [0.711,0.867]; **位置不一致率 0.87** [0.79,0.92], 硬翻转 0.77, B 位总胜率 0.81
- 双序合并 v4 胜率 0.49 ≈ 平局; judge pairwise 存在极端第二位锚定, 一切 pairwise 结论必须双序平均

### E1b 跨评委 (32k vs 8k, n=300)
- **Spearman ρ=0.945** (overall; accuracy 0.948, triz 0.945); v4-base 逐题差值符号一致率 **0.94**; 排序一致 v2≈v4>base; 均差 ≈0 (base 1.57/1.50, v2 2.58/2.57, v4 2.57/2.57)

### E1c 翻转率 (800 裁决)
- **T=0: 翻转率 0.000** (80 cell 全一致) — judge 完全确定, 无需多次投票 (独立可发表发现)
- **T=0.7: 翻转率 0.015**, 远低于文献 13.6%; 收敛 k=1 0.950 / k=3 0.988 / k=5 1.000; 最终裁决 v4 胜 25-27/40, tie 7

### E3 ARIZ rubric 重判
- **关键词漏判率 19/60 = 0.317** (kw<0.5 而 rubric≥0.5), 反向高估仅 1/60
- rubric 覆盖: v4 0.883 ≈ v2 0.875 > base 0.675; v4 vs base +0.208 [0.133,0.300] ✅; v2 vs base +0.200 [0.067,0.333] ✅; v4 vs v2 +0.008 [-0.083,0.125] 持平
- "v2 ARIZ 回退"在 rubric 轨下消失 → 证实关键词伪影假说; v4 的 ARIZ 优势在语义轨下对 base 成立、对 v2 为持平

### E2 concept 退化终判
- 数据侧: cap 门保留集显著更长 (MWU p=5.3e-25) 但幅度仅 +5%
- kw 轨 v4 vs v2 -0.083 [-0.149,-0.023] 显著, 但 McNemar n=15 p=0.5 不显著; judge 重判 v4 vs v2 +0.07 [-0.2,0.33] 持平, v4 vs base +1.2 [0.8,1.53] 显著为正
- 关键词重分类: **synonym 伪影 0 条**, 真缺失 8 词次 (功能分析×3 等工具术语)
- **终判: 非噪声也非伪影 — v4 真实减少了对 TRIZ 工具术语的点名复诵 (真缺失), 但回答质量未退化 (judge 持平 v2); 属术语表述层面的真实行为偏移, n=15 下决策门"保留 v2"仍稳健, v5 修复方向 = CE 样本保留术语枚举式答案**
