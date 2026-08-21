# E0 报告: v4 金标评测 base 锚点 think 污染修复与干净重跑

- 执行人: 实验工程师_E0 · 日期: 2026-07-24 · 状态: **✅ 完成**
- 关键数字: v4−base judge 差 +1.00 [0.80,1.19] (污染) → **−0.30 [−0.46,−0.14] (干净, 反号)**;
  决策门: **保留 v2** (理由链已更新, 见 §4.4)

## 1. 问题 (W3 发现)

v4 金标评测 (`eval_v4_base_gold_20260723_105438.json`) 中, base 模型 100 条回答里
**91 条是未闭合的英文 `<think>` 草稿** —— 1024 token 预算被思考过程烧光, 正式答案从未产生。
关键词轨与 judge 轨评的都是草稿, 导致 "v4 vs base: judge +1.00 [0.80, 1.19] /
kw +0.1907 [0.1297, 0.2520]" 的提升被严重高估, 不能写入论文。

## 2. 根因定位 (本任务新发现)

`eval_harness.py:render_prompt` 在 `apply_chat_template(enable_thinking=False)` 之后
**把渲染出的空 think 块 `<think>\n\n</think>\n\n` 剥掉了**。Qwen3.6-35B-A3B 是
thinking-native 基座, 失去"思考已结束"锚点后 100/100 自吐 `<think>` 英文草稿。

诊断实验 (`results/e0_basefix/diag*.log`, 3 题冒烟):

| 策略 | 结果 |
|---|---|
| 原 harness 渲染 (剥空 think) + 2048 tok | ❌ 未闭合英文 think 草稿, 答案为空 |
| + assistant prefill 强制开场 | ❌ 立即 `<|im_end|>` (空) |
| + bad_words_ids 禁 think token | ❌ 立即结束回合 (空) |
| **保留空 think 块 (不剥)** | ✅ **直接产出正常中文结构化 TRIZ 作答 (3/3)** |

## 3. 修复方案

新 tag `base_goldfix`, 生成脚本 `results/e0_basefix/e0_gen_basefix.py`:
prompt 保留 `enable_thinking=False` 渲染的空 think 块; max_new_tokens=2048;
生成后剥离任何闭合 think 块; 校验非空 + 中文字符占比 ≥0.3; 个别失败者用
bad_words_ids 兜底重生成一次。不覆盖任何既有缓存; 冒烟 3 题 + 全量首题均为
正常中文作答 (首题 2947 字符, 含矛盾矩阵/工程参数分析)。

## 4. 修复后结果 (终稿, 2026-07-24; 统计: paired bootstrap 10000 次 seed=42, stdlib
与 eval_harness 同实现 + McNemar 精确 + Wilson CI; 全数据 `e0_stats.json`,
逐题明细 `eval_v4_base_goldfix_20260724_055459.json`)

### 4.1 生成与评分质量门 (全部通过)

- 生成 100/100, **mode 全部 direct** (0 兜底、0 invalid); 无 think 残留;
  无过短 (<100 字符); 长度 1246–4080, 均值 3250 字符。
- judge (moonshot-v1-32k, T=0, RPM=3 退避): 100/100 无缺失; **干净 base judge 均值
  2.87, kw 均值 0.5642** —— 远高于污染草稿的 1.57 / 0.3661。

### 4.2 四模型双轨总览 (n=100)

| 模型 | kw 均值 | kw pass [Wilson] | judge 均值 | judge pass [Wilson] |
|---|---|---|---|---|
| **base_goldfix (干净)** | **0.5642** | 0.670 [0.573, 0.754] | **2.87** | 0.830 [0.745, 0.891] |
| v2 | 0.5483 | 0.630 [0.532, 0.718] | 2.58 | 0.620 [0.522, 0.709] |
| v4 | 0.5568 | 0.650 [0.553, 0.736] | 2.57 | 0.630 [0.532, 0.718] |
| base_polluted (旧) | 0.3661 | 0.350 [0.264, 0.447] | 1.57 | 0.120 [0.070, 0.198] |

### 4.3 污染修复前后对比 (核心结论)

| 指标 | 污染 base 锚点 (旧) | 干净 base_goldfix 锚点 (新) | 变化 |
|---|---|---|---|
| **v4 − base judge 差** | **+1.00 [+0.80, +1.19]** ⚠️ | **−0.30 [−0.46, −0.14] 显著为负** | **反号** |
| **v4 − base kw 差** | **+0.1907 [+0.1297, +0.2520]** ⚠️ | **−0.0074 [−0.0497, +0.0345] 不显著** | 优势消失 |
| v4 − base McNemar (judge pass) | 55/4, p=1.7e-12 | 10/30, p=0.0022 (**反方向显著**) | 反号 |
| **v2 − base judge 差** | (未测, 同污染) | **−0.29 [−0.42, −0.16] 显著为负** | v2 同样低于干净 base |
| v2 − base kw 差 | (未测, 同污染) | −0.0158 [−0.0550, +0.0236] 不显著 | — |
| v4 − v2 judge 差 | −0.01 [−0.13, +0.11] | **−0.01 [−0.13, +0.11] (复算一致)** | 不变 |
| v4 − v2 kw 差 | +0.008 [−0.025, +0.043] | **+0.008 [−0.024, +0.044] (复算一致)** | 不变 |

**干净锚点下, v4 与 v2 在 judge 轨上双双显著低于 base 0.29-0.30 分, 关键词轨与
base 无差异。** "微调带来大幅提升" 的旧叙事彻底不成立——它是 base 草稿污染的测量伪影。
逐子集差值 (6 子集 × 2 轨 × 3 对比) 见 `e0_stats_report.md`; 显著项:
v4−base judge principle_recommendation −0.60 [−0.95, −0.25];
v2−base judge ariz −0.45 / concept −0.40 / principle −0.45 (均显著);
v4−base kw: ariz +0.144 (正) 与 concept_explanation −0.144 (负) 对冲;
v4−v2 kw concept_explanation −0.083 [−0.149, −0.023] (与 W3 一致, 与锚点无关)。

### 4.4 决策门 (干净锚点重判)

- 规则: v4 judge overall 显著 > base 且两轨所有子集无显著退化 → 建议替代 v2, 否则保留 v2。
- judge overall 显著为正: **否** (−0.30, 显著为负); judge 子集显著退化:
  principle_recommendation; kw 子集显著退化: concept_explanation。
- **判定: 保留 v2** —— 结论与旧报告相同, 但理由链完全改变: 旧理由是 "v4 相对
  (污染) base 有提升但 CE-kw 退化"; 新理由是 "v4 judge 轨显著低于干净 base,
  根本不满足替代条件, 且 CE-kw 退化依然显著"。

### 4.5 解读与注意事项

1. **为什么干净 base judge 分这么高**: base 回答长 (均值 3250 字符 vs v2/v4 ~300),
   内容详尽结构化, judge 的 completeness 维度受益; 同时 base 未经微调, 措辞泛化但
   TRIZ 知识本底扎实。judge 输入截断 1500 字符 (base 仅 ~46% 可见, v2/v4 100% 可见)
   的旧不对称在新方向下反而**压低**了 base 分——即 base 真实优势可能比 −0.30 更大。
2. **kw 轨两模型与 base 持平**: 微调没有带来表层关键词命中变化; v4/v2 的价值体现在
   回答简洁 (~1/10 长度) 与同分质量 (judge 仅差 0.29/4 ≈ 7%), 而非绝对质量提升。
3. **v4 ≈ v2 结论不受影响**: 两版差异全部不显著 (CE-kw 子集除外), 与 W3 一致。
4. **对论文叙事**: 主结论必须从 "v4/v2 显著优于 base (judge +1.00)" 改写为
   "LoRA 微调将 TRIZ 领域回答压缩至 ~1/10 长度且保持关键词水平, judge 轨质量
   略低于 verbose 的 base (−0.29~-0.30/4); v4 与 v2 等效, 决策门保留 v2"。
   旧数字只能作为测量污染案例进入方法学讨论。

## 5. 受影响结论清单: 需用 base_goldfix 重跑验证 (E1a/E1c/E3 API 包注意)

以下既有结论全部基于**污染的 base 缓存** (`results/results/v4_gen_base_gold.jsonl` /
`v4_judge_base_gold.json` / `eval_v4_base_gold_20260723_105438.json`), 必须由对应
实验负责人用 `base_goldfix` 缓存重跑验证 (本任务不代为重跑 API 包实验):

| 结论/实验 | 污染暴露面 | 重跑建议 |
|---|---|---|
| E1a / E1c / E3 等 API 包中所有含 base 臂的对比 | 若其 base 响应取自 `v4_gen_base_gold.jsonl` 或同 pipeline 的 base 生成 (同 render_prompt 缺陷), 其 base 臂同样是英文 think 草稿 | 换用 `results/results/v4_gen_base_goldfix.jsonl` 的干净响应重跑 base 臂; 若 API 包自己调 harness 生成 base, 须先打"保留空 think 块"补丁 (见 `e0_gen_basefix.py:render_prompt`) 再重生成 |
| v4_final_report.md 全部 "vs base" 数字 (judge +1.00、kw +0.19、McNemar 55/4、各子集差值、决策门触发条件) | base 锚点即污染源 | 已被本任务 E0 的 `e0_stats.json` 正式取代 (见 §4) |
| W3 stats_review.md 中 base 的两轨相关 (r=-0.076)、kw 均值 0.3661、judge 均值 1.68 | 同上 | 可用 base_goldfix 的 100 条重算, 预计两轨相关恢复正相关 |
| 论文中一切 "相对 base 提升 X%" 的表述 | 同上 | 全部改引 §4 干净锚点数字 |

**通用原则**: 凡输入包含 "base 模型对 100 金标题的回答" 的分析, 在核对所用缓存文件
mtime/tag 不是 `base_goldfix` 之前, 一律视为受污染。

## 6. 对论文结论的影响 (已由 §4 数据确认)

- 所有 "相对 base 的提升幅度" (judge +1.00、kw +0.19、McNemar 55/4) **已被证伪**:
  干净锚点下 v4−base judge **−0.30**、v2−base judge **−0.29** (均显著为负),
  kw 轨无差异。旧值只能作为"测量污染案例"写入局限/方法学讨论。
- 论文主结论必须改写为: 微调的价值在于**回答效率** (长度 ~1/10, 直接作答无草稿)
  与风格对齐, 而非绝对质量提升; v4 ≈ v2 (双轨均无显著差异), 决策门**保留 v2**。
- 决策门理由链更新: 不再是 "有提升但 CE 退化", 而是 "v4 judge 显著低于干净 base,
  替代条件不成立; CE-kw 退化 (v4 vs v2 −0.083) 依然存在, 与锚点无关"。
- 待办 (交 E1a/E1c/E3 与写作组): ① API 包 base 臂按 §5 重跑; ② 建议补一个
  "长度控制后" 的公平对比 (如限 base 回答长度或按信息密度归一) 再下最终质量结论;
  ③ base 两轨 Spearman 相关可用干净数据重算 (原 r=−0.076 为污染伪影)。

---
*产物: 远端 `results/e0_basefix/` + `results/v4_gen_base_goldfix.jsonl`;
本地 `paper/experiments/e0/`。续跑步骤见 [STATE.md](STATE.md)。*
