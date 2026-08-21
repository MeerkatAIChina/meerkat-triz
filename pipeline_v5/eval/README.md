# pipeline_v5/eval — v5 评测 harness(协议 §6.1–6.8, diff §11.3)

干净新建,不覆盖 pipeline_v4 任何产物。纯 stdlib 实现(统计/质量门/别名表/
决策门),GPU 与 API 仅在正式评测时启用;`--dry-run` 全链路不碰 GPU/API。

## 模块与协议条款对照

| 模块 | 协议条款 | 要点 |
|---|---|---|
| `render.py` | §6.1 干净锚点 | `render_prompt` **保留空 think 块**(E0 铁律,无 `.replace`);生成后才 `strip_closed_think` 剥闭合块;`assert_empty_think_retained` 冒烟断言 |
| `quality_gates.py` | §6.1 四道门 | ① think 残留 ② 非空 ③ 中文占比≥0.3 ④ 长度≥100 + ④b 英文草稿检测;微调追加 ≥50 字符且≥同题 base 3%;任一不过→`bad_words_ids` 兜底一次→仍不过冻结(exit 3);汇总上报告首页 |
| `keyword_scorer.py` + `keyword_map_v5.json` | §6.2 别名表 | 子串+别名表(status=confirmed 生效);更新前/后双分数;漏判审计队列 `results/v5_miss_audit_<tag>.jsonl`(rubric≥0.5 且 kw<0.5);漏判率<5% 建议冻结 |
| `judge_arms.py` | §6.4 judge 双臂 | 臂 A:反冗长条款+输入**不截断**;臂 B:<500/500–1500/1500–3000/>3000 同长度桶配对,不同桶报"无同桶配对"以臂 A 为准;T=0 硬断言;钉死 moonshot-v1-32k,偏离红标;评委家族谱系入 meta |
| `pairwise.py` | §6.3 双序强制 | AB/BA 合并胜率+Wilson CI+位置不一致率+B 位胜率;不一致率>10% 抛 AssertionError(单序作废) |
| `overrefusal.py` + `refusal_templates_v5.json` | sec1_data §5 | 金标回答拒答模板命中率>2% 不过门。⚠️ `trained_templates` 为空占位,待数据侧 300 条 safety_refusal 模板定稿后回填 |
| `decision_gate.py` | §6.7 决策门 2.0 | G0(overrefusal)+G1–G7 全机检;`python decision_gate.py <scores.json>` 输出逐门 PASS/FAIL/SKIP/FREEZE+总判定 |
| `stats_utils.py` | §6.8 统计纪律 | stdlib `Random(42)` 配对 bootstrap 10000 次;McNemar 精确;Wilson;**import 即指纹自检**(McNemar(55,4)=1.6979681549678105e-12 逐位一致 + bootstrap 指纹) |
| `eval_harness_v5.py` | 全部 | 主 harness:生成(缓存续跑)→质量门→臂 A judge(缓存续跑)→双轨+双分数→漏判审计→overrefusal→配对统计→臂 B→报告。锚点谱系 `base_goldfix` 一票否决 |

## 冒烟与测试

```bash
# 单元测试(9 组: E0 回归/质量门/别名表/judge 双臂/pairwise/overrefusal/决策门回溯/统计指纹)
venv_v5/bin/python pipeline_v5/eval/tests/test_v5.py

# dry-run 全链路(不碰 GPU/API, 12 题假数据)
venv_v5/bin/python pipeline_v5/eval/eval_harness_v5.py \
  --config pipeline_v5/eval/configs/eval_v5.json --tag dryrun --dry-run --limit 12 \
  --eval-file pipeline_v5/eval/tests/fixtures_gold_dryrun.jsonl

# 决策门回溯验证(§6.7 验收测试: v4 代入必须复现 keep_v2)
venv_v5/bin/python pipeline_v5/eval/decision_gate.py \
  pipeline_v5/eval/backtest/gate_scores_v4_backtest.json   # 期望 exit 1 + keep_v2

# judge 可用性探测(每条候选 1 条 ping, 正式评测前)
venv_v5/bin/python pipeline_v5/eval/eval_harness_v5.py \
  --config pipeline_v5/eval/configs/eval_v5.json --probe-judge
```

## 回溯验证结果(已复现)

`backtest/gate_scores_v4_backtest.json` 全部数字来自
`paper/experiments/e0/e0_stats.json`(干净锚点 base_goldfix):
G1 FAIL(−0.30 [−0.46,−0.14],CI 上限已低于 −0.15)、
G2 FAIL(−0.01 [−0.13,+0.11] 无增量)、
G3 FAIL(concept_explanation kw −0.083,E2 归因=真缺失)→
总判定 **keep_v2**,与 E0 实际结论一致(§6.7 声明的预期)。
报告: `backtest/BACKTEST_v4_report.md`。

## 正式评测前待办(非本工作包)

1. 金标 200 题(`v5_gold_101–200`)生成后,`configs/eval_v5.json:eval_file` 指向正式金标;
2. base 锚点按 §6.1 重新生成(tag 必须含 `base_goldfix`),`--anchor-check` 接旧缓存做一致性检查;
3. 数据侧 300 条 safety_refusal 定稿后回填 `refusal_templates_v5.json:trained_templates`;
4. 臂 A rubric 先在 v4 缓存试跑,差值移动 >±0.15 时人工校准(开放项 #12);
5. 异源评委终审(GPT-4o)接入 pairwise 终审流程(≤400 裁决/轮)。

## v5b 后新增常驻纪律 (2026-07-31, 决策门终审后固化)

v5b 被 G7 否决换来的四条硬规则, 后续每一代 (v5c+) 必须执行:

### D1 数据契约门 (训练前强制)
任何定向注入/新语料在启动训练前, 必须先过:
`python pipeline_v5/eval/data_contract_check.py --inject <新数据.jsonl> --gold <金标.jsonl> --subset <子集>`
判定 WARN 即阻止全量训练, 回炉修数据。回溯证据: 该检对 v5b 的 408 条
注入数据判 WARN (框架词 TRIZ 25%/发明原理 6% 覆盖), 本可在训练前预言
keyword −0.104 反噬, 避免 29h GPU 空耗。

### D2 smoke 先行 (全量训练前置)
定向数据变更一律先跑 0.5 epoch 冒烟 + 快速双轨评测 (smoke40 协议),
方向正确才放行全量训练。v5b 直接全量是纪律退步, 不得重演。

### D3 发货 dtype 兜底 (已入 train.py)
续跑路径可致 LoRA 参数 F32 化 (v5b 实测: best/ 与 checkpoint 全部 F32;
v5a 单跑从未触发)。train.py 现有双重防线: 训练开始回调再校正 +
ship_adapter 后发货前 safetensors 统一转 BF16 (幂等)。

### D4 PR 度量备注
keyword 逐词命中对 principle_recommendation 为部分失效度量
(编码单一参考答案措辞)。原理集合 F1 (pipeline_v5/eval/principle_f1.py)
为 benchmark v2 候选; v5b 三方对比显示该度量下 base/v5a/v5b 无显著差,
与评委判定一致, 坐实 keyword −0.104 以伪影为主
(归因: paper/external_review_v5b/attribution_pr_keyword_v5b.md)。

### D5 等长对照门 (judge 轨显著结论的强制伴随实验, 2026-08-01 增)
任何"judge 轨显著为正"的结论, 在采信/发布前必须伴随一次等长对照:
对照臂按候选臂逐题长度 ±10% 重答同一套题, 其余协议逐字不变。
判定: 等长后差值 CI 仍显著为正 → 质量增益成立; 等长后 n.s. →
收益重述为"等质 N× 简洁", 不得主张"质量超对照"。
回溯证据 (eqlen_report.md): v5a−base 无约束 +0.393 显著, 等长后
−0.070 n.s., base 仅靠压长指令自身 +0.463——v5a 头条基本全是长度伪影,
本规若在 v5 线立项时存在, 叙事从第一天起就不会跑偏。
长度遵从允许部分控制 (v5a 对照臂超写 1.29×), 如实披露即可, 不迭代追完美。

### D6 评委仪器钉版与迁移桥接 (2026-07-31 增)
judge 是测量仪器, 不是工具。三条铁律:
1. **钉快照**: harness 配置中评委必须写具体版本号 (如 moonshot-v1-32k,
   k3-YYYYMMDD), 禁止 "latest"/浮动别名——浮动别名会在评测中途静默漂移。
2. **换评委必桥接, 不切换**: 任何评委升级 (如 v1-32k → k3, tensoris 面板
   换代) 按一次"测量代"处理——现有生成臂 (base/候选/等长臂) 新旧评委
   平行打分, 出一致性报告 (逐题 Spearman + 均值偏移), 并重跑 T=0 翻转率
   (≤0.02 才可钉版) 与等长对照 (§5.3 结论的跨仪器稳健性检验)。
   新旧评委并行报数一代后, 旧版归档。
3. **同族定义不变**: 家族内升级 (v1→k3) 不改变同族属性——回避制、G8、
   "同族读数不可跨家族外推"照旧适用。新评委不等于更客观的评委。
4. **非确定评委的噪声账 (2026-07-31 探针增补)**: 翻转率 >0.02 的评委
   (实测: tensoris 网关全部外部评委 0.18–0.80; kimi-k3 禁 T=0),
   其显著性结论必须附复跑噪声传播后的合成 CI
   (sd_rerun = √(2·σ²_item/n), 与 bootstrap sd 按 RSS 合成),
   或对关键结论 N=3 重复取均值。已发表外部数字经此核算仍显著
   (gpt +0.104 → [+0.006, +0.202]; claude +0.094 → [+0.015, +0.173]),
   依据: paper/bridge/external_judge_determinism_report.md。
迁移全流程见 paper/v6_redesign_plan.md §2b。
