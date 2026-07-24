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
