# STATE_workerB — Day 1 评测 harness 改造(Worker B)

状态: **完成** (2026-07-24,单窗口内完成,无需 resume)

## 交付清单(8 项任务)

| # | 任务 | 状态 | 证据 |
|---|---|---|---|
| 1 | 干净锚点硬化 | ✅ | `render.py` 保留空 think 块(无 v4 的 `.replace`);`tests/test_v5.py::test_e0_regression_think_retained` 3 题冒烟断言 + 剥离路径拦截;微调模型生成后同过 `has_think_residue` |
| 2 | 生成质量门四道 | ✅ | `quality_gates.py`:think 残留/非空/中文≥0.3/长度≥100 + 英文草稿检测;微调追加 ≥50 字符且≥同题 base 3%;兜底 bad_words_ids 一次→仍不过 exit 3 冻结;`gate_summary_markdown` 上报告首页 |
| 3 | 关键词轨升级 | ✅ | `keyword_scorer.py` 子串+别名表;`keyword_map_v5.json` 外置,E3 19 条漏判清单初始化(9 关键词 confirmed,每条附证据);更新前/后双分数;漏判审计队列 `v5_miss_audit_<tag>.jsonl` |
| 4 | judge 双臂 | ✅ | `judge_arms.py`:臂 A 反冗长条款+输入不截断;臂 B 四桶(<500/500-1500/1500-3000/>3000)同桶配对,无同桶报"以臂 A 为准";T=0 硬断言;钉死 moonshot-v1-32k 偏离红标;谱系入 meta |
| 5 | pairwise 双序 | ✅ | `pairwise.py`:合并胜率+Wilson+位置不一致率+B 位胜率;>10% `AssertionError` 单序作废 |
| 6 | overrefusal | ✅ | `overrefusal.py`:>2% 不过门;单测覆盖 2%/3% 边界 |
| 7 | 决策门 2.0 | ✅ | `decision_gate.py` G0+G1–G7 全机检;回溯 v4 复现 **keep_v2**(G1/G2/G3 FAIL,数字全部取自 `paper/experiments/e0/e0_stats.json`,见 `backtest/BACKTEST_v4_report.md`) |
| 8 | 落远端+README+冒烟 | ✅ | 远端 `pipeline_v5/eval/`(未碰 pipeline_v4);README 模块-条款对照表;dry-run 12 题双臂冒烟通过 |

## 关键计数

- 单元测试: 9/9 组通过(远端 venv_v5 复跑同结果)
- dry-run: 质量门 12/12 过门, invalid 0;kw 别名表前 0.9000→后 0.9833;臂 B 同桶 12/12;overrefusal 0/12 过门
- 回溯验证: G1 −0.30[−0.46,−0.14] FAIL;G2 −0.01[−0.13,+0.11] FAIL;G3 CE-kw −0.083 真缺失 FAIL → keep_v2(与 §6.7 预期一致)
- 统计指纹: McNemar(55,4)=1.6979681549678105e-12 逐位一致;bootstrap stdlib 指纹自检 import 即过

## 产物路径

- 远端: `/home/meerkat/mongoose_ai/pipeline_v5/eval/`(13 文件);冒烟产物 `results/v5/eval_v5_dryrun*20260724_094006*`
- 本地: `paper/v5_execution/day1/harness/pipeline_v5/eval/`(镜像)+ `smoke_outputs/`(远端冒烟回传)

## 阻塞/交接项(非本包可解)

1. `refusal_templates_v5.json:trained_templates` 为空占位 —— 等 Worker A 300 条 safety_refusal 模板定稿后回填(README 已红字标注)
2. 臂 A rubric 需在 v4 缓存试跑校准(开放项 #12,差值移动>±0.15 人工校准)
3. 金标 200 题/探针 120 题未就绪,G6 回溯场景 SKIP(设计内)
4. 远端 `/home/meerkat/mongoose_ai` 实为 `/home/chinux/jupyterlab/meerkatai` 的符号链接,产物实际落后者(同一项目根,与 v4 一致)
