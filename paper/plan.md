# 论文化评估计划 — Meerkat-AI TRIZ 领域微调训练

> 目标:以科研视角评估 v1→v4 训练迭代,凝练论文级科学意义,设计并执行补充实验,产出论文骨架与评估报告。

## 已有事实(Orchestrator 已核实)
- 项目:Qwen3.6-35B-A3B(Gated DeltaNet + Gated Attention + MoE 混合架构)在 DGX Spark GB10(121GB 统一内存)上的 TRIZ 领域 LoRA 微调。
- 四个版本:v1(2.6K 语料,无质量门,整体回退)→ v2(8.5K 多角度,overall +0.060 但 ARIZ 回退)→ v3(+ARIZ 定向数据)→ v4(干净管线:质量门/去污/completion-only loss/BF16)。
- 关键结果(远端已跑完):
  - eval2 四方报告(results/eval2/report_20260723_024941):v3 vs v2 关键词轨显著更好、judge 轨显著更差(指标轨分歧);v1/v2 的 principle_accuracy、ariz_step_coverage 显著低于 base;general_probe 无显著遗忘(v2 除外)。
  - v4 金标终报(results/v4_final_report.md):v4 judge overall 2.57 vs base 1.57(差 +1.00,95%CI [0.80,1.19],McNemar p=1.7e-12);关键词 overall +0.19;但 keyword/concept_explanation 显著退化 → 决策门判"保留 v2"。
- 远端:ssh chinux@spark-855a,项目 /home/meerkat/mongoose_ai,GPU 当前空闲。
- 本地已有:eval_methodology_research.md(评测方法论文献调研)、docs/training_retrospective_2026-07-20.md(七维复盘)、results/METRICS_LEDGER.md。

## Stage 1 — 并行分析与设计(4 workers,互不依赖)
- W1 证据矿工(explore):汇总本地+远端全部实验数据 → paper/evidence_table.md(版本×数据×超参×全部指标×产物哈希的单一证据表)。
- W2 文献定位员(explore):相关工作与新颖性定位 → paper/related_work.md(五个候选贡献点逐一做新颖性核查,给可引用文献)。
- W3 统计审查员(coder):拉取远端原始 scores json 做独立统计复核 → paper/stats_review.md(效应量、检验功效、轨迹分歧量化、judge 可信度)。
- W4 实验设计师(plan):补充实验矩阵 → paper/experiment_plan.md(每个实验:假设/设计/成本/可发表性,分"必做/建议/可选")。

## Stage 2 — 补充实验执行(Stage-Gate:审阅 W4 方案后启动,coder,SSH 到 DGX)
候选(按 W4 定稿):E1 judge 可靠性包(位置交换/多评委/重复翻转率,纯 API);E2 concept_explanation 退化归因(数据分析,CPU);E3 ARIZ 指标伪影分析(judge 重判);E4 数据消融训练(若证据需要且成本可控,排队执行)。
产物全部回传本地 paper/experiments/。

## Stage 3 — 论文骨架与评估报告(coder)
paper/paper_skeleton.md(中英文摘要、贡献点、章节结构、图表清单)+ paper/assessment_report.md(给用户的完整评估)。

## Stage 4 — 交付(docx 技能)
评估报告转 .docx 交付。
