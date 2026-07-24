# STATE_WorkerA — Day 1 数据门主链(CPU)

> Worker A ｜ 状态:**已完成** ｜ 远端脚本:`pipeline_v5/src/data_build_v5.py`(确定性 seed=42,可 `--resume` 续跑,重跑计数一致)

## 产物(远端 `/home/meerkat/mongoose_ai/data/processed/v5_data/` ↔ 本地 `paper/v5_execution/day1/`)

| 文件 | 条数 | 说明 |
|---|---|---|
| gated_corpus.jsonl | 8,613 | 过门语料(v2 10,327 + ariz boost 594 → 门1-5) |
| cleaned_seeds.jsonl | **59** | 种子三规则清洗后(见⚠️) |
| styleC_longanswer_sampling.jsonl | 3,445 | 按子集分层随机 40% 长答生成清单,含 group_id |
| v5_gates_report.json | — | 每门计数/术语覆盖/IP 声明/sha256 |
| _checkpoints/ | — | 分 stage 报告 + .done 断点 |

## 关键计数

- **摸底**:全部输入路径确认。v2 语料 `checkpoint_corpus_sft_v2/corpus_sft_checkpoint.json`(10,327,字段 subset/instruction/input/output);ariz boost `corpus_sft_v2_ariz_boost/ariz_guidance.json`(674);种子 = `synthetic/*_synthetic.json` 中 `source=='seed'` 合计 385 ✓;v4_gold 100 / sample_data_expanded 465 / general_probe 30 均在位。v4 `data_build.py` 门函数直接 import 复用。
- **ariz 594**:由 `sample_data_expanded.ariz_guidance(102) − sample_data.ariz_guidance(22) = 80` 差集确定性还原评测预留,674−80=594 入训 ✓(与 `build_v3_and_evalset.py` seed=42 口径一致,80 条全部在 boost 中匹配)。
- **语料门**:10,921 → think 剥离 0 → 长度丢 329 → 精确去重丢 4 + 冲突组 30 组/68 条(与 v4 完全一致)→ 近重复丢 69(v4 为 72,输入差 80 条所致)→ 门5 cap 2500:concept 3,527→2,500、innovation 3,311→2,500(覆盖 1500 + 三桶随机 1000)→ ariz 占比 878/8,613=10.2% <15% 未触发下采 → **8,613**。
- **术语保底未达(如实)**:7 词未达每词≥30:Functionality(3/0)、Cost(9/11)、物场分析(10/9)、资源分析(26/27)、改善参数(15/5)、恶化参数(20/8)、通用工程参数(11/1)(concept/innovation 两侧计数);原因 = 语料池内含这些字面串的样本总数不足 30(中文语料多用"功能性/成本"等表述),非脚本缺陷;明细在 report `gate5_rebalance_v5.*.floor_unmet`。
- **长度分布(字符)**:mean 243.8 / p50 235 / p95 349 / p99 439 / max 660 → 短答侧无 2048 token 风险;长答 1200–2500 字符的 token 检查属后续渲染阶段(裁决项 #15 输入)。

## ⚠️ 需 Orchestrator/Owner 知晓的偏差

1. **种子清洗存活仅 59/385**。R1 冲突组 10 组/20 条 ✓ 对上 retrospective;但种子 output 整体极短(原始 p50=103 字符、max=181、<150 占 292/385),R2 截尾 196 条截后全部 <150 按规则整条弃,R3 再弃 110 → 存活 59。**×3 上采样仅 177 等效样本(~2% 而非方案预期 ~15% 锚定占比)**。规则严格执行无误(残留率=0 ✓,抽查截断只切模板尾句),方案"385 条种子"的隐含预期(清洗后仍数百条)与现实数据不符,需 Owner 裁决(放宽 150 下限?降上采样目标?如实接受?)。
2. **<60 字符计数 23 vs retrospective 26**:差 3 条,retrospective 未注明统计口径(是否去空白/统计时点),本结果为原始 output 字符数、NFKC 前。
3. **IP 边界(裁决项 #2)**:样本级**无 source/chunk 标识,无法区分**客户案例派生样本(构建脚本未保留 chunk 映射);chunk 级 `triz_corpus.jsonl` 3,914 chunks 有 source_path,疑似客户目录(4-大华/阳光电源/创新方法大赛/四级案例等)。已按方案"如实继承"条款声明,未编造区分结果;若须严格执行剔除,需重建带 chunk 映射的语料(超出本工作包)。

## 不做(按任务书)

去污(双重)、划分、manifest、长答/Refusal API 生成 —— 待后续阶段;本链产物即其输入。

## 复跑方式

```bash
ssh chinux@spark-855a
cd /home/meerkat/mongoose_ai
venv_v5/bin/python pipeline_v5/src/data_build_v5.py --resume   # 或 --force 全量重跑(确定性)
```
