# Meerkat-AI 实验证据总表（W1 证据矿工）

> 编制日期：2026-07-24。本表汇总 v1–v4 四个 LoRA 微调版本的全部实验证据。
> 本地路径前缀 `LOCAL:` = `/Volumes/2nd-HD/claude/Meerkat-AI`；远端路径前缀 `REMOTE:` = `chinux@spark-855a:/home/meerkat/mongoose_ai`。
> 所有数字均直接抄录来源文件，精度与来源一致；拿不到的明确标注"未获取"。

---

## ① 版本总表（v1–v4 × 数据 / 质量门 / 超参 / 训练时长 / 硬件）

| 维度 | v1 (corpus_sft_v1) | v2 (corpus_sft_v2) | v3 (v2+ARIZ boost) | v4（干净重建管线） |
|---|---|---|---|---|
| 训练日期 | 2026-06-19 | 2026-07-21 | 2026-07-22 | 2026-07-23 ~ 07-24 |
| 训练数据 | 2,662 train / 313 val / 157 test（`REMOTE:data/processed/train.jsonl` wc -l 实测 2,662；划分见 `LOCAL:results/METRICS_LEDGER.md:8`） | 8,458 train / 995 val / 498 test（`REMOTE:data/processed/v2_{train,validation,test}.jsonl` wc -l 实测；`LOCAL:results/adapter_info_v2.json:10-13`） | 8,963 train / 1,054 val / 528 test（`REMOTE:data/processed/v3_*.jsonl` wc -l 实测；= v2 9,951 + boost 674 − 评测预留 80 = 10,545，见 `LOCAL:build_v3_and_evalset.py:1-13`） | 5,739 train / 637 val / 314 test（`REMOTE:results/v4_data_report.json` → `final.counts`） |
| 数据质量门 | 零质量门（`METRICS_LEDGER.md:8`） | 基础质量门（min_output_chars=150、门内去重，见 `build_v3_and_evalset.py` 引用 `apply_v2_quality_gates`） | 同 v2 门 + 三重去重（vs v2 语料 / vs 既有评测集 / 门内去重）+ 80 条评测预留防污（`LOCAL:build_v3_and_evalset.py:43-69`） | 完整质量门链：think 剥离(0) → 长度门(丢329) → 精确去重(丢4)+冲突组丢弃(30组/68条) → 3-gram 近重复去重(J≥0.7，丢72) → 金标去污(J≥0.5，丢0) → 子集再平衡(cap 1500，丢3,838) → 长度门 2048 token(丢0)（`REMOTE:results/v4_data_report.json` → `gates`） |
| Epochs（配置/实际） | 2 / 2（666 步）（`REMOTE:models/meerkat_triz_adapter_v1/adapter_info.json`） | 2 / 2（2,116 步）（`LOCAL:results/adapter_info_v2.json:5-6`） | 4 / 1.249（1,400 步，`REMOTE:results/train_log_v3.json` 末条 epoch=1.24902376436461） | 4 / 1.393（1,000 步，早停 patience=3；`REMOTE:models/meerkat_triz_adapter_v4/adapter_info.json` → `training`） |
| LoRA 配置 | r=64, α=128, dropout=0.0, bias=none, 12 个 target_modules（q/k/v/o_proj + in_proj_qkv/z/b/a + out_proj + gate/up/down_proj）（`LOCAL:config.py:48-66`；v4 adapter_info 同） | 同左 | 同左 | 同左（`REMOTE:models/meerkat_triz_adapter_v4/adapter_info.json` → `training.lora`） |
| 学习率 / 调度 | 2e-4（adapter_info_v1）；cosine + warmup 0.05（`LOCAL:config.py:75-77`） | 2e-4 cosine：step10 lr=1.698e-5（warmup）→ 峰值 2e-4 → step2110 lr=5.985e-9（`LOCAL:results/train_log_v2.json` 首/末条） | 2e-4 cosine：step10 lr=8.0e-6（`REMOTE:results/train_log_v3.json` 首条） | 2e-4 cosine（按 max_steps=2872 设计）：step10 lr=1.25e-5 → step1000 lr=1.553e-4（早停时仍高）（`REMOTE:results/train_log_v4.json` 首/末 train 条） |
| 批次 | 1 × grad_accum 8（有效 8）（`LOCAL:config.py:72-74`） | 同左 | 同左 | 同左（`LOCAL:pipeline_v4/configs/train_v4.json`） |
| Loss 口径 | 全文（system+user+assistant）（`METRICS_LEDGER.md:8` 备注） | 全文（同口径备注） | 全文（同一训练脚本族） | **completion-only**（trl 1.5.1 SFTTrainer prompt/completion 自动启用，train.py 断言；`LOCAL:pipeline_v4/configs/train_v4.json` → `_notes`） |
| 精度 | FP16 加载（评测 json `base_fp16`；v1 训练 crash 史见 `METRICS_LEDGER.md:9` peft 0.19 fp32 转换 OOM） | 同左 | 同左 | **纯 BF16 不量化**（`train_v4.json` → `_notes.precision`；adapter_info `precision: bf16 (no quantization)`） |
| 训练时长 | 1 小时 52 分（10.0 s/step × 666 步）（`LOCAL:docs/training_retrospective_2026-07-20.md:65`；checkpoint 时间戳 14:10→15:29） | 6h46m（`METRICS_LEDGER.md:9`；checkpoint 时间戳 00:54→07:01） | 19,327.9961 s ≈ 5h22m（train_runtime，`REMOTE:results/train_log_v3.json` 末条） | 11,034.9843 s ≈ 3h04m（train_runtime，`REMOTE:results/train_log_v4.json` 末条） |
| Best eval_loss | 1.3979（无基座对照）（`METRICS_LEDGER.md:8`；adapter_info_v1 1.3978878259658813） | 1.0673376321792603 @ step 2000（`LOCAL:results/adapter_info_v2.json:9`；轨迹 1.1576(step200)→1.0673(step2000)→1.0675(step2116)） | 1.0705283880233765 @ step 1100（`REMOTE:results/train_log_v3.json` eval 序列；轨迹 1.1682(step100)→1.0705(step1100)→回升至 1.0848(step1400)） | 1.5592304468154907 @ eval_step 700（completion-only 口径，不可与 v1–v3 比；`REMOTE:models/meerkat_triz_adapter_v4/adapter_info.json` → `best_checkpoint`） |
| 适配器产物 | 161.57 MB（169,417,040 B），sha256 `1f909cb0…`（adapter_info_v1） | 161.57 MB，sha256 `a7bb48a4f90eae03ff12f79e9e14476d4e5969dcedd71364b666300eb31bfd86`（`LOCAL:results/adapter_info_v2.json:93`） | 161.57 MB，sha256 `0335987e…`（修复后）；原全零 lora_B 版 `f00ef15b…`（adapter_info_v3） | 169.4 MB，620 tensors（A=310/B=310 全 BF16 非零），sha256 adapter_model `86402293…`（adapter_info_v4 → `validation`） |
| 硬件 | NVIDIA DGX Spark（GB10，121GB 统一内存；`LOCAL:config.py:311-317` 记 128GB/20 核 Grace） | 同左 | 同左 | 同左 |

**基座模型**：Qwen3.6-35B-A3B（35B 总参 / 3B 活跃 / 256 专家 / 262K 上下文；Gated DeltaNet 30 层 + Gated Attention 10 层 + MoE 40 层）（`LOCAL:config.py:24-25, 52-55`）。

---

## ② 全部评测指标长表（指标 × base/v1/v2/v3/v4 × 评测集 × 日期 × 来源）

### ②-A 修复版指标四连评测（Layer 2 TRIZ + Layer 3 性能，评测集 `data/sample_data.json`，2026-07-21）

来源：`LOCAL:results/adapter_vs_base_v1_20260721_150041.json`、`adapter_vs_base_v2_20260721_140022.json`、`triz_eval_results_20260721_{134010,135837,144139,145947}.json`、`METRICS_LEDGER.md:11-23`。

| 指标 | base | v1 | v2 | v3 | v4 |
|---|---|---|---|---|---|
| overall_score | 0.5290277778 | 0.4718055556（Δ −0.0572） | 0.5893750000（Δ +0.0603） | 未获取 | 未获取 |
| principle_accuracy | 0.9（9/10） | 0.6（Δ −0.30） | 0.9（Δ 0，天花板） | 未获取 | 未获取 |
| contradiction_resolution | 0.3625 | 0.4625（Δ +0.100） | 0.4145833333（Δ +0.0521） | 未获取 | 未获取 |
| case_coverage | 0.3625 | 0.4875（Δ +0.125） | 0.725（Δ +0.3625） | 未获取 | 未获取 |
| ariz_completeness | 0.3888888889 | 0.2777777778（Δ −0.1111） | 0.2500000000（Δ −0.1389） | 未获取 | 未获取 |
| BLEU（sacrebleu zh, n=10） | 0.7557359877 | 4.2703460132 | 2.2972699473 | 未获取 | 未获取 |
| ROUGE-1 / 2 / L | 0.02775/0.00143/0.02444 | 0.30179/0.07752/0.22213 | 0.15798/0.07083/0.15798 | 未获取 | 未获取 |
| 吞吐 tok/s | 17.70–17.90 | 15.22 | 15.10 | 未获取 | 未获取 |
| p50 延迟 ms | 28,617–28,904 | 11,259 | 33,905 | 未获取 | 未获取 |
| 峰值显存 GB | 64.68 | 65.06 | 65.06 | 未获取 | 未获取 |

注：两次独立基座评测分数完全一致（可复现性确认，`METRICS_LEDGER.md:12-13`）。v1/v2 的 ARIZ 回退为共性短板（−0.111 / −0.139），是 v3 定向补数据的直接动机。

### ②-B held-out PPL 对比（2026-07-21）

来源：`LOCAL:results/ppl_adapter_vs_base_v1_20260721_150954.json`（test.jsonl 157 条）、`ppl_adapter_vs_base_v2_20260721_141350.json`（v2_test.jsonl 498 条）。

| 指标 | base@v1集 | v1 | base@v2集 | v2 | v3 | v4 |
|---|---|---|---|---|---|---|
| avg_loss | 3.2787 | 1.4107（Δ −1.8680） | 3.5276 | 1.0663（Δ −2.4613） | 未获取 | 未获取 |
| perplexity | 25.8425 | 4.2225（Δ −21.6200） | 34.0862 | 2.9005（Δ −31.1856） | 未获取 | 未获取 |
| 样本/tokens | 157 / 31,729 | 同 | 498 / 103,376 | 同 | — | — |

⚠️ 两版 PPL 评测集不同（157 条 vs 498 条），绝对值不可跨版本互比。

### ②-C eval2 四方对比（评测集 `sample_data_expanded.json`（6 子集 465 条）+ `general_probe.json`（30 题），judge=moonshot-v1-8k，2026-07-22 ~ 07-23）

来源：`REMOTE:results/eval2/report_20260723_024941.json`（统计：配对 bootstrap 10,000 次 seed=42、McNemar 精确检验、Wilson CI）；流水线说明 `LOCAL:eval_pipeline_v2/README.md`。

| 指标 | base | v1 | v2 | v3 |
|---|---|---|---|---|
| principle_accuracy（0/1 全覆盖，n=100） | 0.28 | 0.24 | 0.22 | 0.29 |
| principle_coverage | 0.5467 | 0.5458 | 0.4933 | 0.5808 |
| contradiction_coverage（n=40） | 0.1350 | 0.2733 | 0.2683 | 0.3317 |
| case_coverage（n=56） | 0.3607 | 0.4107 | 0.3643 | 0.4000 |
| ariz_step_coverage（n=102） | 0.2843 | 0.1650 | 0.1699 | 0.2124 |
| concept_coverage（n=81，参考轨） | 0.9877 | 0.9877 | 0.9877 | 0.9877 |
| general_probe_coverage（n=30） | 0.9667 | 0.8333 | 1.0000 | 0.9667 |
| judge_contradiction_coverage | 0.1367 | 0.2067 | 0.3700 | 0.1000 |
| judge_ariz_step_coverage | 0.2631 | 0.2990 | 0.3676 | 0.2908 |
| **overall_kw** | 0.2535 | 0.2691 | 0.2533 | 0.3090 |
| **overall_judge** | 0.2498 | 0.2759 | 0.3234 | 0.2552 |
| BLEU（参考轨，n_pairs=56） | 1.7487 | 2.5050 | 2.4956 | 1.9320 |
| ROUGE-L（参考轨） | 0.0207 | 0.0374 | 0.0340 | 0.0172 |

**配对显著性（95% CI，✅=显著）**：

- v2 vs base：principle_accuracy −0.06 ✅、ariz_step_coverage −0.114 ✅（显著退化）；judge_contradiction +0.233 ✅、judge_ariz +0.105 ✅、**overall_judge +0.0736 [0.0310, 0.1162] ✅**（显著更优）；overall_kw −0.0002（不显著）。McNemar p=0.03125。
- v3 vs v2：principle_accuracy +0.07 ✅、principle_coverage +0.0875 ✅、ariz_step +0.0425 ✅、**overall_kw +0.0556 [0.0010, 0.1099] ✅**；judge_contradiction −0.27 ✅、judge_ariz −0.0768 ✅、**overall_judge −0.0682 [−0.1078, −0.0296] ✅**——**两轨结论相反**。McNemar p=0.0390625。
- v1 vs base：overall_kw +0.0156（不显著）、overall_judge +0.0262（不显著）；ariz_step −0.119 ✅、general_probe −0.133 ✅。
- v3 vs base：overall_kw +0.0555 [0.0156, 0.0980] ✅；overall_judge +0.0054（不显著）。

### ②-D v3 修复版四连评测（评测集 sample_data_expanded.json，2026-07-22 / 07-23）

来源：`REMOTE:results/adapter_vs_base_v3exp_20260723_052037.json`（另 `adapter_vs_base_v1exp_20260722_090542.json`、`adapter_vs_base_v2exp_20260722_095247.json` 数值与 07-21 sample_data 版完全一致）。

| 指标 | base | v3（修复后） | Δ |
|---|---|---|---|
| overall_score | 0.5290277778 | 0.5220833333 | −0.0069 |
| principle_accuracy | 0.9 | 0.8 | −0.1 |
| contradiction_resolution | 0.3625 | 0.4458333333 | +0.0833 |
| case_coverage | 0.3625 | 0.325 | −0.0375 |
| ariz_completeness | 0.3888888889 | 0.4166666667 | +0.0278 |
| 吞吐 tok/s | 17.74 | 15.36 | −2.38 |
| p50 延迟 ms | 28,846.78 | 8,632.59 | — |
| 峰值显存 GB | 64.68 | 65.06 | — |

⚠️ 同日早些时候的 `adapter_vs_base_v3exp_20260722_082000.json` 全部 Δ=0——该次评测命中**全零 lora_B** 适配器（详见缺陷清单 #5），数据无效但保留为事故证据。

### ②-E v4 金标终报（100 题金标集 `v4_gold.jsonl`，judge=moonshot-v1-32k，2026-07-23 ~ 07-24）

来源：`REMOTE:results/v4_final_report.md`（生成时间 2026-07-24T01:16:48）；各模型原始评测 `REMOTE:results/eval_v4_{base,v2,v3,v4}_gold_*.json`。

| 模型 | 关键词轨均值 | judge 轨均值 | 关键词 pass 率 | judge pass 率 |
|---|---|---|---|---|
| base | 0.3661 | 1.5700 | 0.350 | 0.120 |
| v2 | 0.5483 | 2.5800 | 0.630 | 0.620 |
| v3 | 0.5716 | 2.2800 | 0.640 | 0.430 |
| v4 | 0.5568 | 2.5700 | 0.650 | 0.630 |

子集（关键词轨 / judge 轨）：

| 子集 | base | v2 | v3 | v4 |
|---|---|---|---|---|
| ariz_guidance | 0.4437 / 1.70 | 0.6439 / 2.65 | 0.6938 / 2.45 | **0.7494 / 2.85**（双轨四方最高） |
| case_generation | 0.2190 / 1.47 | 0.5492 / 2.47 | 0.5683 / 2.33 | 0.5190 / 2.53 |
| concept_explanation | 0.5317 / 1.73 | 0.5187 / 2.80 | 0.5257 / 2.53 | **0.4356 / 2.80**（关键词显著退化） |
| contradiction_analysis | 0.3645 / 1.00 | 0.4987 / 2.55 | 0.5401 / 2.10 | 0.5233 / 2.35 |
| innovation_assessment | 0.4748 / 2.10 | 0.6914 / 2.80 | 0.6748 / 2.50 | 0.7081 / 2.90 |
| principle_recommendation | 0.2217 / 1.70 | 0.4524 / 2.35 | 0.4661 / 1.95 | 0.4412 / 2.20 |

决策门：v4−base judge overall **+1.0000 [+0.8000, +1.1900]**，McNemar（judge pass）p=1.6979681549678105e-12；v4−base 关键词 overall **+0.1907 [+0.1297, +0.2520]**；judge overall 显著优于 base = 是；显著退化子指标 = keyword/concept_explanation → **判定：保留 v2**。

### ②-F 已作废（修复前 buggy 指标）评测——仅存档，不得引用

| 文件 | 问题 |
|---|---|
| `LOCAL:results/adapter_vs_base_20260719_220108.json`（v1，07-19） | 分母 bug：principle 分母 40 题分子 10 题（上限 25%）；BLEU/ROUGE 恒为空 `{}`；peak_memory 恒 null |
| `LOCAL:results/eval_v2_20260721_081323.json`（v2，07-21 早） | 同上 |

---

## ③ 训练效率表

| 版本 | 步数 | 时长 | 均速 | HF 报告吞吐 | 峰值显存（推理评测） | 来源 |
|---|---|---|---|---|---|---|
| v1 | 666（2 epochs） | 1h52m | 10.0 s/step | 未获取（逐步 loss 日志已丢失） | 65.06 GB | `docs/training_retrospective_2026-07-20.md:65`；`adapter_vs_base_v1_20260721_150041.json` |
| v2 | 2,116（2 epochs） | 6h46m | ≈11.5 s/step | 未获取（train_log_v2.json 无 train_runtime 汇总条目） | 65.06 GB | `METRICS_LEDGER.md:9` |
| v3 | 1,400（1.249/4 epochs） | 19,328.0 s（≈5h22m） | 13.8 s/step | 1.855 samples/s；0.232 steps/s；total_flos 4.9791e+17 | 65.06 GB（07-23 复测）/ 65.22 GB（07-22 零权重版） | `REMOTE:results/train_log_v3.json` 末条 |
| v4 | 1,000（1.393/4 epochs，早停） | 11,035.0 s（≈3h04m） | 11.0 s/step | 2.08 samples/s；0.26 steps/s；total_flos 3.6718e+17 | 未获取 | `REMOTE:results/train_log_v4.json` 末条 |

训练 token 吞吐佐证：v2 全程 num_tokens 累计 3,555,340（step 2110，`train_log_v2.json`）；v4 累计 1,787,443（step 1000，`train_log_v4.json`）。v4 mean_token_accuracy 0.3954→0.6889；v2 0.4813→0.7919。

---

## ④ 数据规模与配比表

| 层级 | 规模 | 来源 |
|---|---|---|
| 种子数据 | **385 条**（原记 548 条，2026-06-18 提交 2f72fa6 移除 163 条完全重复后为 385；6 子集每集约 100 条规划值） | `LOCAL:README.md:233,256-257`；`docs/training_retrospective_2026-07-20.md:49`；`config.py:99-130` |
| 02b 合成路径总量 | 6,286 条（385 真实 + ~5,901 合成），真实占比 **6.1%**（低于 20–30% 理论目标，成本优先的有意决策） | `LOCAL:README.md:256-260` |
| 合成扩展倍数 | concept/ariz ×6，principle/innovation ×11，case/contradiction ×16（改写/混合/全新策略） | `LOCAL:config.py:162-178` |
| corpus 路径（v1） | 3,132 条（2,662/313/157）；TRIZ-raw 教材片段 grounding + Moonshot 生成 Q&A | `METRICS_LEDGER.md:8` |
| v2 语料 checkpoint | 10,327 条 → 划分 9,951（8,458/995/498） | `REMOTE:results/v4_data_report.json` → `inputs`；jsonl wc -l |
| ARIZ boost | 674 条（674 − 80 评测预留 = 594 入训练） | `REMOTE:results/v4_data_report.json` → `inputs`；`LOCAL:build_v3_and_evalset.py:5-8,30` |
| v3 总训练 | 10,545 条（8,963/1,054/528） | jsonl wc -l 实测 |
| v4 构建漏斗 | 11,001 输入 → think 剥离 11,001 → 长度门 10,672 → 精确去重+冲突 10,600 → 近重复 10,528 → 金标去污 10,528（0 命中）→ 再平衡 6,690 → 划分 5,687/670/333 → 交叉检查移回 52 条 → **最终 5,739/637/314** | `REMOTE:results/v4_data_report.json` → `gates`/`split`/`final` |
| v4 训练集子集分布 | ariz 831 / case 1,989 / concept 1,283 / contradiction 218 / innovation 1,279 / principle 139 | `REMOTE:results/v4_data_report.json` → `final.subset_distribution.train` |
| 评测集 | sample_data.json（修复版四连）；sample_data_expanded.json（6 子集 465 条：concept 127 / contradiction 40 / principle 100 / case 56 / ariz 102 / innovation 40）+ general_probe.json 30 题（eval2）；v4_gold.jsonl **100 题**（金标终报） | jsonl/脚本实测 |
| 真实占比口径 | corpus 路径下 Q&A **100% 由 Moonshot 生成**，"真实"仅体现于教材片段 grounding（`docs/training_retrospective_2026-07-20.md:61`）；6.1% 仅适用 02b 路径 | 见各来源 |

---

## ⑤ 已知缺陷与口径注意事项清单

1. **Loss 口径不可跨代比较**：v1/v2/v3 为全文 loss（system+user+assistant，`METRICS_LEDGER.md:8`），v4 为 completion-only loss（trl 1.5.1 自动，`train_v4.json` → `_notes`）。v4 best eval_loss 1.5592 vs v2 1.0673 **不表示 v4 更差**。
2. **指标 bug 修复史**（`docs/training_retrospective_2026-07-20.md:85-88,117-118`；git log）：① 原理识别分母 bug（`benchmark_utils.py:538`，分母 30–40 题/分子 10 题，2026-06-04 基线被污染）；② BLEU/ROUGE 恒被跳过（长度判断恒 False）；③ `eval_adapter_vs_base.py:55` 键名 bug（peak_memory 恒 None）；④ 温度 0.7→0.0；⑤ sacrebleu 2.x `BLEUScore.signature` 兼容（commit 22c1a32）。**修复前文件（②-F 表）一律作废**。
3. **judge 模型版本差异**：eval2 用 **moonshot-v1-8k**（response 截断：README 记 500 字符 vs report config 记 `judge_resp_chars=1000`——两处口径不一致，以 report json 为准；批量 README 记 10 条 vs report 记 5 条，同前）；金标终报用 **moonshot-v1-32k**。两套 judge 分数不可直接互比。
4. **v1 逐步 loss 日志永久丢失**（`results/README.md`）；v2 train_log 无 train_runtime/train_loss 汇总条目（两次 crash 后续跑），ledger 所记 train_loss 0.9872 来源为控制台日志，以 ledger 为准。
5. **v3 全零 lora_B 事故**：2026-07-22 05:45 发货的 v3 适配器 lora_B 全零（`adapter_model.safetensors.zerobak`，sha256 `f00ef15b…`），导致 08:20 评测 Δ 全 0；10:28 从 checkpoint-1200 恢复（sha256 `0335987e…`，`REMOTE:models/meerkat_triz_adapter_v3/adapter_info.json` → `restored_from`）。此事故直接催生 v4 的"全零 lora_B 校验"质量门。
6. **v3 adapter_info 内部不一致**：`best_checkpoint` 记 checkpoint-1200，但 `best_eval_loss` 1.07053 对应 eval step 1100；`shipped_from` 记"末步内存态"（即出事故的那版）。
7. **v1exp/v2exp 数值完全复现疑云**：07-22 在 sample_data_expanded.json 上的 v1exp/v2exp 评测数值与 07-21 sample_data.json 版逐位一致，疑命中相同评测子集或缓存，引用时需注明。
8. **PPL 评测集不同**：v1 用 test.jsonl 157 条、v2 用 v2_test.jsonl 498 条，绝对值不可互比；v3/v4 未跑 PPL。
9. **三套评测体系分数不可互比**：修复版四连（overall 0.472–0.589 档）/ eval2（overall_kw 0.25–0.31 档，8k judge）/ 金标终报（0.37–0.57 档 + 1.57–2.58 judge，32k judge）——权重、评测集、judge 均不同。
10. **eval2 concept n=81 与 expanded 集 127 条不一致**（concept_coverage 仅参考轨不入 overall），口径待核。
11. **v4 早停与学习率**：cosine 按 max_steps=2,872 设计，step 1,000 早停时 lr 仍有 1.553e-4；actual_epochs=1.393。v4 划分因语料无 source/chunk 标识**降级**为 instruction 前缀（12 字符）聚类分组（`v4_data_report.json` → `split.degraded`）。
12. **v2 训练 crash 史**：两次 crash 后成功，根因 peft 0.19 fp32 转换 OOM（`METRICS_LEDGER.md:9`）。
13. **config.py 名义 QLoRA 4bit 配置与实际不符**：`config.py:40-46` 留有 4bit nf4 量化配置，但 v1–v3 实际 FP16 加载、v4 纯 BF16 不量化；引用 config.py 时须以 adapter_info/评测 json 为准。

---

## 附：v4 全链时间线（`REMOTE:data/processed/v4_chain_state/*.done` 时间戳，UTC+8）

| 里程碑 | 时间 |
|---|---|
| wait_gold | 2026-07-23 10:54:29 |
| data_build | 2026-07-23 10:54:38 |
| eval_base / eval_v2 / eval_v3（与训练并行） | 07-23 12:48 / 13:20 / 13:54 |
| best checkpoint 连续晋升 step100→700 | 07-23 21:58 → 23:48 |
| train_v4（早停 @ step 1000） | 2026-07-24 00:43:55 |
| eval_v4 / final_report | 2026-07-24 01:16:48 |
