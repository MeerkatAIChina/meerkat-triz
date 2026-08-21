# METRICS_LEDGER — 跨版本效果台账

每次训练 run 完成后必须登记一行；评测补跑后更新对应行。原始数据文件存于 `results/`（命名见 [README.md](./README.md)）。
"未跑"是合法值，但 v1.1 起训练后评测（EVAL-14）为强制项，新 run 不允许留空。

| Run 名 | 日期 | 数据集版本 | Epochs | Best eval_loss | Layer 1 delta | Layer 2 分数 | Layer 3 指标 | 适配器 SHA-256 | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| corpus_sft_v1 | 2026-06-19 | TRIZ-raw corpus SFT v1（2,662 train / 313 val / 157 test） | 2（666 steps） | 1.3979（无基座对照） | Δloss −1.868 / ΔPPL −21.62（base PPL 25.84 → 4.22，test.jsonl 157 条） | overall 0.472（base 0.529，**−0.057**）；原理 0.9→0.6（−0.30）；矛盾 +0.100；案例 +0.125；ARIZ −0.111 | 吞吐 17.7→15.2 tok/s；p50 延迟 28.9→11.3 s；峰值显存 64.7→65.1 GB | `1f909cb0…`（169 MB，`models/meerkat_triz_adapter_v1/`） | checkpoint 验证 200/400/600/666 全 PASSED；loss 为全文口径（system+user+assistant），与 completion-only 不可比；逐步 loss 日志已丢失。修复版指标复测（2026-07-21）：**整体回退**，仅矛盾/案例子项提升 |
| corpus_sft_v2 | 2026-07-21 | TRIZ-raw corpus SFT v2 多角度（8,458 train / 995 val / 498 test，text-only schema） | 2（2,116 steps，6h46m） | 1.0673（step 2000；轨迹 1.1576→1.0673 见 train_log_v2.json） | Δloss −2.461 / ΔPPL −31.19（base PPL 34.09 → 2.90，v2_test.jsonl 498 条） | overall 0.589（base 0.529，**+0.060**）；原理 0.9→0.9（0，天花板）；矛盾 +0.052；案例 **+0.363**；ARIZ **−0.139** | 吞吐 17.9→15.1 tok/s；p50 延迟 28.6→33.9 s；峰值显存 64.7→65.1 GB | `a7bb48a4…`（161.6 MB，`models/meerkat_triz_adapter_v2/`） | 经 /tmp/train_v2.py 实跑（两次 crash 后成功，根因 peft 0.19 fp32 转换 OOM）；train_loss 0.9872；eval_mean_token_accuracy 0.7436。修复版指标（2026-07-21）：整体与案例显著提升，**ARIZ 回退**（v1 同样 −0.111，共性短板，需定向补 ARIZ 数据） |
| v6_qwen38_main | 2026-08-18 | v5_data（v5_train_v5a.jsonl 11,096 train / 1,050 val，prompt/completion schema） | 4 配置 → 1.15 实际（1,600 步 early-stopping，9.73h） | 1.5058（completion-only 口径 @ step1300） | 未测（通用基准欠账，待 lm-eval 补） | judge armA +0.6033（moonshot 同族，vs 锚点 2.93）；异族三评委 claude-sonnet-4-6 +0.031(n.s.)/gpt-5.4 +0.094/gemini-3.5-flash +0.107，**噪声账后全部 n.s.**；keyword −0.001(n.s.) | 峰值显存 58.9 GB（吞吐/延迟未测） | `7af119ba…`（933 MB，`models/meerkat_triz_adapter_v6_qwen38/`） | 换基座 Qwen3.8-27B（qwen3_5 混合架构）；训练多次崩溃（BF16 optimizer foreach + caching_allocator_warmup OOM），最终全新干净 run 完成；同族 +0.60 → 异族 +0.03~+0.11（~1/6），复现 v5a 评委家族膨胀；300/300 质量门、overrefusal 0% |

## 基座对照（EVAL-12，修复版指标，2026-07-21 实测）

两次独立评测（v1/v2 链）基座分数完全一致，可复现性确认：

| 指标 | 基座值 |
|---|---|
| overall_score | 0.529 |
| principle_accuracy | 0.9 |
| contradiction_resolution | 0.3625 |
| case_coverage | 0.3625 |
| ariz_completeness | 0.389 |
| 吞吐 / p50 延迟 / 峰值显存 | 17.7–17.9 tok/s / 28.6–28.9 s / 64.7 GB |
| PPL（test.jsonl 157 条 / v2_test.jsonl 498 条） | 25.84 / 34.09 |

<!-- 新 run 在此追加行。-->
