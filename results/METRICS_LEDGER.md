# METRICS_LEDGER — 跨版本效果台账

每次训练 run 完成后必须登记一行；评测补跑后更新对应行。原始数据文件存于 `results/`（命名见 [README.md](./README.md)）。
"未跑"是合法值，但 v1.1 起训练后评测（EVAL-14）为强制项，新 run 不允许留空。

| Run 名 | 日期 | 数据集版本 | Epochs | Best eval_loss | Layer 1 delta | Layer 2 分数 | Layer 3 指标 | 适配器 SHA-256 | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| corpus_sft_v1 | 2026-06-19 | TRIZ-raw corpus SFT v1（2,662 train / 313 val / 157 test） | 2（666 steps） | 1.3979（无基座对照） | 未跑 | 未跑 | 未跑 | `1f909cb0…`（169 MB，`models/meerkat_triz_adapter_v1/`） | checkpoint 验证 200/400/600/666 全 PASSED；loss 为全文口径（system+user+assistant），与 completion-only 不可比；唯一完整 run，逐步 loss 日志已丢失 |

<!-- 新 run 在此追加行。基座对照 eval_loss（EVAL-12）测得后，补充在下面控制行：

| base_model_control | YYYY-MM-DD | 同 v1 val 集（313 条） | — | <base eval_loss> | — | — | — | — | EVAL-12 基座对照 |
-->
