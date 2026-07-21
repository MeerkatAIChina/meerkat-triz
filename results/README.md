# results/ — 训练日志与评测结果

本目录收录**必须入库（git 跟踪）**的训练与评测产物：

- 训练日志（`trainer.state.log_history` 落盘 JSON，v1 的逐步 loss 已永久丢失，不得再犯）
- 各层评测结果（Layer 1 通用基准 / Layer 2 TRIZ 基准 / Layer 3 性能）
- 基座 vs 适配器对照评测、幻觉检测、人工评审记录
- 跨版本效果台账：[`METRICS_LEDGER.md`](./METRICS_LEDGER.md)

## 命名约定

```
<type>_<run>_<ts>.json
```

- `<type>`：结果类型，如 `eval`（对照评测）、`layer1`、`layer2`、`layer3`、`hallucination`、`train_log`、`human_review`
- `<run>`：训练 run 名，如 `corpus_sft_v1`、`corpus_sft_v2`
- `<ts>`：时间戳 `YYYYMMDD_HHMMSS`

示例：`eval_corpus_sft_v2_20260801_143000.json`

人工评审记录与台账使用 Markdown（`*.md`）。

## .gitignore 口径

`results/*.json` 与 `results/*.md` 被 git 跟踪；大文件（`*.bin`、`*.safetensors`、`*.pt` 等）继续排除，请勿放入本目录。
