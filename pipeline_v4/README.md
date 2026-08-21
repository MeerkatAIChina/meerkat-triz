# pipeline_v4 — 干净重建的 TRIZ LoRA 训练管线

推翻 v1→v3 旧管线后的干净版。基座模型 `models/Qwen3.6-35B-A3B`（MoE，33.25B 专家参数），
目标硬件 NVIDIA GB10（121GB 统一内存），环境 `venv_v5`
（peft 0.19.1 / transformers 5.10.1 / trl 1.5.1 / torch 2.12.0）。

## 旧管线事故清单与 v4 修复对照

| # | 旧事故/缺陷 | v4 修复 |
|---|---|---|
| 1 | peft 0.19.1 × transformers 5.10.1 `WeightConverter` API 不兼容；`PeftModel.from_pretrained(model.unload(), ckpt)` 先污染内存再抛异常，fallback 保存了 310 个全零 lora_B（F32） | `src/compat.py` import 即打 monkey-patch（提取自 `scripts/eval_adapter_vs_base.py` L26-33）；且 v4 **发货不再从内存加载/保存模型**，只复制磁盘文件 |
| 2 | `save_total_limit=3` 把真正最优 checkpoint 轮转删除；`find_best_checkpoint` 扫幸存 checkpoint 的全量 log_history，eval_loss 张冠李戴 | `BestCheckpointCallback`：eval 创新低**立即**把对应 checkpoint 复制到 `best/`（不参与轮转）；归因只采纳 `log_history` 中 `step == checkpoint 步数` 的 eval_loss 条目；`save_total_limit=8` |
| 3 | 旧验证只查文件存在 + ≥1MB + 前向能跑，全零 lora_B 全部 PASSED | `checkpointing.validate_adapter_dir`：断言**全部 lora_B 非零**（手解 safetensors 逐字节检查）、**dtype 全 BF16**、**sha256 记录**、adapter_config 完整 |
| 4 | 训练结束 fallback 保存被污染的内存态 | 任何 best checkpoint 不可用的情况，fallback = 复制 Trainer 落盘的 checkpoint 目录文件（Trainer 落盘是好的），**绝不保存训练内存态** |
| 5 | `assistant_only_loss` 未启用（chat template 无 `{% generation %}`），全文计 loss | 数据集改为 **prompt/completion 格式**，trl 1.5.1 `SFTTrainer` 自动启用 `completion_only_loss`（见下节）；train.py 启动时断言必须为 True 否则退出 |
| 6 | 4-bit 量化未生效且是事故链源头 | 纯 BF16 加载 + BF16 LoRA，不用 bitsandbytes（121GB 内存够，旧 FP16 加载实测 64.6GB） |
| 7 | 元数据不诚实 | `adapter_info.json` 如实记录实际 epoch 数、实际步数、早停、best 的 eval step/loss、sha256、逐条验证结果；验证 FAILED 时写明 FAILED 且退出码非零 |

## completion-only loss 在 trl 1.5.1 的实际机制（源码确认）

- `SFTConfig.completion_only_loss` 默认 `None`；`SFTTrainer.__init__` 中：
  `args.completion_only_loss is None` → 按训练数据集首样本是否同时含
  `"prompt"` 和 `"completion"` 键自动判定（`sft_trainer.py` L1140-1145）。
- 标准（字符串）prompt-completion 格式的处理（L1452-1488）：
  分别 tokenize `prompt` 与 `prompt+completion`，构造 `completion_mask`
  （prompt 部分为 0），`DataCollatorForLanguageModeling` 将 mask 为 0 的
  labels 置 -100。**不应用 chat template**，直接字符串拼接；EOS 自动附加到
  completion 末尾（若尚未以 EOS 结尾）。
- **关键推论**：`prompt` 字段必须是完整格式化文本（含 ChatML 标记），
  由数据构建侧负责。见 `configs/data_v4.json`。
- `formatting_func` 与 `completion_only_loss=True` 不兼容（TRL 会 raise），
  因此 train.py 绝不传 `formatting_func`。

## 目录结构

```
pipeline_v4/
  README.md
  configs/
    train_v4.json    # 全部训练超参（eval_steps==save_steps=100 强制相等）
    data_v4.json     # 数据构建参数（v4.1: 与 pipeline_v4_remote/configs/data_v4.json 同步, 产物 v4_1_*.jsonl）
    eval_v4.json     # 评测参数（占位，评测代理填充）
  src/
    compat.py        # WeightConverter monkey-patch，import 即生效
    checkpointing.py # best-ckpt 归因/另存/验证/发货（torch-free）
    train.py         # BF16 LoRA 训练主脚本
  run/
    train_v4.sh      # 单入口（tee 日志到 checkpoints/train_v4.log）
```

## 复现步骤

```bash
# 0. 前置: 数据构建产出 data/processed/v4_train.jsonl / v4_validation.jsonl
#    （prompt/completion 格式, 见 configs/data_v4.json）

# 1. 干跑验证（不加载模型、不碰 GPU）
venv_v5/bin/python pipeline_v4/src/train.py \
  --config pipeline_v4/configs/train_v4.json --dry-run

# 2. 训练（单入口, 日志 tee 到 checkpoints/train_v4.log）
bash pipeline_v4/run/train_v4.sh

# 3. 产物
#    checkpoints/qlora_triz_v4/best/        ← 训练中实时另存的最优 checkpoint
#    models/meerkat_triz_adapter_v4/        ← 发货适配器 + adapter_info.json
#    results/train_log_v4.json              ← 完整 log_history
```

## 训练/评测细节

TODO：训练完成后补充 —— 实际步数/耗时/best eval_loss 曲线/内存占用实测。

TODO：评测方案由评测代理按 `configs/eval_v4.json` 填充后补充结果与结论。
