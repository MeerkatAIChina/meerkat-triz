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
    data_v4.json     # 数据构建参数（v4.1: term_coverage_random 再平衡, 产物 v4_1_*.jsonl）
    eval_v4.json     # 评测参数（占位，评测代理填充）
  src/
    compat.py        # WeightConverter monkey-patch，import 即生效
    checkpointing.py # best-ckpt 归因/另存/验证/发货（torch-free）
    train.py         # BF16 LoRA 训练主脚本
  run/
    train_v4.sh      # 训练单入口（tee 日志到 checkpoints/train_v4.log）
    chain_v4.sh      # 串行总链（等金标→数据构建→base/v2/v3评测→训练→v4评测→汇总, 可续跑）
    chain_v4_1.sh    # v4.1 续跑链（复用金标+四方评测 → 修复版数据构建→v4.1训练→v4.1评测(锚点v2)→五方汇总）
  src/ (续)
    data_build.py    # 数据构建（质量门/去污/再平衡/分层分组划分/ChatML 渲染）
    gold_gen.py      # 金标评测集生成（Moonshot API, 断点续跑）
    eval_harness.py  # 金标双轨评测 harness（关键词轨 + judge 轨 + 配对统计）
    final_report.py  # 四方对比汇总 + 决策门判定
```

## 复现步骤

```bash
# 0a. 金标评测集生成（RPM=3, 约 30-60 分钟, 断点续跑; 产物 data/processed/v4_gold.jsonl）
venv_v5/bin/python pipeline_v4/src/gold_gen.py --config pipeline_v4/configs/eval_v4.json

# 0b. 数据构建（纯 CPU, 秒级; 产出 v4_train/v4_validation/v4_test.jsonl + 报告）
venv_v5/bin/python pipeline_v4/src/data_build.py --config pipeline_v4/configs/data_v4.json

# 0c. 评测冒烟（不碰 GPU/API: 假数据跑通全链路; judge 探测实发一条 ping/模型）
venv_v5/bin/python pipeline_v4/src/eval_harness.py --config pipeline_v4/configs/eval_v4.json \
  --tag dryrun --dry-run --limit 12
venv_v5/bin/python pipeline_v4/src/eval_harness.py --config pipeline_v4/configs/eval_v4.json \
  --probe-judge

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

## 数据（data_build.py）

输入 v2 语料（10327 条）+ ariz boost（674 条），全部参数见 `configs/data_v4.json`。
质量门按序执行（每门计数进 `results/v4_data_report.json`）：

1. 去 output 中 `<think>...</think>` 块；output < 150 字符丢弃。
2. 归一化 instruction 精确去重；**同 instruction 多答案冲突组整组丢弃**（旧管线同题不同答混入的修复）。
3. **近重复去重**：instruction 级 token 3-gram Jaccard ≥ 0.7（token = 拉丁/数字连续段或单汉字；稀有 token 签名分桶 + 桶内两两比较，纯 stdlib，11k 条秒级完成）。
4. **去污**：与金标集 `v4_gold.jsonl` 的 question 做 3-gram Jaccard ≥ 0.5，命中剔除（金标集不存在则跳过并在报告注明，金标完成后必须重跑）。
5. 子集再平衡：concept_explanation / innovation_assessment 各 cap 2500，其余子集全保留。
   **v4.1 起选取规则改为两阶段**（`strategy=term_coverage_random`）：TRIZ 工具术语言表
   贪心最大覆盖（60% 配额）+ output 长度三桶分层随机补足（40%，seed=42）；v4 初版的
   "cap 内优先保留 output 更长者"（`strategy=longest_first`，保留作回滚/对照）已证实
   造成 keyword/concept_explanation 退化（v4 0.4356 vs v2 0.5187；E2 归因：保留集长度
   漂移 MWU p=5.3e-25、术语真缺失 8 词次），修复依据 `paper/v5_plan/sec1_data.md` §4。
6. 分层划分 85/10/5（seed=42）：v2 语料无 source/chunk 标识 → **退化为按归一化 instruction 前缀（12 字符）聚类分组**，同组同侧、按子集分层，报告注明退化；划分后 test/validation 与 train 再做 3-gram Jaccard ≥ 0.5 交叉检查，命中者移回 train。
7. ChatML 渲染：`apply_chat_template(system+user, add_generation_prompt=True, enable_thinking=False)` 后剥空 think 块，prompt 以 `<|im_start|>assistant\n` 结尾；completion 为原始 output（EOS 由 TRL 附加）；prompt+completion > 2048 token 丢弃。

与旧方法差异：旧质量门只有长度+精确去重（无近重复、无冲突检测、无去污、随机划分致泄漏）；
v4 全部补齐，且评测 reference 由独立金标集承担，不再与训练数据同批生成。

## 评测（gold_gen.py + eval_harness.py）

- **金标集**：从 `triz_corpus.jsonl`（3914 chunks）按 category 分层随机抽 chunk，moonshot-v1-8k
  生成 100 题（principle 20 / contradiction 20 / ariz 20 / case 15 / concept 15 / innovation 10），
  每题含 question / reference_answer(≥200字) / 5-8 个期望关键词（必须在 reference 中出现）/
  source chunk id。逐条追加 `data/processed/v4_gold.jsonl` 断点续跑；解析失败丢弃补抽；
  人工抽检队列 `data/processed/v4_gold_review.md`。
- **双轨评分**：关键词轨（期望关键词命中率）+ judge 轨（0-4 分 rubric：准确性/完整性/TRIZ
  正确性/结构，批量 5 条/请求）。judge **不用 moonshot-v1-8k**：按
  `kimi-k2-0711-preview → moonshot-v1-32k → moonshot-v1-8k` 顺序 ping 探测选第一个可用者，
  兜底同源时结果大字标注"仅供参考"。两轨各自报告，不混合加权。
- **统计**：与 base 逐题配对 —— paired bootstrap 10000 次 95% CI（overall + 各子集）、
  Wilson CI（pass 率）、McNemar 精确检验（干净重实现，参考 `/tmp/eval_pipeline_v2/eval2.py`
  L582-647）。生成结果与 judge 打分均有缓存，可断点续跑。
- **校准模式**：`--calibrate` 只对 base 跑双轨并导出 judge 打分明细供人工一致性检查。
- 与旧方法差异：旧评测 n=6~16 且扩充集未真正进入评测；judge 与数据生成器同源未校准。
  v4 固定 100 题金标、judge 异源 + 校准模式 + 完整配对统计。

## 串行总链（chain_v4.sh）

`bash pipeline_v4/run/chain_v4.sh`（建议 tmux 内）：等金标满 100 题（超时 2h 报错）→
数据构建 → base/v2/v3 金标评测 → v4 训练 → v4 评测 → `results/v4_final_report.md`
（四方对比 + 决策门：v4 judge 轨 overall 显著 > base 且所有子指标无显著退化 → "建议替代 v2"，
否则 "保留 v2"）。每步完成写 `data/processed/v4_chain_state/<step>.done`，重跑自动续跑。

## v4.1 续跑链（chain_v4_1.sh）— concept_explanation 退化修复版

`bash pipeline_v4/run/chain_v4_1.sh`（建议 tmux 内）：前置检查金标与四方评测结果齐备
（不重新生成）→ 修复版数据构建（`term_coverage_random`，cap 2500 →
`data/processed/v4_1_*.jsonl`，v4 原始 jsonl 保留可复查）→ v4.1 训练
（`run_name=v4.1` → `checkpoints/qlora_triz_v4_1` / `models/meerkat_triz_adapter_v4_1`，
超参与 v4 完全一致，唯一自变量是 rebalance 策略）→ v4.1 金标评测
（**锚点 = v2**：stats_review §2.3 证实 base 锚点被 think 污染，vs-base 提升幅度高估不可用）
→ `results/v4_1_final_report.md`（五方对比 + 决策门，candidate=v4_1_gold）。

判读重点：修复成功的标志是 **keyword/concept_explanation 相对 v2 的差值 CI 回到包含 0**
（v4 为 -0.083 [-0.148, -0.023]），而非必须显著超过 v2；若出现 kw 升 judge 降的两轨
反向（v3 堆关键词覆辙），按 `configs/data_v4.json` rebalance.note 的回退预案处理。
状态目录独立：`data/processed/v4_1_chain_state/<step>.done`，与 v4 链互不干扰。
训练步支持**断点自动续训**：存在 `checkpoints/qlora_triz_v4_1/checkpoint-*` 时自动从最新
一个 `--resume`（v4_1 曾在 2026-07-27 step~200 被外部 kill，无错误现场，疑似 GPU 争抢）。
GPU 排队：`run/wait_gpu_then_chain_v4_1.sh` 看守进程等 v5a 训练退出后自动接力本链
（2026-07-28 起挂在 tmux 会话 `v4_1_queue`，日志 `data/processed/v4_1_queue.log`）。
**2026-07-28 优先级调整**（主人指令）：v5a 金标评测插队在先——v4_1 于 step 283 SIGTERM
暂停（checkpoint-200 在盘），`run/eval_v5a_then_chain_v4_1.sh` 跑完 v5a 评测
（tag=v5a_gold，300 题，锚点 base_goldfix_v5；旧 tag v5_gold 的缓存属 backup 适配器不可复用）
后自动接回本链续训。

## 训练/评测细节

TODO：训练完成后补充 —— 实际步数/耗时/best eval_loss 曲线/内存占用实测。

TODO：chain_v4 跑完后补充金标评测结果与决策门结论。
