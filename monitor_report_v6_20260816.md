# Meerkat-TRIZ v6 换基座监控报告 — 2026-08-16

## 本次运行结论：训练已启动 ✅

---

## 一、锚点评测状态

| 项目 | 结果 |
|------|------|
| 进程状态 | 已完成（2026-08-15 02:36 ~ 08-16 14:59，约 36 小时） |
| 生成缓存 | 300/300 行（100%） |
| 报告产物 | `results/v5/eval_v5_base_goldfix_v5_qwen38_20260815_023643.json` |
| | `results/v5/eval_v5_base_goldfix_v5_qwen38_20260815_023643.md` |

## 二、决策门判定

| 门 | 阈值 | 实测值 | 状态 |
|----|------|--------|------|
| **门 A** | judge armA overall < 3.4 | **2.9300** | ✅ 通过 |
| **门 B** | invalid 占比 ≤ 10% | **0%** (0/300) | ✅ 通过 |
| **门 C** | overrefusal ≤ 2% | **null/0%** | ✅ 通过 |

> 判定：三扇决策门全部通过，**准予启动 v6 训练**。

### 锚点详细指标
- **keyword overall**: 0.6245
- **judge armA overall**: 2.9300
- **pass 率 (keyword)**: 0.740 [0.688, 0.786]
- **pass 率 (judge)**: 0.787 [0.737, 0.829]
- **漏判率**: 23.3% (70/300)，进入审计队列

## 三、v6 训练启动状态

| 项目 | 值 |
|------|-----|
| 训练 PID | **518663** |
| 运行时长 | ~10 分钟（截至检查时刻） |
| CPU 占用 | 97.6% |
| 当前步数 | ~13 / 5548 |
| 日志路径 | `checkpoints/qlora_triz_v6_qwen38/train.log` |

### 启动断言检查结果

| 断言项 | 状态 | 详情 |
|--------|------|------|
| max_length = 2048 | ✅ | 配置一致，截断率已由 P2 预检确认 0% |
| eval_steps == save_steps | ✅ | 100 == 100 |
| 首 batch labels prompt 区段全 -100 | ✅ PASSED | 序列长 169, prompt 46 token 全 -100, 监督 123 token |
| completion 无 <|im_start|> 泄漏 | ✅ PASSED | 无泄漏 |
| loss_smoke | ✅ PASSED | completion_only_loss = True (trl 1.5.1 自动判定) |

### 模型加载信息
- 基座模型：`models/Qwen3.8-27B`（BF16，851 shards）
- 显存占用：50.1 GB
- LoRA 可训练参数：466,911,232 / 27,362,909,696 (1.7064%)
- LoRA 配置：r=64, alpha=128, dropout=0.0, use_rslora=False

### 训练参数
- 训练数据：11,096 条（train）/ 1,050 条（val）
- epochs：4（安全帽）
- 每 epoch 步数：1,387
- **cosine horizon**：1,387 × 2 = **2,774 步**
- warmup：138 步（5%）
- 学习率：2e-4
- 优化器：adamw_torch
- 评估/保存间隔：100 步
- EarlyStopping：patience=3, threshold=0.002
- save_total_limit：8

### 速度估算
- 当前速度：~15.5 秒/步（预热中，可能加快）
- horizon (2,774 步) 预计：**~12 小时**
- 完整 4 epochs (5,548 步) 预计：**~24 小时**
- 实际可能因 early stopping 提前结束

## 四、异常/警告记录

1. **[非致命]** `torch_dtype is deprecated! Use dtype instead!` — transformers 版本警告，不影响训练
2. **[非致命]** `warmup_ratio is deprecated...` — transformers 版本警告
3. **[非致命]** tokenizer PAD/BOS/EOS tokens 与 model config 不同，已自动对齐
4. **[注意]** `Using EarlyStoppingCallback without load_best_model_at_end=True` — 训练结束后不会自动加载最佳模型，需手动选择 checkpoint

## 五、后续监控建议

| 时间点 | 操作 |
|--------|------|
| **每 100 步** | 检查 eval loss 趋势，确认无发散 |
| **第 100 步 (第一个 eval)** | 重点观察 val loss 是否下降，确认学习有效 |
| **horizon 中点 (~1,387 步)** | 检查 cosine decay 是否生效，lr 是否下降到 ~1e-4 |
| **horizon 终点 (~2,774 步)** | 决定是否继续（若 early stopping 未触发） |
| **训练结束后** | 用 `eval_v5_qwen38_anchor.json` 跑适配器评测，与锚点做差值对比 |

## 六、备注

- 无需对 qwen3_5 多模态架构做修补：模型加载正常，vision tower 参数未卷入训练（仅 LoRA 目标模块命中 language_model 侧）
- 无 PATCH_NOTES.md 需要记录
- vLLM 服务保持停止状态
