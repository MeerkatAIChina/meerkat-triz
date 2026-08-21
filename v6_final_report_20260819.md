# Meerkat-TRIZ v6 换基座任务 — 完整链路报告

## 任务概览

| 阶段 | 状态 | 时间 |
|------|------|------|
| 预检（P1/P2/P3） | ✅ 通过 | 2026-08-14 |
| 锚点评测 | ✅ 完成 | 2026-08-15 02:36 ~ 08-16 14:59 |
| v6 训练 | ✅ 完成 | 2026-08-16 ~ 08-18（历经两次恢复） |
| v6 适配器评测 | ✅ 完成 | 2026-08-18 11:30 ~ 08-19 04:58 |

---

## 一、锚点评测（基线）

**报告**: `results/v5/eval_v5_base_goldfix_v5_qwen38_20260815_023643.md`

| 指标 | 值 |
|------|-----|
| keyword overall | 0.6245 |
| judge armA overall | **2.9300** |
| pass 率 (keyword) | 0.740 [0.688, 0.786] |
| pass 率 (judge) | 0.787 [0.737, 0.829] |
| invalid | 0/300 (0%) |
| overrefusal | 0/300 (0%) |
| 漏判率 | 23.3% (70/300) |

**决策门判定**：
- 门 A (judge < 3.4): 2.93 ✅ → 微调空间充足
- 门 B (invalid ≤ 10%): 0% ✅
- 门 C (overrefusal ≤ 2%): 0% ✅

---

## 二、v6 训练

**适配器**: `models/meerkat_triz_adapter_v6_qwen38/` (934 MB)
**配置**: `pipeline_v5/configs/train_v6_qwen38.json`

| 参数 | 值 |
|------|-----|
| 基座 | Qwen3.8-27B (BF16, 50.1 GB) |
| LoRA r/alpha | 64 / 128 |
| 可训练参数 | 466,911,232 (1.7064%) |
| 学习率 | 2e-4 |
| horizon | 2,774 步 (2 epochs) |
| 最终 checkpoint | checkpoint-1600 + best/ |

**训练危机记录**（详见 `PATCH_NOTES.md`）：
1. **step 500 恢复**: BF16 optimizer 状态与 Torch Adam foreach 不兼容 → 删除 optimizer.pt，设 `TORCH_OPTIM_FOREACH=0` 恢复
2. **step 1000+ 恢复**: CUDA OOM (caching_allocator_warmup) → 同样操作从 checkpoint-1000 恢复

---

## 三、v6 适配器评测（微调后）

**报告**: `results/v5/eval_v5_v6_gold_20260818_113035.md`

| 指标 | v6 适配器 | 锚点 | 差值 | 显著性 |
|------|-----------|------|------|--------|
| keyword overall | **0.6236** | 0.6245 | -0.0009 | 不显著 (p=1) |
| judge armA overall | **3.5333** | 2.9300 | **+0.6033** | **显著** (p=3e-12) |
| invalid | 0/300 (0%) | 0/300 (0%) | — | — |
| overrefusal | 0/300 (0%) | 0/300 (0%) | — | — |
| pass 率 (keyword) | 0.743 [0.691, 0.789] | 0.740 | +0.003 | — |
| pass 率 (judge) | 0.960 [0.931, 0.977] | 0.787 | **+0.173** | — |

### 子集 judge armA 对比

| 子集 | n | 锚点 | v6 | 差值 | 95% CI |
|------|---|------|-----|------|--------|
| ariz_guidance | 60 | 3.1500 | 3.7500 | **+0.6000** | [+0.4333, +0.7667] |
| case_generation | 45 | 2.8889 | 3.5333 | **+0.6444** | [+0.4222, +0.9111] |
| concept_explanation | 45 | 2.6000 | 3.5333 | **+0.9333** | [+0.6000, +1.2667] |
| contradiction_analysis | 60 | 2.6500 | 3.4167 | **+0.7667** | [+0.4833, +1.0667] |
| innovation_assessment | 30 | 2.9667 | 3.7000 | **+0.7333** | [+0.4667, +1.0667] |
| principle_recommendation | 60 | 3.2500 | 3.3500 | +0.1000 | [-0.0833, +0.3000] |

### McNemar 配对检验
- 翻正（基线错 → v6 对）: **57 题**
- 翻负（基线对 → v6 错）: **5 题**
- p 值: **3.066e-12**（极显著）

---

## 四、关键结论

### ✅ 换基座任务成功

1. **Qwen3.8-27B 基座兼容**: 无需多模态架构修补，直接加载成功
2. **微调有效**: judge armA 从 2.93 → 3.53（+0.60 分，p<1e-11）
3. **生成质量稳定**: invalid = 0%，overrefusal = 0%，keyword 基本持平
4. **所有子集均有提升**: concept_explanation 提升最大（+0.93），principle_recommendation 提升较小（+0.10，不显著）

### ⚠️ 注意事项

1. **keyword 指标未改善**: 0.6245 → 0.6236（不显著），说明微调主要改善了 judge 评分而非 keyword 匹配
2. **漏判率略升**: 23.3% → 25.3%，需关注审计队列
3. **训练稳定性**: 两次恢复（BF16 optimizer + CUDA OOM），建议后续优化环境配置

---

## 五、产物清单

| 产物 | 路径 |
|------|------|
| 锚点评测报告 | `results/v5/eval_v5_base_goldfix_v5_qwen38_20260815_023643.{md,json}` |
| v6 适配器 | `models/meerkat_triz_adapter_v6_qwen38/` |
| v6 评测报告 | `results/v5/eval_v5_v6_gold_20260818_113035.{md,json}` |
| 训练日志 | `checkpoints/qlora_triz_v6_qwen38/train.log` |
| 补丁记录 | `checkpoints/qlora_triz_v6_qwen38/PATCH_NOTES.md` |
| Checkpoints | `checkpoints/qlora_triz_v6_qwen38/checkpoint-{900,1000,...,1600}/` + `best/` |

---

报告生成时间: 2026-08-19
