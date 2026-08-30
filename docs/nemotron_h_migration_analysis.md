# Meerkat-TRIZ 迁移至 Nemotron-H MTP 基座 — 可行性分析报告

> 分析日期: 2026-08-30  
> 当前基座: Qwen3.6-35B-A3B / Ornith-1.5-35B-A3B (qwen3_5_moe 架构)  
> 目标基座: NVIDIA Nemotron-H 系列 (hybrid Mamba-2 + Transformer + LatentMoE + MTP)

---

## 一、当前状态速览

| 维度 | 当前配置 |
|------|---------|
| **基座模型** | Qwen3.6-35B-A3B (v1) / Qwen3.8-27B (v6) / Ornith-1.5-35B-A3B (v7) |
| **总参 / 活跃参** | 35B / ~3B (MoE, 256 experts) |
| **架构** | qwen3_5_moe: Gated DeltaNet + Gated Attention + MoE MLP |
| **训练方式** | LoRA SFT, r=64, α=128, BF16, dropout=0 |
| **Target Modules** | 12个: q/k/v/o_proj, in_proj_qkv/z/b/a, out_proj, gate/up/down_proj |
| **数据集** | v5b, ~11,421 条 TRIZ 领域 SFT 样本 |
| **训练框架** | Transformers 5.10.1 + PEFT 0.19.1 + TRL 1.5.1 + Torch 2.12.0 |
| **部署** | vLLM 0.25.0 (MARLIN MoE backend) + Open WebUI |
| **硬件** | DGX Spark (GB10, 128GB unified memory) |

---

## 二、Nemotron-H 架构概览

Nemotron-H (`model_type: nemotron_h`) 是 NVIDIA 推出的混合架构，核心特征：

### 2.1 三层架构混合

```
┌─────────────────────────────────────────┐
│  Nemotron-H Layer Stack                 │
│                                         │
│  ┌─────────────┐  ┌─────────────┐      │
│  │ Mamba-2     │  │ Attention   │      │
│  │ (SSM)       │  │ (Transformer)│     │
│  │ linear-time │  │ quadratic   │      │
│  │ context     │  │ attention   │      │
│  └──────┬──────┘  └──────┬──────┘      │
│         └────────┬────────┘              │
│                  ▼                       │
│         ┌─────────────┐                 │
│         │ LatentMoE   │                 │
│         │ (compressed │                 │
│         │  latent dim)│                 │
│         └──────┬──────┘                 │
│                ▼                         │
│         ┌─────────────┐                 │
│         │ MTP Layer   │ ← baked-in     │
│         │ (multi-token│   speculative   │
│         │  prediction)│   decoding      │
│         └─────────────┘                 │
└─────────────────────────────────────────┘
```

### 2.2 与当前 Qwen3.5_moe 架构的关键差异

| 特征 | Qwen3.5_moe (当前) | Nemotron-H (目标) |
|------|-------------------|-------------------|
| **核心层** | Gated DeltaNet + Gated Attention | Mamba-2 SSM + Standard Attention |
| **MoE 类型** | Standard MoE (256 routed experts) | **LatentMoE** (latent space routing) |
| **MoE 维度** | 全维度专家 (d=4096) | 压缩维度专家 (d=4096 → ℓ=1024) |
| **推理加速** | MTP (外部, vLLM speculative) | **MTP 内建** (baked into checkpoint) |
| **KV Cache** | 标准 KV cache | KV cache + **Mamba SSM state cache** |
| **上下文** | 262K | **1M tokens** |
| **训练精度** | BF16 | **NVFP4 pretraining** + BF16/FP8 inference |

### 2.3 推荐的 Nemotron-H 变体

| 模型 | 总参 | 活跃参 | DGX Spark 可行性 | 备注 |
|------|------|--------|-----------------|------|
| Nemotron-3-Nano-4B | 4B | 4B (dense) | ✅ 轻松 | 规模太小，领域效果可能不足 |
| Nemotron-3.5-Lightning-30B-A3B | **30B** | **~3B** | ✅ **官方支持** | **推荐** — 与当前 Qwen3.6-35B-A3B 规格最接近 |
| Nemotron-3-Super-120B-A12B | 120B | ~12B | ⚠️ 可能 OOM | 活跃参 ×4，推理内存压力大 |
| Nemotron-3-Ultra-550B-A55B | 550B | ~55B | ❌ 不可行 | 远超 DGX Spark 容量 |

> **推荐基座**: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`  
> 理由: 30B/3B 与当前 35B/3B 处于同一量级，DGX Spark 有官方部署指南，NVIDIA 明确将其列为 DGX Spark 部署目标。

---

## 三、迁移工作量分析

### 3.1 工作量矩阵

| 模块 | 工作量 | 风险 | 说明 |
|------|--------|------|------|
| **训练框架适配** | 🔴 高 | 中 | 需验证 PEFT 对 nemotron_h 的兼容性，或迁移至 NeMo Automodel |
| **Target Modules 重探测** | 🟡 中 | 低 | 架构完全不同，需重新扫描线性层 |
| **Mamba 层 LoRA 限制** | 🟡 中 | 中 | Mamba `out_proj` 不接受 LoRA (custom kernel 限制) |
| **LatentMoE LoRA** | 🟡 中 | 中 | `fc1_latent_proj` / `fc2_latent_proj` / `e_score_correction_bias` 特殊处理 |
| **MTP 层处理** | 🟢 低 | 低 | 训练时 MTP 层通常冻结或不参与 LoRA |
| **Tokenizer / Chat Template** | 🟡 中 | 低 | Nemotron 使用不同 template，需重新适配 |
| **数据格式验证** | 🟢 低 | 低 | 当前 v5b 数据格式 (prompt/completion) 通用，大概率兼容 |
| **评测 Harness 适配** | 🟡 中 | 低 | 需验证评测脚本在 nemotron_h 上的加载 |
| **部署配置 (vLLM)** | 🟢 低 | 低 | vLLM 已支持 nemotron_h LoRA (prefix remapping resolved) |
| **量化口径对齐** | 🟡 中 | 中 | 当前 BF16 训练 → NVFP4 基座，需评估量化漂移 |

### 3.2 关键风险项

#### 风险 1: PEFT 兼容性 (🔴 高)

当前训练栈 (`transformers 5.10.1 + PEFT 0.19.1 + TRL 1.5.1`) 对 `nemotron_h` 的支持状态不明。已知信息：

- ✅ NeMo Automodel (NVIDIA 官方): 完全支持，有 YAML 配置示例
- ✅ AgileRL: 已添加 Nemotron-H 架构支持 + LoRA
- ⚠️ HuggingFace PEFT: 需验证 `model_type=nemotron_h` 是否被正确识别
- ⚠️ TRL SFTTrainer: 需验证与 Nemotron-H 的 `trust_remote_code` 兼容性

**建议路径**:
- **短期 PoC**: 先用 HuggingFace PEFT 尝试，若遇到兼容性问题再切 NeMo
- **长期**: 若 NeMo Automodel 性能显著更优，可考虑迁移至 NeMo 训练栈

#### 风险 2: Mamba 层 LoRA 限制 (🟡 中)

NVIDIA 官方文档明确指出:

> "Mamba layers use custom kernels that take in the `out_proj.weight` directly, thus **LoRA doesn't work here**."  
> — [transformers/docs/.../nemo_automodel_finetuning.md](https://github.com/huggingface/transformers/blob/main/docs/source/en/community_integrations/nemo_automodel_finetuning.md)

这意味着 Nemotron-H 的 Mamba-2 层无法挂载 LoRA，只有 Attention 层和部分 MoE 投影层可训练。

**影响**: 可训练参数比例可能低于当前 Qwen3.6 配置（当前 12 个模块全覆盖）。

#### 风险 3: 评测基线重置 (🟡 中)

迁移至全新基座 = 所有历史评测数据失去直接可比性。需：
- 重新跑 base model 基线（Layer 1/2/3）
- 重新跑外部评委等长臂评审
- 更新 METRICS_LEDGER 的基座对照行

---

## 四、技术方案

### 4.1 方案 A: 保守迁移 (HuggingFace 训练栈)

保持当前 `transformers + PEFT + TRL` 训练栈，仅替换基座模型。

```python
# 关键变更点
model = AutoModelForCausalLM.from_pretrained(
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",  # ← 新基座
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# 重新探测 target_modules (Nemotron-H 架构完全不同)
target_modules = find_all_linear_names(model)
# 预期结果: 包含 attention q/k/v/o_proj, mlp gate/up/down_proj
#           但 mamba 层的 out_proj 需手动排除
```

**预计 target_modules** (基于 Nemotron-H 结构推断):

```python
# Attention 层 (标准 Transformer)
"q_proj", "k_proj", "v_proj", "o_proj",

# MLP 层 (LatentMoE 前后的投影)
"gate_proj", "up_proj", "down_proj",

# LatentMoE 特有
"fc1_latent_proj",   # latent space 压缩投影
"fc2_latent_proj",   # latent space 还原投影

# ⚠️ 排除项 (已知不接受 LoRA)
# "out_proj" in Mamba layers — 需按层类型区分
```

**NeMo Automodel 的推荐配置** (参考官方文档):

```yaml
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  match_all_linear: false
  exclude_modules:
    - "*vision_tower*"
    - "*lm_head*"
    - "*.out_proj"  # ← mamba layers 排除
  dim: 64
  alpha: 128
  use_triton: true
```

### 4.2 方案 B: 官方迁移 (NeMo Automodel)

切换到 NVIDIA NeMo Automodel 训练栈，获得官方优化。

**优势**:
- NVIDIA 官方维护，对 Nemotron-H 有第一方支持
- 集成 DeepEP / TransformerEngine 内核，吞吐更高
- 支持 FSDP2 + Expert Parallelism
- 与 NVIDIA 推理栈 (TensorRT-LLM) 无缝衔接

**劣势**:
- 学习曲线陡峭，YAML 配置体系与当前 Python 脚本不同
- 需安装 NeMo 依赖链，可能与现有 venv 冲突
- 评测 harness 需重写以适配 NeMo checkpoint 格式

**NeMo 配置示例**:

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
  torch_dtype: torch.bfloat16

peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  dim: 64
  alpha: 128
  exclude_modules: ["*.out_proj"]  # mamba 层排除

distributed:
  strategy: fsdp2
  ep_size: 1  # DGX Spark 单卡，expert parallelism=1

step_scheduler:
  global_batch_size: 8
  local_batch_size: 1
  max_steps: 1600  # ~4 epochs on 11k samples

optimizer:
  _target_: torch.optim.AdamW
  lr: 2e-4
  weight_decay: 0.01
  betas: [0.9, 0.95]
```

### 4.3 方案对比

| 维度 | 方案 A: HF 栈 | 方案 B: NeMo |
|------|-------------|-------------|
| **开发成本** | 低 (保持现有代码) | 高 (全新框架) |
| **性能天花板** | 中等 (PEFT eager) | 高 (TE/DeepEP 优化) |
| **官方支持** | 社区维护 | NVIDIA 第一方 |
| **风险** | PEFT 兼容性问题 | 学习成本 + 评测适配 |
| **推荐阶段** | **PoC / 快速验证** | **生产级长期** |

---

## 五、部署兼容性

### 5.1 vLLM 推理

| 维度 | 当前 (Qwen3.6) | 目标 (Nemotron-H) |
|------|---------------|-------------------|
| **vLLM 版本** | 0.25.0 | 0.25.0+ (可能需要 nightly) |
| **LoRA 支持** | ✅ MARLIN backend | ✅ **已解决** (prefix remapping) |
| **MTP 推理** | 外部 speculative | **内建 MTP 层** (原生支持) |
| **Mamba cache** | N/A | `mamba_ssm_cache` (float32) |
| **量化** | NVFP4 (unsloth) | NVFP4 (NVIDIA 官方) |

**关键变更**:

```bash
# 当前启动 (Qwen3.6)
vllm serve unsloth/Qwen3.6-35B-A3B-NVFP4-Fast \
  --enable-lora \
  --lora-modules Meerkat-TRIZ-v1=/path/to/adapter

# 目标启动 (Nemotron-H)
vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --enable-lora \
  --lora-modules Meerkat-TRIZ-v1-nemotron=/path/to/adapter \
  --speculative-model "[ngram]"  # MTP 层自动生效，无需外部 draft
```

### 5.2 DGX Spark 特定注意事项

根据 [NVIDIA 官方 DGX Spark 部署指南](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Super/SparkDeploymentGuide/README.md):

1. **Tensor Parallel = 1**: DGX Spark 单 GPU，LatentMoE 的 all-to-all 通信无需 EP
2. **Mamba SSM cache**: 使用 `float32`，与 KV cache 分开管理
3. **MTP 层**: 内建于 checkpoint，vLLM 自动检测，无需额外配置
4. **reasoning_parser.py**: Nemotron-3 系列需要特定的 reasoning parser 处理思考链

---

## 六、迁移路线图

```
Phase 0: 技术预研 (1 周)
├── 1.1 下载 Nemotron-3.5-Lightning-30B-A3B-BF16，验证本地加载
├── 1.2 用 find_all_linear_names() 扫描 target_modules
├── 1.3 尝试 PEFT LoRA attach，验证是否报错
├── 1.4 跑单个 training step smoke test
└── 决策点: PEFT 是否兼容？→ 兼容走方案 A，不兼容走方案 B

Phase 1: 基线重建 (1 周)
├── 2.1 跑 base model 在 v5 gold (300 items) 的评测
├── 2.2 记录新基线到 METRICS_LEDGER
├── 2.3 对比 keyword track / judge track 与 Qwen3.6 base
└── 2.4 如有显著差异，分析归因 (tokenizer / template / architecture)

Phase 2: 训练适配 (1-2 周)
├── 3.1 适配 chat template (Nemotron 使用 <extra_id_1> 格式)
├── 3.2 冻结超参数 (lr=2e-4, r=64, α=128, 4 epochs)
├── 3.3 训练并监控 eval_loss 轨迹
└── 3.4 early-stopping 触发的 step 点对比

Phase 3: 评测与决策 (1 周)
├── 4.1 完整 Layer 2 评测 (keyword + judge dual-track)
├── 4.2 外部评委等长臂评审
├── 4.3 通用能力探针 (Layer 1)
└── 决策点: 是否显著优于 Qwen3.6 base？→ 决定 v7 是否替换基座

Phase 4: 部署更新 (3-5 天)
├── 5.1 更新 docker-compose.yml (新镜像/模型ID)
├── 5.2 更新 pi-models.json (provider 配置)
├── 5.3 生产环境切换 + smoke test
└── 5.4 HuggingFace 发布 (如通过评测)
```

---

## 七、预期收益与成本

### 7.1 潜在收益

| 收益项 | 预期效果 | 置信度 |
|--------|---------|--------|
| **推理速度** | MTP 内建推测解码，C1 吞吐预计 +20-40% | 高 |
| **上下文长度** | 1M tokens (vs 当前 262K) | 高 |
| **长文本能力** | Mamba-2 线性时间注意力，长 context 更稳定 | 中 |
| **Agentic 能力** | Nemotron 系列针对 agentic reasoning 优化 | 中 |
| **生态系统** | NVIDIA 官方持续迭代，vLLM/TRT-LLM 第一方支持 | 高 |

### 7.2 潜在成本

| 成本项 | 预估 | 说明 |
|--------|------|------|
| **开发时间** | 3-5 周 | 含框架适配、基线重建、评测 |
| **可训练参数减少** | -10~30% | Mamba 层无法接受 LoRA |
| **基线不可比** | 永久 | 新基座 = 新评测基准 |
| **量化漂移风险** | 需验证 | NVFP4 基座 + BF16 LoRA vs 纯 BF16 |

---

## 八、结论与建议

### 8.1 总体评估

| 维度 | 评分 | 说明 |
|------|------|------|
| **技术可行性** | ✅ 可行 | Nemotron-H LoRA 有成熟支持路径 |
| **工作量** | 🟡 中等偏大 | 约 3-5 周，主要在框架适配和基线重建 |
| **风险可控性** | 🟡 中 | PEFT 兼容性为主要不确定因素 |
| **预期 ROI** | 🟡 中 | 推理速度提升确定，领域效果提升不确定 |

### 8.2 分阶段建议

1. **立即执行 (PoC, 3-5 天)**:
   - 下载 `Nemotron-3.5-Lightning-30B-A3B-BF16`
   - 验证 PEFT LoRA attach 是否成功
   - 跑 1 个 training step 的 smoke test
   - **决策点**: PEFT 兼容性是否通过？

2. **条件执行 (基线评测, 1 周)**:
   - 若 PoC 通过 → 跑完整 base model 评测
   - 对比 keyword/judge track 与 Qwen3.6 base
   - **决策点**: Nemotron base 是否 ≥ Qwen3.6 base？

3. **完整迁移 (2-3 周)**:
   - 若前两个阶段通过 → 启动完整训练
   - 更新训练/评测/部署全套配置
   - 发布 `Meerkat-TRIZ-v1-Nemotron-3.5-Lightning-30B-A3B`

### 8.3 替代方案

若 Nemotron-H 迁移工作量过大，可考虑 **渐进式路径**:
- **v7_ornith 先完成**: Ornith-1.5-35B-A3B 与 Qwen3.6 同架构，迁移成本低，可快速发布
- **Nemotron 作为 v8**: 将 Nemotron-H 作为下一代基座实验，不阻塞当前 v7 发布节奏

---

## 附录: 参考资源

- [Nemotron-3-Super DGX Spark Deployment Guide](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/usage-cookbook/Nemotron-3-Super/SparkDeploymentGuide)
- [Nemotron-H Fine-Tune in NeMo Automodel](https://docs.nvidia.com/nemo/automodel/model-coverage/large-language-models/nvidia/nemotron-h)
- [TensorRT-LLM Nemotron v3 Deployment Guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/deployment-guide/deployment-guide-for-nemotron-3-on-trtllm.md)
- [PEFT Nemotron Support (AgileRL)](https://github.com/AgileRL/AgileRL/releases/tag/v2.9.0)
- [Nemotron 3 Super Paper](https://arxiv.org/abs/2604.12374)
- [Nemotron-3.5-Lightning on DGX Spark (实测)](https://github.com/kubesimplify/website/blob/main/content/blog/nemotron-3-5-lightning-on-dgx-spark.md)
