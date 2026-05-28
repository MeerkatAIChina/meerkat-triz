# 猫鼬AI DGX Spark 大模型训练与评测套件

在 NVIDIA DGX Spark (GB10, 128GB Unified Memory) 上完成 TRIZ 领域模型微调与评测的完整代码套件。

## 硬件环境

| 组件 | 规格 |
|------|------|
| GPU | NVIDIA GB10 Grace Blackwell |
| 统一内存 | 128 GB |
| 内存带宽 | 273 GB/s |
| FP4 算力 | 1 PFLOPS |
| CPU | 20-core Grace CPU |

## 项目结构

```
mongoose_ai/
├── README.md                          # 本文件
├── requirements.txt                   # Python依赖
├── config.py                          # 全局配置（超参数、路径）
│
├── utils/                             # 工具函数包
│   ├── __init__.py
│   ├── benchmark_utils.py             # 三层评测体系实现
│   ├── data_utils.py                  # 数据加载、ChatML转换、合成数据
│   └── training_utils.py              # 模型加载、QLoRA配置、训练
│
├── notebooks/                         # Jupyter Notebook（按顺序执行）
│   ├── 01_download_and_setup.ipynb    # 环境搭建与模型准备
│   ├── 02_data_preparation.ipynb      # 训练数据准备
│   ├── 03_model_benchmark.ipynb       # 微调前模型评测（基线）
│   ├── 04_qlora_finetune.ipynb        # QLoRA微调训练
│   └── 05_model_evaluation.ipynb      # 微调后模型评估
│
├── data/                              # 数据目录
│   ├── raw/                           # 原始技术手册/案例
│   └── processed/                     # ChatML格式数据集
│
├── models/                            # 模型目录
│   └── meerkat_triz_adapter_v1/       # 训练好的LoRA适配器
│
└── results/                           # 评测结果
```

## 快速开始

### 1. 上传项目到 DGX Spark

将本套件上传到 DGX Spark 的 JupyterLab 环境：

```bash
# 在DGX Spark终端中
mkdir -p /home/meerkat/mongoose_ai
cd /home/meerkat/mongoose_ai

# 上传所有文件后，创建必要的目录
mkdir -p data/raw data/processed models outputs checkpoints results
```

### 2. 安装依赖

打开 `01_download_and_setup.ipynb`，按顺序运行单元格安装依赖。

或手动安装：

```bash
pip install -r requirements.txt
```

### 3. 按顺序执行 Notebook

| 顺序 | Notebook | 内容 | 预计时间 |
|------|----------|------|---------|
| 1 | `01_download_and_setup.ipynb` | 检查硬件、安装依赖、加载模型 | 30分钟 |
| 2 | `02_data_preparation.ipynb` | 加载数据、转ChatML、划分数据集 | 10分钟 |
| 3 | `03_model_benchmark.ipynb` | 微调前评测（建立基线） | 2-4小时 |
| 4 | `04_qlora_finetune.ipynb` | QLoRA微调训练 | 8-15小时 |
| 5 | `05_model_evaluation.ipynb` | 微调后评测（对比效果） | 2-4小时 |

### 4. 配置说明

所有可调整参数集中在 `config.py`：

```python
# 基座模型选择
BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"  # 35B总参数 / 3B活跃参数 / 262K上下文

# QLoRA超参数
QLORA_CONFIG = {
    "lora": {
        "r": 64,           # LoRA秩 (32-128)
        "lora_alpha": 128, # 缩放因子
        "lora_dropout": 0.0,  # MoE架构兼容性：dropout可能干扰专家路由
    },
    "training": {
        "num_train_epochs": 2,          # 训练轮数
        "learning_rate": 2e-4,          # 学习率
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
    }
}
```

## Qwen3.6 架构适配说明

### 混合架构的 target_modules 问题

Qwen3.6 使用 **Gated DeltaNet + Gated Attention + MoE** 混合架构，与传统 Transformer 的模块名称不同：

| 层类型 | 占比 | 模块名称 |
|--------|------|----------|
| **Gated Attention** | 10/40层 (25%) | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| **Gated DeltaNet** | 30/40层 (75%) | `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj` |
| **MoE MLP** | 全部40层 | `gate_proj`, `up_proj`, `down_proj` |

### 解决方案 (已内置)

本套件已内置三种适配方案，**推荐使用显式模块列表**（`"all-linear"` 在混合架构上存在已知兼容性问题）：

```python
# 方案1: 手动指定Qwen3.6模块列表 (推荐)
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",           # Gated Attention (10/40层)
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",  # Gated DeltaNet (30/40层)
    "gate_proj", "up_proj", "down_proj",              # MoE MLP (全部40层)
]

# 方案2: PEFT自动检测 (不推荐，已知可能错误包含lm_head)
# target_modules = "all-linear"

# 方案3: 运行时自动扫描 (用于验证)
from utils.training_utils import find_all_linear_names
target_modules = find_all_linear_names(model)
```

### 自动检测函数

在 Notebook 04 的训练步骤中，会自动执行检测并输出实际模块列表：

```python
from utils.training_utils import find_all_linear_names

# 加载模型后运行
detected_modules = find_all_linear_names(model)
print(f"检测到 {len(detected_modules)} 个线性层: {detected_modules}")
```

## 关键技术参数

### QLoRA微调配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 量化方式 | NF4 (4-bit) | 内存占用减少75% |
| LoRA Rank | 64 | 复杂领域适配推荐 |
| LoRA Alpha | 128 | 2 * rank |
| 目标模块 | q/k/v/o_proj + gate/up/down_proj | 全量注意力+FFN |
| 训练轮数 | 2 | TRIZ领域通常足够 |
| 有效Batch Size | 8 | 1 * 8梯度累积 |
| 学习率 | 2e-4 | LoRA推荐范围 |
| 优化器 | paged_adamw_8bit | 分页优化器省内存 |

### 内存需求估算

| 组件 | 内存占用 |
|------|---------|
| 35B模型 (4-bit) | ~18-20 GB |
| LoRA适配器 | ~0.5 GB |
| 激活值/梯度 | ~25-35 GB |
| 优化器状态 | ~15-25 GB |
| **总计** | **~60-80 GB** |

DGX Spark 的 128GB 统一内存充裕，可同时容纳模型训练和其他任务。

## 评测体系

### Layer 1: 通用能力基准

使用 `lm-eval-harness` 运行：
- **MMLU-Pro**: 大学级别多学科知识
- **GPQA**: 研究生级别科学问答
- **HumanEval**: Python代码生成
- **MATH**: 数学推理
- **BBH**: 大基准难题

### Layer 2: TRIZ定制评测

专用评测器评估：
- **原理识别准确率**: 40个发明原理识别
- **矛盾解决能力**: 技术矛盾分析与化解
- **案例生成质量**: 创新方案质量 (BLEU/ROUGE)
- **ARIZ完整性**: ARIZ算法步骤覆盖度

### Layer 3: 工程性能基准

实测指标：
- **吞吐量**: tokens/second
- **P50延迟**: 毫秒
- **峰值内存**: GB

## 数据格式

### ChatML格式示例

```
<|im_start|>system
You are Meerkat-AI, an expert innovation consultant...<|im_end|>
<|im_start|>user
请解释TRIZ的分割原理...<|im_end|>
<|im_start|>assistant
分割原理（Principle 1 - Segmentation）是TRIZ 40个发明原理中的第一个...
包含以下三个指导方向：
1. 将物体分成独立的部分
2. 使物体成为可拆卸的
3. 增加物体的分割程度<|im_end|>
```

### 原始数据格式

```json
[
  {
    "instruction": "问题描述",
    "input": "补充输入（可选）",
    "output": "专家回答"
  }
]
```

## 合成数据生成

本项目使用 Moonshot API 从 548 条种子数据生成约 6000 条合成训练数据：

```bash
# 设置 Moonshot API Key
export MOONSHOT_API_KEY="your-api-key"

# 在 Notebook 02b 中执行生成
```

生成策略按子集区分：
- **改写问题** (concept_explanation, ariz_guidance): 保持答案不变，改写问题表述
- **全新生成** (case_generation, contradiction_analysis): 基于种子灵感生成全新Q&A对
- **混合策略** (principle_recommendation, innovation_assessment): 部分改写 + 部分全新

质量关卡：
- 去重：种子数据中去除重复模板后再生成
- 长度过滤：超过3500 tokens的样本被过滤
- 格式验证：所有输出使用 `tokenizer.apply_chat_template()` 转换为ChatML
- 困惑度过滤：可选，使用基座模型计算样本困惑度，过滤高困惑度样本（默认关闭，需要约20GB内存加载模型）
- 多样性评分：计算n-gram distinct-1/2指标，自动去重低多样性样本（默认开启，纯文本处理）

**真实数据比例说明：**

当前配置下真实种子数据占比约 **8.7%**（548条真实 / 6286条总计），低于理论目标的20-30%。
这是有意为之的设计决策：为了在Moonshot API成本可控的前提下（约￥5-10元）获得足够的样本总量（~6K条），
我们选择了以样本总量优先的策略。如需提高真实数据比例，可在 `config.py` 中降低 `multipliers` 值
（例如将 `case_generation` 和 `contradiction_analysis` 的倍数从16降至3-4），
但这会将总样本数降至约1500-2500条。

## 训练数据子集

| 子集 | 种子数据 | 目标数据 | 内容 |
|------|---------|---------|------|
| concept_explanation | 100 | 1,000 | TRIZ核心概念解释 |
| contradiction_analysis | 100 | 1,000 | 技术矛盾识别与分析 |
| principle_recommendation | 100 | 1,000 | 发明原理推荐 |
| case_generation | 100 | 1,000 | 创新案例生成 |
| ariz_guidance | 100 | 1,000 | ARIZ算法指导 |
| innovation_assessment | 100 | 1,000 | 创新方案评估 |
| **合计** | **600** | **6,000** | |

## 常见问题

### Q: DGX Spark 128GB内存是否足够？
A: 完全足够。Qwen3.6-35B-A3B模型4-bit量化后约18-20GB，加上训练所需激活值和优化器状态，峰值约60-80GB，仅占128GB的60%左右。剩余内存可同时运行评测或其他任务。

### Q: 训练时间多长？
A: 约8-15小时/epoch（35B模型比72B模型快约50%）。建议先运行1个epoch验证，再决定是否继续。

### Q: 如何恢复中断的训练？
A: 使用 checkpoint 恢复：
```python
from utils.training_utils import resume_from_checkpoint
trainer.train(resume_from_checkpoint="checkpoints/qlora_trtiz_v1/checkpoint-XXX")
```

### Q: 适配器可以迁移到其他基座模型吗？
A: 不可以。LoRA适配器与特定基座模型绑定。如需更换基座，需要重新训练。

## 版本信息

- 套件版本: v1.0
- 适配 PyTorch: >= 2.3.0
- 适配 Transformers: >= 4.40.0
- 适配 PEFT: >= 0.11.0
- 创建日期: 2026-05-23

## 许可证

内部使用 | 猫鼬AI技术团队
