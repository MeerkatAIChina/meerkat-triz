"""
猫鼬AI DGX Spark 项目配置文件
所有超参数和路径集中管理，便于统一调整
"""

import os
from pathlib import Path

# ==================== 路径配置 ====================
# 基础目录 (根据DGX Spark实际环境调整)
BASE_DIR = Path("/home/meerkat/mongoose_ai")
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

# 自动创建目录
for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, CHECKPOINTS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==================== 模型配置 ====================
# 基座模型 (HuggingFace模型ID或本地路径)
BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"  # 首选: Qwen3.6-35B-A3B MoE
# 模型规格: 35B总参数 / 3B活跃参数 / 256专家 / 262K上下文 / Apache 2.0
# 备选模型:
# BASE_MODEL = "Qwen/Qwen3-72B"  # 更大的模型，需要更多内存
# BASE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"  # 英文场景

# 模型加载配置
MODEL_CONFIG = {
    "trust_remote_code": True,
    "torch_dtype": "auto",  # 自动选择float16/bfloat16
    "device_map": "auto",   # 自动分配GPU/CPU内存
}

# ==================== QLoRA 配置 ====================
QLORA_CONFIG = {
    # 量化配置 (BitsAndBytes)
    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",           # Normal Float 4，精度最优
        "bnb_4bit_compute_dtype": "float16",     # 计算用float16
        "bnb_4bit_use_double_quant": True,       # 嵌套量化进一步省内存
    },
    
    # LoRA适配器配置
    "lora": {
        "r": 64,                    # LoRA秩: 复杂领域推荐64
        "lora_alpha": 128,          # 缩放因子: 通常为2*r
        # Qwen3.6 混合架构的 target_modules:
        # - Gated Attention 层 (10/40层): q_proj, k_proj, v_proj, o_proj
        # - Gated DeltaNet 层 (30/40层): in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj
        # - MoE MLP 层 (全部40层): gate_proj, up_proj, down_proj
        # 显式指定12个模块，不使用"all-linear"（已知兼容性问题）
        # 手动指定所有模块名，确保覆盖混合架构的全部线性层
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",           # Gated Attention (10/40层)
            "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",  # Gated DeltaNet (30/40层)
            "gate_proj", "up_proj", "down_proj",              # MoE MLP (全部40层)
        ],
        "lora_dropout": 0.0,        # Dropout率: 0.0 for MoE architecture compatibility (dropout can destabilize expert routing)
        "bias": "none",             # 不训练偏置
        "task_type": "CAUSAL_LM",   # 因果语言模型
        "use_rslora": False,        # 大rank时建议开启rsLoRA
    },
    
    # 训练超参数
    "training": {
        "output_dir": str(CHECKPOINTS_DIR / "qlora_triz_v1"),
        "num_train_epochs": 4,              # 04_worked实证best eval_loss在末步(欠训练), 2→4; 10s/step下约3.7h
        "per_device_train_batch_size": 1,   # DGX Spark单卡batch_size=1
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,   # 有效batch_size=8
        "learning_rate": 2e-4,              # LoRA推荐学习率
        "warmup_ratio": 0.05,               # 5% warmup (QLoRA推荐范围)
        "lr_scheduler_type": "cosine",      # 余弦退火
        "logging_steps": 10,
        "save_steps": 200,
        "eval_steps": 100,                # 配合4 epochs与EarlyStopping(patience=3), 200→100
        "save_total_limit": 3,              # 最多保留3个checkpoint
        "load_best_model_at_end": False,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "tensorboard",         # 或 "wandb"
        "bf16": False,                      # DGX Spark可能不支持bf16
        "fp16": False,                      # 与实际训练运行一致 (04_worked: fp16/bf16均关闭, 避免GradScaler+BF16冲突)
        "optim": "paged_adamw_8bit",        # 分页优化器省内存
    }
}

# ==================== 数据配置 ====================
DATA_CONFIG = {
    # 原始数据路径
    "raw_data_dir": str(DATA_DIR / "raw"),
    "processed_data_dir": str(DATA_DIR / "processed"),
    
    # 数据集子集配置 (6个子集)
    "subsets": {
        "concept_explanation": {    # 概念解释
            "seed_count": 100,
            "target_count": 1000,
            "description": "TRIZ核心概念、40个发明原理、39个工程参数的解释"
        },
        "contradiction_analysis": {  # 矛盾分析
            "seed_count": 100,
            "target_count": 1000,
            "description": "技术矛盾识别、物理矛盾分析、矛盾矩阵查询"
        },
        "principle_recommendation": {  # 原理推荐
            "seed_count": 100,
            "target_count": 1000,
            "description": "根据问题推荐发明原理、原理组合策略"
        },
        "case_generation": {  # 案例生成
            "seed_count": 100,
            "target_count": 1000,
            "description": "基于真实案例生成创新解决方案"
        },
        "ariz_guidance": {  # ARIZ指导
            "seed_count": 100,
            "target_count": 1000,
            "description": "ARIZ算法步骤指导、问题转化、理想解构建"
        },
        "innovation_assessment": {  # 创新评估
            "seed_count": 100,
            "target_count": 1000,
            "description": "创新方案评估、专利可行性分析、技术成熟度评价"
        },
    },
    
    # ChatML格式配置
    "chatml": {
        "system_message": (
            "You are Meerkat-AI, an expert innovation consultant specializing in TRIZ "
            "(Theory of Inventive Problem Solving). You help users analyze technical contradictions, "
            "recommend invention principles, generate innovative solutions, and guide them through "
            "the ARIZ algorithm. Always provide structured, actionable advice grounded in TRIZ methodology."
        ),
        "max_length": 4096,  # 最大序列长度
    },
    
    # 训练/验证/测试划分
    "split_ratio": {
        "train": 0.85,
        "validation": 0.10,
        "test": 0.05,
    }
}

# ==================== 合成数据生成配置 ====================
SYNTHETIC_CONFIG = {
    "api": {
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
        "rpm": 3,  # Tier 0默认速率限制
        "batch_size": 5,  # 每个API请求包含的种子数
        "max_tokens_per_sample": 1500,
        "temperature": 0.8,
    },
    # 按子集的扩展倍数（D-03: 事实类子集低倍数，多样性类子集高倍数）
    "multipliers": {
        "concept_explanation": 6,      # 25% real目标 → 实际约14%
        "ariz_guidance": 6,            # 25% real目标 → 实际约14%
        "principle_recommendation": 11,  # ~15% real目标 → 实际约8%
        "innovation_assessment": 11,     # ~15% real目标 → 实际约8%
        "case_generation": 16,         # 10% real目标 → 实际约6%
        "contradiction_analysis": 16,  # 10% real目标 → 实际约6%
    },
    # 生成策略（D-01: 按子集采用不同策略）
    "strategies": {
        "concept_explanation": "rephrase",       # 改写问题，保持答案
        "ariz_guidance": "rephrase",             # 改写问题，保持答案
        "principle_recommendation": "mixed",     # 混合：改写+全新
        "innovation_assessment": "mixed",        # 混合：改写+全新
        "case_generation": "generate_new",       # 全新Q&A对
        "contradiction_analysis": "generate_new", # 全新Q&A对
    },
    "quality_gates": {
        "max_tokens": 3500,           # 超过此长度的样本将被过滤
        "deduplicate": True,          # 去重种子数据中的重复项
        # 困惑度过滤 (可选: 需要加载基座模型, 内存占用约20GB)
        "perplexity": {
            "enabled": False,         # 默认关闭 (可在Notebook 02b中手动开启)
            "percentile": 80,         # 保留困惑度最低的80%样本
            "device": None,           # None=自动选择 (cuda优先, 可设为"cpu")
        },
        # 多样性评分 (纯文本处理, 无需模型)
        "diversity": {
            "enabled": True,          # 默认开启
            "min_distinct_1": 0.30,   # 最低unigram多样性
            "min_distinct_2": 0.15,   # 最低bigram多样性
            "field": "instruction",   # 用于多样性计算的字段
        },
    },
    "output_dir": str(DATA_DIR / "processed" / "synthetic"),
    "checkpoint_dir": str(DATA_DIR / "processed" / "checkpoint"),
}

# ==================== 语料库构建配置 (TRIZ-raw 原始材料) ====================
CORPUS_CONFIG = {
    # 原始TRIZ材料目录 (DGX Spark上应复制到 BASE_DIR/TRIZ-raw/)
    "raw_dir": str(BASE_DIR / "TRIZ-raw"),
    # 输出目录
    "output_dir": str(DATA_DIR / "processed" / "corpus"),
    # 文件名
    "output_filename": "triz_corpus.jsonl",
    "stats_filename": "triz_corpus_stats.json",
    "failed_files_filename": "failed_files.json",
    # 分块配置
    "chunk": {
        "target_tokens": 2048,
        "max_tokens": 4096,
        # 中文字符token估算: 1 token ~ 1字符 (保守估计)
        "chars_per_token": 1.0,
    },
    # 质量关卡
    "quality_gates": {
        "min_chars": 50,
        "deduplicate": True,
        "language_filter": False,
    },
    # 支持提取的文件扩展名
    "supported_extensions": [".pdf", ".docx", ".pptx", ".doc"],
    # 默认跳过的文件扩展名 (图片/视频/压缩包/表格)
    "skip_extensions": [
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".jfif",
        ".mov", ".mp4", ".avi", ".mkv",
        ".zip", ".rar", ".7z",
        ".xlsx", ".xls", ".csv",
    ],
    # OCR配置 (用于扫描版PDF)
    "ocr": {
        "enabled": True,
        "min_text_chars": 20,
    },
}

# ==================== 评测配置 ====================
BENCHMARK_CONFIG = {
    # 评测输出目录
    "output_dir": str(RESULTS_DIR),
    
    # Layer 1: 通用能力基准 (使用lm-eval-harness)
    "general_benchmarks": {
        "mmlu_pro": {
            "num_fewshot": 5,
            "batch_size": 1,
            "description": "大学级别多学科知识 (MMLU-Pro)"
        },
        "gpqa": {
            "num_fewshot": 0,
            "batch_size": 1,
            "description": "研究生级别科学问答 (GPQA)"
        },
        "humaneval": {
            "num_fewshot": 0,
            "batch_size": 1,
            "description": "Python代码生成 (HumanEval)"
        },
        "math": {
            "num_fewshot": 4,
            "batch_size": 1,
            "description": "数学推理 (MATH)"
        },
        "bbh": {
            "num_fewshot": 3,
            "batch_size": 1,
            "description": "大基准难题 (BBH)"
        },
    },
    
    # Layer 2: TRIZ定制评测
    "triz_benchmarks": {
        "principle_accuracy": {
            "description": "40个发明原理识别准确率",
            "metric": "accuracy",
        },
        "contradiction_resolution": {
            "description": "技术矛盾解决能力",
            "metric": "accuracy",
        },
        "case_quality": {
            "description": "创新案例生成质量",
            "metric": "bleu+rouge",
        },
        "ariz_completeness": {
            "description": "ARIZ步骤完整性",
            "metric": "accuracy",
        },
    },
    
    # Layer 3: 工程性能基准
    "performance_benchmarks": {
        "tokens_per_second": {
            "description": "推理吞吐量 (tokens/s)",
            "target": ">50",
        },
        "latency_p50": {
            "description": "P50延迟 (ms)",
            "target": "<2000",
        },
        "memory_usage": {
            "description": "峰值内存占用 (GB)",
            "target": "<100",
        },
    }
}

# ==================== DGX Spark 硬件配置 ====================
HARDWARE_CONFIG = {
    "device": "cuda",
    "gpu_memory": 128,       # GB (统一内存)
    "cpu_cores": 20,         # Grace CPU
    "recommended_batch_size": 1,
    "gradient_checkpointing": True,  # 必须开启以节省内存
}

# ==================== 日志配置 ====================
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": str(OUTPUTS_DIR / "training.log"),
}
