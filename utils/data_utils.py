"""
数据处理工具函数集
支持：原始数据加载、ChatML格式转换、合成数据生成、数据集划分
"""

import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datasets import Dataset, DatasetDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 原始数据加载 ====================

def load_raw_data(data_dir: str) -> Dict[str, List[Dict]]:
    """
    从原始数据目录加载所有数据
    
    期望的数据结构:
    data_dir/
        concept_explanation.json
        contradiction_analysis.json
        principle_recommendation.json
        case_generation.json
        ariz_guidance.json
        innovation_assessment.json
    
    每个JSON文件格式:
    [
        {
            "instruction": "问题描述",
            "input": "补充输入（可选）",
            "output": "专家回答"
        },
        ...
    ]
    
    Returns:
        按子集分类的数据字典
    """
    data_dir = Path(data_dir)
    subsets = {}
    
    if not data_dir.exists():
        logger.warning(f"数据目录不存在: {data_dir}，将创建示例数据")
        return create_sample_data()
    
    for json_file in sorted(data_dir.glob("*.json")):
        subset_name = json_file.stem
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            subsets[subset_name] = data
            logger.info(f"加载 {subset_name}: {len(data)} 条样本")
        except Exception as e:
            logger.error(f"加载 {json_file} 失败: {e}")
    
    return subsets


def create_sample_data(data_dir: Optional[str] = None) -> Dict[str, List[Dict]]:
    """
    加载示例训练数据

    从JSON文件加载预生成的548条高质量TRIZ领域训练数据，
    覆盖6个子集：概念解释、矛盾分析、原理推荐、案例生成、ARIZ指导、创新评估。

    如果JSON文件不存在，则创建仅含12条最小示例数据的回退数据集
    (仅用于验证数据流程，不足以支撑有意义的训练)。

    Args:
        data_dir: 数据目录路径 (默认: 项目根目录下的data/)

    Returns:
        按子集分类的数据字典
    """
    import os

    # 默认数据目录
    if data_dir is None:
        # 尝试从项目根目录查找
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "..", "data"),
            "/home/meerkat/mongoose_ai/data",
            "./data",
        ]
        for p in possible_paths:
            if os.path.exists(p):
                data_dir = p
                break
        else:
            data_dir = "./data"

    json_path = os.path.join(data_dir, "sample_data.json")

    # 优先从JSON文件加载完整数据
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                sample_data = json.load(f)
            total = sum(len(v) for v in sample_data.values())
            logger.info(f"从JSON加载示例数据: {total} 条，覆盖 {len(sample_data)} 个子集")
            return sample_data
        except Exception as e:
            logger.warning(f"加载JSON数据失败 ({e})，使用回退数据")

    # 回退：创建最小示例数据 (仅用于验证流程)
    logger.warning("使用最小示例数据 (12条) - 仅用于验证数据流程，不足以支撑训练")
    return _create_fallback_sample_data()


def _create_fallback_sample_data() -> Dict[str, List[Dict]]:
    """创建最小示例数据（仅12条，用于验证数据流程）"""
    return {
        "concept_explanation": [
            {"instruction": "请解释TRIZ的发明原理1——分割原理", "input": "", "output": "分割原理是将物体分成独立部分..."},
            {"instruction": "解释TRIZ中的技术矛盾", "input": "", "output": "技术矛盾是改善一个参数导致另一个参数恶化..."},
        ],
        "contradiction_analysis": [
            {"instruction": "分析：手机大屏与续航的矛盾", "input": "", "output": "矛盾矩阵推荐分割、动态化原理..."},
        ],
        "principle_recommendation": [
            {"instruction": "提高空调能效的原理推荐", "input": "", "output": "反馈原理、自服务原理..."},
        ],
        "case_generation": [
            {"instruction": "为医疗器械生成方案", "input": "便携心电监测", "output": "分割原理+嵌套原理..."},
        ],
        "ariz_guidance": [
            {"instruction": "ARIZ分析质量与成本", "input": "", "output": "步骤1-问题分析..."},
        ],
        "innovation_assessment": [
            {"instruction": "评估无人机运输专利", "input": "", "output": "新颖性中等，实用性高..."},
        ],
    }


# ==================== ChatML格式转换 ====================

def convert_to_chatml(
    data: Dict[str, List[Dict]],
    tokenizer=None,
    system_message: Optional[str] = None
) -> DatasetDict:
    """
    将原始数据转换为对话格式的HuggingFace Dataset

    优先使用 tokenizer.apply_chat_template() 生成对话格式，
    确保训练数据格式与模型推理时的chat template完全一致。
    如果tokenizer不可用，则回退到硬编码的ChatML格式。

    Args:
        data: 原始数据字典 (包含instruction, input, output字段)
        tokenizer: 模型的tokenizer (推荐使用，确保格式一致性)
        system_message: 系统提示词 (默认使用TRIZ专家角色设定)

    Returns:
        DatasetDict，包含train/validation/test分割
    """
    default_system = (
        "You are Meerkat-AI, an expert innovation consultant specializing in TRIZ "
        "(Theory of Inventive Problem Solving). You help users analyze technical contradictions, "
        "recommend invention principles, generate innovative solutions, and guide them through "
        "the ARIZ algorithm. Always provide structured, actionable advice grounded in TRIZ methodology."
    )
    system_message = system_message or default_system

    # 检查tokenizer是否支持apply_chat_template
    use_chat_template = tokenizer is not None and hasattr(tokenizer, 'apply_chat_template')
    if use_chat_template:
        logger.info("使用tokenizer.apply_chat_template()生成对话格式 (推荐)")
    else:
        logger.warning("tokenizer不可用，使用硬编码ChatML格式 (格式可能不一致)")

    logger.info("转换数据格式...")
    all_samples = []

    for subset_name, samples in data.items():
        for sample in samples:
            instruction = sample.get("instruction", "")
            input_text = sample.get("input", "")
            output = sample.get("output", "")
            sample_system = sample.get("system", system_message)

            # 构建完整问题
            if input_text:
                full_question = f"{instruction}\n\n{input_text}"
            else:
                full_question = instruction

            # 使用tokenizer的chat template 或 硬编码格式
            if use_chat_template:
                messages = [
                    {"role": "system", "content": sample_system},
                    {"role": "user", "content": full_question},
                    {"role": "assistant", "content": output},
                ]
                try:
                    chatml_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                except Exception as e:
                    logger.warning(f"apply_chat_template失败 ({e})，回退到硬编码格式")
                    chatml_text = _build_chatml_hardcoded(sample_system, full_question, output)
            else:
                chatml_text = _build_chatml_hardcoded(sample_system, full_question, output)

            all_samples.append({
                "text": chatml_text,
                "subset": subset_name,
                "instruction": instruction,
                "input": input_text,
                "output": output,
                "system": sample_system,
                "length": len(chatml_text),
            })

    logger.info(f"总共 {len(all_samples)} 条对话格式样本")

    # 创建Dataset并划分
    dataset = Dataset.from_list(all_samples)
    dataset = split_dataset(dataset)

    return dataset


def _build_chatml_hardcoded(system: str, question: str, answer: str) -> str:
    """硬编码ChatML格式（回退方案）"""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer}<|im_end|>"
    )


def format_messages(
    tokenizer,
    user_content: str,
    system_message: Optional[str] = None,
    assistant_content: Optional[str] = None,
    add_generation_prompt: bool = False,
) -> str:
    """
    Format messages using tokenizer.apply_chat_template().

    Replaces all hardcoded ChatML strings with a single utility that ensures
    token ID alignment with the model's native chat template.

    Args:
        tokenizer: Model tokenizer with chat_template support.
        user_content: The user's question/prompt.
        system_message: System prompt. If None, uses DATA_CONFIG default.
        assistant_content: If provided, includes assistant response (for training data).
        add_generation_prompt: If True, appends assistant start token for inference.
            If False, returns complete conversation for training data.

    Returns:
        Formatted chat string.
    """
    from config import DATA_CONFIG

    if system_message is None:
        system_message = DATA_CONFIG['chatml']['system_message']

    messages = [{"role": "system", "content": system_message}]
    messages.append({"role": "user", "content": user_content})
    if assistant_content is not None:
        messages.append({"role": "assistant", "content": assistant_content})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    test_ratio: float = 0.05,
    seed: int = 42
) -> DatasetDict:
    """
    划分数据集为训练/验证/测试集
    
    Args:
        dataset: 输入数据集
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    
    Returns:
        DatasetDict
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须等于1"
    
    dataset = dataset.shuffle(seed=seed)
    
    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    test_dataset = dataset.select(range(train_size + val_size, total))
    
    logger.info(f"数据集划分: 训练集={len(train_dataset)}, 验证集={len(val_dataset)}, 测试集={len(test_dataset)}")
    
    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset,
    })


# ==================== 合成数据生成 ====================

def create_synthetic_data(
    seed_data: List[Dict],
    multiplier: int = 10,
    variation_strategy: str = "paraphrase"
) -> List[Dict]:
    """
    基于种子数据生成合成训练数据
    
    策略：
    1. paraphrase: 改写问题表述，保持答案不变
    2. extend: 扩展答案内容
    3. combine: 组合多个问题
    
    Args:
        seed_data: 种子数据
        multiplier: 扩展倍数
        variation_strategy: 变化策略
    
    Returns:
        扩展后的数据列表
    """
    logger.info(f"使用策略 '{variation_strategy}' 生成合成数据，扩展倍数: {multiplier}x")
    
    synthetic_data = []
    
    # 问题改写模板
    question_templates = [
        "请详细解释{topic}",
        "如何用TRIZ方法分析{topic}",
        "在{topic}方面，TRIZ提供了哪些解决方案？",
        "请举例说明{topic}的实际应用",
        "{topic}的核心思想是什么？",
    ]
    
    for sample in seed_data:
        # 保留原始样本
        synthetic_data.append(sample)
        
        # 生成变体
        for i in range(multiplier - 1):
            varied_sample = vary_sample(sample, i, variation_strategy)
            synthetic_data.append(varied_sample)
    
    logger.info(f"合成数据生成完成: {len(seed_data)} → {len(synthetic_data)} 条")
    return synthetic_data


def vary_sample(sample: Dict, variant_id: int, strategy: str) -> Dict:
    """生成单条数据的变体"""
    instruction = sample.get("instruction", "")
    output = sample.get("output", "")
    
    if strategy == "paraphrase":
        # 简单改写：添加前缀/后缀
        prefixes = [
            "从TRIZ角度分析，",
            "请运用创新方法论回答：",
            "作为一个TRIZ专家，",
            "在工程创新实践中，",
        ]
        prefix = prefixes[variant_id % len(prefixes)]
        new_instruction = prefix + instruction
        
    elif strategy == "extend":
        # 扩展输出
        new_instruction = instruction
        output = output + "\n\n[补充说明] 这一方法在现代工业中有广泛应用，建议结合具体场景灵活运用。"
        
    else:
        new_instruction = instruction
    
    return {
        "instruction": new_instruction,
        "input": sample.get("input", ""),
        "output": output,
    }


# ==================== 数据集保存与加载 ====================

def save_dataset(dataset: DatasetDict, output_dir: str):
    """保存DatasetDict到磁盘"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_dataset in dataset.items():
        split_path = output_path / f"{split_name}.jsonl"
        split_dataset.to_json(str(split_path))
        logger.info(f"保存 {split_name}: {len(split_dataset)} 条 → {split_path}")


def load_processed_dataset(data_dir: str) -> DatasetDict:
    """从磁盘加载处理后的数据集"""
    data_dir = Path(data_dir)
    dataset_dict = {}
    
    for split in ["train", "validation", "test"]:
        file_path = data_dir / f"{split}.jsonl"
        if file_path.exists():
            dataset_dict[split] = Dataset.from_json(str(file_path))
            logger.info(f"加载 {split}: {len(dataset_dict[split])} 条")
    
    return DatasetDict(dataset_dict)


# ==================== 数据验证 ====================

def validate_chatml_format(dataset: DatasetDict) -> Dict[str, Any]:
    """
    验证ChatML格式数据的正确性
    
    Returns:
        验证报告
    """
    report = {
        "total_samples": 0,
        "valid_samples": 0,
        "invalid_samples": 0,
        "issues": [],
        "stats": {}
    }
    
    required_tags = ["<|im_start|>", "<|im_end|>", "system", "user", "assistant"]
    
    for split_name, split_dataset in dataset.items():
        for idx, sample in enumerate(split_dataset):
            report["total_samples"] += 1
            text = sample.get("text", "")
            
            # 检查必要标签
            missing_tags = [tag for tag in required_tags if tag not in text]
            
            if missing_tags:
                report["invalid_samples"] += 1
                report["issues"].append({
                    "split": split_name,
                    "index": idx,
                    "issue": f"缺少标签: {missing_tags}"
                })
            else:
                report["valid_samples"] += 1
    
    # 统计信息
    all_lengths = []
    for split_dataset in dataset.values():
        for sample in split_dataset:
            all_lengths.append(len(sample.get("text", "")))
    
    if all_lengths:
        report["stats"] = {
            "avg_length": sum(all_lengths) / len(all_lengths),
            "max_length": max(all_lengths),
            "min_length": min(all_lengths),
        }
    
    logger.info(f"验证结果: {report['valid_samples']}/{report['total_samples']} 有效")
    return report
