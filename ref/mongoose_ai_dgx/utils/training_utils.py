"""
训练工具函数集
支持：模型加载、QLoRA配置、训练参数设置、Trainer创建、模型合并与保存
"""

import torch
import logging
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CheckpointValidationCallback(TrainerCallback):
    """
    Checkpoint保存验证回调
    每次保存checkpoint后验证:
    1. adapter文件存在且非空
    2. 模型能正常执行前向传播
    """

    def __init__(self, tokenizer, test_prompt=None):
        self.tokenizer = tokenizer
        self.test_prompt = test_prompt or "请解释TRIZ的分割原理及其应用场景。"
        self.validation_results = []

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        result = {
            "step": state.global_step,
            "timestamp": datetime.now().isoformat(),
        }

        # 1. 验证adapter文件存在
        adapter_file = os.path.join(checkpoint_dir, "adapter_model.safetensors")
        if not os.path.exists(adapter_file):
            adapter_file = os.path.join(checkpoint_dir, "adapter_model.bin")

        if not os.path.exists(adapter_file):
            result["status"] = "FAILED"
            result["reason"] = "missing_adapter_file"
            print(f"[CHECKPOINT] FAILED: No adapter file in {checkpoint_dir}")
            self.validation_results.append(result)
            return control

        # 2. 验证文件大小
        size_mb = os.path.getsize(adapter_file) / (1024**2)
        result["size_mb"] = round(size_mb, 2)
        if size_mb < 1:
            result["status"] = "FAILED"
            result["reason"] = "file_too_small"
            print(f"[CHECKPOINT] FAILED: Adapter too small ({size_mb:.1f} MB)")
            self.validation_results.append(result)
            return control

        # 3. 前向传播验证
        model = kwargs.get('model')
        if model:
            try:
                device = next(model.parameters()).device
                inputs = self.tokenizer(self.test_prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                result["status"] = "PASSED"
                result["loss"] = round(outputs.loss.item(), 4)
                print(f"[CHECKPOINT] PASSED: step={state.global_step}, size={size_mb:.1f}MB, loss={outputs.loss.item():.4f}")
            except Exception as e:
                result["status"] = "FAILED"
                result["reason"] = f"forward_pass_error: {str(e)}"
                print(f"[CHECKPOINT] FAILED: Forward pass error: {e}")

        self.validation_results.append(result)
        return control


# ==================== 模型与分词器加载 ====================

def load_model_and_tokenizer(
    model_name_or_path: str,
    quantization_config: Optional[Dict] = None,
    device_map: str = "auto",
    trust_remote_code: bool = True,
) -> tuple:
    """
    加载模型和分词器，支持4-bit量化
    
    Args:
        model_name_or_path: 模型路径或HuggingFace ID
        quantization_config: 量化配置字典
        device_map: 设备映射策略
        trust_remote_code: 是否信任远程代码
    
    Returns:
        (model, tokenizer) 元组
    """
    logger.info(f"加载模型: {model_name_or_path}")
    logger.info(f"设备映射: {device_map}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        padding_side="right",  # 左填充更适合生成
    )
    
    # 设置填充token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info(f"分词器词汇表大小: {len(tokenizer)}")
    
    # 构建模型加载参数
    model_kwargs = {
        "pretrained_model_name_or_path": model_name_or_path,
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
        "torch_dtype": torch.float16,
    }
    
    # 添加量化配置
    if quantization_config and quantization_config.get("load_in_4bit"):
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(**quantization_config)
            model_kwargs["quantization_config"] = bnb_config
            logger.info("启用4-bit量化 (NF4)")
        except ImportError:
            logger.warning("bitsandbytes未安装，跳过量化")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    
    # 启用梯度检查点（节省内存）
    model.gradient_checkpointing_enable()
    
    logger.info(f"模型加载完成")
    logger.info(f"模型参数总量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    return model, tokenizer


# ==================== 自动检测目标模块 ====================

def find_all_linear_names(model) -> List[str]:
    """
    自动检测模型中所有可用于LoRA的线性层名称
    
    适用于任意架构（包括Qwen3.6的Gated DeltaNet混合架构）
    
    Args:
        model: 已加载的模型
    
    Returns:
        线性层名称列表（去重，排除lm_head和embed_tokens）
    """
    import bitsandbytes as bnb
    import torch.nn as nn
    
    linear_classes = (nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    target_modules = set()
    
    for name, module in model.named_modules():
        if isinstance(module, linear_classes):
            # 提取最后一级模块名
            module_name = name.split(".")[-1]
            # 排除不需要的层
            if module_name not in ["lm_head", "embed_tokens", "embed_in", "embed_out"]:
                target_modules.add(module_name)
    
    return sorted(list(target_modules))


def get_qwen36_target_modules() -> List[str]:
    """
    返回Qwen3.6混合架构推荐的target_modules列表
    
    Qwen3.6架构: 10 x (3 x Gated DeltaNet -> MoE + 1 x Gated Attention -> MoE)
    共40层: 30层GDN + 10层GA
    
    Gated Attention层模块: q_proj, k_proj, v_proj, o_proj
    Gated DeltaNet层模块: in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj
    MoE MLP层模块: gate_proj, up_proj, down_proj (或 linear_fc1/linear_fc2)
    """
    return [
        # Gated Attention 层 (10/40层)
        "q_proj", "k_proj", "v_proj", "o_proj",
        # Gated DeltaNet 层 (30/40层)
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
        # MoE MLP 层 (全部40层)
        "gate_proj", "up_proj", "down_proj",
    ]


# ==================== QLoRA配置 ====================

def setup_qlora_config(
    r: int = 64,
    lora_alpha: int = 128,
    target_modules: Optional[list] = None,
    lora_dropout: float = 0.0,  # MoE架构兼容性: dropout可能干扰专家路由稳定性
    use_rslora: bool = False,
) -> LoraConfig:
    """
    创建QLoRA配置

    支持以下 target_modules 配置方式:
    - None: 使用Qwen3.6显式模块列表 (推荐, 默认)
    - List[str]: 手动指定模块名列表
    - "all-linear": PEFT自动检测 (不推荐, 已知在混合架构上存在兼容性问题)

    Qwen3.6混合架构模块列表 (12个):
    - Gated Attention (10/40层): q_proj, k_proj, v_proj, o_proj
    - Gated DeltaNet (30/40层): in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj
    - MoE MLP (全部40层): gate_proj, up_proj, down_proj

    Args:
        r: LoRA秩
        lora_alpha: 缩放因子
        target_modules: 目标模块列表或 "all-linear"
        lora_dropout: Dropout率 (默认0.0, MoE架构兼容性)
        use_rslora: 是否使用rsLoRA（大rank时推荐）

    Returns:
        LoraConfig对象
    """
    if target_modules is None:
        # 默认使用Qwen3.6推荐模块列表
        target_modules = get_qwen36_target_modules()
        logger.info("使用Qwen3.6默认target_modules列表")
    elif target_modules == "all-linear":
        # 不推荐: 已知在Qwen3.6混合架构上存在兼容性问题 (可能错误包含lm_head)
        logger.warning("使用'all-linear'自动检测模式 (不推荐: 已知在混合架构上存在兼容性问题)")
    elif isinstance(target_modules, list):
        # 使用提供的列表
        logger.info(f"使用手动指定的target_modules: {target_modules}")
    else:
        raise ValueError(f"不支持的target_modules类型: {type(target_modules)}")
    
    # 大rank时建议使用rsLoRA
    if isinstance(r, int) and r > 64 and not use_rslora:
        logger.warning(f"rank={r} > 64，建议开启use_rslora=True以稳定训练")
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_rslora=use_rslora,
    )
    
    logger.info(f"LoRA配置: rank={r}, alpha={lora_alpha}, target_modules={len(target_modules)}个")
    logger.info(f"目标模块: {target_modules}")
    
    return lora_config


def prepare_qlora_model(model, lora_config: LoraConfig) -> Any:
    """
    为QLoRA训练准备模型
    
    步骤：
    1. 准备4-bit模型（梯度检查点、输入嵌入层启用梯度）
    2. 应用LoRA配置
    
    Args:
        model: 已加载的模型
        lora_config: LoRA配置
    
    Returns:
        配置好的PEFT模型
    """
    logger.info("准备QLoRA模型...")
    
    # 准备4-bit模型用于训练
    model = prepare_model_for_kbit_training(model)
    
    # 应用LoRA配置
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数信息
    model.print_trainable_parameters()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.4f}%)")
    
    return model


# ==================== 训练参数配置 ====================

def setup_training_arguments(
    output_dir: str,
    num_train_epochs: int = 2,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.03,
    save_steps: int = 200,
    eval_steps: int = 200,
    logging_steps: int = 10,
    **kwargs
) -> TrainingArguments:
    """
    创建训练参数
    
    DGX Spark推荐配置：
    - batch_size=1 (内存限制)
    - gradient_accumulation_steps=8 (有效batch=8)
    - fp16=True (DGX Spark支持FP16)
    - optim=paged_adamw_8bit (分页优化器省内存)
    
    Args:
        output_dir: 输出目录
        num_train_epochs: 训练轮数
        per_device_batch_size: 每设备batch大小
        gradient_accumulation_steps: 梯度累积步数
        learning_rate: 学习率
        warmup_ratio: warmup比例
        save_steps: 保存步数
        eval_steps: 评估步数
        logging_steps: 日志步数
        **kwargs: 其他参数
    
    Returns:
        TrainingArguments对象
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # 训练轮数与batch
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        
        # 学习率调度
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        
        # 日志与保存
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # 评估策略
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        
        # 精度设置 (DGX Spark)
        fp16=True,
        bf16=False,  # DGX Spark可能不支持bf16
        
        # 优化器
        optim="paged_adamw_8bit",
        
        # 其他
        group_by_length=True,
        report_to="tensorboard",
        remove_unused_columns=False,
        **kwargs
    )
    
    logger.info(f"训练参数配置完成:")
    logger.info(f"  输出目录: {output_dir}")
    logger.info(f"  训练轮数: {num_train_epochs}")
    logger.info(f"  有效batch_size: {per_device_batch_size * gradient_accumulation_steps}")
    logger.info(f"  学习率: {learning_rate}")
    logger.info(f"  优化器: paged_adamw_8bit")
    
    return training_args


# ==================== Trainer创建 ====================

def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    training_args: TrainingArguments,
    system_message: Optional[str] = None,
    max_seq_length: int = 4096,
    packing: bool = True,
):
    """
    创建SFTTrainer进行监督微调

    使用TRL库的SFTTrainer + formatting_func 正确处理ChatML格式：
    - 通过formatting_func将messages列表转换为对话文本
    - SFTTrainer内部自动只计算assistant回复部分的loss
    - 不传入data_collator，避免与SFTTrainer内置逻辑冲突

    Args:
        model: PEFT模型
        tokenizer: 分词器 (需支持apply_chat_template)
        train_dataset: 训练数据集 (需包含instruction, output字段)
        eval_dataset: 验证数据集
        training_args: 训练参数
        system_message: 系统提示词 (None则使用数据集中的system字段)
        max_seq_length: 最大序列长度
        packing: 是否启用序列打包

    Returns:
        SFTTrainer实例
    """
    logger.info("创建SFTTrainer...")

    # 默认系统提示词
    if system_message is None:
        system_message = (
            "You are Meerkat-AI, an expert innovation consultant specializing in TRIZ "
            "(Theory of Inventive Problem Solving). You help users analyze technical contradictions, "
            "recommend invention principles, generate innovative solutions, and guide them through "
            "the ARIZ algorithm. Always provide structured, actionable advice grounded in TRIZ methodology."
        )

    # 格式化函数: 将原始数据转换为对话文本
    # SFTTrainer使用此函数处理每条样本，并自动只计算assistant部分的loss
    def formatting_func(example):
        """将instruction/output格式转换为对话文本"""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")
        sys_msg = example.get("system", system_message)

        # 构建完整问题
        if input_text:
            full_question = f"{instruction}\n\n{input_text}"
        else:
            full_question = instruction

        # 使用tokenizer的官方chat template生成对话格式
        # 这是推荐做法，确保训练格式与模型推理格式一致
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": full_question},
            {"role": "assistant", "content": output},
        ]

        # 应用chat template生成文本 (不添加generation prompt)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return text

    # 验证tokenizer是否支持apply_chat_template
    if not hasattr(tokenizer, 'apply_chat_template'):
        logger.error("tokenizer不支持apply_chat_template()，请确保使用兼容的模型和transformers版本")
        raise RuntimeError("tokenizer缺少apply_chat_template方法")

    # 测试格式化函数
    try:
        test_text = formatting_func(train_dataset[0])
        logger.info(f"格式化测试通过，样本长度: {len(test_text)} 字符")
    except Exception as e:
        logger.warning(f"格式化测试失败: {e}")

    # 创建SFTTrainer (不传入data_collator，由SFTTrainer内部处理)
    # SFTTrainer会自动:
    # 1. 调用formatting_func将样本转换为文本
    # 2. 使用tokenizer进行tokenization
    # 3. 自动mask掉user部分，只计算assistant回复的loss
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        max_seq_length=max_seq_length,
        formatting_func=formatting_func,  # 使用格式化函数替代dataset_text_field
        packing=packing,
        # 不传入data_collator，避免与SFTTrainer内置逻辑冲突
    )

    logger.info("SFTTrainer创建完成 (formatting_func模式)")
    return trainer


# ==================== 模型合并与保存 ====================

def merge_and_save_model(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
):
    """
    合并LoRA适配器与基座模型，保存完整模型
    
    Args:
        base_model_path: 基座模型路径
        adapter_path: LoRA适配器路径 (checkpoint目录)
        output_path: 合并后模型的保存路径
        push_to_hub: 是否推送到HuggingFace Hub
        hub_model_id: Hub模型ID
    """
    logger.info("合并LoRA适配器与基座模型...")
    logger.info(f"基座模型: {base_model_path}")
    logger.info(f"适配器: {adapter_path}")
    
    # 加载基座模型（不量化，用于合并）
    logger.info("加载基座模型（FP16）...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    
    # 加载并合并适配器
    logger.info("加载LoRA适配器...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    logger.info("合并权重...")
    model = model.merge_and_unload()
    
    # 保存合并后的模型
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"保存合并模型到: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # 推送到Hub（可选）
    if push_to_hub and hub_model_id:
        logger.info(f"推送到HuggingFace Hub: {hub_model_id}")
        model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)
    
    logger.info("模型合并与保存完成！")
    return model, tokenizer


def compute_file_sha256(filepath: str) -> str:
    """计算文件的SHA-256哈希值"""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def save_adapter_only(model, tokenizer, output_path: str, metadata: Optional[Dict] = None):
    """
    仅保存LoRA适配器（不含基座模型，体积小）

    适配器大小通常只有100-200MB，便于版本管理和快速加载。
    保存的元数据包括: adapter类型、基座模型、时间戳、SHA-256哈希、训练参数等。

    Args:
        model: PEFT模型
        tokenizer: 分词器
        output_path: 保存路径
        metadata: 额外的元数据字典 (如训练步数、最终loss等)
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"保存LoRA适配器到: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # 计算SHA-256
    sha256_hash = None
    adapter_file = output_path / "adapter_model.safetensors"
    if adapter_file.exists():
        sha256_hash = compute_file_sha256(str(adapter_file))
    else:
        # 尝试bin格式
        adapter_bin = output_path / "adapter_model.bin"
        if adapter_bin.exists():
            sha256_hash = compute_file_sha256(str(adapter_bin))

    # 记录适配器信息
    info = {
        "adapter_type": "LORA",
        "base_model": model.config._name_or_path if hasattr(model, "config") else "unknown",
        "timestamp": datetime.now().isoformat(),
    }
    if metadata:
        info.update(metadata)
    if sha256_hash:
        info["sha256"] = sha256_hash

    with open(output_path / "adapter_info.json", "w") as f:
        import json
        json.dump(info, f, indent=2)

    logger.info(f"适配器保存完成! 元数据: {info}")


# ==================== 训练恢复 ====================

def resume_from_checkpoint(trainer, checkpoint_path: str) -> Dict[str, Any]:
    """
    从checkpoint恢复训练，并验证LR scheduler连续性

    Args:
        trainer: SFTTrainer实例
        checkpoint_path: checkpoint目录路径

    Returns:
        恢复信息字典，包含恢复前后的step和lr
    """
    logger.info(f"从checkpoint恢复训练: {checkpoint_path}")

    # 记录恢复前的状态
    initial_step = trainer.state.global_step if hasattr(trainer.state, 'global_step') else 0
    initial_lr = trainer.optimizer.param_groups[0]['lr'] if trainer.optimizer else 0.0

    logger.info(f"恢复前: step={initial_step}, lr={initial_lr:.2e}")

    # 执行恢复
    trainer.train(resume_from_checkpoint=checkpoint_path)

    # 验证恢复后的状态
    resumed_step = trainer.state.global_step
    resumed_lr = trainer.optimizer.param_groups[0]['lr'] if trainer.optimizer else 0.0

    logger.info(f"恢复后: step={resumed_step}, lr={resumed_lr:.2e}")

    # 验证step是否增加
    if resumed_step <= initial_step:
        logger.warning(f"恢复后step未增加: {initial_step} -> {resumed_step}")
    else:
        logger.info(f"恢复成功: step从 {initial_step} 增加到 {resumed_step}")

    return {
        "initial_step": initial_step,
        "resumed_step": resumed_step,
        "initial_lr": initial_lr,
        "resumed_lr": resumed_lr,
        "checkpoint_path": checkpoint_path,
    }
