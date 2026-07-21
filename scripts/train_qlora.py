"""
QLoRA 训练脚本 (DGX Spark) — 从 notebooks/04_worked.ipynb 提炼的实战路径

与 2026-06-19 成功运行完全一致的流程：
- 4-bit NF4 量化加载基座
- get_peft_model + 显式 12 模块 target_modules
- SFTTrainer + formatting_func (无 packing / 无 data_collator)
- fp16/bf16 均关闭 (避免 GradScaler + BF16 冲突)
- CheckpointValidationCallback + save_adapter_only (SHA-256 元数据)
- 注册到 pipeline_state

2026-07-20 复盘修复:
- EarlyStoppingCallback(patience=3) + eval_steps=100 + epochs=4
- SFTConfig 显式 max_length=2048 (TRL v1 已移除 SFTTrainer.max_seq_length, 防默认静默截断)
- assistant_only_loss: 先检测 chat template {% generation %} 标记, 不含则警告并回退全文 loss
- 训练日志持久化: results/train_log_<run_name>.json (v1 逐步 loss 已永久丢失, 不能再犯)
- 发货 best checkpoint (按 eval_loss 扫描 checkpoints 目录), 无 eval 记录回退末步并警告
- --dry-run: 只装配参数 + 加载数据集 + 打印配置摘要, 不加载模型

2026-07-21 合并 DGX /tmp/train_v2.py (v2 训练实际运行脚本, 2116 步成功) 的实战要素:
- 手动 kbit 准备替代 prepare_model_for_kbit_training: peft 0.19 会把全部 bf16
  参数转 fp32, 本模型 MoE 专家未量化 (33.25B), 全量转换需 ~133GB 必然 OOM
  (crash1/crash2 实锤), 因此只冻结 + 梯度检查点 + enable_input_require_grads + LoRA
- drop_shard_pagecache(): posix_fadvise(DONTNEED) 清理模型分片页缓存 (统一内存环境)
- device_map="cuda:0" (v2 实战验证, 替代 auto)
- 兼容 text-only jsonl (v2 数据实际 schema): 有 instruction/output 用 formatting_func,
  仅 text 字段时交由 TRL 直接使用

用法 (DGX Spark):
    # v2 数据集训练 (默认路径)
    venv_v5/bin/python scripts/train_qlora.py --run-name v2

    # 干跑验证 (不加载模型)
    venv_v5/bin/python scripts/train_qlora.py --run-name v2 --dry-run

    # 断点续训
    venv_v5/bin/python scripts/train_qlora.py --run-name v2 \
        --resume checkpoints/qlora_triz_v2/checkpoint-XXX
"""

import argparse
import ctypes
import gc
import glob
import inspect
import json
import os
import re
import sys
import time
from datetime import datetime

# 项目根目录 = 脚本所在目录的父目录 (替代硬编码 /home/meerkat/mongoose_ai)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MAX_SEQ_LENGTH = 2048  # corpus 样本按 2048-token 目标生成; TRL v1 默认 1024 会静默截断


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def chat_template_has_generation_marker(tokenizer) -> bool:
    """检测 tokenizer chat template 是否含 {% generation %} 标记 (assistant_only_loss 的前提)"""
    template = getattr(tokenizer, "chat_template", None) or ""
    return bool(re.search(r"\{%-?\s*generation\s*-?%\}", template))


def drop_shard_pagecache(model_path):
    """posix_fadvise(DONTNEED) 清理模型分片页缓存 (DGX 统一内存环境, 来自 /tmp/train_v2.py 实战)"""
    import torch

    libc = ctypes.CDLL("libc.so.6")
    for f in glob.glob(os.path.join(model_path, "*.safetensors")):
        fd = os.open(f, os.O_RDONLY)
        libc.posix_fadvise(fd, 0, 0, 4)  # POSIX_FADV_DONTNEED
        os.close(fd)
    gc.collect()
    torch.cuda.empty_cache()


def find_best_checkpoint(ckpt_dir):
    """
    按 eval_loss 扫描 checkpoints 目录找最优 (参考 04_worked.ipynb cell 16)
    返回 (best_ckpt_path, best_eval_loss); 无 eval 记录返回 (None, None)
    """
    best_loss = float("inf")
    best_ckpt = None
    if os.path.isdir(ckpt_dir):
        for ckpt in sorted(os.listdir(ckpt_dir)):
            if not ckpt.startswith("checkpoint-"):
                continue
            state_path = os.path.join(ckpt_dir, ckpt, "trainer_state.json")
            if not os.path.exists(state_path):
                continue
            with open(state_path) as f:
                state = json.load(f)
            for entry in state.get("log_history", []):
                if "eval_loss" in entry and entry["eval_loss"] < best_loss:
                    best_loss = entry["eval_loss"]
                    best_ckpt = ckpt
    if best_ckpt is None:
        return None, None
    return os.path.join(ckpt_dir, best_ckpt), best_loss


def last_train_loss(log_history):
    """从 log_history 末尾向前扫描最后一条含 loss 的条目 (末条可能是 eval_loss 而非 train loss)"""
    for entry in reversed(log_history or []):
        if "loss" in entry:
            return entry["loss"]
    return "N/A"


def build_training_args(cfg, ckpt_dir, epochs, assistant_only_loss=False):
    """
    构造训练参数: 优先 TRL SFTConfig (显式 max_length=2048),
    API 不匹配时防御性回退 TrainingArguments 并打印警告
    """
    try:
        from transformers import TrainingArguments
    except ImportError:
        log("[警告] transformers 不可用, 无法构造训练参数")
        return None

    common = dict(
        output_dir=ckpt_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        eval_steps=cfg["eval_steps"],
        save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=False,
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=cfg["greater_is_better"],
        report_to=cfg["report_to"],
        fp16=False,                      # 04_worked: 避免 GradScaler + BF16 冲突
        bf16=False,
        optim=cfg["optim"],
        save_strategy="steps",
        eval_strategy="steps",           # transformers v5 需显式
        logging_strategy="steps",
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    try:
        from trl import SFTConfig
        params = inspect.signature(SFTConfig.__init__).parameters
        kwargs = dict(common)
        if "max_length" in params:
            kwargs["max_length"] = MAX_SEQ_LENGTH
        elif "max_seq_length" in params:
            kwargs["max_seq_length"] = MAX_SEQ_LENGTH
        else:
            log("[警告] SFTConfig 无 max_length/max_seq_length 参数, 序列长度未被显式设置")
        if assistant_only_loss:
            if "assistant_only_loss" in params:
                kwargs["assistant_only_loss"] = True
            else:
                log("[警告] SFTConfig 不支持 assistant_only_loss, 回退全文 loss")
        args = SFTConfig(**kwargs)
        log(f"训练参数: SFTConfig max_length={kwargs.get('max_length', kwargs.get('max_seq_length', 'N/A'))} "
            f"assistant_only_loss={kwargs.get('assistant_only_loss', False)}")
        return args
    except Exception as e:
        log(f"[警告] SFTConfig 不可用 ({type(e).__name__}: {e}), "
            f"回退 TrainingArguments (max_length 未显式设置)")
        return TrainingArguments(**common)


def main():
    from config import (  # 延迟导入: 保证模块 import 期不依赖 config/torch
        BASE_MODEL, MODELS_DIR, CHECKPOINTS_DIR, DATA_DIR, RESULTS_DIR,
        QLORA_CONFIG, DATA_CONFIG,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="v2", help="run 标识: checkpoints/qlora_triz_<name>, models/meerkat_triz_adapter_<name>")
    parser.add_argument("--train-file", default=os.path.join(DATA_DIR, "processed", "v2_train.jsonl"))
    parser.add_argument("--val-file", default=os.path.join(DATA_DIR, "processed", "v2_validation.jsonl"))
    parser.add_argument("--epochs", type=int, default=QLORA_CONFIG["training"]["num_train_epochs"])
    parser.add_argument("--resume", default=None, help="从 checkpoint 目录恢复训练")
    parser.add_argument("--dry-run", action="store_true",
                        help="只装配参数+加载数据集+打印配置摘要后退出, 不加载模型")
    args = parser.parse_args()

    cfg = QLORA_CONFIG["training"]
    ckpt_dir = os.path.join(CHECKPOINTS_DIR, f"qlora_triz_{args.run_name}")
    adapter_dir = os.path.join(MODELS_DIR, f"meerkat_triz_adapter_{args.run_name}")
    t0 = time.time()

    # ---------- 数据 ----------
    from datasets import Dataset, DatasetDict

    log(f"加载数据集: {args.train_file} / {args.val_file}")
    dataset = DatasetDict({
        "train": Dataset.from_json(args.train_file),
        "validation": Dataset.from_json(args.val_file),
    })
    log(f"train: {len(dataset['train'])} 条 | validation: {len(dataset['validation'])} 条")
    sample = dataset["train"][0]
    # 兼容两种 schema: instruction/input/output (v1 corpus SFT) 或 text-only (v2 实际 schema)
    has_instruct_fields = bool(sample.get("instruction") and sample.get("output"))
    if not has_instruct_fields:
        assert sample.get("text"), \
            f"样本既无 instruction/output 也无 text 字段 (字段: {list(sample.keys())})"
        log(f"数据模式: text-only ChatML (字段: {list(sample.keys())}), 由 TRL 直接使用 text 列")
    else:
        log(f"数据模式: instruction/output (字段: {list(sample.keys())}), 使用 formatting_func 构造 ChatML")

    # ---------- DRY-RUN: 装配参数 + 打印配置摘要后退出 (不加载模型) ----------
    if args.dry_run:
        training_args = build_training_args(cfg, ckpt_dir, args.epochs)
        summary = {
            "run_name": args.run_name,
            "ckpt_dir": ckpt_dir,
            "adapter_dir": adapter_dir,
            "train_file": args.train_file,
            "val_file": args.val_file,
            "train_samples": len(dataset["train"]),
            "val_samples": len(dataset["validation"]),
            "data_mode": "instruction/output + formatting_func" if has_instruct_fields else "text-only",
            "epochs": args.epochs,
            "training_args_type": type(training_args).__name__ if training_args else "构造失败",
            "max_length": getattr(training_args, "max_length",
                                  getattr(training_args, "max_seq_length", "未设置")) if training_args else "N/A",
            "assistant_only_loss": "训练时按 chat template {% generation %} 标记自动检测",
            "early_stopping_patience": 3,
        }
        log("=== DRY-RUN 配置摘要 (未加载模型) ===")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    # ---------- 重依赖 (仅真实训练路径需要) ----------
    import torch
    from peft import PeftModel, get_peft_model
    from transformers import EarlyStoppingCallback
    from trl import SFTTrainer
    from utils.training_utils import (
        load_model_and_tokenizer,
        setup_qlora_config,
        CheckpointValidationCallback,
        save_adapter_only,
    )
    from utils.pipeline_state import PipelineState

    # ---------- 模型 (4-bit) ----------
    model_path = os.path.join(MODELS_DIR, BASE_MODEL.split("/")[-1])
    log(f"加载 4-bit 量化模型: {model_path}")
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=model_path,
        quantization_config=QLORA_CONFIG["quantization"],
        device_map="cuda:0",             # v2 实战验证 (/tmp/train_v2.py); 统一内存环境无需 auto 分片
        trust_remote_code=True,
    )
    log(f"模型加载完成, 显存 {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    model.config.use_cache = False
    drop_shard_pagecache(model_path)
    log(f"pagecache清理后, 显存 {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # ---------- QLoRA ----------
    lora_config = setup_qlora_config(
        r=QLORA_CONFIG["lora"]["r"],
        lora_alpha=QLORA_CONFIG["lora"]["lora_alpha"],
        target_modules=QLORA_CONFIG["lora"]["target_modules"],
        lora_dropout=QLORA_CONFIG["lora"]["lora_dropout"],
        use_rslora=QLORA_CONFIG["lora"].get("use_rslora", False),
    )

    # 手动 kbit 准备 (替代 prepare_model_for_kbit_training, 来自 /tmp/train_v2.py 实战):
    # peft 0.19 会把全部 bf16 参数转 fp32, 本模型 MoE 专家未量化 (33.25B),
    # 全量转换需 ~133GB 必然 OOM (v2 crash1/crash2 实锤), 因此只冻结 + 梯度检查点 + LoRA。
    log("手动准备QLoRA模型 (跳过 prepare_model_for_kbit_training 的 fp32 转换)...")
    for p in model.parameters():
        p.requires_grad = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"可训练参数: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")

    # ---------- 训练参数 (SFTConfig 显式 max_length=2048 + assistant_only_loss 检测) ----------
    use_assistant_only_loss = chat_template_has_generation_marker(tokenizer)
    if use_assistant_only_loss:
        log("chat template 含 {% generation %} 标记, 启用 assistant_only_loss (仅 assistant 回复计 loss)")
    else:
        log("[警告] chat template 不含 {% generation %} 标记, 无法启用 assistant_only_loss, 回退全文 loss")
    training_args = build_training_args(cfg, ckpt_dir, args.epochs,
                                        assistant_only_loss=use_assistant_only_loss)

    # ---------- Trainer (无 packing / 无 data_collator; max_length 由 SFTConfig 承载) ----------
    trainer_kwargs = dict(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
    )
    if has_instruct_fields:
        system_message = DATA_CONFIG["chatml"]["system_message"]

        def formatting_func(example):
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            output = example.get("output", "")
            sys_msg = example.get("system", system_message)
            full_question = f"{instruction}\n\n{input_text}" if input_text else instruction
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": full_question},
                {"role": "assistant", "content": output},
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        trainer_kwargs["formatting_func"] = formatting_func
    # text-only 模式: 不传 formatting_func, TRL 直接使用数据集的 text 列

    trainer = SFTTrainer(**trainer_kwargs)

    checkpoint_callback = CheckpointValidationCallback(
        tokenizer=tokenizer,
        test_prompt="请解释TRIZ的分割原理及其应用场景。",
    )
    trainer.add_callback(checkpoint_callback)
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

    log(f"=== 开始训练: {args.run_name} | epochs={args.epochs} | resume={args.resume} ===")
    trainer.train(resume_from_checkpoint=args.resume)
    log(f"=== 训练完成, 耗时 {(time.time() - t0) / 3600:.1f} 小时 ===")

    # ---------- 训练日志持久化 (v1 逐步 loss 因 notebook 未持久化已永久丢失, 不能再犯) ----------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, f"train_log_{args.run_name}.json")
    with open(log_path, "w") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
    log(f"训练日志已持久化: {log_path}")

    # ---------- 发货 best checkpoint 而非末步内存态 ----------
    best_ckpt, best_eval = find_best_checkpoint(ckpt_dir)
    ship_model = model
    if best_ckpt:
        log(f"最佳 checkpoint: {best_ckpt} (eval_loss={best_eval:.4f}), 从磁盘加载该适配器发货")
        try:
            # unload() 移除训练态 LoRA 模块恢复干净基座, 再从磁盘挂载 best checkpoint 适配器
            ship_model = PeftModel.from_pretrained(model.unload(), best_ckpt)
        except Exception as e:
            log(f"[警告] best checkpoint 加载失败 ({type(e).__name__}: {e}), 回退保存末步内存态")
            ship_model = model
    else:
        log("[警告] 未找到含 eval_loss 记录的 checkpoint, 回退保存末步内存态")

    # ---------- 保存适配器 + 注册 ----------
    training_metadata = {
        "training_steps": trainer.state.global_step,
        "num_train_epochs": args.epochs,
        "learning_rate": cfg["learning_rate"],
        "final_loss": last_train_loss(trainer.state.log_history),
        "best_eval_loss": best_eval if best_eval is not None else "N/A",
        "best_checkpoint": best_ckpt or "N/A",
        "shipped_from": best_ckpt if (best_ckpt and ship_model is not model) else "末步内存态",
        "train_log": log_path,
        "checkpoint_validation": checkpoint_callback.validation_results,
        "train_file": args.train_file,
        "run_name": args.run_name,
    }
    save_adapter_only(ship_model, tokenizer, adapter_dir, metadata=training_metadata)
    log(f"适配器已保存: {adapter_dir}")

    state = PipelineState()
    state.register(
        name=f"adapter_checkpoint_{args.run_name}",
        path=adapter_dir,
        artifact_type="model",
        metadata={
            "base_model": BASE_MODEL,
            "training_steps": trainer.state.global_step,
            "final_loss": training_metadata["final_loss"],
            "best_eval_loss": training_metadata["best_eval_loss"],
            "adapter_dir": adapter_dir,
        },
    )
    log("已注册到 pipeline_state")

    passed = sum(1 for r in checkpoint_callback.validation_results if r.get("status") == "PASSED")
    log(f"Checkpoint 验证: {passed}/{len(checkpoint_callback.validation_results)} PASSED")
    print(json.dumps({
        "run_name": args.run_name,
        "adapter_dir": adapter_dir,
        "best_checkpoint": training_metadata["best_checkpoint"],
        "train_log": log_path,
        "steps": trainer.state.global_step,
        "hours": round((time.time() - t0) / 3600, 2),
        "checkpoint_validation_passed": f"{passed}/{len(checkpoint_callback.validation_results)}",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
