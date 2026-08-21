#!/usr/bin/env python
"""
pipeline_v4 干净 BF16 LoRA 训练脚本 (TRIZ 适配器 v4)。

与旧管线 (v1-v3) 的关键差异:
  - 纯 BF16 加载 + BF16 LoRA, 不使用 bitsandbytes 4-bit 量化
    (旧 4-bit 路径是 lora_B 全零事故链源头; 121GB 统一内存实测 FP16 加载 64.6GB, 够用)。
  - prompt/completion 数据集 → trl 1.5.1 自动启用 completion_only_loss
    (SFTTrainer.__init__ 中: args.completion_only_loss is None 时按数据集是否含
     "prompt"+"completion" 键自动判定; prompt-completion 标准(字符串)格式下
     TRL 分别 tokenize prompt 与 prompt+completion 构造 completion_mask,
     DataCollatorForLanguageModeling 将 prompt 部分 labels 置 -100;
     EOS 自动附加到 completion 末尾。formatting_func 与 completion_only_loss
     不兼容, 因此绝不传 formatting_func)。
  - BestCheckpointCallback: eval 创新低立即另存 best/ 防轮转;
    eval_steps == save_steps 强制相等 (config 校验)。
  - 发货只复制 Trainer 落盘文件, 绝不保存训练内存态;
    发货后完整验证 (lora_B 非零 + BF16 + sha256), PASSED 才写 adapter_info.json。

用法:
    venv_v5/bin/python pipeline_v4/src/train.py --config pipeline_v4/configs/train_v4.json
    venv_v5/bin/python pipeline_v4/src/train.py --config ... --dry-run   # 不加载模型
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 项目根目录 = pipeline_v4/src/ 的上两级; src/ 自身入 path 以 import 同级模块
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import checkpointing  # noqa: E402  (torch-free, 秒级 import)

DEFAULT_CONFIG = "pipeline_v4/configs/train_v4.json"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="pipeline_v4 BF16 LoRA 训练 (TRIZ v4)")
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help=f"训练配置 JSON (默认: {DEFAULT_CONFIG})")
    p.add_argument("--train-file", default=None, help="覆盖 config 中的 train_file")
    p.add_argument("--val-file", default=None, help="覆盖 config 中的 val_file")
    p.add_argument("--resume", default=None, help="从 checkpoint 目录恢复训练")
    p.add_argument("--dry-run", action="store_true",
                   help="只加载数据集 + 打印配置摘要后退出, 不加载模型、不碰 GPU")
    return p.parse_args()


def resolve(path_str: str) -> str:
    """相对路径一律相对项目根目录解析。"""
    p = Path(path_str)
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


def load_config(path: str) -> dict:
    with open(resolve(path) if not os.path.isabs(path) else path) as f:
        cfg = json.load(f)
    # eval/save 对齐是 BestCheckpointCallback 简单可靠实现的前提
    if cfg.get("eval_steps") != cfg.get("save_steps"):
        log(f"[致命] config 要求 eval_steps == save_steps "
            f"(当前 eval_steps={cfg.get('eval_steps')}, save_steps={cfg.get('save_steps')}); "
            f"v4 强制二者相等以保证每个 eval 步都有落盘 checkpoint 可归因")
        sys.exit(2)
    return cfg


def load_datasets(train_file: str, val_file: str):
    """加载 prompt/completion jsonl; 只保留 prompt/completion 两列 (subset 仅用于统计)。"""
    from datasets import Dataset

    for name, path in (("train", train_file), ("validation", val_file)):
        if not os.path.isfile(path):
            log(f"[致命] {name} 数据文件不存在: {path}")
            log("请先运行数据构建 (参见 pipeline_v4/configs/data_v4.json), 再启动训练")
            sys.exit(2)

    train_ds = Dataset.from_json(train_file)
    val_ds = Dataset.from_json(val_file)
    log(f"数据集加载: train={len(train_ds)} 条 ({train_file}) | "
        f"validation={len(val_ds)} 条 ({val_file})")

    for name, ds in (("train", train_ds), ("validation", val_ds)):
        cols = set(ds.column_names)
        missing = {"prompt", "completion"} - cols
        if missing:
            log(f"[致命] {name} 数据集缺少字段 {missing} (实际字段: {sorted(cols)}); "
                f"v4 要求 prompt/completion 格式 jsonl 以启用 completion-only loss")
            sys.exit(2)
        if "subset" in cols:
            counts = {}
            for s in ds["subset"]:
                counts[s] = counts.get(s, 0) + 1
            log(f"{name} subset 分布: {json.dumps(counts, ensure_ascii=False)}")
    # 只保留两列, 避免多余列 (如 subset) 进入 TRL 处理链路
    return (train_ds.select_columns(["prompt", "completion"]),
            val_ds.select_columns(["prompt", "completion"]))


def main():
    args = parse_args()
    cfg = load_config(args.config)

    train_file = resolve(args.train_file or cfg["train_file"])
    val_file = resolve(args.val_file or cfg["val_file"])
    output_dir = resolve(cfg["output_dir"])
    adapter_dir = resolve(cfg["adapter_output_dir"])
    base_model = resolve(cfg["base_model_path"])
    results_dir = PROJECT_ROOT / "results"

    train_ds, val_ds = load_datasets(train_file, val_file)

    # ---------- DRY-RUN: 数据 + 配置摘要, 不加载模型 ----------
    if args.dry_run:
        summary = {
            "base_model_path": base_model,
            "base_model_exists": os.path.isdir(base_model),
            "output_dir": output_dir,
            "adapter_output_dir": adapter_dir,
            "train_file": train_file,
            "val_file": val_file,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "data_format": "prompt/completion → completion_only_loss 自动启用",
            "sample_keys": train_ds.column_names,
            "epochs": cfg.get("num_train_epochs"),
            "max_length": cfg.get("max_length"),
            "eval_steps == save_steps": cfg.get("eval_steps"),
            "save_total_limit": cfg.get("save_total_limit"),
            "early_stopping_patience": cfg.get("early_stopping_patience"),
            "lora": cfg.get("lora"),
        }
        log("=== DRY-RUN 配置摘要 (未加载模型) ===")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    # ---------- 重依赖: compat 补丁必须先于一切模型加载 ----------
    import compat  # noqa: F401  (import 即打 WeightConverter 补丁)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        EarlyStoppingCallback,
        TrainerCallback,
    )
    from trl import SFTConfig, SFTTrainer

    t0 = time.time()
    if not os.path.isdir(base_model):
        log(f"[致命] 基座模型目录不存在: {base_model}")
        sys.exit(2)

    # ---------- 模型 (纯 BF16, 无量化) ----------
    log(f"加载 BF16 基座模型: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",       # v2 实战验证; 统一内存环境无需 auto 分片
        trust_remote_code=True,
    )
    log(f"模型加载完成, 显存 {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    model.config.use_cache = False

    # ---------- LoRA ----------
    lora_cfg = cfg["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    # 显式冻结基座 (peft 默认也会冻结, 双保险), 梯度检查点 + 输入梯度
    for p in model.parameters():
        p.requires_grad = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"可训练参数: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")

    # ---------- 训练参数 ----------
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.get("num_train_epochs", 4),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=cfg.get("learning_rate", 2e-4),
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        max_length=cfg.get("max_length", 2048),
        logging_steps=cfg.get("logging_steps", 10),
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg.get("save_total_limit", 8),
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        load_best_model_at_end=False,   # best 由 BestCheckpointCallback 另存, 不依赖内存回载
        bf16=True,                      # 纯 BF16 训练; bf16 不启用 GradScaler
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=cfg.get("report_to", "none"),
        seed=cfg.get("seed", 42),
        dataset_num_proc=cfg.get("dataset_num_proc", 4),
        completion_only_loss=None,      # 显式 None: 由 SFTTrainer 按 prompt/completion 自动判定
    )

    # ---------- Trainer (绝不传 formatting_func: 与 completion_only_loss 不兼容) ----------
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
    )
    log(f"completion_only_loss = {trainer.completion_only_loss} "
        f"(trl 1.5.1 按 prompt/completion 数据集自动判定)")
    if trainer.completion_only_loss is not True:
        log("[致命] completion_only_loss 未启用, 拒绝训练 (v4 硬性要求 completion-only loss)")
        sys.exit(2)

    best_callback_cls = checkpointing.build_best_checkpoint_callback(
        TrainerCallback, output_dir, log_fn=log)
    best_callback = best_callback_cls()
    trainer.add_callback(best_callback)
    patience = cfg.get("early_stopping_patience", 3)
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=patience))
    log(f"EarlyStopping patience={patience} | eval_steps=save_steps={cfg['eval_steps']} "
        f"| save_total_limit={cfg.get('save_total_limit', 8)}")

    log(f"=== 开始训练 v4 | epochs={cfg.get('num_train_epochs', 4)} "
        f"| resume={args.resume} ===")
    trainer.train(resume_from_checkpoint=args.resume)
    log(f"=== 训练结束, 耗时 {(time.time() - t0) / 3600:.2f} 小时 ===")

    # ---------- 训练日志持久化 (v1 逐步 loss 已永久丢失, 不能再犯) ----------
    results_dir.mkdir(exist_ok=True)
    log_path = results_dir / "train_log_v4.json"
    with open(log_path, "w") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
    log(f"训练日志已持久化: {log_path}")

    # ---------- 发货: 只复制磁盘文件, 绝不保存内存态 ----------
    best_dir = os.path.join(output_dir, "best")
    ship_source = None
    shipped_from = None
    best_eval_loss = best_callback.best_eval_loss
    best_eval_step = best_callback.best_eval_step

    if os.path.isdir(best_dir) and checkpointing.ADAPTER_FILES[0] in os.listdir(best_dir):
        ship_source = best_dir
        shipped_from = f"best/ (eval_step={best_eval_step}, eval_loss={best_eval_loss})"
    else:
        log("[警告] best/ 目录不可用, 回退到按 eval step 精确归因扫描幸存 checkpoint")
        best = checkpointing.find_best_checkpoint(output_dir)
        if best is not None:
            ship_source = best["path"]
            best_eval_loss = best["eval_loss"]
            best_eval_step = best["step"]
            shipped_from = f"{best['path']} (精确归因 eval_loss={best['eval_loss']:.6f} @ step={best['step']})"
        else:
            last = checkpointing.last_checkpoint(output_dir)
            if last is not None:
                ship_source = last["path"]
                shipped_from = f"{last['path']} (末步兜底, 无可归因 eval 记录)"
                log(f"[警告] 无任何可归因 eval 记录, 发货末步 checkpoint: {last['path']}")

    supplement = []
    last = checkpointing.last_checkpoint(output_dir)
    if last is not None:
        supplement.append(last["path"])

    validation = {"status": "FAILED", "checks": [], "sha256": {}, "tensor_stats": {}}
    copied = []
    if ship_source is None:
        checkpointing.log_marked("发货失败: 无任何可用 checkpoint", log)
    else:
        log(f"发货来源: {shipped_from}")
        copied = checkpointing.ship_adapter(ship_source, adapter_dir,
                                            supplement_dirs=supplement, log_fn=log)
        validation = checkpointing.validate_adapter_dir(adapter_dir)
        for c in validation["checks"]:
            log(f"  验证 [{c['status']}] {c['name']}: {c['detail']}")
        checkpointing.log_marked(f"适配器验证结果: {validation['status']}", log)

    # ---------- adapter_info.json: 元数据诚实 ----------
    final_train_loss = None
    for entry in reversed(trainer.state.log_history):
        if "loss" in entry:
            final_train_loss = entry["loss"]
            break
    max_steps = getattr(trainer.state, "max_steps", None)
    info = {
        "version": "v4",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": validation["status"],
        "base_model": cfg["base_model_path"],
        "adapter_dir": adapter_dir,
        "training": {
            "configured_epochs": cfg.get("num_train_epochs", 4),
            "actual_epochs": trainer.state.epoch,
            "actual_steps": trainer.state.global_step,
            "max_steps": max_steps,
            "early_stopped": bool(max_steps is not None
                                  and trainer.state.global_step < max_steps),
            "early_stopping_patience": patience,
            "final_train_loss": final_train_loss,
            "completion_only_loss": True,
            "precision": "bf16 (no quantization)",
            "lora": lora_cfg,
            "learning_rate": cfg.get("learning_rate", 2e-4),
            "lr_scheduler_type": cfg.get("lr_scheduler_type", "cosine"),
            "per_device_train_batch_size": cfg.get("per_device_train_batch_size", 1),
            "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps", 8),
            "max_length": cfg.get("max_length", 2048),
            "train_file": train_file,
            "val_file": val_file,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "train_log": str(log_path),
        },
        "best_checkpoint": {
            "eval_step": best_eval_step,
            "eval_loss": best_eval_loss,
            "promotion_history": best_callback.promotion_history,
        },
        "shipped_from": shipped_from,
        "shipped_files": copied,
        "sha256": validation["sha256"],
        "validation": validation,
    }
    if ship_source is not None:
        info_path = os.path.join(adapter_dir, "adapter_info.json")
        with open(info_path, "w") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        log(f"adapter_info.json 已写入: {info_path} (status={validation['status']})")
    else:
        log(f"adapter_info (未发货, 仅打印): {json.dumps(info, ensure_ascii=False, indent=2)}")

    print(json.dumps({
        "status": validation["status"],
        "adapter_dir": adapter_dir if ship_source else None,
        "best_eval_step": best_eval_step,
        "best_eval_loss": best_eval_loss,
        "actual_steps": trainer.state.global_step,
    }, ensure_ascii=False, indent=2))

    if validation["status"] != "PASSED":
        checkpointing.log_marked("v4 训练流程结束: 验证 FAILED, 退出码 1", log)
        sys.exit(1)
    checkpointing.log_marked("v4 训练流程结束: 验证 PASSED, 适配器已发货", log)


if __name__ == "__main__":
    main()
