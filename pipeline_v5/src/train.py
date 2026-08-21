#!/usr/bin/env python
"""
pipeline_v5 BF16 LoRA 训练脚本 (TRIZ 适配器 v5)。

基于 pipeline_v4/src/train.py 复制演进 (不覆盖 v4); 继承 v4 全部已验证资产:
  - 纯 BF16 加载 + BF16 LoRA (含 LoRA 参数 dtype 校正, v4 首轮发货 dtype 事故修复)
  - prompt/completion 数据集 → trl 1.5.1 自动启用 completion_only_loss + 启动断言
  - BestCheckpointCallback (eval 创新低立即另存 best/, eval_steps==save_steps 强制相等)
  - 发货只复制磁盘文件 + 完整验证 (lora_B 非零 + BF16 + sha256)

v5 相对 v4 的变更 (v5_优化微调方案.md §11.2):
  1. cosine horizon = ceil(D/8) * lr_horizon_epochs(=2) (2-epoch 设计);
     num_train_epochs=4 仅安全帽。实现: Trainer 初始化后用
     create_optimizer_and_scheduler(num_training_steps=horizon) 重建调度器,
     max_steps(若设)只限训练长度, 不缩 horizon。
  2. early_stopping_threshold=0.002 显式传入 EarlyStoppingCallback。
  3. max_length=2048 断言项 (SFTConfig.max_length==2048 否则退出)。
  4. optimizer 显式写入 (optim, 默认 adamw_torch, 不切 8bit)。
  5. loss 冒烟: decode 首 batch labels, 确认 prompt 区段全 -100 且
     completion 区段不含 ChatML 标记 (防 trl 自动判定规则变更的共模失效)。
  6. use_rslora 从配置读取 (P0 小扫臂); 记录进 adapter_info。
  7. 训练末尾: 末步 checkpoint 落盘 + 终点 eval (eval_steps 不对齐末步时
     也能拿到终点 eval_loss, BestCheckpointCallback 正常归因)。
  8. 记录训练时长与显存峰值 (torch.cuda.max_memory_allocated)。

退出码约定: 0=成功; 2=配置/数据致命错误; 3=断言/冒烟类失败 (扫描链遇 3 停链)。

用法:
    venv_v5/bin/python pipeline_v5/src/train.py --config pipeline_v5/configs/train_v5_base.json
    venv_v5/bin/python pipeline_v5/src/train.py --config ... --dry-run   # 不加载模型
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 项目根目录 = pipeline_v5/src/ 的上两级; src/ 自身入 path 以 import 同级模块
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import checkpointing  # noqa: E402  (torch-free, 秒级 import)

DEFAULT_CONFIG = "pipeline_v5/configs/train_v5_base.json"

# 断言/冒烟类失败退出码 (扫描链据此停链)
EXIT_ASSERT = 3


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def fatal(msg: str, code: int = 2) -> None:
    log(f"[致命] {msg}")
    sys.exit(code)


def parse_args():
    p = argparse.ArgumentParser(description="pipeline_v5 BF16 LoRA 训练 (TRIZ v5)")
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
        fatal(f"config 要求 eval_steps == save_steps "
              f"(当前 eval_steps={cfg.get('eval_steps')}, save_steps={cfg.get('save_steps')}); "
              f"v5 强制二者相等以保证每个 eval 步都有落盘 checkpoint 可归因")
    return cfg


def load_datasets(train_file: str, val_file: str):
    """加载 prompt/completion jsonl; 只保留 prompt/completion 两列 (subset 仅用于统计)。"""
    from datasets import Dataset

    for name, path in (("train", train_file), ("validation", val_file)):
        if not os.path.isfile(path):
            fatal(f"{name} 数据文件不存在: {path}; 请先运行数据构建, 再启动训练")

    train_ds = Dataset.from_json(train_file)
    val_ds = Dataset.from_json(val_file)
    log(f"数据集加载: train={len(train_ds)} 条 ({train_file}) | "
        f"validation={len(val_ds)} 条 ({val_file})")

    for name, ds in (("train", train_ds), ("validation", val_ds)):
        cols = set(ds.column_names)
        missing = {"prompt", "completion"} - cols
        if missing:
            fatal(f"{name} 数据集缺少字段 {missing} (实际字段: {sorted(cols)}); "
                  f"v5 要求 prompt/completion 格式 jsonl 以启用 completion-only loss")
        if "subset" in cols:
            counts = {}
            for s in ds["subset"]:
                counts[s] = counts.get(s, 0) + 1
            log(f"{name} subset 分布: {json.dumps(counts, ensure_ascii=False)}")
    # 只保留两列, 避免多余列 (如 subset) 进入 TRL 处理链路
    return (train_ds.select_columns(["prompt", "completion"]),
            val_ds.select_columns(["prompt", "completion"]))


def smoke_check_prompt_mask(trainer, tokenizer, log_fn=log) -> None:
    """loss 冒烟 (§11.2-5): decode 首 batch labels, 确认 prompt 区段全 -100。

    断言:
      1. labels 中同时存在 -100 与被监督 token;
      2. 开头连续 -100 区段 (prompt 区段) decode 后含 ChatML 标记 (<|im_start|>);
      3. 被监督区段 (completion) decode 后不含 <|im_start|> (prompt 未泄漏进 loss)。
    任一失败 → 退出码 EXIT_ASSERT (扫描链停链)。
    """
    dl = trainer.get_train_dataloader()
    batch = next(iter(dl))
    labels = batch["labels"][0].tolist()
    ids = batch["input_ids"][0].tolist()

    n_masked = sum(1 for v in labels if v == -100)
    n_supervised = sum(1 for v in labels if v != -100)
    first_sup = next((i for i, v in enumerate(labels) if v != -100), None)

    if first_sup is None or first_sup == 0:
        fatal(f"loss 冒烟失败: labels 无 -100 前缀 (masked={n_masked}, "
              f"supervised={n_supervised}, first_sup={first_sup}); "
              f"completion-only loss 疑似未生效", EXIT_ASSERT)

    prefix_all_masked = all(v == -100 for v in labels[:first_sup])
    if not prefix_all_masked:
        fatal("loss 冒烟失败: prompt 前缀区段存在非 -100 token", EXIT_ASSERT)

    prompt_text = tokenizer.decode(ids[:first_sup])
    sup_ids = [v for v in labels if v != -100]
    completion_text = tokenizer.decode(sup_ids)

    if "<|im_start|>" not in prompt_text:
        fatal("loss 冒烟失败: -100 前缀区段 decode 后不含 <|im_start|>, "
              "被 mask 的不是 prompt", EXIT_ASSERT)
    if "<|im_start|>" in completion_text:
        fatal("loss 冒烟失败: 被监督区段 decode 后含 <|im_start|>, "
              "prompt 泄漏进 loss 区段", EXIT_ASSERT)

    log_fn(f"loss 冒烟 PASSED: 序列长 {len(labels)}, prompt 区段 {first_sup} token 全 -100, "
           f"被监督 {n_supervised} token (completion); prompt 含 ChatML 标记, "
           f"completion 区段无 <|im_start|> 泄漏")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_name = cfg.get("run_name", "v5")

    train_file = resolve(args.train_file or cfg["train_file"])
    val_file = resolve(args.val_file or cfg["val_file"])
    output_dir = resolve(cfg["output_dir"])
    adapter_dir = resolve(cfg["adapter_output_dir"])
    base_model = resolve(cfg["base_model_path"])
    results_dir = PROJECT_ROOT / "results"

    train_ds, val_ds = load_datasets(train_file, val_file)

    # ---------- DRY-RUN: 数据 + 配置摘要, 不加载模型 ----------
    if args.dry_run:
        eff_batch = (cfg.get("per_device_train_batch_size", 1)
                     * cfg.get("gradient_accumulation_steps", 8))
        steps_per_epoch = math.ceil(len(train_ds) / eff_batch)
        summary = {
            "run_name": run_name,
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
            "max_steps": cfg.get("max_steps", -1),
            "steps_per_epoch": steps_per_epoch,
            "lr_horizon_epochs": cfg.get("lr_horizon_epochs", 2),
            "lr_horizon_steps": steps_per_epoch * cfg.get("lr_horizon_epochs", 2),
            "max_length": cfg.get("max_length"),
            "optim": cfg.get("optim", "adamw_torch"),
            "early_stopping_patience": cfg.get("early_stopping_patience"),
            "early_stopping_threshold": cfg.get("early_stopping_threshold", 0.0),
            "eval_steps == save_steps": cfg.get("eval_steps"),
            "save_total_limit": cfg.get("save_total_limit"),
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
        fatal(f"基座模型目录不存在: {base_model}")

    # ---------- 模型 (纯 BF16, 无量化) ----------
    torch.cuda.reset_peak_memory_stats()
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
        use_rslora=lora_cfg.get("use_rslora", False),   # v5: P0 小扫臂
    )
    # 显式冻结基座 (peft 默认也会冻结, 双保险), 梯度检查点 + 输入梯度
    for p in model.parameters():
        p.requires_grad = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    # PEFT 默认以 FP32 初始化 LoRA 参数, 与 "纯 BF16" 设计不符, 且导致
    # checkpoint/发货落盘为 F32 (首轮 v4 训练发货验证 "lora_B:dtype全BF16"
    # FAILED 的根因)。统一转为 BF16, 使落盘即 BF16。
    n_cast = 0
    for name, p in model.named_parameters():
        if "lora_" in name and p.dtype != torch.bfloat16:
            p.data = p.data.to(torch.bfloat16)
            n_cast += 1
    log(f"LoRA 参数 dtype 校正: {n_cast} 个张量 → BF16")
    model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"可训练参数: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")
    log(f"use_rslora={lora_cfg.get('use_rslora', False)} "
        f"(r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}, "
        f"缩放={'alpha/sqrt(r)' if lora_cfg.get('use_rslora', False) else 'alpha/r'})")

    # ---------- 训练参数 ----------
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.get("num_train_epochs", 4),
        max_steps=cfg.get("max_steps", -1),   # v5 小扫: 0.5 epoch 硬上限; -1 = 按 epochs
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=cfg.get("learning_rate", 2e-4),
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        max_length=cfg.get("max_length", 2048),
        optim=cfg.get("optim", "adamw_torch"),   # v5: 显式写入, 不切 8bit
        logging_steps=cfg.get("logging_steps", 10),
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg.get("save_total_limit", 8),
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        load_best_model_at_end=False,   # best 由 BestCheckpointCallback 另存, 不依赖内存回载
        metric_for_best_model="eval_loss",  # transformers 5.x: EarlyStoppingCallback 强制要求
        greater_is_better=False,
        bf16=True,                      # 纯 BF16 训练; bf16 不启用 GradScaler
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=cfg.get("report_to", "none"),
        seed=cfg.get("seed", 42),
        dataset_num_proc=cfg.get("dataset_num_proc", 4),
        completion_only_loss=None,      # 显式 None: 由 SFTTrainer 按 prompt/completion 自动判定
    )

    # v5 §11.2-3: max_length 从配置项升级为断言项
    if training_args.max_length != 2048:
        fatal(f"max_length 断言失败: SFTConfig.max_length={training_args.max_length} != 2048 "
              f"(v5 显式锁定; v1 静默截断前科, 防回归)", EXIT_ASSERT)

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
        fatal("completion_only_loss 未启用, 拒绝训练 (v5 硬性要求 completion-only loss)",
              EXIT_ASSERT)

    # v5 §11.2-5: loss 冒烟 (decode 首 batch labels, prompt 区段须全 -100)
    smoke_check_prompt_mask(trainer, tokenizer)

    # v5 §11.2-1: cosine horizon = ceil(D/8) * lr_horizon_epochs (2-epoch 设计);
    # max_steps 只限训练长度, 不缩 horizon。重建 optimizer+scheduler (训练未开始, 无副作用)。
    eff_batch = (cfg.get("per_device_train_batch_size", 1)
                 * cfg.get("gradient_accumulation_steps", 8))
    steps_per_epoch = math.ceil(len(train_ds) / eff_batch)
    horizon_epochs = cfg.get("lr_horizon_epochs", 2)
    if cfg.get("lr_scheduler_type", "cosine") == "cosine":
        horizon_steps = steps_per_epoch * horizon_epochs
        trainer.create_optimizer_and_scheduler(num_training_steps=horizon_steps)
        log(f"cosine horizon 重建: {steps_per_epoch} 步/epoch × {horizon_epochs} epoch "
            f"= {horizon_steps} 步 (warmup={int(horizon_steps * cfg.get('warmup_ratio', 0.05))} 步; "
            f"max_steps={cfg.get('max_steps', -1)} 仅训练上限, 不缩 horizon)")

    best_callback_cls = checkpointing.build_best_checkpoint_callback(
        TrainerCallback, output_dir, log_fn=log)
    best_callback = best_callback_cls()
    trainer.add_callback(best_callback)
    patience = cfg.get("early_stopping_patience", 3)
    threshold = cfg.get("early_stopping_threshold", 0.0)
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=patience,
                                               early_stopping_threshold=threshold))
    log(f"EarlyStopping patience={patience} threshold={threshold} "
        f"| eval_steps=save_steps={cfg['eval_steps']} "
        f"| save_total_limit={cfg.get('save_total_limit', 8)}")

    # v5b 教训(2026-07-31): 续跑后内存 LoRA 参数退化为 F32 (best/ 与
    # checkpoint 落盘全部 F32, 发货验证 lora_B:dtype全BF16 FAILED; v5a 单跑
    # 无续跑从未触发)。resume 加载发生在 trainer.train() 内部、
    # on_train_begin 之前, 故在回调里把加载后的 LoRA 参数重新校正为 BF16,
    # 恢复训练期数值保真, 而不只是发货时补救。
    class _LoraDtypeRecastCallback(TrainerCallback):
        def on_train_begin(self, args_, state, control, model=None, **kw):
            if model is None:
                return
            n = 0
            for name, p in model.named_parameters():
                if "lora_" in name and p.dtype != torch.bfloat16:
                    p.data = p.data.to(torch.bfloat16)
                    n += 1
            if n:
                log(f"[resume] LoRA dtype 再校正: {n} 个张量 → BF16 (v5b 教训)")
    trainer.add_callback(_LoraDtypeRecastCallback())

    log(f"=== 开始训练 {run_name} | epochs={cfg.get('num_train_epochs', 4)} "
        f"| max_steps={cfg.get('max_steps', -1)} | resume={args.resume} ===")
    trainer.train(resume_from_checkpoint=args.resume)
    elapsed_h = (time.time() - t0) / 3600
    log(f"=== 训练结束, 耗时 {elapsed_h:.2f} 小时 ===")

    # v5 §11.2-7: 末步 checkpoint 落盘 + 终点 eval (末步不对齐 eval_steps 时补终点值;
    # save_model 落盘 adapter 文件, on_evaluate 时 BestCheckpointCallback 可正常归因)
    final_step = trainer.state.global_step
    final_ckpt = os.path.join(output_dir, f"checkpoint-{final_step}")
    if not os.path.isdir(final_ckpt):
        trainer.save_model(final_ckpt)
        log(f"末步 checkpoint 落盘: {final_ckpt} (供 best 精确归因)")
    final_metrics = trainer.evaluate()
    final_eval_loss = final_metrics.get("eval_loss")
    log(f"终点评估 @ step {final_step}: eval_loss={final_eval_loss}")

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
    log(f"显存峰值: {peak_mem_gb:.1f} GB")

    # ---------- 训练日志持久化 (v1 逐步 loss 已永久丢失, 不能再犯) ----------
    results_dir.mkdir(exist_ok=True)
    log_path = results_dir / f"train_log_{run_name}.json"
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
        # v5b 教训(2026-07-31): 发货前兜底把 adapter safetensors 统一校正为
        # BF16 (幂等)。无论训练/续跑路径造成什么 dtype 漂移, 发货物必须满足
        # validate_adapter_dir 的 lora_B:dtype全BF16 检查, 不再靠事后手工修复。
        _st_path = os.path.join(adapter_dir, "adapter_model.safetensors")
        if os.path.isfile(_st_path):
            from safetensors.torch import load_file as _st_load
            from safetensors.torch import save_file as _st_save
            _tensors = _st_load(_st_path)
            _n_cast = sum(1 for v in _tensors.values()
                          if v.dtype != torch.bfloat16)
            if _n_cast:
                _tensors = {k: (v.to(torch.bfloat16)
                                if v.dtype != torch.bfloat16 else v)
                            for k, v in _tensors.items()}
                _st_save(_tensors, _st_path, metadata={"format": "pt"})
                log(f"[ship] 发货 dtype 校正: {_n_cast} 个张量 → BF16 (v5b 教训)")
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
    eval_trajectory = [
        {"step": e["step"], "eval_loss": e["eval_loss"]}
        for e in trainer.state.log_history if "eval_loss" in e
    ]
    max_steps = getattr(trainer.state, "max_steps", None)
    info = {
        "version": "v5",
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": validation["status"],
        "base_model": cfg["base_model_path"],
        "adapter_dir": adapter_dir,
        "training": {
            "configured_epochs": cfg.get("num_train_epochs", 4),
            "configured_max_steps": cfg.get("max_steps", -1),
            "actual_epochs": trainer.state.epoch,
            "actual_steps": trainer.state.global_step,
            "max_steps": max_steps,
            "early_stopped": bool(max_steps is not None
                                  and trainer.state.global_step < max_steps),
            "early_stopping_patience": patience,
            "early_stopping_threshold": threshold,
            "final_train_loss": final_train_loss,
            "final_eval_loss_at_end": final_eval_loss,
            "eval_trajectory": eval_trajectory,
            "elapsed_hours": round(elapsed_h, 3),
            "peak_mem_gb": round(peak_mem_gb, 2),
            "completion_only_loss": True,
            "loss_smoke": "PASSED",
            "precision": "bf16 (no quantization)",
            "lora": lora_cfg,
            "learning_rate": cfg.get("learning_rate", 2e-4),
            "lr_scheduler_type": cfg.get("lr_scheduler_type", "cosine"),
            "lr_horizon_epochs": horizon_epochs,
            "steps_per_epoch": steps_per_epoch,
            "optim": cfg.get("optim", "adamw_torch"),
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

    # 单组运行摘要 (扫描汇总脚本直接消费)
    run_summary = {
        "run_name": run_name,
        "learning_rate": cfg.get("learning_rate", 2e-4),
        "use_rslora": lora_cfg.get("use_rslora", False),
        "status": validation["status"],
        "actual_steps": trainer.state.global_step,
        "elapsed_hours": round(elapsed_h, 3),
        "peak_mem_gb": round(peak_mem_gb, 2),
        "eval_trajectory": eval_trajectory,
        "final_eval_loss": final_eval_loss,
        "best_eval_step": best_eval_step,
        "best_eval_loss": best_eval_loss,
        "early_stopped": info["training"]["early_stopped"],
        "adapter_dir": adapter_dir if ship_source else None,
    }
    summary_path = results_dir / f"run_summary_{run_name}.json"
    with open(summary_path, "w") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)
    log(f"运行摘要已写入: {summary_path}")

    print(json.dumps({
        "status": validation["status"],
        "adapter_dir": adapter_dir if ship_source else None,
        "best_eval_step": best_eval_step,
        "best_eval_loss": best_eval_loss,
        "final_eval_loss": final_eval_loss,
        "actual_steps": trainer.state.global_step,
        "elapsed_hours": round(elapsed_h, 3),
        "peak_mem_gb": round(peak_mem_gb, 2),
    }, ensure_ascii=False, indent=2))

    if validation["status"] != "PASSED":
        checkpointing.log_marked(f"{run_name} 训练流程结束: 验证 FAILED, 退出码 1", log)
        sys.exit(1)
    checkpointing.log_marked(f"{run_name} 训练流程结束: 验证 PASSED, 适配器已发货", log)


if __name__ == "__main__":
    main()
