import argparse, logging, os, sys, json, gc, glob, ctypes
sys.path.append('/home/meerkat/mongoose_ai')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger('train_v2')

import torch
from datetime import datetime
from datasets import load_dataset
from trl import SFTTrainer
from peft import get_peft_model
from config import QLORA_CONFIG, MODELS_DIR, CHECKPOINTS_DIR, DATA_DIR, BASE_MODEL
from utils.training_utils import (
    load_model_and_tokenizer, setup_qlora_config,
    setup_training_arguments, CheckpointValidationCallback, save_adapter_only,
)

def drop_shard_pagecache(model_path):
    libc = ctypes.CDLL('libc.so.6')
    for f in glob.glob(os.path.join(model_path, '*.safetensors')):
        fd = os.open(f, os.O_RDONLY)
        libc.posix_fadvise(fd, 0, 0, 4)  # POSIX_FADV_DONTNEED
        os.close(fd)
    gc.collect()
    torch.cuda.empty_cache()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-name', default='v2')
    args = ap.parse_args()
    run = args.run_name

    train_path = str(DATA_DIR / 'processed' / f'{run}_train.jsonl')
    val_path = str(DATA_DIR / 'processed' / f'{run}_validation.jsonl')
    output_dir = str(CHECKPOINTS_DIR / f'qlora_trtiz_{run}')
    adapter_dir = str(MODELS_DIR / f'meerkat_triz_adapter_{run}')

    logger.info(f'=== QLoRA 训练启动 run={run} {datetime.now().isoformat()} ===')
    logger.info(f'train={train_path} val={val_path}')
    logger.info(f'output_dir={output_dir} adapter_dir={adapter_dir}')

    ds = load_dataset('json', data_files={'train': train_path, 'validation': val_path})
    logger.info(f'数据集: train={len(ds["train"])} validation={len(ds["validation"])}')
    logger.info(f'样本字段: {list(ds["train"][0].keys())}')

    model_path = os.path.join(MODELS_DIR, BASE_MODEL.split('/')[-1])
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=model_path,
        quantization_config=QLORA_CONFIG['quantization'],
        device_map='cuda:0',
        trust_remote_code=True,
    )
    logger.info(f'模型加载完成, 显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB')
    logger.info(f'量化检测: is_quantized={getattr(model, "is_quantized", False)}')

    drop_shard_pagecache(model_path)
    logger.info(f'pagecache清理后, 显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB')

    # 禁用KV缓存 (统一内存环境)
    model.config.use_cache = False

    lora_config = setup_qlora_config(
        r=QLORA_CONFIG['lora']['r'],
        lora_alpha=QLORA_CONFIG['lora']['lora_alpha'],
        target_modules=QLORA_CONFIG['lora']['target_modules'],
        lora_dropout=QLORA_CONFIG['lora']['lora_dropout'],
        use_rslora=QLORA_CONFIG['lora'].get('use_rslora', False),
    )

    # 手动准备 (替代 prepare_model_for_kbit_training):
    # peft 0.19 会把全部 bf16 参数转 fp32, 本模型 MoE 专家未量化(33.25B),
    # 全量转换需 ~133GB 必然 OOM, 因此只冻结 + 梯度检查点 + LoRA。
    logger.info('手动准备QLoRA模型 (跳过分到fp32的转换)...')
    for p in model.parameters():
        p.requires_grad = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f'可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)')
    logger.info(f'prepare后显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB')

    tcfg = QLORA_CONFIG['training']
    training_args = setup_training_arguments(
        output_dir=output_dir,
        num_train_epochs=tcfg['num_train_epochs'],
        per_device_batch_size=tcfg['per_device_train_batch_size'],
        gradient_accumulation_steps=tcfg['gradient_accumulation_steps'],
        learning_rate=tcfg['learning_rate'],
        warmup_ratio=tcfg['warmup_ratio'],
        save_steps=tcfg['save_steps'],
        eval_steps=tcfg['eval_steps'],
        logging_steps=tcfg['logging_steps'],
        save_total_limit=tcfg['save_total_limit'],
        load_best_model_at_end=False,
        metric_for_best_model=tcfg.get('metric_for_best_model', 'eval_loss'),
        greater_is_better=tcfg.get('greater_is_better', False),
        report_to=tcfg.get('report_to', 'tensorboard'),
        fp16=False,
        bf16=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        args=training_args,
    )
    cb = CheckpointValidationCallback(tokenizer=tokenizer)
    trainer.add_callback(cb)
    logger.info('Trainer创建完成 (含checkpoint验证回调), 开始训练')

    trainer.train()
    logger.info('训练完成, 保存适配器')

    evals = [h['eval_loss'] for h in trainer.state.log_history if 'eval_loss' in h]
    metadata = {
        'training_steps': trainer.state.global_step,
        'num_train_epochs': tcfg['num_train_epochs'],
        'learning_rate': tcfg['learning_rate'],
        'final_loss': trainer.state.log_history[-1].get('loss', 'N/A') if trainer.state.log_history else 'N/A',
        'best_eval_loss': min(evals) if evals else 'N/A',
        'dataset': {'train': len(ds['train']), 'validation': len(ds['validation'])},
        'checkpoint_validation': cb.validation_results,
    }
    save_adapter_only(model, tokenizer, adapter_dir, metadata=metadata)
    logger.info(f'TRAINING_DONE adapter={adapter_dir}')

if __name__ == '__main__':
    main()
