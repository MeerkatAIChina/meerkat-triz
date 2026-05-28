# Phase 02: Baseline & Training Execution - Pattern Map

**Mapped:** 2026-05-28
**Files analyzed:** 7
**Analogs found:** 7 / 7

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `notebooks/03_model_benchmark.ipynb` | notebook | request-response | `notebooks/03_model_benchmark.ipynb` (existing) | exact |
| `notebooks/04_qlora_finetune.ipynb` | notebook | batch | `notebooks/04_qlora_finetune.ipynb` (existing) | exact |
| `utils/benchmark_utils.py` | utility | CRUD | `utils/benchmark_utils.py` (existing) | exact |
| `utils/training_utils.py` | utility | batch | `utils/training_utils.py` (existing) | exact |
| `utils/pipeline_state.py` | utility | CRUD | `utils/pipeline_state.py` (existing) | exact |
| `config.py` | config | static | `config.py` (existing) | exact |
| `notebooks/02b_synthetic_generation.ipynb` | notebook | batch | `notebooks/02b_synthetic_generation.ipynb` (existing) | exact |

---

## Pattern Assignments

### `notebooks/03_model_benchmark.ipynb` (notebook, request-response)

**Analog:** `ref/mongoose_ai_dgx/notebooks/03_model_benchmark.ipynb`

**Notebook setup pattern** (cell-2):
```python
import sys
sys.path.append('/home/meerkat/mongoose_ai')

import torch
from utils.training_utils import load_model_and_tokenizer
from config import BASE_MODEL, MODELS_DIR, RESULTS_DIR

# 确定模型路径 (基座模型或微调后模型)
model_path = os.path.join(MODELS_DIR, BASE_MODEL.split('/')[-1])

print(f"评测模型: {model_path}")

# 加载模型 (4-bit量化以节省内存)
model, tokenizer = load_model_and_tokenizer(
    model_name_or_path=model_path,
    quantization_config={
        'load_in_4bit': True,
        'bnb_4bit_quant_type': 'nf4',
        'bnb_4bit_compute_dtype': 'float16',
        'bnb_4bit_use_double_quant': True,
    },
    device_map='auto',
    trust_remote_code=True,
)
```

**Layer 1 benchmark invocation pattern** (cell-4):
```python
from utils.benchmark_utils import run_lm_evaluation

# 选择要评测的任务
tasks = ["mmlu_pro", "gpqa", "humaneval", "math", "bbh"]

# 运行评测
general_results = run_lm_evaluation(
    model_path=model_path,
    tasks=tasks,
    output_dir=RESULTS_DIR,
    num_fewshot=5,
    batch_size=1,
)
```

**Pipeline state registration pattern** (from 02b cell-16):
```python
from utils.pipeline_state import PipelineState

state = PipelineState()

state.register(
    name="baseline_results",
    path=str(result_file),
    artifact_type="benchmark",
    metadata={
        "tasks": tasks,
        "model_path": model_path,
        "timestamp": datetime.now().isoformat(),
    }
)
```

**Memory cleanup pattern** (cell-12):
```python
# 清理显存，为训练做准备
del model
del tokenizer
torch.cuda.empty_cache()

print("显存已清理")
print(f"当前显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
```

---

### `notebooks/04_qlora_finetune.ipynb` (notebook, batch)

**Analog:** `ref/mongoose_ai_dgx/notebooks/04_qlora_finetune.ipynb`

**Notebook setup + preflight pattern** (from 01 cell-6, 02b cell-2):
```python
import sys
sys.path.append('/home/meerkat/mongoose_ai')

from config import (
    BASE_MODEL, MODELS_DIR, DATA_DIR, OUTPUTS_DIR,
    QLORA_CONFIG, DATA_CONFIG
)
from utils.pipeline_state import PipelineState
from utils.training_utils import (
    load_model_and_tokenizer, setup_qlora_config,
    prepare_qlora_model, setup_training_arguments,
    create_trainer, save_adapter_only
)
from utils.data_utils import load_processed_dataset

# Pre-flight: verify baseline exists, data artifact exists, GPU memory > 60GB
state = PipelineState()
errors = state.preflight(
    required_artifacts=["baseline_results", "processed_dataset"],
    required_packages={"transformers": "4.45.0", "trl": "0.9.6"}
)
if errors:
    raise RuntimeError(f"预飞行检查失败: {errors}")

# Check GPU memory
if torch.cuda.is_available():
    free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
    free_gb = free_memory / (1024**3)
    if free_gb < 60:
        raise RuntimeError(f"可用GPU内存不足: {free_gb:.1f}GB < 60GB")
```

**Model loading pattern (4-bit for training)** (cell-4):
```python
model_path = os.path.join(MODELS_DIR, BASE_MODEL.split('/')[-1])

print(f"加载模型: {model_path}")
print("启用4-bit量化以节省内存...")

model, tokenizer = load_model_and_tokenizer(
    model_name_or_path=model_path,
    quantization_config=QLORA_CONFIG['quantization'],
    device_map='auto',
    trust_remote_code=True,
)

print(f"\n模型加载完成!")
print(f"显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
```

**QLoRA config + preparation pattern** (cell-8):
```python
lora_config = setup_qlora_config(
    r=QLORA_CONFIG['lora']['r'],
    lora_alpha=QLORA_CONFIG['lora']['lora_alpha'],
    target_modules=QLORA_CONFIG['lora']['target_modules'],
    lora_dropout=QLORA_CONFIG['lora']['lora_dropout'],
    use_rslora=QLORA_CONFIG['lora'].get('use_rslora', False),
)

model = prepare_qlora_model(model, lora_config)
```

**Training arguments pattern** (cell-10):
```python
training_args = setup_training_arguments(
    output_dir=QLORA_CONFIG['training']['output_dir'],
    num_train_epochs=QLORA_CONFIG['training']['num_train_epochs'],
    per_device_batch_size=QLORA_CONFIG['training']['per_device_train_batch_size'],
    gradient_accumulation_steps=QLORA_CONFIG['training']['gradient_accumulation_steps'],
    learning_rate=QLORA_CONFIG['training']['learning_rate'],
    warmup_ratio=QLORA_CONFIG['training']['warmup_ratio'],
    save_steps=QLORA_CONFIG['training']['save_steps'],
    eval_steps=QLORA_CONFIG['training']['eval_steps'],
    logging_steps=QLORA_CONFIG['training']['logging_steps'],
)
```

**Trainer creation + training pattern** (cell-12 / cell-2):
```python
trainer = create_trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    training_args=training_args,
    system_message=DATA_CONFIG['chatml']['system_message'],
    max_seq_length=DATA_CONFIG['chatml']['max_length'],
    packing=True,
)

print("Trainer创建完成，开始训练...")
print("="*60)

# 开始训练
trainer.train()

print("="*60)
print("训练完成!")
```

**Checkpoint resume pattern** (from training_utils.py lines 544-553):
```python
from utils.training_utils import resume_from_checkpoint

# Cell-level checkpoint resume: load latest checkpoint and resume
checkpoint_dir = QLORA_CONFIG['training']['output_dir']
latest_checkpoint = None

# Find the latest checkpoint
if os.path.exists(checkpoint_dir):
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
    if checkpoints:
        latest = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
        latest_checkpoint = os.path.join(checkpoint_dir, latest)

if latest_checkpoint:
    print(f"从checkpoint恢复: {latest_checkpoint}")
    resume_from_checkpoint(trainer, latest_checkpoint)
else:
    print("未找到checkpoint，从头开始训练")
    trainer.train()
```

**Adapter save + pipeline state registration pattern** (cell-14 + from 02b cell-16):
```python
from utils.training_utils import save_adapter_only

adapter_output_dir = os.path.join(MODELS_DIR, 'meerkat_triz_adapter_v1')
save_adapter_only(model, tokenizer, adapter_output_dir)

print(f"\n适配器已保存到: {adapter_output_dir}")

# Register to pipeline state
state.register(
    name="adapter_checkpoint",
    path=adapter_output_dir,
    artifact_type="model",
    metadata={
        "base_model": BASE_MODEL,
        "training_steps": trainer.state.global_step,
        "final_loss": trainer.state.log_history[-1].get('loss', 'N/A') if trainer.state.log_history else 'N/A',
    }
)
```

---

### `utils/benchmark_utils.py` (utility, CRUD)

**Analog:** `ref/mongoose_ai_dgx/utils/benchmark_utils.py`

**lm-eval-harness invocation pattern** (lines 20-83):
```python
def run_lm_evaluation(
    model_path: str,
    tasks: List[str],
    output_dir: str,
    num_fewshot: Optional[int] = None,
    batch_size: int = 1,
    device: str = "cuda"
) -> Dict[str, Any]:
    try:
        import lm_eval
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        logger.error("lm-eval 未安装，请先运行: pip install lm-eval>=0.4.3")
        raise

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 配置模型
    model_args = f"pretrained={model_path},dtype=float16,device={device}"

    # 运行评测
    results = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        write_out=True,
        output_path=str(output_path),
    )

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_path / f"lm_eval_results_{timestamp}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results
```

**Results summary printing pattern** (lines 86-100):
```python
def print_evaluation_summary(results: Dict[str, Any], tasks: List[str]):
    print("\n" + "=" * 60)
    print("通用能力评测结果摘要")
    print("=" * 60)
    for task in tasks:
        if task in results.get("results", {}):
            task_results = results["results"][task]
            for key, value in task_results.items():
                if "acc" in key or "score" in key:
                    print(f"  {task:20s} | {key:30s} | {value:.4f}")
    print("=" * 60 + "\n")
```

**Performance benchmark pattern** (lines 361-475):
```python
def run_performance_benchmark(
    model, tokenizer, output_dir: str,
    test_prompts: Optional[List[str]] = None, max_tokens: int = 512
) -> Dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device

    # 预热
    warmup_prompt = "Hello, this is a warmup prompt."
    inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=50)
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # 性能测试
    latencies = []
    token_counts = []
    memory_before = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0

    for i, prompt in enumerate(test_prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_tokens = inputs["input_ids"].shape[1]

        start_time = time.time()
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_tokens,
                temperature=0.7, top_p=0.9, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()

        latency_ms = (end_time - start_time) * 1000
        output_tokens = outputs.shape[1] - input_tokens
        tokens_per_sec = output_tokens / (end_time - start_time)
        latencies.append(latency_ms)
        token_counts.append(output_tokens)

    memory_peak = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0

    results = {
        "latency_p50_ms": sorted(latencies)[len(latencies)//2],
        "latency_avg_ms": sum(latencies) / len(latencies),
        "throughput_tokens_per_sec": sum(token_counts) / sum([l/1000 for l in latencies]),
        "memory_peak_gb": memory_peak,
    }
    return results
```

---

### `utils/training_utils.py` (utility, batch)

**Analog:** `ref/mongoose_ai_dgx/utils/training_utils.py`

**Model loading pattern** (lines 31-93):
```python
def load_model_and_tokenizer(
    model_name_or_path: str,
    quantization_config: Optional[Dict] = None,
    device_map: str = "auto",
    trust_remote_code: bool = True,
) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {
        "pretrained_model_name_or_path": model_name_or_path,
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
        "torch_dtype": torch.float16,
    }
    if quantization_config and quantization_config.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(**quantization_config)
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model.gradient_checkpointing_enable()
    return model, tokenizer
```

**QLoRA config pattern** (lines 150-209):
```python
def setup_qlora_config(
    r: int = 64, lora_alpha: int = 128,
    target_modules: Optional[list] = None,
    lora_dropout: float = 0.0,
    use_rslora: bool = False,
) -> LoraConfig:
    if target_modules is None:
        target_modules = get_qwen36_target_modules()
    elif target_modules == "all-linear":
        logger.warning("使用'all-linear'自动检测模式 (不推荐)")

    lora_config = LoraConfig(
        r=r, lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_rslora=use_rslora,
    )
    return lora_config
```

**SFTTrainer creation pattern (CRITICAL)** (lines 338-441):
```python
def create_trainer(
    model, tokenizer, train_dataset, eval_dataset,
    training_args: TrainingArguments,
    system_message: Optional[str] = None,
    max_seq_length: int = 4096,
    packing: bool = True,
):
    def formatting_func(example):
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")
        sys_msg = example.get("system", system_message)

        if input_text:
            full_question = f"{instruction}\n\n{input_text}"
        else:
            full_question = instruction

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": full_question},
            {"role": "assistant", "content": output},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        return text

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        args=training_args,
        max_seq_length=max_seq_length,
        formatting_func=formatting_func,
        packing=packing,
        # 不传入data_collator，避免与SFTTrainer内置逻辑冲突
    )
    return trainer
```

**Adapter save pattern** (lines 507-539):
```python
def save_adapter_only(model, tokenizer, output_path: str):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    info = {
        "adapter_type": "LORA",
        "base_model": model.config._name_or_path if hasattr(model, "config") else "unknown",
    }
    with open(output_path / "adapter_info.json", "w") as f:
        import json
        json.dump(info, f, indent=2)
```

**Checkpoint resume pattern** (lines 544-553):
```python
def resume_from_checkpoint(trainer, checkpoint_path: str):
    logger.info(f"从checkpoint恢复训练: {checkpoint_path}")
    trainer.train(resume_from_checkpoint=checkpoint_path)
```

---

### `utils/pipeline_state.py` (utility, CRUD)

**Analog:** `ref/mongoose_ai_dgx/utils/pipeline_state.py`

**Artifact registration pattern** (lines 45-66):
```python
def register(
    self, name: str, path: str,
    artifact_type: str,
    metadata: Optional[Dict[str, Any]] = None,
):
    artifact = {
        "name": name,
        "path": str(path),
        "type": artifact_type,
        "created_at": datetime.now().isoformat(),
        "metadata": metadata or {},
    }
    self.state["artifacts"] = [
        a for a in self.state["artifacts"] if a["name"] != name
    ]
    self.state["artifacts"].append(artifact)
    self._save()
    logger.info(f"注册工件: {name} ({artifact_type}) -> {path}")
```

**Pre-flight check pattern** (lines 88-138):
```python
def preflight(
    self,
    required_artifacts: Optional[List[str]] = None,
    required_packages: Optional[Dict[str, str]] = None,
) -> List[str]:
    errors = []
    if required_artifacts:
        for name in required_artifacts:
            if not self.verify(name):
                artifact = self.get(name)
                if artifact:
                    errors.append(f"工件路径不存在: {name} -> {artifact['path']}")
                else:
                    errors.append(f"未注册工件: {name}")

    if required_packages:
        for pkg_name, min_ver in required_packages.items():
            try:
                mod = __import__(pkg_name)
                actual_ver = getattr(mod, "__version__", "unknown")
                if actual_ver != "unknown":
                    if version.parse(actual_ver) < version.parse(min_ver):
                        errors.append(f"{pkg_name} 版本过低: {actual_ver} < {min_ver}")
                else:
                    errors.append(f"{pkg_name} 无法获取版本信息")
            except ImportError:
                errors.append(f"未安装包: {pkg_name}")
    return errors
```

---

### `config.py` (config, static)

**Analog:** `ref/mongoose_ai_dgx/config.py`

**Path configuration pattern** (lines 9-20):
```python
from pathlib import Path

BASE_DIR = Path("/home/meerkat/mongoose_ai")
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, CHECKPOINTS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
```

**QLoRA config pattern** (lines 38-91):
```python
QLORA_CONFIG = {
    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_use_double_quant": True,
    },
    "lora": {
        "r": 64,
        "lora_alpha": 128,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "use_rslora": False,
    },
    "training": {
        "output_dir": str(CHECKPOINTS_DIR / "qlora_trtiz_v1"),
        "num_train_epochs": 2,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "save_steps": 200,
        "eval_steps": 200,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "tensorboard",
        "bf16": False,
        "fp16": True,
        "optim": "paged_adamw_8bit",
        "group_by_length": True,
    }
}
```

**Benchmark config pattern** (lines 202-270):
```python
BENCHMARK_CONFIG = {
    "output_dir": str(RESULTS_DIR),
    "general_benchmarks": {
        "mmlu_pro": {"num_fewshot": 5, "batch_size": 1},
        "gpqa": {"num_fewshot": 0, "batch_size": 1},
        "humaneval": {"num_fewshot": 0, "batch_size": 1},
        "math": {"num_fewshot": 4, "batch_size": 1},
        "bbh": {"num_fewshot": 3, "batch_size": 1},
    },
    "triz_benchmarks": { ... },
    "performance_benchmarks": { ... },
}
```

---

### `notebooks/02b_synthetic_generation.ipynb` (notebook, batch)

**Analog:** `ref/mongoose_ai_dgx/notebooks/02b_synthetic_generation.ipynb`

**Data loading from pipeline_state pattern** (cell-2, cell-4):
```python
import sys
sys.path.append('/home/meerkat/mongoose_ai')

from config import DATA_DIR, DATA_CONFIG, SYNTHETIC_CONFIG, BASE_MODEL, MODELS_DIR
from utils.data_utils import load_raw_data, convert_to_chatml, split_dataset, save_dataset
from utils.pipeline_state import PipelineState
from utils.training_utils import load_model_and_tokenizer

# Load seed data
seed_data = load_raw_data(DATA_CONFIG['raw_data_dir'])
if not seed_data or sum(len(v) for v in seed_data.values()) == 0:
    from utils.data_utils import create_sample_data
    seed_data = create_sample_data()
```

**Dataset save + pipeline state registration pattern** (cell-16):
```python
processed_dir = DATA_CONFIG['processed_data_dir']
save_dataset(dataset, processed_dir)

state = PipelineState()
state.register(
    name="synthetic_dataset",
    path=processed_dir,
    artifact_type="dataset",
    metadata={
        "splits": {k: len(v) for k, v in dataset.items()},
        "format": "chatml",
        "total": sum(len(v) for v in dataset.values()),
    }
)
```

---

## Shared Patterns

### Notebook Setup (sys.path + config imports)
**Source:** All notebooks
**Apply to:** All notebook files
```python
import sys
sys.path.append('/home/meerkat/mongoose_ai')

from config import (
    BASE_MODEL, MODELS_DIR, DATA_DIR, OUTPUTS_DIR, RESULTS_DIR,
    QLORA_CONFIG, DATA_CONFIG, BENCHMARK_CONFIG, HARDWARE_CONFIG
)
```

### Model Loading (FP16 for inference, 4-bit for training)
**Source:** `utils/training_utils.py` lines 31-93
**Apply to:** Notebook 03 (FP16), Notebook 04 (4-bit)
```python
# For inference (FP16, no quantization):
model_kwargs = {
    "pretrained_model_name_or_path": model_path,
    "trust_remote_code": True,
    "device_map": "auto",
    "torch_dtype": torch.float16,
}

# For training (4-bit NF4):
if quantization_config and quantization_config.get("load_in_4bit"):
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(**quantization_config)
    model_kwargs["quantization_config"] = bnb_config
```

### SFTTrainer + formatting_func (NO data_collator)
**Source:** `utils/training_utils.py` lines 338-441
**Apply to:** Notebook 04
**CRITICAL:** Do NOT pass `data_collator` to SFTTrainer. Use `formatting_func` only.
```python
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=train_dataset, eval_dataset=eval_dataset,
    args=training_args,
    max_seq_length=max_seq_length,
    formatting_func=formatting_func,
    packing=packing,
)
```

### Pipeline State Registration
**Source:** `utils/pipeline_state.py`
**Apply to:** Notebook 03 (baseline_results), Notebook 04 (adapter_checkpoint)
```python
state = PipelineState()
state.register(name="...", path="...", artifact_type="...", metadata={...})
```

### Memory Cleanup
**Source:** All notebooks
**Apply to:** All notebooks after heavy operations
```python
del model
del tokenizer
torch.cuda.empty_cache()
print(f"当前显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
```

### Checkpoint Save/Resume
**Source:** `utils/training_utils.py` lines 544-553, `config.py` QLORA_CONFIG
**Apply to:** Notebook 04
```python
# Config: save every 200 steps, keep only 3 most recent
"save_steps": 200,
"save_total_limit": 3,

# Resume:
trainer.train(resume_from_checkpoint=checkpoint_path)
```

### Logging Pattern
**Source:** All utility modules
**Apply to:** All new utility modules
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("信息日志")
logger.warning("警告日志")
logger.error("错误日志")
```

---

## No Analog Found

No files in this phase require patterns without existing analogs. All patterns are well-established in the codebase.

---

## Metadata

**Analog search scope:** `ref/mongoose_ai_dgx/notebooks/`, `ref/mongoose_ai_dgx/utils/`, `ref/mongoose_ai_dgx/config.py`
**Files scanned:** 10 (5 notebooks, 5 utils, 1 config)
**Pattern extraction date:** 2026-05-28
