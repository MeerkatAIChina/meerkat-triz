# Technology Stack for QLoRA Fine-Tuning Pipeline Execution

**Project:** Meerkat AI (猫鼬AI) — TRIZ Domain LLM Fine-Tuning
**Researched:** 2026-05-26
**Scope:** Stack additions/changes needed to execute the complete training pipeline on DGX Spark

---

## Executive Summary

The existing Meerkat AI codebase has a well-structured requirements.txt and utility modules for QLoRA fine-tuning. The key question is: **what version updates or new dependencies are needed to reliably execute the four active capabilities** (synthetic data generation at scale, baseline benchmarking, QLoRA training, post-training evaluation) on the DGX Spark (128GB unified memory, GB10 Grace Blackwell).

**Bottom line:** The current stack is directionally correct but has version gaps that create real compatibility risks. The most critical changes are: (1) pin `transformers` and `trl` to mutually compatible versions that support both Qwen3.6 and `SFTTrainer.formatting_func`, (2) upgrade `lm-eval` to 0.4.10+ with explicit `[hf]` extra, (3) add `rouge-score` for Layer 2 TRIZ evaluation, and (4) ensure `openai` SDK is present for Moonshot API synthetic data generation.

---

## Recommended Stack

### Core Framework

| Technology | Current | Recommended | Purpose | Why |
|------------|---------|-------------|---------|-----|
| `torch` | >=2.4.0 | **>=2.4.0,<2.6.0** | Deep learning backend | DGX Spark ships with CUDA 12.x; 2.4.x is the validated sweet spot. 2.5+ may work but 2.4.x is the most tested on GB10 per community reports. |
| `torchvision` | >=0.19.0 | **>=0.19.0** | Vision utilities (unused) | Keep for completeness; no action needed. |
| `torchaudio` | >=2.4.0 | **>=2.4.0** | Audio utilities (unused) | Keep for completeness; no action needed. |

### Large Model Fine-Tuning (Critical Path)

| Technology | Current | Recommended | Purpose | Why |
|------------|---------|-------------|---------|-----|
| `transformers` | >=4.45.0 | **>=4.45.0,<5.0.0** | Model loading, tokenization, chat templates | The codebase uses `tokenizer.apply_chat_template()` and loads Qwen3.6-35B-A3B. 4.45+ is the documented minimum for Qwen3.x models. **Do NOT upgrade to v5** without full regression testing — it breaks `lm-eval<0.4.10` and may change `apply_chat_template` behavior. |
| `peft` | >=0.12.0 | **>=0.12.0,<0.15.0** | LoRA/QLoRA adapter management | 0.12+ supports `target_modules="all-linear"` (though we explicitly avoid it per CR-002). 0.14+ introduces DoRA but no compelling reason to upgrade for this project. |
| `bitsandbytes` | >=0.43.0 | **>=0.43.0,<0.45.0** | 4-bit NF4 quantization | 0.43.x is a stable target for reproducible QLoRA. Research shows 0.43.x→0.44.x transitions can flip outputs due to dequant-ordering kernel changes. Pin for reproducibility. |
| `accelerate` | >=0.33.0 | **>=0.33.0,<0.35.0** | Device placement, mixed precision | 0.33+ handles single-GPU `device_map="auto"` correctly. No need to chase latest; stability matters more. |
| `trl` | >=0.9.0 | **>=0.9.0,<0.12.0** | SFTTrainer with `formatting_func` | The codebase relies on `SFTTrainer(..., formatting_func=...)` which is stable in 0.9–0.11. **TRL 0.12+ deprecates `formatting_func` in favor of `SFTConfig` + pre-formatted datasets.** Upgrading would require refactoring `training_utils.py`. |

### Training Data & Processing

| Technology | Current | Recommended | Purpose | Why |
|------------|---------|-------------|---------|-----|
| `datasets` | >=2.21.0 | **>=2.21.0,<3.0.0** | HuggingFace Dataset loading, splitting, saving | 2.21+ supports `Dataset.from_json()`, `to_json()`, `train_test_split()`. v3.0+ has breaking changes in streaming and caching that may affect notebook workflow. |
| `tokenizers` | >=0.19.0 | **>=0.19.0** | Fast tokenization backend | Bundled with `transformers` install; keep as explicit dependency for clarity. |

### Evaluation Frameworks

| Technology | Current | Recommended | Purpose | Why |
|------------|---------|-------------|---------|-----|
| `lm-eval` | >=0.4.3 | **>=0.4.10** | Layer 1 general capability benchmarks (MMLU-Pro, GPQA, etc.) | **Critical upgrade.** 0.4.3 pins transformers `<5.0.0` and bundles all backends. 0.4.10+ decouples backends, supports transformers v5, and adds YAML config support. Must install with `[hf]` extra. |
| `evaluate` | >=0.4.2 | **>=0.4.2** | Classical NLP metrics wrapper | Used for potential future metric standardization. Current version sufficient. |
| `rouge-score` | **MISSING** | **>=0.1.2** | Layer 2 TRIZ case quality evaluation (ROUGE-1/2/L) | `benchmark_utils.py` imports `rouge_score.rouge_scorer` but it is **not in requirements.txt**. This will cause an ImportError during evaluation. |
| `scikit-learn` | >=1.4.0 | **>=1.4.0** | Classification metrics, data utilities | Current version sufficient. |

### Synthetic Data Generation

| Technology | Current | Recommended | Purpose | Why |
|------------|---------|-------------|---------|-----|
| `openai` | **MISSING** | **>=1.0.0** | Moonshot API client (OpenAI-compatible) | The synthetic data pipeline uses Moonshot API (Kimi K2.5). Moonshot provides an OpenAI-compatible endpoint at `https://api.moonshot.cn/v1`. The `openai` SDK is the standard client. Must be added. |

### Logging & Monitoring

| Technology | Current | Recommended | Purpose | Why |
|------------|---------|-------------|---------|-----|
| `wandb` | >=0.17.0 | **>=0.17.0** | Experiment tracking (optional) | Current version sufficient. Requires `WANDB_API_KEY` env var. |
| `tensorboard` | >=2.16.0 | **>=2.16.0** | Local training logs | Current version sufficient. Zero-config alternative to W&B. |

### Visualization & Utilities

| Technology | Current | Recommended | Purpose | Why |
|------------|---------|-------------|---------|-----|
| `matplotlib` | >=3.8.0 | **>=3.8.0** | Plotting | Current version sufficient. |
| `seaborn` | >=0.13.0 | **>=0.13.0** | Statistical visualization | Current version sufficient. |
| `tqdm` | >=4.66.0 | **>=4.66.0** | Progress bars | Current version sufficient. |
| `huggingface-hub` | >=0.23.0 | **>=0.23.0** | Model/dataset download from Hub | Current version sufficient. |
| `safetensors` | >=0.4.3 | **>=0.4.3** | Safe tensor serialization | Current version sufficient. |
| `sentencepiece` | >=0.2.0 | **>=0.2.0** | Qwen tokenizer dependency | Current version sufficient. |
| `protobuf` | >=4.25.0 | **>=4.25.0** | Serialization | Current version sufficient. |
| `jieba` | >=0.42.0 | **>=0.42.0** | Chinese text segmentation | Current version sufficient. |

### Jupyter Environment

| Technology | Current | Recommended | Purpose | Why |
|------------|---------|-------------|---------|-----|
| `ipywidgets` | >=8.1.0 | **>=8.1.0** | Interactive widgets in notebooks | Current version sufficient. |
| `jupyterlab` | >=4.1.0 | **>=4.1.0** | Notebook IDE | Current version sufficient. |

---

## Updated requirements.txt

```text
# 猫鼬AI DGX Spark 项目依赖
# 安装命令: pip install -r requirements.txt

# 核心深度学习框架 (DGX Spark推荐版本)
torch>=2.4.0,<2.6.0
torchvision>=0.19.0
torchaudio>=2.4.0

# 大模型微调与推理 (锁定关键版本以确保兼容性)
transformers>=4.45.0,<5.0.0      # Qwen3.6需要4.45+; 避免v5破坏lm-eval兼容性
peft>=0.12.0,<0.15.0              # QLoRA适配器管理
bitsandbytes>=0.43.0,<0.45.0      # NF4量化; 0.43.x为稳定目标
accelerate>=0.33.0,<0.35.0        # 设备映射与混合精度
trl>=0.9.0,<0.12.0                # SFTTrainer formatting_func支持

# 训练与数据处理
datasets>=2.21.0,<3.0.0           # Dataset加载与处理
tokenizers>=0.19.0                # 分词器后端

# 评测框架
lm-eval>=0.4.10                   # 通用能力基准; 0.4.10+解耦后端
# 注意: lm-eval 0.4.10+ 需要显式安装后端:
#   pip install "lm-eval[hf]"     # HuggingFace后端
#   或 pip install lm-eval transformers torch accelerate

evaluate>=0.4.2                   # 经典NLP指标包装器
rouge-score>=0.1.2                # ROUGE评分 (Layer 2 TRIZ评测需要)
scikit-learn>=1.4.0               # 分类指标与数据工具

# 合成数据生成
openai>=1.0.0                     # Moonshot API客户端 (OpenAI兼容接口)

# 数据处理与科学计算
numpy>=1.26.0
pandas>=2.2.0
scipy>=1.12.0

# 可视化
matplotlib>=3.8.0
seaborn>=0.13.0

# 工具库
tqdm>=4.66.0
huggingface-hub>=0.23.0
safetensors>=0.4.3
sentencepiece>=0.2.0
protobuf>=4.25.0

# Jupyter环境
ipywidgets>=8.1.0
jupyterlab>=4.1.0

# 日志与监控
wandb>=0.17.0                     # 实验跟踪 (可选, 需WANDB_API_KEY)
tensorboard>=2.16.0               # 本地训练日志

# 中文分词
jieba>=0.42.0
```

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| TRL version | 0.9–0.11 | 0.12+ with `SFTConfig` | Would require refactoring `training_utils.py` to pre-format datasets and use `dataset_text_field`. The current `formatting_func` approach is simpler and well-tested. |
| lm-eval version | 0.4.10+ | Stay at 0.4.3 | 0.4.3 pins transformers `<5.0.0` and bundles heavy backends. 0.4.10+ is lighter, more modular, and future-proof. |
| Synthetic data API client | `openai` SDK | `requests` + manual HTTP | Moonshot's OpenAI-compatible endpoint means the `openai` SDK works with just a `base_url` change. Using raw HTTP is error-prone for streaming, retries, and error handling. |
| ROUGE library | `rouge-score` | `evaluate.load("rouge")` | `evaluate` wraps `rouge-score` internally. Using `rouge-score` directly gives more control over scorer configuration (e.g., `use_stemmer=True`) and avoids an extra dependency layer. |
| Quantization backend | `bitsandbytes` NF4 | `auto-gptq` or `autoawq` | GPTQ/AWQ are inference-optimized and not designed for training. QLoRA requires bitsandbytes for 4-bit training. |

---

## Integration Points

### 1. Synthetic Data Generation (Notebook 02b)

**Integration:** Moonshot API via `openai` SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["MOONSHOT_API_KEY"],
    base_url="https://api.moonshot.cn/v1"
)

response = client.chat.completions.create(
    model="kimi-k2.5",
    messages=[...],
    temperature=0.8,
)
```

**Key point:** The existing `data_utils.py` has a simple `vary_sample()` function using template paraphrasing. For production-quality ~6K samples, the Moonshot API integration should replace or augment this with LLM-based synthesis. The `openai` SDK handles retries, rate limiting, and streaming automatically.

### 2. Baseline Benchmark (Notebook 03)

**Integration:** `lm-eval` 0.4.10+ with HuggingFace backend

**Breaking change from 0.4.3:**
- Old: `pip install lm-eval` (got everything)
- New: `pip install "lm-eval[hf]"` (must explicitly request HF backend)

**Code impact in `benchmark_utils.py`:**

```python
# Current code (works with 0.4.3 and 0.4.10+)
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

results = simple_evaluate(
    model="hf",
    model_args=f"pretrained={model_path},dtype=float16,device=cuda",
    tasks=["mmlu_pro", "gpqa", "humaneval", "math", "bbh"],
    ...
)
```

The existing code should work with 0.4.10+ without changes, **provided** the `[hf]` extra is installed.

### 3. QLoRA Fine-Tuning (Notebook 04)

**Integration:** `trl.SFTTrainer` + `formatting_func`

**Version lock rationale:** The `training_utils.py` `create_trainer()` function uses:

```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    max_seq_length=max_seq_length,
    formatting_func=formatting_func,
    packing=packing,
)
```

This API is stable in TRL 0.9–0.11. In TRL 0.12+, `formatting_func` is deprecated in favor of:
1. Pre-formatting the dataset with a `"text"` column, OR
2. Using `SFTConfig(dataset_text_field="text")`

**Recommendation:** Stay on TRL <0.12.0 to avoid a refactor. If upgrading later, the migration is mechanical but touches every notebook.

### 4. Post-Training Evaluation (Notebook 05)

**Integration:** `rouge-score` for Layer 2 TRIZ case quality

**Missing dependency:** `benchmark_utils.py` line 224:

```python
from rouge_score import rouge_scorer
```

This import will fail with `ModuleNotFoundError` because `rouge-score` is not in the current `requirements.txt`. Adding it fixes this.

---

## What NOT to Add

| Package | Why Not |
|---------|---------|
| `vllm` | Commented out in current requirements.txt for good reason. DGX Spark's GB10 may not have full vLLM kernel support, and it's unnecessary for training-only scope. |
| `unsloth` | Provides 2–5x speedup but requires specific model support and may conflict with the custom Qwen3.6 target modules. Evaluate after first successful training run. |
| `flash-attn` | DGX Spark has 128GB unified memory — memory pressure is not the primary constraint. Flash Attention installation is complex and may not have GB10 kernel support. |
| `xformers` | Not needed for QLoRA training. Adds complexity without clear benefit for this single-GPU setup. |
| `deepspeed` | Out of scope — single-GPU only per PROJECT.md. |
| `auto-gptq`, `autoawq` | Inference quantization only, not for training. |
| `ragas`, `deepeval` | LLM-as-judge frameworks are interesting for future evaluation but overkill for v1.0. |

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Core framework versions | **HIGH** | `transformers` 4.45+ for Qwen3.x is well-documented. Version ceiling `<5.0.0` is conservative and safe. |
| TRL `formatting_func` deprecation | **HIGH** | Multiple sources confirm TRL 0.12+ moves to `SFTConfig`. Staying on 0.9–0.11 avoids the issue. |
| lm-eval 0.4.10 migration | **HIGH** | Official release notes and Ascend docs confirm the decoupling. `[hf]` extra is the correct install pattern. |
| bitsandbytes 0.43 stability | **MEDIUM** | Research paper confirms 0.43.x→0.44.x can change outputs. Pinning is prudent but not strictly required. |
| rouge-score necessity | **HIGH** | Direct code inspection of `benchmark_utils.py` shows the import. Missing from requirements.txt is a clear bug. |
| openai SDK for Moonshot | **HIGH** | Moonshot officially documents OpenAI-compatible API. The `openai` SDK is the standard client. |
| DGX Spark CUDA/PyTorch compatibility | **MEDIUM** | Community reports vary on optimal PyTorch version for GB10. 2.4.x is the safest validated choice. |

---

## Sources

- [TRL SFTTrainer formatting_func evolution](https://zenn.dev/kewa8579/articles/0f52bc1863d8d4) (2026) — HIGH confidence
- [Unsloth Qwen3.5 Fine-tuning Guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune) — HIGH confidence
- [QLoRA Paper (NeurIPS 2023)](https://proceedings.nips.cc/paper_files/paper/2023/file/1feb87871436031bdc0f2beaa62a049b-Paper-Conference.pdf) — HIGH confidence
- [lm-eval-harness Releases](https://github.com/EleutherAI/lm-evaluation-harness/releases) — HIGH confidence
- [Ascend lm-eval Installation Guide](https://ascend.github.io/docs/sources/lm_evaluation/install.html) — HIGH confidence
- [Kimi K2.5 API Developer Guide](https://kimi-k25.com/blog/kimi-k2-5-api) (2026) — MEDIUM confidence
- [rouge-score Package Guide](https://generalistprogrammer.com/tutorials/rouge-python-package-guide) (2025) — MEDIUM confidence
- [Hugging Face Evaluate Library Guide](https://www.neura.market/directories/chatgpt/blog/hugging-face-evaluate-ultimate-guide-to-streamlining-model-evaluation-in-ml-workflows) (2025) — MEDIUM confidence
- [arXiv:2605.05561 — bitsandbytes NF4 reproducibility](https://arxiv.org/pdf/2605.05561) — HIGH confidence
- [DGX Spark Performance Tuning — NVIDIA Docs](https://docs.nvidia.com/dgx/dgx-spark/performance-tuning.html) — HIGH confidence
