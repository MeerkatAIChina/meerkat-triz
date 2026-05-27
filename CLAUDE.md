# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains the **猫鼬AI (Meerkat AI)** project — a TRIZ (Theory of Inventive Problem Solving) domain large language model fine-tuning system. The project fine-tunes Qwen3.6-35B-A3B using QLoRA on an NVIDIA DGX Spark (128GB Unified Memory) to create a specialized TRIZ innovation consultant AI.

**Project location on DGX Spark:** `/home/meerkat/mongoose_ai`

## Project Structure

Source code lives in `ref/mongoose_ai_dgx/`:

```
ref/mongoose_ai_dgx/
├── config.py              # Global configuration: hyperparameters, paths, hardware
├── requirements.txt       # Python dependencies
├── utils/
│   ├── __init__.py
│   ├── data_utils.py      # Data loading, ChatML conversion, synthetic data generation
│   ├── training_utils.py  # Model loading, QLoRA config, SFTTrainer creation
│   └── benchmark_utils.py # 3-layer evaluation (general + TRIZ + performance)
├── notebooks/             # Execute in strict order
│   ├── 01_download_and_setup.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_model_benchmark.ipynb      # Pre-fine-tuning baseline
│   ├── 04_qlora_finetune.ipynb       # Main training (8-15 hours)
│   └── 05_model_evaluation.ipynb     # Post-fine-tuning evaluation
├── data/
│   └── sample_data.json   # 548 TRIZ domain samples across 6 subsets
└── README.md              # Chinese-language project documentation
```

Reference documents in `ref/` (audit reports, strategic plans, evaluation schemes) provide business and technical context but are not executable code.

## Development Workflow

All work happens inside Jupyter notebooks on the DGX Spark. There is no traditional build/test command — the workflow is notebook-driven:

1. **Environment setup** (Notebook 01): Install deps, load model
2. **Data preparation** (Notebook 02): Load `sample_data.json`, convert to ChatML, split train/val/test
3. **Baseline benchmark** (Notebook 03): Run lm-eval-harness for general capability baseline
4. **QLoRA fine-tuning** (Notebook 04): The main training step
5. **Evaluation** (Notebook 05): TRIZ custom benchmarks + performance benchmarks

### Key Commands

```bash
# Install dependencies
pip install -r ref/mongoose_ai_dgx/requirements.txt

# Add project path (used in notebooks)
import sys
sys.path.append('/home/meerkat/mongoose_ai')

# From project root on DGX Spark:
cd /home/meerkat/mongoose_ai
```

## Architecture & Key Design Decisions

### Base Model
- **Qwen/Qwen3.6-35B-A3B** — 35B total params, 3B active (MoE), 262K context, Apache 2.0 license
- 4-bit quantized (NF4) for training; ~18-20GB model size
- Peak memory during training: ~60-80GB (well within 128GB)

### QLoRA Configuration
- Rank: 64, Alpha: 128, Dropout: 0.05
- Learning rate: 2e-4, Epochs: 2, Effective batch size: 8 (1 × 8 gradient accumulation)
- Optimizer: `paged_adamw_8bit`, Scheduler: cosine, Warmup: 5%

### Target Modules (Critical)
Qwen3.6 uses a hybrid architecture (Gated DeltaNet + Gated Attention + MoE). The `target_modules` must cover all three layer types:
- Gated Attention (10/40 layers): `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Gated DeltaNet (30/40 layers): `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`
- MoE MLP (all 40 layers): `gate_proj`, `up_proj`, `down_proj`

**Do NOT use `"all-linear"`** — it has known compatibility issues with this hybrid architecture and may incorrectly include `lm_head`. Use the explicit manual list above.

### Data Format
Training data uses ChatML format via `tokenizer.apply_chat_template()` (not hardcoded). Each sample has:
- `instruction`: Question text
- `input`: Optional supplementary input
- `output`: Expert answer
- `system`: TRIZ expert persona prompt

Six data subsets: `concept_explanation`, `contradiction_analysis`, `principle_recommendation`, `case_generation`, `ariz_guidance`, `innovation_assessment`.

### Training: SFTTrainer + formatting_func
The `create_trainer()` function in `training_utils.py` uses:
- `SFTTrainer` from `trl` with `formatting_func` parameter
- **No `data_collator` passed** — passing one conflicts with SFTTrainer's internal label-masking logic
- SFTTrainer automatically masks user tokens and only computes loss on assistant responses

### Three-Layer Evaluation
1. **Layer 1 — General benchmarks**: `lm-eval-harness` (MMLU-Pro, GPQA, HumanEval, MATH, BBH)
2. **Layer 2 — TRIZ custom**: Principle accuracy, contradiction resolution, case quality (BLEU/ROUGE), ARIZ completeness
3. **Layer 3 — Performance**: Throughput (tokens/s), P50 latency, peak memory

## Hardware Environment

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GB10 Grace Blackwell |
| Unified Memory | 128 GB |
| Memory Bandwidth | 273 GB/s |
| FP4 Compute | 1 PFLOPS |
| CPU | 20-core Grace CPU |

## Known Issues & Audit History

An independent audit (2026-05-23, see `ref/审计报告_猫鼬AI训练方案.md`) identified several critical issues that have been addressed in the current code:

- **CR-001 (FIXED)**: SFTTrainer + DataCollatorForLanguageModeling conflict — resolved by using `formatting_func` instead
- **CR-002 (FIXED)**: `target_modules="all-linear"` risk — resolved by using explicit manual module list
- **CR-003 (FIXED)**: Hardcoded ChatML format — resolved by using `tokenizer.apply_chat_template()`
- **MA-001 (FIXED)**: Sample data expanded from 12 to 548 samples in `sample_data.json`

Remaining concern: Synthetic data generation (`vary_sample()`) uses simple paraphrase templates. For production-quality training, a GPT-4o-based synthesis pipeline with expert review is recommended.

## Behavioral Guidelines

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
