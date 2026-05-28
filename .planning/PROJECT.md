# PROJECT.md

## What This Is

**猫鼬AI (Meerkat AI)** — A TRIZ domain large language model fine-tuning system. Fine-tunes Qwen3.6-35B-A3B using QLoRA on an NVIDIA DGX Spark (128GB Unified Memory) to create a specialized TRIZ innovation consultant AI.

## Core Value

Transform a general-purpose 35B-parameter MoE model into a world-class TRIZ innovation consultant through targeted domain fine-tuning, with rigorous three-layer evaluation (general capability, TRIZ expertise, performance).

## Requirements

### Validated
- [x] Qwen3.6-35B-A3B base model with QLoRA fine-tuning
- [x] 548 seed TRIZ samples across 6 subsets
- [x] Synthetic data generation pipeline (6-stage, Moonshot API)
- [x] Notebook-driven workflow (01–05 + 02b)
- [x] DGX Spark hardware environment (128GB unified memory)

### Active
- [ ] Generate ~6K synthetic training samples from 548 seeds
- [ ] Baseline benchmark (03) before training
- [ ] QLoRA fine-tuning run (04)
- [ ] Post-training evaluation (05)

### Out of Scope
- Full fine-tuning (resource constraints; QLoRA only)
- Multi-GPU training (single DGX Spark node)
- Deployment/inference serving (training focus only)

## Key Decisions

| Date | Decision | Outcome |
|------|----------|---------|
| 2026-05-26 | Use SSH (not HTTPS) for GitHub remote | SSH avoids credential prompts in non-interactive sessions |
| 2026-05-26 | Use git filter-branch to rewrite history and remove venv | GitHub 2GB pack limit enforced; original 3GB repo could not push |
| 2026-05-26 | Add .gitignore excluding mai/, __pycache__/, .ipynb_checkpoints/ | Prevent future accidental commits of large/generated files |

## Current Milestone: v1.0 First Training Run

**Goal:** Generate synthetic training data, establish baseline, fine-tune Qwen3.6-35B-A3B with QLoRA, and evaluate results.

**Phase 02 complete (2026-05-28):** Training utilities enhanced with checkpoint validation, comprehensive adapter metadata, and verified resume capability. Notebook 03 (baseline benchmark) upgraded to FP16 loading with three-layer benchmarks and pipeline_state persistence. Notebook 04 (QLoRA training) rebuilt as a complete 21-cell self-contained training pipeline with pre-flight checks, CheckpointValidationCallback, and checkpoint resume.

**Target features:**
- Complete synthetic data generation (~6K samples from 548 seeds)
- Baseline benchmark (03_model_benchmark.ipynb) — notebook ready, execution on DGX Spark pending
- QLoRA fine-tuning run (04_qlora_finetune.ipynb) — notebook ready, execution on DGX Spark pending
- Post-training evaluation (05_model_evaluation.ipynb)

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

## Constraints

- Single NVIDIA GB10 (128GB unified memory)
- Training run: 8–15 hours for 2 epochs
- GitHub repo: coidea-ai/mai
