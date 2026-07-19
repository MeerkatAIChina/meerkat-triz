# PROJECT.md

## What This Is

**猫鼬AI (Meerkat AI)** — A TRIZ domain large language model fine-tuning system. Fine-tunes Qwen3.6-35B-A3B using QLoRA on an NVIDIA DGX Spark (128GB Unified Memory) to create a specialized TRIZ innovation consultant AI.

## Core Value

Transform a general-purpose 35B-parameter MoE model into a world-class TRIZ innovation consultant through targeted domain fine-tuning, with rigorous three-layer evaluation (general capability, TRIZ expertise, performance).

## Requirements

### Validated
- [x] Qwen3.6-35B-A3B base model with QLoRA fine-tuning — v1.0
- [x] 548 seed TRIZ samples across 6 subsets — v1.0
- [x] Synthetic data generation pipeline (6-stage, Moonshot API) — v1.0
- [x] Notebook-driven workflow (01–05 + 02b) — v1.0
- [x] DGX Spark hardware environment (128GB unified memory) — v1.0
- [x] Generate ~6K synthetic training samples from 548 seeds — v1.0
- [x] Baseline benchmark (03) before training — v1.0
- [x] QLoRA fine-tuning run (04) — v1.0
- [x] Post-training evaluation (05) — v1.0
- [x] Raw corpus builder workflow (02c/02d) validated on DGX Spark — v1.0 post-close
- [x] End-to-end QLoRA training run on the TRIZ-raw corpus SFT dataset (2,662 train / 313 val / 157 test): 666 steps, 2 epochs, all checkpoint validations PASSED, best eval_loss 1.3979 — 2026-06-19

### Active
- [ ] Open-source the fine-tuned LoRA adapter on Hugging Face with model card, license, and dataset attribution
- [ ] Expand TRIZ test set from 5 to 50–100 questions for more robust Layer 2 evaluation
- [ ] Run full Layer 1 suite (MMLU-Pro, GPQA, HumanEval, MATH, BBH) post-training
- [ ] Evaluate and align `requirements.txt` with Notebook 01/04 dependency specifications

### Out of Scope
- Full fine-tuning (resource constraints; QLoRA only)
- Multi-GPU training (single DGX Spark node)
- Deployment/inference serving (training focus only)
- vLLM, Unsloth, Flash Attention, DeepSpeed, or RAGAS integration

## Key Decisions

| Date | Decision | Outcome |
|------|----------|---------|
| 2026-05-26 | Use SSH (not HTTPS) for GitHub remote | SSH avoids credential prompts in non-interactive sessions |
| 2026-05-26 | Use git filter-branch to rewrite history and remove venv | GitHub 2GB pack limit enforced; original 3GB repo could not push |
| 2026-05-26 | Add .gitignore excluding mai/, __pycache__/, .ipynb_checkpoints/ | Prevent future accidental commits of large/generated files |
| 2026-05-28 | Set `lora_dropout=0.0` for MoE architecture compatibility | Dropout can destabilize expert routing |
| 2026-05-28 | Use explicit 12-module `target_modules` list instead of `all-linear` | Avoids known compatibility issues with Qwen3.6 hybrid architecture |
| 2026-05-28 | Use `SFTTrainer` with `formatting_func` + `packing=True`; no `data_collator` | Resolves conflict with SFTTrainer's internal label-masking logic |
| 2026-06-30 | Insert Phase 3.1 to close audit blocker BLK-01 | Layer 1 baseline comparison now computes real lm-eval deltas |
| 2026-07-12 | Move source code from `ref/mongoose_ai_dgx/` to repo root | Repo layout matches deployed DGX Spark path |
| 2026-07-12 | Use TRIZ-raw corpus as the training dataset for the open-source release | Aligns the published model with real TRIZ source material rather than synthetic samples |
| 2026-07-18 | Sync DGX Spark drift back into repo (2 test files with paths fixed to root layout, Notebook 01 4-bit cell, 04_worked.ipynb run record) | Remote `.git` has no commits; local repo is canonical and drift must be pulled manually |

## Current State

**Milestone v1.0 MVP shipped (2026-07-12).**

All 4 phases (1, 2, 3, 3.1) are complete with 18/18 plans summarized. The notebook-driven QLoRA fine-tuning pipeline is fully implemented:

- **Data**: `utils/synthetic_pipeline.py` + Notebook 02b generate ~6K synthetic TRIZ samples from 548 seeds with perplexity/diversity gates and pipeline-state registration. The raw corpus builder workflow (02c/02d) has also been validated on the DGX Spark.
- **Baseline**: Notebook 03 loads the model in FP16, runs all three benchmark layers, and persists baseline results.
- **Training**: Notebook 04 executes QLoRA fine-tuning with checkpoint validation, SHA-256 metadata, and verified resume.
- **Evaluation**: Notebook 05 loads baseline and adapter, runs Layer 2/3 TRIZ benchmarks, and — after Phase 3.1 — displays per-task Layer 1 lm-eval deltas.

Source code now lives at the repo root, matching the DGX Spark deployment layout at `/home/meerkat/mongoose_ai`.

A full QLoRA training run completed on the DGX Spark on **2026-06-19** using the TRIZ-raw corpus SFT dataset (`data/processed/train.jsonl`, 2,662 samples): 666 steps over 2 epochs, checkpoint validations PASSED at steps 200/400/600/666, best eval_loss 1.3979. The final adapter (169 MB, SHA-256 `1f909cb0…`) is registered in pipeline_state as `adapter_checkpoint` at `models/meerkat_triz_adapter_v1/`; the executed run record is preserved in `notebooks/04_worked.ipynb`.

Deployment sync verified 2026-07-18: all source files (`config.py`, `utils/`, `scripts/`, notebooks 02–05) are SHA-256-identical between repo and DGX Spark. Drift items were pulled back into the repo: `tests/test_metrics.py` + `tests/test_report.py` (paths fixed from the old `ref/mongoose_ai_dgx/` layout), Notebook 01's working 4-bit loading cell, and `notebooks/04_worked.ipynb`. Note: the remote `.git` has no commits, so the local repo (GitHub coidea-ai/mai) is canonical.

## Next Milestone Goals

The v1.1 milestone should focus on **execution, open-source release, and hardening**:

1. ~~Train the LoRA adapter on the TRIZ-raw corpus (via Notebooks 02c/02d → 04).~~ ✅ Completed 2026-06-19 (666 steps, best eval_loss 1.3979).
2. Open-source the fine-tuned adapter on Hugging Face with a complete model card, Apache 2.0 license, and TRIZ-raw dataset attribution.
3. Fix any runtime issues discovered during DGX Spark training runs (e.g., 4-bit quantization, dependency alignment) — Notebook 01's 4-bit loading fix synced back 2026-07-18.
4. Expand evaluation coverage (Layer 1 post-training, larger TRIZ test set).

## Context

- ~5,925 lines of Python across `utils/`, `scripts/`, and `tests/`.
- 13 pytest tests mock heavy dependencies (torch, transformers, datasets) so they can run locally.
- Known deferred items at v1.0 close: Phase 03.1 UAT/verification artifacts remain partial; documented in STATE.md.

## Constraints

- Single NVIDIA GB10 (128GB unified memory)
- Training run: 8–15 hours for 2 epochs
- GitHub repo: coidea-ai/mai

---
*Last updated: 2026-07-18 after DGX Spark deployment sync*
