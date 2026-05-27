---
phase: 01-foundation-data-pipeline
plan: "01"
subsystem: infrastructure
tags: [config, dependencies, documentation, qlora, synthetic-data]
dependency_graph:
  requires: []
  provides: [INFRA-04, INFRA-08, TRAIN-02, TRAIN-03]
  affects: [01-02-PLAN.md, 01-03-PLAN.md]
tech_stack:
  added: []
  patterns: [centralized-config, pinned-dependencies, explicit-module-list]
key_files:
  created: []
  modified:
    - ref/mongoose_ai_dgx/config.py
    - ref/mongoose_ai_dgx/requirements.txt
    - ref/mongoose_ai_dgx/README.md
decisions:
  - "Set lora_dropout=0.0 for MoE architecture compatibility (dropout can destabilize expert routing)"
  - "Use explicit 12-module target_modules list instead of all-linear (known compatibility issues with Qwen3.6 hybrid architecture)"
  - "Pin all critical package versions to prevent supply-chain breaking changes"
  - "Add SYNTHETIC_CONFIG to config.py for centralized synthetic generation pipeline settings"
metrics:
  duration_minutes: 4
  completed_date: "2026-05-27T22:07:10Z"
  tasks_completed: 3
  files_modified: 3
  lines_changed: 80
---

# Phase 01 Plan 01: Fix Pre-Training Configuration Files Summary

**One-liner:** Fixed critical pre-training config bugs — lora_dropout=0.0 for MoE stability, pinned all dependencies, replaced all-linear recommendation with explicit 12-module list, and added SYNTHETIC_CONFIG for the data generation pipeline.

## What Was Done

### Task 1: Fix config.py (Commit: 488f352)

Modified `ref/mongoose_ai_dgx/config.py` with three changes:

1. **lora_dropout: 0.05 → 0.0** — Dropout can destabilize expert routing in MoE architectures. Added explanatory comment.
2. **Verified target_modules** — Confirmed exactly 12 explicit module names covering all three Qwen3.6 layer types (Gated Attention, Gated DeltaNet, MoE MLP). Added comment warning against `"all-linear"`.
3. **Added SYNTHETIC_CONFIG section** — Complete configuration for the synthetic data generation pipeline including:
   - Moonshot API settings (base_url, model, rpm, batch_size, temperature)
   - Per-subset multipliers (6x for fact-based subsets, 11x for mixed, 16x for diversity-based)
   - Generation strategies (rephrase / mixed / generate_new)
   - Quality gates (max_tokens=3500, deduplicate=True)
   - Output and checkpoint directory paths

### Task 2: Update requirements.txt (Commit: 5959b63)

Replaced unpinned `>=` versions with exact `==` pins for all critical packages:

- `transformers==4.45.0` (was `>=4.45.0`)
- `peft==0.12.0` (was `>=0.12.0`)
- `bitsandbytes==0.43.3` (was `>=0.43.0`)
- `accelerate==0.33.0` (was `>=0.33.0`)
- `datasets==2.21.0` (was `>=2.21.0`)
- `trl==0.9.6` (was `>=0.9.0`)
- `lm-eval==0.4.10` (was `>=0.4.3`)
- Added `openai==1.35.0` — Moonshot API client (OpenAI-compatible)
- Added `rouge-score==0.1.2` — BLEU/ROUGE quality scoring for synthetic data gates
- Removed the `all-linear` / rsLoRA comment from the peft line

### Task 3: Fix README.md (Commit: 8cdfe5a)

Updated documentation with three changes:

1. **Replaced "all-linear" as default recommendation** — Now recommends explicit 12-module list as primary approach. Marked `"all-linear"` as not recommended (known to incorrectly include `lm_head` in hybrid architectures).
2. **Updated QLoRA config example** — Changed `lora_dropout` from `0.05` to `0.0` with MoE compatibility explanation.
3. **Added 合成数据生成 section** — Documents the Moonshot API setup, per-subset generation strategies, and quality gates (deduplication, length filtering, ChatML format validation).

## Deviations from Plan

None — plan executed exactly as written.

## Threat Flags

No new security-relevant surface introduced. All changes are configuration/documentation fixes within existing trust boundaries.

## Known Stubs

No stubs introduced. All changes are concrete configuration values and documentation.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | 488f352 | feat(01-01): fix config.py — lora_dropout=0.0, verify 12 target_modules, add SYNTHETIC_CONFIG |
| 2 | 5959b63 | chore(01-01): pin all critical dependency versions and add new packages |
| 3 | 8cdfe5a | docs(01-01): fix README.md — replace all-linear recommendation, add synthetic data section |

## Self-Check: PASSED

- [x] `config.py` has `lora_dropout=0.0`
- [x] `config.py` has no `lora_dropout=0.05`
- [x] `config.py` has exactly 12 `target_modules`
- [x] `config.py` has `SYNTHETIC_CONFIG` with API params, multipliers, strategies
- [x] `requirements.txt` has `transformers==4.45.0`
- [x] `requirements.txt` has `trl==0.9.6`
- [x] `requirements.txt` has `peft==0.12.0`
- [x] `requirements.txt` has `bitsandbytes==0.43.3`
- [x] `requirements.txt` has `openai==1.35.0`
- [x] `requirements.txt` has `rouge-score==0.1.2`
- [x] `requirements.txt` has `lm-eval==0.4.10`
- [x] `requirements.txt` has no `all-linear` mention
- [x] `README.md` recommends explicit module list (显式模块列表)
- [x] `README.md` marks all-linear as not recommended (不推荐)
- [x] `README.md` has no old "默认使用 all-linear" text
- [x] `README.md` shows `lora_dropout=0.0` in config example
- [x] `README.md` has 合成数据生成 section
- [x] `README.md` documents `MOONSHOT_API_KEY` setup
