# Roadmap: Meerkat AI (猫鼬AI)

## Overview

This roadmap delivers the v1.0 milestone: a complete end-to-end QLoRA fine-tuning pipeline for Qwen3.6-35B-A3B on the TRIZ domain. The journey moves from data generation (the critical path blocker) through baseline establishment and training execution, to post-training validation with before/after comparison. Each phase produces observable artifacts that the researcher can verify in the DGX Spark notebook environment.

## Phases

- [x] **Phase 1: Foundation & Data Pipeline** - Fix configs, build synthetic data pipeline, generate ~6K training samples
- [ ] **Phase 2: Baseline & Training Execution** - Run pre-training benchmarks, execute 15-hour QLoRA fine-tuning
- [ ] **Phase 3: Evaluation & Hardening** - Validate fine-tuning results, produce comparison report, harden cross-notebook integration
- [ ] **Phase 3.1: Close gap: BLK-01 — fix Layer 1 baseline comparison in Notebook 05** (INSERTED) - Urgent audit closure

## Phase Details

### Phase 1: Foundation & Data Pipeline
**Goal**: The researcher can generate ~6K high-quality synthetic TRIZ training samples and all pre-training configuration is verified correct.
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, INFRA-04, INFRA-05, INFRA-06, INFRA-07, INFRA-08, INFRA-09, TRAIN-02, TRAIN-03
**Success Criteria** (what must be TRUE):
  1. `pip install` from updated `requirements.txt` succeeds with all pinned versions on DGX Spark
  2. `utils/synthetic_pipeline.py` can generate semantically varied TRIZ samples via Moonshot API with batching and rate limiting
  3. `02b_synthetic_generation.ipynb` produces ~6K synthetic samples from 548 seeds with quality gates (perplexity filtering, diversity scoring, 20-30% real data ratio)
  4. Notebook 02 shows token length histogram that catches any samples exceeding model context limit before training
  5. `config.py` has `lora_dropout=0.0` and explicit 12-module `target_modules` list (verified by inspection)
  6. `utils/pipeline_state.py` persists JSON artifact registry accessible across all notebooks
  7. README.md no longer recommends `"all-linear"` and documents the explicit module list
  8. Notebook pre-flight checks verify paths, artifacts, and version compatibility before execution
**Plans**: 7 plans in 2 waves
  - [x] `01-01-PLAN.md` -- Config Hardening: fix config.py, requirements.txt, README.md
  - [x] `01-02-PLAN.md` -- Infrastructure Modules: create pipeline_state.py, synthetic_pipeline.py, update __init__.py
  - [x] `01-03-PLAN.md` -- Synthetic Generation Notebook: create 02b_synthetic_generation.ipynb
  - [x] `01-04-PLAN.md` -- Notebook Enhancements: add token profiling to Notebook 02, pre-flight checks to Notebook 01
  - [x] `01-05-PLAN.md` -- [GAP] Fix training_utils.py docstring and defaults
  - [x] `01-06-PLAN.md` -- [GAP] Add perplexity/diversity quality gates + document real ratio deviation
  - [x] `01-07-PLAN.md` -- [GAP] Create pytest test stubs

### Phase 2: Baseline & Training Execution
**Goal**: The researcher has valid pre-training baselines and a completed QLoRA fine-tuning run with verified checkpoints.
**Depends on**: Phase 1
**Requirements**: BENCH-01, BENCH-02, BENCH-03, BENCH-04, BENCH-05, BENCH-06, TRAIN-01, TRAIN-04, TRAIN-05, TRAIN-06, TRAIN-07, TRAIN-08, TRAIN-09, TRAIN-10
**Success Criteria** (what must be TRUE):
  1. Notebook 03 runs baseline benchmark with model loaded in FP16 (not 4-bit), producing Layer 2 TRIZ and Layer 3 performance scores
  2. Baseline results are automatically persisted to `pipeline_state` registry for later comparison
  3. Notebook 04 executes QLoRA fine-tuning end-to-end for 2 epochs without manual intervention
  4. Training saves checkpoints every 200 steps with `save_total_limit=3` and comprehensive adapter metadata
  5. Immediate load-and-forward-pass verification succeeds after each checkpoint save
  6. Checkpoint resume works correctly: interrupted training can resume with data iterator state and LR scheduler continuity verified
  7. Training uses `SFTTrainer` with `formatting_func` + `packing=True`, no `data_collator` passed
**Plans**: 3 plans in 2 waves
  - [ ] `02-01-PLAN.md` -- Utility Enhancements: add CheckpointValidationCallback, SHA-256 metadata, LR-verified resume to training_utils.py
  - [ ] `02-02-PLAN.md` -- Baseline Benchmark Notebook: enhance Notebook 03 with FP16 loading, all 3 benchmark layers, pipeline_state persistence
  - [ ] `02-03-PLAN.md` -- QLoRA Training Notebook: rebuild Notebook 04 with pre-flight checks, checkpoint callback, resume cell, adapter metadata

### Phase 3: Evaluation & Hardening
**Goal**: The researcher can verify fine-tuning improved TRIZ capabilities and the pipeline is hardened for reproducible future runs.
**Depends on**: Phase 2
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05
**Success Criteria** (what must be TRUE):
  1. Notebook 05 automatically loads baseline results from `pipeline_state` and runs post-training evaluation without manual score entry
  2. Before/after comparison report is generated showing Layer 2 TRIZ and Layer 3 performance deltas
  3. Unified `format_messages()` utility replaces all hardcoded ChatML strings in `benchmark_utils.py` and evaluation paths
  4. Adapter loads correctly via `AutoPeftModelForCausalLM` for evaluation inference
  5. TRIZ case quality scoring computes BLEU/ROUGE metrics for generated cases
  6. (Optional) One Layer 1 task spot-check runs if time permits
**Plans**: 4 plans in 3 waves
  - [ ] `03-01-PLAN.md` -- format_messages Utility + sacrebleu dependency
  - [ ] `03-02-PLAN.md` -- Benchmark Utils Hardening: unified prompts, BLEU/ROUGE, before/after report
  - [ ] `03-03-PLAN.md` -- Notebook 05 Evaluation Orchestration: pre-flight, adapter/base eval, comparison report
  - [ ] `03-04-PLAN.md` -- Test Suite: format_messages, metrics, report structure

### Phase 3.1: Close gap: BLK-01 — fix Layer 1 baseline comparison in Notebook 05 (INSERTED)
**Goal**: Notebook 05 loads actual Layer 1 benchmark scores from the baseline result file so the before/after comparison report computes meaningful deltas.
**Depends on**: Phase 3
**Requirements**: EVAL-01, EVAL-05, BENCH-05
**Success Criteria** (what must be TRUE):
  1. Notebook 03 registers `baseline_results` with either (a) Layer 1 scores embedded in metadata or (b) a `path` that Notebook 05 reads.
  2. Notebook 05 loads `general_results` from the actual Layer 1 benchmark JSON, not from `baseline['metadata']`.
  3. `aggregate_results(before_results=..., after_results=...)` receives valid `layer1_general` scores for both base and adapter runs.
  4. The before/after report displays Layer 1 metric deltas (± and % change) for each general benchmark task.
**Plans**: 4 plans in 3 waves
  - [ ] `03.1-00-PLAN.md` -- Wave 0 Tests: create pytest fixtures and failing stubs for Layer 1 delta behavior
  - [ ] `03.1-01-PLAN.md` -- Utility Layer: extend aggregate_results() with Layer 1 delta support
  - [ ] `03.1-02-PLAN.md` -- Notebook 03: register baseline_results with metadata.layer1_path and layer1_summary
  - [ ] `03.1-03-PLAN.md` -- Notebook 05: load validated Layer 1 baseline, re-run adapter Layer 1, display deltas

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation & Data Pipeline | 7/7 | Complete | 2026-05-28 |
| 2. Baseline & Training Execution | 0/3 | Planned | - |
| 3. Evaluation & Hardening | 0/4 | Planned | - |
| 3.1. Close gap: BLK-01 — fix Layer 1 baseline comparison in Notebook 05 | 2/4 | In Progress|  |
