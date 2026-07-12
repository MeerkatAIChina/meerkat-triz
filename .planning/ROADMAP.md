# Roadmap: Meerkat AI (猫鼬AI)

## Milestones

- ✅ **v1.0 MVP** — Phases 1-3.1 (shipped 2026-07-12)

## Phases

<details>
<summary>✅ v1.0 MVP (Phases 1-3.1) — SHIPPED 2026-07-12</summary>

### Phase 1: Foundation & Data Pipeline
**Goal**: The researcher can generate ~6K high-quality synthetic TRIZ training samples and all pre-training configuration is verified correct.
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, INFRA-04, INFRA-05, INFRA-06, INFRA-07, INFRA-08, INFRA-09, TRAIN-02, TRAIN-03
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
**Plans**: 3 plans in 2 waves
  - [x] `02-01-PLAN.md` -- Utility Enhancements: add CheckpointValidationCallback, SHA-256 metadata, LR-verified resume to training_utils.py
  - [x] `02-02-PLAN.md` -- Baseline Benchmark Notebook: enhance Notebook 03 with FP16 loading, all 3 benchmark layers, pipeline_state persistence
  - [x] `02-03-PLAN.md` -- QLoRA Training Notebook: rebuild Notebook 04 with pre-flight checks, checkpoint callback, resume cell, adapter metadata

### Phase 3: Evaluation & Hardening
**Goal**: The researcher can verify fine-tuning improved TRIZ capabilities and the pipeline is hardened for reproducible future runs.
**Depends on**: Phase 2
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05
**Plans**: 4 plans in 3 waves
  - [x] `03-01-PLAN.md` -- format_messages Utility + sacrebleu dependency
  - [x] `03-02-PLAN.md` -- Benchmark Utils Hardening: unified prompts, BLEU/ROUGE, before/after report
  - [x] `03-03-PLAN.md` -- Notebook 05 Evaluation Orchestration: pre-flight, adapter/base eval, comparison report
  - [x] `03-04-PLAN.md` -- Test Suite: format_messages, metrics, report structure

### Phase 3.1: Close gap: BLK-01 — fix Layer 1 baseline comparison in Notebook 05 (INSERTED)
**Goal**: Notebook 05 loads actual Layer 1 benchmark scores from the baseline result file so the before/after comparison report computes meaningful deltas.
**Depends on**: Phase 3
**Requirements**: EVAL-01, EVAL-05, BENCH-05
**Plans**: 4 plans in 3 waves
  - [x] `03.1-00-PLAN.md` -- Wave 0 Tests: create pytest fixtures and failing stubs for Layer 1 delta behavior
  - [x] `03.1-01-PLAN.md` -- Utility Layer: extend aggregate_results() with Layer 1 delta support
  - [x] `03.1-02-PLAN.md` -- Notebook 03: register baseline_results with metadata.layer1_path and layer1_summary
  - [x] `03.1-03-PLAN.md` -- Notebook 05: load validated Layer 1 baseline, re-run adapter Layer 1, display deltas

</details>

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Foundation & Data Pipeline | v1.0 | 7/7 | Complete | 2026-05-28 |
| 2. Baseline & Training Execution | v1.0 | 3/3 | Complete | 2026-05-28 |
| 3. Evaluation & Hardening | v1.0 | 4/4 | Complete | 2026-07-11 |
| 3.1. Close gap: BLK-01 | v1.0 | 4/4 | Complete | 2026-06-30 |
