# Phase 02: baseline-training-execution - Context

**Gathered:** 2026-05-28
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers:
1. Pre-training baseline benchmarks (Notebook 03) with model loaded in FP16
2. A completed QLoRA fine-tuning run (Notebook 04) for 2 epochs (~15 hours)
3. Verified checkpoints with immediate forward-pass validation
4. All results persisted to pipeline_state registry for Phase 3 comparison

Scope does NOT include: synthetic data generation (Phase 01), post-training evaluation (Phase 03), deployment.
</domain>

<decisions>
## Implementation Decisions

### Benchmark Scope (D-01)
- **Baseline (Notebook 03) runs Layer 1 general benchmarks only** — MMLU-Pro, GPQA, HumanEval, MATH, BBH via lm-eval-harness
- **Layer 2 TRIZ custom benchmarks run ONLY post-training** (Phase 03, Notebook 05)
- Rationale: Layer 2 requires inference on the trained adapter; running it at baseline with the base model is possible but not required by success criteria. Phase 03's before/after comparison will use the baseline's Layer 1 + post-training Layer 1 + post-training Layer 2.

### Training Data Readiness (D-02)
- **Phase 02 assumes synthetic data is ready** (~6K samples from Phase 01)
- If synthetic data is not generated when Phase 02 starts, Notebook 02b must be run first
- Notebook 04 reads from the pipeline_state registry to locate the processed dataset artifact

### Checkpoint Verification (D-03)
- **Every checkpoint save (every 200 steps) gets immediate forward-pass validation**
- Verification: load adapter checkpoint, run a single forward pass on a test sample, confirm no errors
- `save_total_limit=3` — only the 3 most recent checkpoints are kept
- Checkpoint metadata includes: step, loss, timestamp, SHA-256 of adapter weights

### Training Interruption Handling (D-04)
- **Dual resume strategy:**
  1. SFTTrainer's built-in `resume_from_checkpoint` for automatic recovery
  2. Cell-level checkpoint resume in Notebook 04 — a dedicated cell that loads the latest checkpoint and resumes training with data iterator state and LR scheduler continuity
- Training is idempotent: running the same notebook twice with the same data produces the same adapter (deterministic seed fixed)

### Notebook Execution Order (D-05)
- **Strict sequence:** 01 (setup) -> 02b (generate synthetic data if needed) -> 03 (baseline) -> 04 (training)
- Notebook 03 must complete and persist baseline to pipeline_state before Notebook 04 starts
- Notebook 04 pre-flight cell verifies: baseline exists in pipeline_state, data artifact exists, GPU memory > 60GB available

### Claude's Discretion
- Specific lm-eval-harness task selection (which subsets of MMLU-Pro, exact HumanEval pass@k)
- Progress bar/logging verbosity during training
- Exact checkpoint metadata format (beyond step/loss/timestamp/SHA-256)
- Whether to log to Weights & Biases or similar (not required by success criteria)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Architecture & Config
- `ref/mongoose_ai_dgx/config.py` — QLORA_CONFIG, DATA_CONFIG, SYNTHETIC_CONFIG, SYSTEM_PROMPT
- `ref/mongoose_ai_dgx/utils/training_utils.py` — setup_qlora_config(), load_model_and_tokenizer(), create_trainer()
- `ref/mongoose_ai_dgx/utils/benchmark_utils.py` — baseline evaluation functions
- `ref/mongoose_ai_dgx/utils/pipeline_state.py` — artifact registry for cross-notebook state

### Notebooks
- `ref/mongoose_ai_dgx/notebooks/03_model_benchmark.ipynb` — existing baseline benchmark notebook
- `ref/mongoose_ai_dgx/notebooks/04_qlora_finetune.ipynb` — existing training notebook
- `ref/mongoose_ai_dgx/notebooks/02b_synthetic_generation.ipynb` — synthetic data generation (if needed)

### Requirements
- `.planning/REQUIREMENTS.md` — BENCH-01 through BENCH-06, TRAIN-01 through TRAIN-10

### Prior Phase Context
- `.planning/phases/01-foundation-data-pipeline/01-CONTEXT.md` — decisions about target_modules, lora_dropout, data format
- `.planning/phases/01-foundation-data-pipeline/01-RESEARCH.md` — known pitfalls (SFTTrainer + data collator conflict, all-linear risk)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `utils/training_utils.py` — setup_qlora_config(), load_model_and_tokenizer(), create_trainer() already implemented
- `utils/benchmark_utils.py` — benchmark infrastructure already exists
- `utils/pipeline_state.py` — artifact registry for persisting baseline results
- Notebooks 03 and 04 already exist with structure; need enhancement not rewrite

### Established Patterns
- Notebook-driven workflow: each notebook is self-contained with setup cells
- SFTTrainer + formatting_func (no data_collator) — established in Phase 01
- ChatML format via tokenizer.apply_chat_template() — not hardcoded
- FP16 for inference, 4-bit (NF4) for training

### Integration Points
- Notebook 03 -> pipeline_state.register("baseline_results", ...)
- Notebook 04 -> pipeline_state.get("processed_dataset") for training data path
- Notebook 04 -> pipeline_state.register("adapter_checkpoint", ...) after training
- Checkpoint resume cell in Notebook 04 -> reads latest checkpoint from output_dir

</code_context>

<specifics>
## Specific Ideas

- Training success criteria from ROADMAP: 2 epochs, checkpoints every 200 steps, save_total_limit=3, immediate forward-pass verification after each save
- Baseline must use FP16 (not 4-bit) for accurate pre-training scores
- Training uses SFTTrainer with packing=True, formatting_func, no data_collator
- Hardware: DGX Spark with 128GB unified memory, ~60-80GB peak during training

</specifics>

<deferred>
## Deferred Ideas

- Weights & Biases integration for training visualization — belongs in future enhancement phase
- Multi-GPU training — out of scope (single DGX Spark node)
- Layer 2 baseline benchmarking with base model — possible but not required by success criteria
- Real-time training monitoring dashboard — future enhancement

</deferred>

---

*Phase: 02-baseline-training-execution*
*Context gathered: 2026-05-28*
