# Phase 03: Evaluation & Hardening - Context

**Gathered:** 2026-05-28
**Status:** Ready for planning
**Source:** discuss-phase

<domain>
## Phase Boundary

This phase delivers:
1. Post-training evaluation (Notebook 05) with automatic baseline loading from `pipeline_state`
2. Before/after comparison report showing Layer 2 TRIZ and Layer 3 performance deltas
3. Unified `format_messages()` utility replacing all hardcoded ChatML strings in `benchmark_utils.py` and evaluation paths
4. Adapter loading via `AutoPeftModelForCausalLM` for evaluation inference
5. BLEU/ROUGE metrics for TRIZ case quality scoring
6. (Optional) One Layer 1 task spot-check if time permits

Scope does NOT include: synthetic data generation (Phase 01), baseline benchmarking (Phase 02), training execution (Phase 02), deployment.

**Note on Phase 2 D-01 conflict:** Phase 2 decided Layer 2 TRIZ benchmarks run ONLY post-training. Phase 3 overrides this for evaluation comparison purposes — Notebook 05 re-runs Layer 2 on the base model (without adapter) to produce true before/after deltas.
</domain>

<decisions>
## Implementation Decisions

### Before/After Comparison Scope (D-01)
- **Notebook 05 loads the base model (without adapter) and runs the full Layer 2 TRIZ benchmarks for comparison**
- This produces true before/after deltas for principle accuracy, contradiction resolution, case quality, and ARIZ completeness
- Adds ~5-10 minutes to evaluation time but satisfies Phase 3 success criteria #2
- Layer 1 general benchmarks are loaded from `pipeline_state` baseline registry (not re-run)
- Layer 3 performance benchmarks run on both base model and adapter for throughput/latency/memory comparison

### Comparison Report Format (D-02)
- **Structured JSON report saved to `results/` plus rich inline display in Notebook 05**
- JSON file: `results/evaluation_report_YYYYMMDD_HHMMSS.json` with full before/after data
- Notebook inline display: formatted tables showing deltas with +/- indicators and percentage changes
- Report includes all three layers: Layer 1 (from pipeline_state), Layer 2 (re-run on base model), Layer 3 (re-run on both)

### format_messages() Utility (D-03 — Claude's Discretion)
- **User deferred to Claude's discretion on location and implementation**
- Must replace hardcoded ChatML strings in:
  - `benchmark_utils.py` `_build_prompt()` method (currently hardcodes `<|im_start|>system...`)
  - Notebook 05 cell 6 (hardcoded ChatML for test case prompts)
- Must use `tokenizer.apply_chat_template()` — never hardcode ChatML tokens
- Must pull system prompt from `DATA_CONFIG['chatml']['system_message']` in `config.py`
- Should handle both evaluation prompts (single-turn Q&A) and training data formatting

### Missing Baseline Handling (D-04)
- **If `pipeline_state` has no baseline results, Notebook 05 auto-runs a quick baseline on-the-fly**
- Quick baseline = minimal Layer 1 (one representative task) + Layer 3 performance on base model
- Full Layer 2 TRIZ benchmarks are always run on base model during evaluation (for comparison), so those don't need to be in pipeline_state
- Auto-run adds ~3-5 minutes but makes Notebook 05 self-contained
- Print a warning: "Baseline not found in pipeline_state — running quick baseline now. Run Notebook 03 for full baseline."

### Adapter Loading (D-05)
- **Use `AutoPeftModelForCausalLM.from_pretrained()` for evaluation inference**
- `torch_dtype=torch.float16`, `device_map='auto'`, `trust_remote_code=True`
- Adapter path: `MODELS_DIR / 'meerkat_triz_adapter_v1'` (from config.py)
- Base model path: `MODELS_DIR / BASE_MODEL.split('/')[-1]`

### Notebook Execution Order (D-06)
- **Strict sequence maintained:** 01 (setup) -> 02b (data if needed) -> 03 (baseline) -> 04 (training) -> 05 (evaluation)
- Notebook 05 pre-flight checks verify: adapter exists, base model exists, pipeline_state accessible
- If training was interrupted and resumed, Notebook 05 uses the final adapter checkpoint (not intermediate ones)

### Claude's Discretion
- Specific `format_messages()` function signature and implementation details
- Exact JSON schema for the evaluation report
- Which Layer 1 task to use for spot-check (if time permits)
- Visual formatting of delta tables in notebook output (color coding, etc.)
- Exact BLEU/ROUGE reference data strategy (seed outputs vs generated references)
- Progress bar/logging verbosity during evaluation

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Architecture & Config
- `ref/mongoose_ai_dgx/config.py` — DATA_CONFIG['chatml']['system_message'], BENCHMARK_CONFIG, MODELS_DIR, BASE_MODEL
- `ref/mongoose_ai_dgx/utils/benchmark_utils.py` — TRIZBenchmark class, _build_prompt() (hardcoded ChatML), evaluate_case_quality() (incomplete ROUGE)
- `ref/mongoose_ai_dgx/utils/data_utils.py` — convert_to_chatml(), load_raw_data()
- `ref/mongoose_ai_dgx/utils/pipeline_state.py` — artifact registry for baseline loading

### Notebooks
- `ref/mongoose_ai_dgx/notebooks/05_model_evaluation.ipynb` — existing evaluation notebook with hardcoded ChatML and hardcoded before_score
- `ref/mongoose_ai_dgx/notebooks/03_model_benchmark.ipynb` — baseline notebook that persists to pipeline_state

### Requirements
- `.planning/REQUIREMENTS.md` — EVAL-01 through EVAL-05

### Prior Phase Context
- `.planning/phases/01-foundation-data-pipeline/01-CONTEXT.md` — decisions about target_modules, lora_dropout, data format
- `.planning/phases/02-baseline-training-execution/02-CONTEXT.md` — D-01 (Layer 2 post-training only), D-03 (checkpoint verification), D-05 (notebook execution order)
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `utils/benchmark_utils.py` — TRIZBenchmark class with principle_accuracy, contradiction_resolution, case_quality, ariz_completeness methods
- `utils/benchmark_utils.py` — run_performance_benchmark() for Layer 3 metrics
- `utils/benchmark_utils.py` — aggregate_results() for report generation (needs before/after delta support)
- `utils/pipeline_state.py` — artifact registry for loading baseline results
- `utils/data_utils.py` — convert_to_chatml() for ChatML formatting via tokenizer

### Established Patterns
- Notebook-driven workflow: each notebook is self-contained with setup cells
- ChatML format via tokenizer.apply_chat_template() — not hardcoded (established in Phase 01, violated in benchmark_utils.py)
- FP16 for inference (base model and adapter both loaded in FP16)
- SFTTrainer + formatting_func (no data_collator) — established in Phase 01
- Pipeline state registry for cross-notebook artifact tracking

### Integration Points
- Notebook 05 -> pipeline_state.get("baseline_results") for before/after comparison
- Notebook 05 -> pipeline_state.get("adapter_checkpoint") for adapter path
- Notebook 05 -> base model load (without adapter) for Layer 2 before/after comparison
- benchmark_utils.py -> config.py for system prompt and paths
- benchmark_utils.py -> data_utils.py for format_messages() utility (if placed there)
</code_context>

<specifics>
## Specific Ideas

- Current `_build_prompt()` in benchmark_utils.py (line 309) hardcodes ChatML tokens: `f"<|im_start|>system\n{system_msg}<|im_end|>\n..."`
- Notebook 05 cell 10 hardcodes `before_score = 0.35` — must load from pipeline_state instead
- `evaluate_case_quality()` imports `rouge_scorer` but never computes actual ROUGE scores (only keyword coverage)
- BLEU requires `sacrebleu` or `nltk`; ROUGE requires `rouge-score` (already in requirements.txt per INFRA-04)
- `aggregate_results()` exists but produces a single report, not a before/after comparison
- `AutoPeftModelForCausalLM` already used in Notebook 05 cell 2 — correct per EVAL-04
</specifics>

<deferred>
## Deferred Ideas

- Weights & Biases integration for evaluation visualization — belongs in future enhancement phase
- Full Layer 1 suite post-training (MMLU-Pro, GPQA, HumanEval, MATH, BBH) — too time-consuming for v1.0, listed in FUTURE-01
- Real-time inference serving — out of scope per PROJECT.md
</deferred>

---

*Phase: 03-evaluation-and-hardening*
*Context gathered: 2026-05-28*
