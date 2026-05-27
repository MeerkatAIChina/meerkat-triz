# Phase 1: Foundation & Data Pipeline - Context

**Gathered:** 2026-05-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Generate ~6K high-quality synthetic TRIZ training samples from 548 seed samples and verify all pre-training configuration is correct. This phase delivers:

- Fixed `config.py` (`lora_dropout=0.0`, explicit 12-module `target_modules`)
- Updated `requirements.txt` with pinned compatible versions
- `utils/synthetic_pipeline.py` -- Moonshot API client with batching, rate limiting, output validation
- `utils/pipeline_state.py` -- JSON artifact registry for cross-notebook state tracking
- `02b_synthetic_generation.ipynb` -- orchestrates ~6K sample generation with quality gates
- Notebook pre-flight checks for paths, artifacts, and version compatibility
- `README.md` no longer recommends `"all-linear"`

</domain>

<decisions>
## Implementation Decisions

### Synthetic Data Generation Strategy
- **D-01:** Hybrid approach by subset:
  - **Rephrase-in-place** (keep answers grounded): `concept_explanation`, `ariz_guidance`
  - **Generate entirely new Q&A pairs**: `case_generation`, `contradiction_analysis`
  - **Mix of both**: `principle_recommendation`, `innovation_assessment`
- **D-02:** Moonshot API (not template paraphrasing) for true semantic variation. Current `vary_sample()` template approach is insufficient for production.

### Real vs Synthetic Ratio
- **D-03:** Variable ratio by subset:
  - **25% real**: `concept_explanation`, `ariz_guidance` (factual accuracy priority)
  - **~15% real**: `principle_recommendation`, `innovation_assessment`
  - **10% real**: `case_generation`, `contradiction_analysis` (diversity priority)
- **D-04:** Total target ~6K samples. Exact per-subset counts derived from seed distribution and ratio targets.

### Notebook 02b Design
- **D-05:** Checkpoint-based resumability: save progress after each subset/batch so generation can resume if interrupted (rate limit, crash, restart).
- **D-06:** Cost monitoring: display estimated Moonshot API cost and token count before each subset generation starts.

### Claude's Discretion
- Multiplier per seed (variable by subset, balancing diversity vs API cost)
- Quality gate implementation details (pragmatic two-gate: perplexity + length; diversity enforced via generation strategy)
- Failed sample handling strategy
- Notebook 02b cell structure and interaction flow
- Pipeline state registry schema and scope
- Config values beyond required fixes (`lora_dropout`, `target_modules`)
- Pre-flight check comprehensiveness by notebook phase
- Requirements pinning approach (exact vs minimum versions)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Audit & Requirements
- `ref/审计报告_猫鼬AI训练方案.md` -- Independent audit identifying CR-001 through CR-003 and MA-001; informs config fixes and data pipeline design
- `.planning/REQUIREMENTS.md` -- Full requirement traceability (DATA-01 through DATA-05, INFRA-04 through INFRA-09, TRAIN-02, TRAIN-03)

### Existing Codebase
- `ref/mongoose_ai_dgx/config.py` -- Current hyperparameters, paths, and hardware config; `lora_dropout` and `target_modules` need fixing
- `ref/mongoose_ai_dgx/requirements.txt` -- Current dependency list; needs version pinning
- `ref/mongoose_ai_dgx/utils/data_utils.py` -- Existing data loading, ChatML conversion, and template-based synthetic generation (to be replaced)
- `ref/mongoose_ai_dgx/utils/training_utils.py` -- Model loading, LoRA config, SFTTrainer creation
- `ref/mongoose_ai_dgx/data/sample_data.json` -- 548 seed TRIZ samples across 6 subsets
- `ref/mongoose_ai_dgx/README.md` -- Currently recommends `"all-linear"`; needs correction

### Notebook Templates
- `ref/mongoose_ai_dgx/notebooks/02_data_preparation.ipynb` -- Existing data prep notebook; 02b will extend it
- `ref/mongoose_ai_dgx/notebooks/01_download_and_setup.ipynb` -- Setup notebook where pre-flight checks should first appear

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `data_utils.load_raw_data()`: Loads subset JSON files from directory; reusable for synthetic pipeline input
- `data_utils.convert_to_chatml()`: Converts instruction/output format to ChatML via `tokenizer.apply_chat_template()`; reusable for final dataset formatting
- `data_utils.split_dataset()`: 85/10/5 train/val/test split; reusable for combined real+synthetic data
- `data_utils.save_dataset()`: Saves DatasetDict to JSONL; reusable for persisting processed data
- `training_utils.load_model_and_tokenizer()`: Model loading with 4-bit quantization; reusable for perplexity gate computation
- `training_utils.get_qwen36_target_modules()`: Returns explicit 12-module list; the correct approach (not `"all-linear"`)

### Established Patterns
- **ChatML via `tokenizer.apply_chat_template()`**: The canonical format for training data. No hardcoded ChatML strings.
- **Notebook-driven workflow**: All execution happens in Jupyter notebooks on DGX Spark. No CLI or server architecture.
- **Central config in `config.py`**: All hyperparameters live in one file. New pipeline modules should import from config, not hardcode values.
- **SFTTrainer + `formatting_func`**: Training uses SFTTrainer with `formatting_func`, no `data_collator` passed.

### Integration Points
- `utils/synthetic_pipeline.py` will read from `data/sample_data.json` and write synthetic outputs to `data/processed/`
- `utils/pipeline_state.py` will write JSON registry accessible across all notebooks
- `02b_synthetic_generation.ipynb` will import both `synthetic_pipeline` and `pipeline_state`, and use `data_utils` for final formatting
- Notebook pre-flight checks should verify `pipeline_state` exists and required prior artifacts are registered
- All notebooks should append to `pipeline_state` after producing artifacts

</code_context>

<specifics>
## Specific Ideas

- The existing `vary_sample()` in `data_utils.py` uses simple prefix templates (e.g., "从TRIZ角度分析，"). This is explicitly acknowledged as insufficient in the audit. The new pipeline must use Moonshot API for true semantic variation.
- 548 real seed samples exist. Target ~6K total. Real sample distribution across subsets may not match the `config.py` `seed_count=100` per subset -- actual counts should be read from `sample_data.json`.
- Moonshot API rate limits and cost are noted as a blocker/concern in `STATE.md`. The pipeline must implement batching, rate limiting, and cost estimation.

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope.

</deferred>

---

*Phase: 01-foundation-data-pipeline*
*Context gathered: 2026-05-27*
