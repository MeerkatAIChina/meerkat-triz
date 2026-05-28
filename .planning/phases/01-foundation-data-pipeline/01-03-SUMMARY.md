---
phase: 01-foundation-data-pipeline
plan: 03
subsystem: data-pipeline
tags: [jupyter, notebook, moonshot-api, synthetic-data, chatml, checkpoint-resumability]

requires:
  - phase: 01-01
    provides: config.py with SYNTHETIC_CONFIG, data_utils.py with load_raw_data
  - phase: 01-02
    provides: synthetic_pipeline.py with MoonshotSyntheticClient and SyntheticPipeline, pipeline_state.py with PipelineState

provides:
  - 02b_synthetic_generation.ipynb: user-facing orchestration notebook for ~6K synthetic sample generation
  - Complete cell-by-cell workflow: imports, seed loading, cost estimation, generation loop, quality gates, ChatML conversion, dataset split, save, pipeline state registration

affects:
  - 01-foundation-data-pipeline
  - 02-training

tech-stack:
  added: []
  patterns:
    - "Notebook-driven workflow with sys.path.append for project imports"
    - "Cell-by-cell execution with markdown section headers"
    - "Checkpoint-resumable generation via SyntheticPipeline.generate_subset()"
    - "Quality gate filtering with tokenizer.encode() before ChatML conversion"
    - "PipelineState artifact registration for cross-notebook tracking"

key-files:
  created:
    - ref/mongoose_ai_dgx/notebooks/02b_synthetic_generation.ipynb
  modified: []

key-decisions:
  - "Notebook loads tokenizer on CPU (device_map='cpu') for data processing to avoid GPU memory contention"
  - "Quality gate runs before ChatML conversion to filter long samples early, reducing unnecessary tokenization work"
  - "Combined raw synthetic data saved as combined_raw.json for reference and pipeline state registration"

patterns-established:
  - "02b notebook pattern: imports -> seed loading -> cost estimation -> generation with checkpoints -> quality gates -> ChatML -> split -> save -> state registration"

requirements-completed:
  - INFRA-07
  - DATA-01
  - DATA-02
  - DATA-04
  - DATA-05

# Metrics
duration: 5min
completed: 2026-05-27
---

# Phase 01 Plan 03: Synthetic Generation Notebook Summary

**18-cell Jupyter notebook orchestrating ~6K synthetic TRIZ training sample generation from 548 seeds via Moonshot API, with checkpoint resumability, quality gates, ChatML conversion, and pipeline state registration.**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-05-27T22:16:44Z
- **Completed:** 2026-05-27T22:17:11Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Created 02b_synthetic_generation.ipynb with 18 cells (8 code, 10 markdown)
- Notebook imports and uses MoonshotSyntheticClient, SyntheticPipeline, PipelineState
- Cost estimation displayed before generation starts (Cell 6)
- Subset-by-subset generation with automatic checkpoint recovery (Cell 8)
- Quality gate filters samples exceeding 3500 tokens (Cell 12)
- ChatML conversion via convert_to_chatml() with tokenizer.apply_chat_template() (Cell 14)
- 85/10/5 train/val/test split via split_dataset() (embedded in convert_to_chatml)
- Pipeline state registration for synthetic_raw and synthetic_dataset artifacts (Cell 16)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create 02b_synthetic_generation.ipynb** - `8fc7344` (feat)

## Files Created/Modified
- `ref/mongoose_ai_dgx/notebooks/02b_synthetic_generation.ipynb` - Complete synthetic data generation orchestration notebook

## Decisions Made
- Notebook loads tokenizer on CPU (`device_map='cpu'`) for data processing to avoid GPU memory contention during generation phase
- Quality gate runs before ChatML conversion to filter long samples early, reducing unnecessary tokenization work
- Combined raw synthetic data saved as `combined_raw.json` for reference and pipeline state registration

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

**Moonshot API key required.** Before running the notebook:
1. Obtain API key from Moonshot Platform (platform.kimi.ai) -> API Keys
2. Set environment variable: `export MOONSHOT_API_KEY='your-key'`
3. Verify: `echo $MOONSHOT_API_KEY` should display the key

## Known Stubs

None. The notebook is a complete orchestration artifact. All data sources are wired to the existing pipeline modules (synthetic_pipeline.py, data_utils.py, pipeline_state.py).

## Threat Flags

| Flag | File | Description |
|------|------|-------------|
| T-01-03 (mitigated) | Cell 6 | MOONSHOT_API_KEY read from env var; cell warns if unset |
| T-01-04 (mitigated) | Cell 8 | Checkpoint saved after every batch; re-run resumes from last checkpoint |
| T-01-05 (mitigated) | Cell 12 | Quality gate filters samples > 3500 tokens; convert_to_chatml validates format |

## Next Phase Readiness

- Synthetic generation notebook complete and ready for execution on DGX Spark
- Pipeline modules (synthetic_pipeline.py, data_utils.py, pipeline_state.py) from Plans 01-01 and 01-02 are prerequisites
- After generation completes, 03_model_benchmark.ipynb can be run for baseline evaluation
- After baseline, 04_qlora_finetune.ipynb can be run for training

---
*Phase: 01-foundation-data-pipeline*
*Plan: 03*
*Completed: 2026-05-27*
