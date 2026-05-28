---
phase: 01-foundation-data-pipeline
plan: 06
subsystem: synthetic-data-pipeline
tags: [quality-gates, perplexity, diversity, data-pipeline]
dependency_graph:
  requires: [01-01, 01-02, 01-03, 01-04, 01-05]
  provides: [DATA-02]
  affects: [synthetic_pipeline.py, config.py, README.md, notebooks/02b]
tech_stack:
  added: []
  patterns: [n-gram diversity scoring, model perplexity filtering, config-driven thresholds]
key_files:
  created: []
  modified:
    - ref/mongoose_ai_dgx/utils/synthetic_pipeline.py
    - ref/mongoose_ai_dgx/config.py
    - ref/mongoose_ai_dgx/README.md
    - ref/mongoose_ai_dgx/notebooks/02b_synthetic_generation.ipynb
decisions:
  - "Perplexity filtering disabled by default to avoid mandatory 20GB model load during data generation"
  - "Diversity scoring enabled by default (pure text processing, no model required)"
  - "Real data ratio ~8.7% documented as intentional volume-prioritized design decision"
metrics:
  duration: 428
  completed_date: "2026-05-28"
---

# Phase 01 Plan 06: Add Quantitative Quality Gates to Synthetic Data Pipeline

**One-liner:** Added perplexity-based filtering (optional, model-backed) and n-gram diversity scoring (model-free) as quantitative quality gates to the synthetic data pipeline, plus documented the intentional ~8.7% real data ratio deviation.

## Tasks Completed

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 | Add perplexity and diversity quality gate functions to synthetic_pipeline.py | f391ce2 | synthetic_pipeline.py |
| 2 | Update config.py SYNTHETIC_CONFIG with quality gate thresholds | b2b3217 | config.py |
| 3 | Document real data ratio deviation in README.md | 77adcff | README.md |
| 4 | Update Notebook 02b cell 12 to call quality gate filter functions | 59488af | 02b_synthetic_generation.ipynb |

## Deviations from Plan

None - plan executed exactly as written.

## Auth Gates

None.

## Known Stubs

None. All functions are fully implemented:
- `compute_perplexity()`: full model forward pass with CPU fallback and graceful failure
- `filter_by_perplexity()`: percentile-based filtering with optional skip
- `compute_diversity_score()`: n-gram distinct metrics using pure Python
- `filter_by_diversity()`: threshold-based filtering with dedup fallback

## Threat Flags

None. All security-relevant surface was already covered in the plan's threat model:
- T-01-06-01 (DoS via model load): mitigated by disabled-by-default + CPU fallback + inf return on failure
- T-01-06-02 (info disclosure via perplexity scores): accepted (internal metrics only)
- T-01-06-03 (tampering via incorrect diversity filtering): mitigated by conservative thresholds

## Self-Check: PASSED

- [x] synthetic_pipeline.py has compute_perplexity() with model forward pass, CPU fallback, and graceful failure (returns inf)
- [x] synthetic_pipeline.py has filter_by_perplexity() with optional skip when model is None, percentile-based threshold
- [x] synthetic_pipeline.py has compute_diversity_score() using n-gram distinct metrics, no model required
- [x] synthetic_pipeline.py has filter_by_diversity() with threshold-based filtering and dedup fallback
- [x] config.py quality_gates includes perplexity config (disabled by default) and diversity config (enabled by default)
- [x] README.md documents the ~8.7% real data ratio with rationale for prioritizing volume over ratio
- [x] Notebook 02b cell 12 imports and calls both filter functions with config-driven thresholds
- [x] All 4 commits verified in git log
