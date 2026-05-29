---
phase: 03-evaluation-and-hardening
plan: 02
status: complete
completed: 2026-05-29
---

## Summary

Refactored `benchmark_utils.py` to replace hardcoded ChatML with the unified `format_messages()` utility, implemented BLEU/ROUGE metrics for TRIZ case quality scoring, and extended `aggregate_results()` to support before/after comparison reports.

## What Was Built

- **`_build_prompt()`** — Now delegates to `format_messages()` with `add_generation_prompt=True`. All hardcoded ChatML tokens and the hardcoded system message removed.
- **`_compute_bleu()`** — Corpus-level BLEU using `sacrebleu.corpus_bleu` with `tokenize='zh'` for Chinese-aware scoring.
- **`_compute_rouge()`** — ROUGE-1/2/L using `rouge_scorer` with `use_stemmer=False` and `jieba.cut` segmentation.
- **`evaluate_case_quality()`** — Collects predictions and references, computes BLEU/ROUGE when references are available, and still reports keyword coverage.
- **`TRIZBenchmark.__init__()`** — Accepts `test_data_path` parameter.
- **`_load_test_questions()`** — Loads external `case_generation` samples from `sample_data.json` when `test_data_path` is provided.
- **`run_triz_evaluation()`** — Passes `test_data_path` through to `TRIZBenchmark`.
- **`_compute_deltas()`** — Recursive helper computing `{before, after, delta, delta_pct}` for all numeric metrics.
- **`aggregate_results()`** — New signature supports `before_results` and `after_results`, producing flat JSON with delta structure. Saves to `evaluation_report_YYYYMMDD_HHMMSS.json`.

## Commits

- `3a41aa2` — feat(03-02): refactor benchmark_utils with format_messages, BLEU/ROUGE, before/after comparison

## Key Files

| File | Change |
|------|--------|
| `ref/mongoose_ai_dgx/utils/benchmark_utils.py` | Major refactor: unified prompts, BLEU/ROUGE, deltas, before/after aggregation |

## Verification

- ✓ `grep "format_messages"` — matches (in _build_prompt)
- ✓ `grep "<|im_start|>system"` — 0 matches (hardcoded ChatML removed)
- ✓ `grep "def _compute_bleu"` — matches
- ✓ `grep "def _compute_rouge"` — matches
- ✓ `grep "corpus_bleu"` — matches
- ✓ `grep "use_stemmer=False"` — matches
- ✓ `grep "jieba.cut"` — matches
- ✓ `grep "tokenize='zh'"` — matches
- ✓ `grep "test_data_path"` — 8 matches (>= 3 required)
- ✓ `grep "def aggregate_results"` — matches
- ✓ `grep "before_results"` — matches
- ✓ `grep "after_results"` — matches
- ✓ `grep "def _compute_deltas"` — matches
- ✓ `grep "delta_pct"` — matches
- ✓ `grep "evaluation_report_"` — matches
- ✓ `grep "re-run_on_both_models"` — matches

## Self-Check: PASSED
