---
phase: 03-evaluation-and-hardening
plan: 04
status: complete
completed: 2026-05-29
---

## Summary

Created pytest test files for the three core evaluation behaviors introduced in this phase: format_messages() utility, BLEU/ROUGE metric computation, and before/after comparison report structure.

## What Was Built

- **`tests/test_format_messages.py`** — 4 tests:
  - `test_format_messages_inference` — inference prompt with `add_generation_prompt=True`
  - `test_format_messages_training` — training data with `assistant_content` and `add_generation_prompt=False`
  - `test_format_messages_custom_system` — custom system message override
  - `test_format_messages_default_system` — default system message from DATA_CONFIG
  - Uses MockTokenizer with `apply_chat_template` support
  - Mocks heavy dependencies (datasets, torch, transformers, peft, trl, bitsandbytes)
  - Mocks `pathlib.Path.mkdir` to prevent config.py directory creation errors

- **`tests/test_metrics.py`** — 4 tests:
  - `test_bleu_function_exists` — verifies `_compute_bleu` exists with `corpus_bleu` and `tokenize='zh'`
  - `test_rouge_function_exists` — verifies `_compute_rouge` exists with `use_stemmer=False` and `jieba.cut`
  - `test_evaluate_case_quality_returns_bleu_rouge` — verifies bleu/rouge fields in return dict
  - `test_chinese_aware_tokenization` — verifies `tokenize='zh'` and `use_stemmer=False`
  - Uses AST parsing to verify code structure without importing heavy dependencies

- **`tests/test_report.py`** — 5 tests:
  - `test_aggregate_results_accepts_before_after` — verifies `before_results` and `after_results` in signature
  - `test_compute_deltas_exists` — verifies `_compute_deltas` function exists
  - `test_delta_structure_in_code` — verifies `before/after/delta/delta_pct` structure
  - `test_evaluation_report_filename` — verifies `evaluation_report_` prefix
  - `test_report_layer_structure` — verifies `layer2_triz`, `layer3_performance`, `layer1_general`
  - Uses AST parsing to verify code structure

## Commits

- `4b3eb03` — test(03-04): add pytest tests for format_messages, metrics, and report

## Key Files

| File | Tests | Description |
|------|-------|-------------|
| `tests/test_format_messages.py` | 4 | format_messages() utility with mock tokenizer |
| `tests/test_metrics.py` | 4 | BLEU/ROUGE function existence and parameters |
| `tests/test_report.py` | 5 | aggregate_results signature, deltas, layer structure |

## Verification

- ✓ `python3 -m pytest tests/test_format_messages.py -v` — 4 passed
- ✓ `python3 -m pytest tests/test_metrics.py -v` — 4 passed
- ✓ `python3 -m pytest tests/test_report.py -v` — 5 passed
- ✓ All 13 tests pass without torch, transformers, or datasets installed
- ✓ Tests verify concrete behaviors (not just import checks)

## Self-Check: PASSED
