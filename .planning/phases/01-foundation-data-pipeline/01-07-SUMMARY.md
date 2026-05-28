---
phase: 01-foundation-data-pipeline
plan: 07
type: execute
subsystem: test-infrastructure
tags: [pytest, mocks, gap-closure, validation]
dependency_graph:
  requires: [01-06]
  provides: [test-stubs-for-verification]
  affects: []
tech_stack:
  added: [pytest, unittest.mock, openai]
  patterns: [importlib.util direct imports, FakeModel callable class, pathlib monkey-patch]
key_files:
  created:
    - ref/mongoose_ai_dgx/tests/conftest.py
    - ref/mongoose_ai_dgx/tests/test_config.py
    - ref/mongoose_ai_dgx/tests/test_pipeline_state.py
    - ref/mongoose_ai_dgx/tests/test_synthetic_pipeline.py
    - ref/mongoose_ai_dgx/tests/test_quality_gates.py
  modified: []
decisions:
  - "Use importlib.util direct module loading to avoid utils/__init__.py torch dependency in test environment"
  - "Use FakeModel callable class instead of unittest.mock.Mock for model mocking because Mock.__call__ ignores return_value"
  - "Monkey-patch pathlib.Path.mkdir to noop in test_config.py to allow config.py import on non-DGX environments"
  - "Install openai and torch as test environment dependencies (not runtime deps for the DGX target)"
metrics:
  duration: 35
  completed_date: "2026-05-28"
---

# Phase 01 Plan 07: Test Infrastructure Gap Closure Summary

**One-liner:** Created 5 pytest test files with fully mocked dependencies to close the VALIDATION.md test stub gap, enabling automated verification of config, pipeline state, synthetic pipeline, and quality gates without API keys or model loading.

## Tasks Completed

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 | Create tests/conftest.py with shared fixtures | `27e00d9` | `ref/mongoose_ai_dgx/tests/conftest.py` |
| 2 | Create tests/test_config.py | `1656d26` | `ref/mongoose_ai_dgx/tests/test_config.py` |
| 3 | Create tests/test_pipeline_state.py | `7e6b5b6` | `ref/mongoose_ai_dgx/tests/test_pipeline_state.py` |
| 4 | Create tests/test_synthetic_pipeline.py | `1b540da` | `ref/mongoose_ai_dgx/tests/test_synthetic_pipeline.py` |
| 5 | Create tests/test_quality_gates.py | `98a8b96` | `ref/mongoose_ai_dgx/tests/test_quality_gates.py` |

## Test Results

All 17 tests pass with `pytest -x`:

```
ref/mongoose_ai_dgx/tests/test_config.py::test_target_modules_count_is_12 PASSED
ref/mongoose_ai_dgx/tests/test_config.py::test_lora_dropout_is_zero PASSED
ref/mongoose_ai_dgx/tests/test_config.py::test_synthetic_config_exists PASSED
ref/mongoose_ai_dgx/tests/test_config.py::test_quality_gates_config PASSED
ref/mongoose_ai_dgx/tests/test_pipeline_state.py::test_register PASSED
ref/mongoose_ai_dgx/tests/test_pipeline_state.py::test_get PASSED
ref/mongoose_ai_dgx/tests/test_pipeline_state.py::test_verify PASSED
ref/mongoose_ai_dgx/tests/test_pipeline_state.py::test_preflight PASSED
ref/mongoose_ai_dgx/tests/test_pipeline_state.py::test_summary PASSED
ref/mongoose_ai_dgx/tests/test_quality_gates.py::test_token_length_filter PASSED
ref/mongoose_ai_dgx/tests/test_quality_gates.py::test_diversity_score_computation PASSED
ref/mongoose_ai_dgx/tests/test_quality_gates.py::test_perplexity_filter_mocked PASSED
ref/mongoose_ai_dgx/tests/test_quality_gates.py::test_filter_by_diversity PASSED
ref/mongoose_ai_dgx/tests/test_synthetic_pipeline.py::test_deduplicate_seeds PASSED
ref/mongoose_ai_dgx/tests/test_synthetic_pipeline.py::test_estimate_cost PASSED
ref/mongoose_ai_dgx/tests/test_synthetic_pipeline.py::test_generate_variations_mocked PASSED
ref/mongoose_ai_dgx/tests/test_synthetic_pipeline.py::test_checkpoint_save_load PASSED
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Mock.__call__ ignores return_value, breaking torch.exp(loss)**
- **Found during:** Task 5 (test_quality_gates.py)
- **Issue:** `unittest.mock.Mock` instances always return a new Mock from `__call__`, ignoring any `return_value` set on `forward`. The `compute_perplexity` function calls `model(input_ids, labels=input_ids)` which invokes `__call__`, not `forward`. This returned a Mock for `outputs`, and `outputs.loss` was also a Mock. `torch.exp(Mock)` raised `TypeError`, causing the exception handler to return `float('inf')`.
- **Fix:** Replaced `Mock`-based model with a `FakeModel` callable class that has a real `__call__` method returning an object with a real `torch.tensor` loss attribute.
- **Files modified:** `ref/mongoose_ai_dgx/tests/conftest.py`
- **Commit:** `98a8b96`

**2. [Rule 3 - Blocking] utils/__init__.py imports torch-dependent modules**
- **Found during:** Task 3 (test_pipeline_state.py)
- **Issue:** `from utils.pipeline_state import PipelineState` triggers `utils/__init__.py`, which imports `benchmark_utils` which imports `torch`. Torch is not available in the test runner environment.
- **Fix:** Used `importlib.util.spec_from_file_location` to import `pipeline_state.py` and `synthetic_pipeline.py` directly, bypassing `utils/__init__.py`.
- **Files modified:** `ref/mongoose_ai_dgx/tests/test_pipeline_state.py`, `ref/mongoose_ai_dgx/tests/test_synthetic_pipeline.py`, `ref/mongoose_ai_dgx/tests/test_quality_gates.py`, `ref/mongoose_ai_dgx/tests/conftest.py`
- **Commit:** `7e6b5b6`, `1b540da`, `98a8b96`

**3. [Rule 3 - Blocking] config.py tries to create /home/meerkat directories at import time**
- **Found during:** Task 2 (test_config.py)
- **Issue:** `config.py` line 20 calls `d.mkdir(parents=True, exist_ok=True)` on paths under `/home/meerkat/mongoose_ai`. On macOS, `/home` is a symlink to `/System/Volumes/Data/home` which does not exist, causing `OSError: [Errno 45] Operation not supported`.
- **Fix:** Monkey-patched `pathlib.Path.mkdir` to a no-op before importing `config`, then restored it.
- **Files modified:** `ref/mongoose_ai_dgx/tests/test_config.py`
- **Commit:** `1656d26`

**4. [Rule 3 - Blocking] Missing openai and torch packages in test environment**
- **Found during:** Task 4 (test_synthetic_pipeline.py) and Task 5 (test_quality_gates.py)
- **Issue:** `synthetic_pipeline.py` imports `openai` at module level, and `compute_perplexity` imports `torch` inside the function. Neither package was installed in the test runner environment.
- **Fix:** Installed `openai` and `torch` via `python3 -m pip install --break-system-packages`.
- **Files modified:** none (environment change)
- **Commit:** N/A (environment setup)

**5. [Rule 1 - Bug] sys module has no `__version__`, breaking preflight package check test**
- **Found during:** Task 3 (test_pipeline_state.py)
- **Issue:** `test_preflight` used `{"sys": "0.0.1"}` as a required package. `sys` has no `__version__` attribute, so `pipeline_state.py` falls through to the "无法获取版本信息" branch and appends an error.
- **Fix:** Changed test to use `{"packaging": "0.0.1"}` which has `__version__`.
- **Files modified:** `ref/mongoose_ai_dgx/tests/test_pipeline_state.py`
- **Commit:** `7e6b5b6`

## Known Stubs

None. All tests verify real production code paths with mocked external dependencies only.

## Threat Flags

None. All test code uses mocked dependencies; no real API keys or network calls are made.

## Self-Check: PASSED

- [x] `ref/mongoose_ai_dgx/tests/conftest.py` exists
- [x] `ref/mongoose_ai_dgx/tests/test_config.py` exists
- [x] `ref/mongoose_ai_dgx/tests/test_pipeline_state.py` exists
- [x] `ref/mongoose_ai_dgx/tests/test_synthetic_pipeline.py` exists
- [x] `ref/mongoose_ai_dgx/tests/test_quality_gates.py` exists
- [x] All 5 files have valid Python syntax
- [x] All 17 tests pass with `pytest -x`
- [x] Commit `27e00d9` exists
- [x] Commit `1656d26` exists
- [x] Commit `7e6b5b6` exists
- [x] Commit `1b540da` exists
- [x] Commit `98a8b96` exists
