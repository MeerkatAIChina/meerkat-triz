---
phase: 01
slug: foundation-data-pipeline
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-27
---

# Phase 01 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | none — Wave 0 installs |
| **Quick run command** | `pytest tests/test_synthetic_pipeline.py -x` |
| **Full suite command** | `pytest tests/ -x` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_{module}.py -x`
- **After every plan wave:** Run `pytest tests/ -x`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | DATA-01 | — | N/A | unit | `pytest tests/test_synthetic_pipeline.py::test_generate_variations -x` | ❌ W0 | ⬜ pending |
| 01-02-01 | 02 | 1 | DATA-02 | — | N/A | unit | `pytest tests/test_quality_gates.py::test_perplexity_filter -x` | ❌ W0 | ⬜ pending |
| 01-03-01 | 03 | 1 | DATA-03 | — | N/A | integration | Notebook 02 cell execution | ❌ W0 | ⬜ pending |
| 01-04-01 | 04 | 1 | INFRA-04 | — | N/A | smoke | `pip install -r requirements.txt` | ❌ W0 | ⬜ pending |
| 01-05-01 | 05 | 1 | INFRA-05 | — | N/A | unit | `pytest tests/test_synthetic_pipeline.py::test_rate_limiting -x` | ❌ W0 | ⬜ pending |
| 01-06-01 | 06 | 1 | INFRA-06 | — | N/A | unit | `pytest tests/test_pipeline_state.py -x` | ❌ W0 | ⬜ pending |
| 01-07-01 | 07 | 1 | TRAIN-02 | — | N/A | unit | `pytest tests/test_config.py::test_target_modules -x` | ❌ W0 | ⬜ pending |
| 01-08-01 | 08 | 1 | TRAIN-03 | — | N/A | unit | `pytest tests/test_config.py::test_lora_dropout -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_synthetic_pipeline.py` — stubs for DATA-01, INFRA-05
- [ ] `tests/test_pipeline_state.py` — stubs for INFRA-06
- [ ] `tests/test_quality_gates.py` — stubs for DATA-02
- [ ] `tests/test_config.py` — stubs for TRAIN-02, TRAIN-03
- [ ] `tests/conftest.py` — shared fixtures (mock Moonshot client, temp directories)
- [ ] `pytest` install — if none detected

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Token length histogram visualization | DATA-03 | Requires visual inspection of matplotlib output | Run notebook 02 cell, verify histogram displays with red max_length line |
| Moonshot API cost estimation | DATA-01 | Requires actual API pricing data | Run notebook 02b cost estimation cell, verify displayed cost is reasonable |
| Notebook pre-flight checks | INFRA-09 | Requires notebook runtime environment | Run pre-flight cell in each notebook, verify all checks pass |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
