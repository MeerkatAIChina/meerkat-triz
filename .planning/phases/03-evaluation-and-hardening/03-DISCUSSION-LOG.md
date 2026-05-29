# Phase 03: Evaluation & Hardening - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-28
**Phase:** 03-evaluation-and-hardening
**Areas discussed:** Before/after comparison scope, Comparison report format, format_messages() utility design, Missing baseline handling

---

## Before/after comparison scope

| Option | Description | Selected |
|--------|-------------|----------|
| Re-run Layer 2 on base model | Notebook 05 loads base model and runs full TRIZ benchmarks for true before/after deltas | ✓ |
| Limit to Layer 1 + Layer 3 | Match Phase 2 D-01 intent, no Layer 2 before/after | |
| Quick Layer 2 spot-check | Abbreviated TRIZ eval on base model (5 questions only) | |

**User's choice:** Re-run Layer 2 on base model during evaluation
**Notes:** This resolves the Phase 2 D-01 conflict with Phase 3 success criteria #2. Adds ~5-10 minutes but produces true before/after deltas for all Layer 2 metrics.

---

## Comparison report format

| Option | Description | Selected |
|--------|-------------|----------|
| Notebook cell output only | Rich tables inline, no extra files | |
| Saved JSON + notebook display | Structured JSON to results/ + inline display | ✓ |
| Markdown report file + notebook display | Human-readable Markdown saved to results/ | |

**User's choice:** Saved JSON + notebook display
**Notes:** JSON file enables programmatic access and future tooling. Notebook display provides immediate visual feedback during DGX Spark sessions.

---

## format_messages() utility design

| Option | Description | Selected |
|--------|-------------|----------|
| data_utils.py shared | Shared across training + eval, pulls system prompt from config.py | |
| benchmark_utils.py eval-only | Local to evaluation, training already has its own path | |
| You decide | Implementation detail — Claude decides cleanest location | ✓ |

**User's choice:** You decide (Claude's Discretion)
**Notes:** User deferred implementation location to Claude. Requirement: replace hardcoded ChatML in benchmark_utils.py _build_prompt() and Notebook 05 cell 6. Must use tokenizer.apply_chat_template() and pull system prompt from config.py.

---

## Missing baseline handling

| Option | Description | Selected |
|--------|-------------|----------|
| Fail with clear error | Raise error explaining baseline must be run first | |
| Skip comparison, show post-training only | Continue with warning, no before/after | |
| Auto-run quick baseline on-the-fly | Minimal Layer 1 + Layer 3 on base model before adapter load | ✓ |

**User's choice:** Auto-run quick baseline on-the-fly
**Notes:** Makes Notebook 05 self-contained. Quick baseline = minimal Layer 1 (one task) + Layer 3 performance. Warning printed to user recommending full baseline via Notebook 03.

---

## Claude's Discretion

- format_messages() function signature, location, and implementation details
- Exact JSON schema for evaluation report
- Which Layer 1 task for optional spot-check
- Visual formatting of delta tables in notebook
- BLEU/ROUGE reference data strategy

## Deferred Ideas

- Weights & Biases integration for evaluation visualization — future enhancement phase
- Full Layer 1 suite post-training (MMLU-Pro, GPQA, etc.) — deferred to v1.1 (FUTURE-01)
- Real-time inference serving — out of scope per PROJECT.md
