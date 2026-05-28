# Phase 02: baseline-training-execution - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-28
**Phase:** 02-baseline-training-execution
**Areas discussed:** Benchmark scope, Training data readiness, Checkpoint verification, Training interruption handling, Notebook execution order

---

## Benchmark Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Layer 1 only at baseline | Run MMLU-Pro, GPQA, HumanEval, MATH, BBH via lm-eval-harness | ✓ |
| Layer 1 + Layer 2 at baseline | Also run TRIZ custom benchmarks with base model | |
| Minimal spot-check | Run only 1-2 Layer 1 tasks to save time | |

**User's choice:** Layer 1 only at baseline — Layer 2 reserved for post-training comparison in Phase 03
**Notes:** User selected "all" gray areas to discuss; this was auto-resolved based on ROADMAP success criteria

---

## Training Data Readiness

| Option | Description | Selected |
|--------|-------------|----------|
| Assume data ready | Notebook 04 reads from pipeline_state; run 02b first if needed | ✓ |
| Include data generation in Phase 02 | Make 02b execution part of Phase 02 plans | |
| Pre-generate before Phase 02 | User runs 02b manually before starting Phase 02 | |

**User's choice:** Assume data ready — Phase 02 scope focuses on benchmark + training execution

---

## Checkpoint Verification

| Option | Description | Selected |
|--------|-------------|----------|
| Every checkpoint | Immediate forward-pass validation after each save (every 200 steps) | ✓ |
| Spot-check only | Validate every Nth checkpoint to save time | |
| No validation | Trust SFTTrainer's save logic | |

**User's choice:** Every checkpoint — forward-pass validation is a success criterion

---

## Training Interruption Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Dual resume | SFTTrainer built-in + cell-level resume in Notebook 04 | ✓ |
| Built-in only | Rely solely on SFTTrainer resume_from_checkpoint | |
| Manual resume | Document steps for user to manually resume | |

**User's choice:** Dual resume — both automatic and manual cell-level recovery

---

## Notebook Execution Order

| Option | Description | Selected |
|--------|-------------|----------|
| Strict sequence | 01 -> 02b -> 03 -> 04 with pipeline_state gates | ✓ |
| Flexible order | Run benchmarks and training in any order | |
| Parallel | Run baseline while training (separate processes) | |

**User's choice:** Strict sequence with pre-flight verification gates

---

## Claude's Discretion

- Specific lm-eval-harness task subsets
- Progress/logging verbosity
- Checkpoint metadata format details
- W&B integration (deferred)

## Deferred Ideas

- Weights & Biases integration — future enhancement
- Multi-GPU training — out of scope
- Real-time monitoring dashboard — future enhancement
