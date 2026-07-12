# Milestones

## v1.0 MVP (Shipped: 2026-07-12)

**Phases completed:** 4 phases, 18 plans, 20 tasks

**Known deferred items at close:** 2 (see STATE.md Deferred Items)

**Key accomplishments:**

- Hardened training configuration with explicit 12-module QLoRA targets, `lora_dropout=0.0`, and pinned dependency versions.
- Built `utils/synthetic_pipeline.py` and Notebook 02b to generate ~6K synthetic TRIZ training samples from 548 seeds via Moonshot API, with quality gates and checkpoint resumability.
- Added token length profiling to Notebook 02 and structured pre-flight environment checks to Notebook 01.
- Implemented QLoRA training utilities with `CheckpointValidationCallback`, SHA-256 adapter integrity verification, and LR-verified checkpoint resume.
- Created three-layer evaluation utilities including unified `format_messages()` (no hardcoded ChatML), BLEU/ROUGE case-quality scoring, and before/after aggregate reporting.
- Closed audit blocker BLK-01: Notebook 05 now loads actual Layer 1 lm-eval baseline scores and renders per-task before/after deltas.

---
