# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 — MVP

**Shipped:** 2026-07-12
**Phases:** 4 | **Plans:** 18 | **Tasks:** 20

### What Was Built
- A complete notebook-driven QLoRA fine-tuning pipeline for Qwen3.6-35B-A3B on the TRIZ domain.
- `utils/synthetic_pipeline.py` + Notebook 02b for ~6K synthetic TRIZ sample generation via Moonshot API.
- Training utilities with `CheckpointValidationCallback`, SHA-256 adapter integrity, and LR-verified resume.
- Three-layer evaluation utilities (`benchmark_utils.py`) with unified `format_messages()`, BLEU/ROUGE scoring, and before/after aggregate reporting.
- Closure of audit blocker BLK-01: Notebook 05 now loads real Layer 1 lm-eval scores and displays per-task deltas.
- Source code moved from `ref/mongoose_ai_dgx/` to repo root to match DGX Spark deployment layout.

### What Worked
- Inserting Phase 3.1 as a focused gap-closure phase kept the BLK-01 fix scoped and verifiable.
- Mock-heavy pytest tests allowed local validation of training/evaluation utilities without DGX Spark hardware.
- Preserving git history with `git mv` during the source restructure made the move reviewable.

### What Was Inefficient
- Planning files (`REQUIREMENTS.md`, `ROADMAP.md`) became stale after Phase 2 and required reconciliation before milestone close.
- The original Phase 3 lacked a `VERIFICATION.md`, which blocked milestone completion until it was created retroactively.
- Phase 03.1 UAT and verification artifacts remain incomplete; these were deferred to close the milestone.

### Patterns Established
- Use `git mv` for directory restructures to preserve history.
- Mock torch/transformers/datasets in tests so local macOS development can validate logic without DGX Spark.
- Decimal phase numbering (e.g., 3.1) for urgent gap closures inserted after a parent phase.

### Key Lessons
1. Close verification artifacts immediately when a phase finishes; retroactive creation slows milestone close.
2. Keep `requirements.txt` and notebook dependency cells in sync to avoid audit warnings.
3. Pipeline-state consumers should read artifact files from disk rather than relying on embedded metadata.

### Cost Observations
- Model mix: Not tracked for v1.0
- Sessions: Multiple sessions across planning, execution, verification, and source restructure
- Notable: Significant context was spent reconciling stale planning state before milestone close

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | Multiple | 4 | Established notebook-driven pipeline and three-layer evaluation |

### Cumulative Quality

| Milestone | Tests | Coverage | Zero-Dep Additions |
|-----------|-------|----------|-------------------|
| v1.0 | 13+ | Mock-based | pytest suite that runs without DGX hardware |

### Top Lessons (Verified Across Milestones)

1. Verification artifacts are part of the definition of done.
2. Planning files must be updated at phase boundaries, not just at milestone close.
