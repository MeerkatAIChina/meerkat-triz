---
phase: 01-foundation-data-pipeline
plan: 05
subsystem: training_utils
wave: 1
type: gap_closure
phase_number: 1
plan_number: 5
---

# Phase 01 Plan 05: Fix training_utils.py Docstring/Defaults Gap Summary

**One-liner:** Closed documentation gap in `setup_qlora_config` by aligning default `lora_dropout` (0.05 → 0.0) and docstring guidance with `config.py` source of truth, preventing researchers from inadvertently using incompatible `all-linear` on Qwen3.6 hybrid architecture.

## What Was Built

Updated `ref/mongoose_ai_dgx/utils/training_utils.py` with three surgical changes to `setup_qlora_config()`:

1. **Default `lora_dropout` changed from 0.05 to 0.0** (line 154) — with inline comment explaining MoE architecture compatibility rationale (`dropout可能干扰专家路由稳定性`).
2. **Docstring rewritten** (lines 157–179) — now lists the explicit 12-module Qwen3.6 target list as the recommended default, documents all three layer types (Gated Attention, Gated DeltaNet, MoE MLP), and downgrades `"all-linear"` from "recommended" to "not recommended (known compatibility issues on hybrid architecture)".
3. **`all-linear` log level changed from `info` to `warning`** (lines 184–186) — with explicit compatibility warning referencing the audit finding (may incorrectly include `lm_head`).

## Verification Results

| Check | Command | Result |
|-------|---------|--------|
| Default dropout is 0.0 | `grep "lora_dropout: float = 0.0"` | MATCH (line 154) |
| No remaining 0.05 default | `grep -c "lora_dropout.*0.05"` | 0 |
| Docstring recommends explicit list | `grep "显式模块列表"` | MATCH (line 161) |
| Docstring warns against all-linear | `grep "不推荐.*all-linear"` | MATCH (line 163) |
| Log uses warning + compatibility note | `grep "已知在混合架构上存在兼容性问题"` | 2 MATCHES (docstring + log) |
| MoE compatibility mentioned | `grep "MoE架构兼容"` | MATCH (line 174) |
| Other defaults unchanged | `grep "r: int = 64"`, `lora_alpha: int = 128`, `use_rslora: bool = False` | ALL MATCH |

## Decisions Made

None — this was a deterministic gap-closure task with no architectural choices.

## Deviations from Plan

None — plan executed exactly as written. All three specified changes applied with no additional modifications.

## Threat Surface

No new threat surface introduced. This change reduces tampering risk (T-01-05-01) and DoS risk (T-01-05-02) by correcting misleading documentation that could lead researchers to use incompatible settings.

## Known Stubs

None.

## Metrics

| Metric | Value |
|--------|-------|
| Tasks completed | 1/1 |
| Files modified | 1 |
| Lines changed | +14 / −9 |
| Duration | < 5 minutes |
| Commits | 1 |

## Commits

| Hash | Message | Files |
|------|---------|-------|
| ce909ff | fix(01-05): update setup_qlora_config docstring and defaults to match config.py | ref/mongoose_ai_dgx/utils/training_utils.py |

## Self-Check: PASSED

- [x] Modified file exists: `ref/mongoose_ai_dgx/utils/training_utils.py`
- [x] Commit exists: `ce909ff`
- [x] No accidental file deletions in commit
- [x] All acceptance criteria verified via grep
