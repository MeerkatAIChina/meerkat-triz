# Phase 1: Foundation & Data Pipeline - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-05-27
**Phase:** 1-Foundation & Data Pipeline
**Areas discussed:** Synthetic generation strategy, Real vs synthetic ratio, Quality gate thresholds, Notebook 02b interaction model, Pipeline state registry design, Config fix scope, Pre-flight check depth, Requirements pinning strategy

---

## Synthetic Generation Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Rephrase questions, keep answers grounded | Moonshot rewrites each question with true semantic variation while keeping the answer closely tied to the seed | |
| Generate entirely new Q&A pairs from seeds | Moonshot reads a seed as inspiration and creates a fresh, semantically related but distinct Q&A pair | |
| Hybrid by subset | Rephrase for concept_explanation & ariz_guidance; new pairs for case_generation & contradiction_analysis; mix for principle_recommendation & innovation_assessment | ✓ |
| You decide | Claude chooses the approach | |

**User's choice:** Hybrid by subset
**Notes:** User confirmed the mapping: concept_explanation and ariz_guidance get rephrase-in-place (factual accuracy critical); case_generation and contradiction_analysis get new Q&A pairs (diversity matters); principle_recommendation and innovation_assessment get a mix.

---

## Scale / Multiplier Per Seed

| Option | Description | Selected |
|--------|-------------|----------|
| ~10x per seed (548 → ~6K total) | Balanced diversity without excessive API cost | |
| ~5x per seed (548 → ~3K total) | Lower cost, but may not be enough data | |
| Variable by subset | Higher multiplier for diversity subsets, lower for factual subsets | |
| You decide | Claude selects based on domain best practices and cost constraints | ✓ |

**User's choice:** You decide
**Notes:** User deferred multiplier selection to Claude. Recommendation: variable by subset (higher for case_generation/contradiction_analysis, lower for concept_explanation/ariz_guidance).

---

## Real vs Synthetic Ratio

| Option | Description | Selected |
|--------|-------------|----------|
| Reduce total target to ~2,000 (27% real) | Scale back synthetic generation so real samples dominate | |
| Upsample real data to ~1,500 via manual augmentation | Use Moonshot API to create high-quality augmented variants of real samples | |
| Accept ~9% real for v1.0 | Proceed with 548 real + ~5,500 synthetic; document as known limitation | |
| Variable ratio by subset | High-real ratio (30-40%) for factual subsets; lower (5-10%) for diversity subsets | ✓ |

**User's choice:** Variable ratio by subset
**Notes:** User refined to "Factual subsets: 25% real, Diversity subsets: 10% real." Mapping: concept_explanation and ariz_guidance at 25%; principle_recommendation and innovation_assessment at ~15%; case_generation and contradiction_analysis at 10%.

---

## Quality Gate Thresholds

| Option | Description | Selected |
|--------|-------------|----------|
| Perplexity + Diversity + Length | Full three-gate system; most rigorous but adds compute overhead | |
| Perplexity + Length only | Two-gate system; simpler, less risk of over-filtering | |
| Minimal -- length check only | Fastest but no content quality filtering | |
| You decide | Claude designs a pragmatic gate system | ✓ |

**User's choice:** You decide
**Notes:** User deferred quality gate design to Claude. Recommendation: pragmatic two-gate system (perplexity + length) for v1.0, with diversity enforced through generation strategy rather than post-hoc scoring.

---

## Failed Sample Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Reject and retry | Drop bad sample and request new variation from Moonshot | |
| Reject and skip | Drop bad sample and reduce subset count | |
| Include with warning flag | Keep sample but flag it in pipeline_state for researcher review | |
| You decide | Claude selects based on reliability and cost considerations | ✓ |

**User's choice:** You decide
**Notes:** User deferred failed sample handling to Claude.

---

## Notebook 02b Interaction Model

| Option | Description | Selected |
|--------|-------------|----------|
| Single 'Generate All' cell | One cell runs full pipeline end-to-end | |
| Step-by-step with checkpoints | Separate cells for each phase; researcher can inspect and resume | |
| Interactive wizard | Cell-by-cell with inline review after each batch | |
| You decide | Claude chooses based on best practices | ✓ |

**User's choice:** You decide
**Notes:** User deferred notebook interaction model to Claude.

---

## Resumability

| Option | Description | Selected |
|--------|-------------|----------|
| Yes -- save progress after each subset/batch | Write checkpoint files to disk; skip already-generated samples on restart | ✓ |
| No -- start fresh each time | Simpler code but wastes API calls if interrupted | |

**User's choice:** Yes -- save progress after each subset/batch
**Notes:** Explicitly requested by user. Essential for long-running generation with external API dependencies.

---

## Cost Monitoring

| Option | Description | Selected |
|--------|-------------|----------|
| Yes -- show estimated cost before each subset | Researcher sees cost before committing API calls | ✓ |
| No -- just run, log actual cost afterward | Simpler code but researcher takes cost risk | |

**User's choice:** Yes -- show estimated cost and token count before each subset
**Notes:** Explicitly requested by user. Recommended for external API usage.

---

## Pipeline State Registry Design

| Option | Description | Selected |
|--------|-------------|----------|
| Minimal -- data paths + checkpoint paths only | Smallest registry, limited traceability | |
| Standard -- data + checkpoints + benchmark results + generation metadata | Covers full v1.0 pipeline | |
| Comprehensive -- all of above + API cost logs + quality gate stats + training metrics + hardware config | Full audit trail; might be overkill for v1.0 | |
| You decide | Claude selects based on cross-notebook integration needs | ✓ |

**User's choice:** You decide
**Notes:** User deferred registry scope to Claude.

---

## Config Fix Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Fix only what's required (dropout + modules) | Minimal changes | |
| Also lock data config (split ratios, max_length) | Ensure data pipeline uses same values as downstream notebooks | |
| Also lock training config (LR, epochs, batch size) | All hyperparameters that affect training outcome | |
| You decide | Claude identifies all config values needing fixing or locking | ✓ |

**User's choice:** You decide
**Notes:** User deferred config fix scope to Claude.

---

## Pre-flight Check Depth

| Option | Description | Selected |
|--------|-------------|----------|
| Lightweight -- paths + artifacts + API key | Fast, catches 80% of startup issues | |
| Standard -- add version compatibility | Also check Python package versions match requirements.txt | |
| Thorough -- add memory and model sanity | Also verify model fits in memory, CUDA available, disk space | |
| You decide | Claude designs checks appropriate for notebook phase and DGX Spark | ✓ |

**User's choice:** You decide
**Notes:** User deferred pre-flight check depth to Claude.

---

## Requirements Pinning Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Exact versions (most reproducible) | e.g., transformers==4.45.2 | |
| Minimum versions (more flexible) | e.g., transformers>=4.45.0 | |
| Mixed -- core libs exact, utilities minimum | Pin critical packages exactly, minimum for utilities | |
| You decide | Claude selects based on DGX Spark stability needs | ✓ |

**User's choice:** You decide
**Notes:** User deferred pinning strategy to Claude.

---

## Claude's Discretion

The following areas were explicitly deferred to Claude's discretion during discussion:

1. Multiplier per seed (variable by subset)
2. Quality gate implementation details (pragmatic two-gate system)
3. Failed sample handling strategy
4. Notebook 02b cell structure and interaction flow
5. Pipeline state registry schema and scope
6. Config values beyond required fixes
7. Pre-flight check comprehensiveness by notebook phase
8. Requirements pinning approach

---

## Deferred Ideas

None -- discussion stayed within phase scope.
