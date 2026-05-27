---
phase: 01-foundation-data-pipeline
plan: 02
type: execute
subsystem: utils
requires: []
provides:
  - PipelineState JSON artifact registry
  - MoonshotSyntheticClient API client
  - SyntheticPipeline checkpoint-resumable generation
affects:
  - ref/mongoose_ai_dgx/utils/pipeline_state.py
  - ref/mongoose_ai_dgx/utils/synthetic_pipeline.py
  - ref/mongoose_ai_dgx/utils/__init__.py
tech-stack:
  added:
    - packaging (for semver comparison in preflight)
    - openai (Moonshot API compatible client)
  patterns:
    - JSON artifact registry for cross-notebook state
    - Rate-limited batched API client with checkpointing
    - MD5-based deduplication of seed data
key-files:
  created:
    - ref/mongoose_ai_dgx/utils/pipeline_state.py
    - ref/mongoose_ai_dgx/utils/synthetic_pipeline.py
  modified:
    - ref/mongoose_ai_dgx/utils/__init__.py
decisions:
  - Default state file at /home/meerkat/mongoose_ai/data/processed/pipeline_state.json (matches project path convention)
  - register() replaces existing entries with same name (idempotent)
  - verify() checks both registry entry AND filesystem existence
  - preflight() returns error list (not raising) so callers decide how to handle failures
  - API key from env var MOONSHOT_API_KEY, never hardcoded
  - Batch size of 5 seeds per request (configurable)
  - Rate limit sleep based on configurable RPM (default 3 for Tier 0)
  - Exponential backoff on RateLimitError (wait 60s, retry once)
  - Checkpoint saved after EVERY batch (not just subset)
  - Deduplication by MD5 hash of instruction+output
  - Original seeds included in results with "source": "seed" tag
  - Generated samples have "source": "synthetic" tag
  - Cost estimation in both CNY and USD
  - Response parsing handles JSON array, JSONL, and partial JSON
metrics:
  duration: "~10 minutes"
  completed_at: "2026-05-27"
  tasks: 3
  files_created: 2
  files_modified: 1
---

# Phase 01 Plan 02: Pipeline State and Synthetic Data Infrastructure Summary

**One-liner:** JSON artifact registry for cross-notebook state tracking and Moonshot API client for batched, rate-limited, checkpoint-resumable synthetic TRIZ data generation.

## What Was Built

### 1. PipelineState (pipeline_state.py)

A JSON artifact registry that enables notebooks to discover what prior steps produced:

- **register(name, path, type, metadata)** — idempotent registration (replaces existing entries with same name)
- **get(name)** — retrieve artifact by name
- **verify(name)** — check both registry entry AND filesystem existence
- **list_artifacts(type)** — list all artifacts, optionally filtered by type
- **preflight(required_artifacts, required_packages)** — validate prerequisites before execution; returns error list (not raises) so callers decide failure handling
- **summary()** — return state summary with artifact counts by type

Default state file: `/home/meerkat/mongoose_ai/data/processed/pipeline_state.json`

### 2. MoonshotSyntheticClient (synthetic_pipeline.py)

OpenAI-compatible client for the Moonshot API:

- **generate_variations(seeds, strategy, subset_name, num_variations)** — batch generation with rate limiting
- **estimate_cost(seed_count, batch_size)** — cost and time estimation in CNY/USD
- Configurable RPM (default 3 for Tier 0)
- Rate limit sleep between requests
- Retry on RateLimitError with 60s backoff
- Response parsing handles JSON array, JSONL, and partial JSON
- API key from env var `MOONSHOT_API_KEY`, never hardcoded

### 3. SyntheticPipeline (synthetic_pipeline.py)

High-level pipeline orchestrating generation across subsets:

- **deduplicate_seeds(seeds)** — MD5 hash deduplication by (instruction, output)
- **generate_subset(subset_name, seeds, strategy, multiplier, batch_size)** — checkpoint-resumable generation
  - Loads checkpoint if exists (resumes from completed batches)
  - Saves checkpoint after EVERY batch
  - Includes original seeds in output with `"source": "seed"`
  - Tags synthetic samples with `"source": "synthetic"`
- **save_subset(subset_name, samples)** — persist results to JSON

Three generation strategies:
- `rephrase` — rephrase questions, keep answers grounded
- `generate_new` — generate entirely new Q&A pairs inspired by seeds
- `mixed` — combine both approaches (~1:1 ratio)

### 4. Updated utils/__init__.py

Exports all three new classes alongside existing utilities.

## Deviations from Plan

None — plan executed exactly as written.

## Threat Surface Scan

| Flag | File | Description |
|------|------|-------------|
| threat_flag: api_key_env | synthetic_pipeline.py | API key read from env var MOONSHOT_API_KEY; never logged; never committed |
| threat_flag: response_validation | synthetic_pipeline.py | JSON structure validated; required fields (instruction, output) checked; malformed samples rejected |
| threat_flag: rate_limiting | synthetic_pipeline.py | Configurable RPM sleep; retry on RateLimitError with 60s backoff; cost estimation before generation |
| threat_flag: static_seed_data | synthetic_pipeline.py | Seed data is static JSON from project repo; no user input enters prompts |

All threats from the plan's `<threat_model>` are properly mitigated.

## Known Stubs

None. All functionality is fully implemented and wired.

## Self-Check: PASSED

- [x] `ref/mongoose_ai_dgx/utils/pipeline_state.py` exists
- [x] `ref/mongoose_ai_dgx/utils/synthetic_pipeline.py` exists
- [x] `ref/mongoose_ai_dgx/utils/__init__.py` exports all three classes
- [x] Commit 6447aad: pipeline_state.py created
- [x] Commit 88f45c5: synthetic_pipeline.py created
- [x] Commit 77f1e27: __init__.py updated
- [x] All modules import without errors (verified with mocked openai)
- [x] PipelineState.register/get/verify/preflight/summary tested
- [x] SyntheticPipeline.deduplicate_seeds tested
- [x] MoonshotSyntheticClient.estimate_cost/_parse_response/_format_seeds_for_prompt tested
