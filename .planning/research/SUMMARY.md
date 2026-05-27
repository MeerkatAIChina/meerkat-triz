# Project Research Summary

**Project:** Meerkat AI (猫鼬AI) — TRIZ Domain LLM Fine-Tuning
**Domain:** QLoRA fine-tuning of Qwen3.6-35B-A3B on NVIDIA DGX Spark (128GB unified memory)
**Researched:** 2026-05-26
**Confidence:** HIGH

---

## Executive Summary

Meerkat AI is a domain-specialized LLM fine-tuning project that adapts Qwen3.6-35B-A3B into a TRIZ (Theory of Inventive Problem Solving) innovation consultant using QLoRA on an NVIDIA DGX Spark. The existing codebase has a well-structured notebook-driven pipeline (01 setup → 02 data prep → 03 baseline → 04 training → 05 evaluation) with audited utility modules. The core challenge for the current milestone is not architectural redesign but **execution integration**: bridging the gap between the current skeleton and a runnable end-to-end pipeline that produces a trained LoRA adapter.

The recommended approach is **minimal-change execution integration** (Option A from ARCHITECTURE.md): keep the notebook-driven workflow, add the missing synthetic data generation pipeline (Moonshot API), fix version compatibility gaps in dependencies, and harden against known pitfalls. The existing training logic in `training_utils.py` is complete and audited; the evaluation framework in `benchmark_utils.py` is functional; the config in `config.py` is centralized. What is missing is the ~6K synthetic training samples, cross-notebook state tracking, and several critical config fixes identified by research.

Key risks center on training execution: (1) **lora_dropout=0.05 is incompatible with MoE** — this must be changed to 0.0 or training will silently corrupt gradients; (2) **synthetic data quality** — the current template-based `vary_sample()` will cause model collapse if scaled to 6K samples, requiring Moonshot API integration with quality gates; (3) **checkpoint resume fragility** — 15-hour training runs in Jupyter kernels are vulnerable to interruption with imperfect resume; (4) **4-bit benchmarking skews baselines** — evaluation must use FP16, not 4-bit, for valid pre/post comparison. All four are actionable with clear prevention strategies.

---

## Key Findings

### Recommended Stack

The current `requirements.txt` is directionally correct but has version gaps that create real compatibility risks. The most critical changes are: pin `transformers` to 4.45+ (Qwen3.6 minimum) but <5.0.0 (breaks lm-eval), pin `trl` to 0.9–0.11 (preserves `formatting_func` API), upgrade `lm-eval` to 0.4.10+ with `[hf]` extra, add `rouge-score` (imported by `benchmark_utils.py` but missing from deps), and add `openai` SDK for Moonshot API integration. Do NOT add vLLM, Unsloth, Flash Attention, DeepSpeed, or RAGAS — they are unnecessary or incompatible for this single-GPU training scope.

**Core technologies:**
- `torch>=2.4.0,<2.6.0` — DGX Spark validated sweet spot; 2.4.x most tested on GB10
- `transformers>=4.45.0,<5.0.0` — Qwen3.6 support; v5 breaks lm-eval compatibility
- `trl>=0.9.0,<0.12.0` — preserves `SFTTrainer(..., formatting_func=...)` API; 0.12+ deprecates it
- `bitsandbytes>=0.43.0,<0.45.0` — NF4 quantization; 0.43.x is stable target for reproducibility
- `lm-eval>=0.4.10` — decoupled backends, YAML config; must install with `[hf]` extra
- `rouge-score>=0.1.2` — Layer 2 TRIZ case quality evaluation (currently missing, causes ImportError)
- `openai>=1.0.0` — Moonshot API client (OpenAI-compatible endpoint)

### Expected Features

**Must have (table stakes):**
- Synthetic data generation (~6K samples from 548 seeds) — current naive template approach insufficient
- Baseline benchmark before training (at minimum Layer 2 TRIZ + Layer 3 performance)
- QLoRA fine-tuning execution end-to-end without manual intervention
- Post-training evaluation with before/after comparison
- Checkpoint saving during training (save_steps=200, save_total_limit=3)
- Train/validation/test split (85/10/5, already implemented)
- ChatML format conversion via `tokenizer.apply_chat_template()`
- Memory-efficient training (4-bit NF4, gradient checkpointing, paged AdamW 8-bit)

**Should have (differentiators):**
- Three-layer evaluation (general + TRIZ custom + performance)
- TRIZ domain-specific benchmarks (principle accuracy, contradiction resolution, case quality, ARIZ completeness)
- Hybrid architecture target_modules (explicit 12-module list for Gated DeltaNet + Gated Attention + MoE)
- Notebook-driven reproducible workflow with clear inputs/outputs
- Moonshot API-based synthetic data pipeline (true semantic variation vs. template paraphrasing)
- DGX Spark optimization (config tuned for 128GB unified memory, single GPU, FP4 compute)

**Defer (v2+):**
- Layer 1 full suite post-training (too time-consuming for MVP; spot-check one task if needed)
- Merged model export (out of scope, wastes 140GB disk)
- Expanded TRIZ test set (current 5 questions sufficient for first run; expand to 50-100 in v1.1)
- Production synthetic data pipeline with GPT-4o and expert review (recommended by audit but not required for v1.0)
- Real-time inference serving (out of scope per PROJECT.md)

### Architecture Approach

The project follows a **notebook-driven execution model** with three conceptual tiers: Jupyter notebook orchestration (thin wrappers), a Python `utils/` package (pure functions, no side effects), and external dependencies (transformers, PEFT, TRL, lm-eval). The architecture is already well-structured; the integration work is about filling gaps, not redesign. The single largest gap is **Notebook 02b (synthetic data generation)** — mentioned in PROJECT.md but completely missing from the codebase. State management between notebooks is implicit and fragile (hardcoded paths, manual cleanup, no artifact registry). The recommended approach is minimal change: create `02b_synthetic_generation.ipynb` + `utils/synthetic_pipeline.py`, add `utils/pipeline_state.py` for cross-notebook state tracking, and auto-load baseline results in Notebook 05.

**Major components:**
1. **Notebook Orchestration (01–05 + 02b)** — Cell-by-cell execution, manual checkpointing, human review points
2. **Python Utils Package (config, data_utils, training_utils, benchmark_utils, synthetic_pipeline, pipeline_state)** — Pure functions, no side effects, reusable across notebooks and optional CLI scripts
3. **External Dependencies** — transformers, PEFT, TRL, bitsandbytes, lm-eval-harness, datasets

### Critical Pitfalls

1. **lora_dropout > 0 with MoE architecture** — `lora_dropout=0.05` (current config) causes training instability or silent gradient corruption with MoE models. Community reports enforce `lora_dropout=0.0` for MoE compatibility. **Fix:** Set `lora_dropout=0.0` in `config.py`; increase `weight_decay` if regularization needed.

2. **SFTTrainer + DataCollator conflict (CR-001 regression)** — Passing `data_collator` to `SFTTrainer` causes label-masking conflict, training the model to predict user tokens instead of assistant tokens. **Fix:** Never pass `data_collator`; always use `formatting_func` + `packing=True`.

3. **target_modules="all-linear" on hybrid architecture (CR-002 regression)** — PEFT auto-detection misses Gated DeltaNet layers or incorrectly includes `lm_head`. **Fix:** Use explicit 12-module list covering all three layer types (Gated Attention, Gated DeltaNet, MoE MLP).

4. **Synthetic data pipeline produces collapsed distribution** — Template-based `vary_sample()` scaled to 6K samples creates high lexical redundancy and model collapse. **Fix:** Use Moonshot API for true semantic variation; add perplexity filtering, diversity scoring, and maintain 20-30% real/human-curated data ratio.

5. **4-bit quantization skews baseline benchmarks** — Loading model in 4-bit for Notebook 03 introduces 1-3% accuracy degradation, making pre/post comparison misleading. **Fix:** Load FP16 for benchmarking; 128GB unified memory can hold 35B FP16 (~70GB) + activations.

6. **Notebook-driven training without robust checkpoint resume** — 15-hour runs in ephemeral Jupyter kernels are vulnerable to interruption with imperfect resume (data iterator state and RNG not preserved). **Fix:** Use `tmux`/`nohup` for long runs; verify checkpoint completeness before resume; save metadata alongside checkpoints.

7. **Chat template mismatch between training and inference** — Training uses `tokenizer.apply_chat_template()` but `benchmark_utils.py` `_build_prompt()` hardcodes ChatML strings. **Fix:** Create single `format_messages()` utility used in all paths; verify token ID equivalence.

8. **UMA double-allocation OOM on model load** — DGX Spark unified memory causes double allocation (page cache + CUDA tensors) at load time. **Fix:** Use `low_cpu_mem_usage=True` with `torch_dtype=torch.float16`; ensure `BitsAndBytesConfig` is passed for 4-bit loads; monitor with `watch -n 1 free -h`.

---

## Implications for Roadmap

Based on combined research, the execution pipeline should be structured in three phases with a hard dependency chain:

### Phase 1: Foundation & Data Pipeline (Unblock Critical Path)
**Rationale:** The ~6K synthetic training samples are the critical path blocker. Without them, training cannot begin. This phase also fixes the most dangerous config issues (lora_dropout, target_modules) before they corrupt a training run.
**Delivers:**
- Updated `requirements.txt` with pinned compatible versions
- `utils/synthetic_pipeline.py` — Moonshot API client, batching, rate limiting, output validation
- `02b_synthetic_generation.ipynb` — orchestration for ~6K sample generation
- `config.py` fixes: `lora_dropout=0.0`, explicit target_modules verified
- Token length profiling in Notebook 02 to catch silent truncation
**Addresses:** Synthetic data generation (table stakes), ChatML conversion (table stakes), memory-efficient training config (table stakes)
**Avoids:** Pitfall 5 (lora_dropout MoE), Pitfall 6 (collapsed synthetic distribution), Pitfall 9 (silent truncation)
**Research flag:** LOW — Moonshot API integration is straightforward OpenAI-compatible client; standard patterns apply. Token length profiling is well-documented in TRL docs.

### Phase 2: Baseline & Training Execution (Core Deliverable)
**Rationale:** Once data is ready, the pipeline must establish a baseline (to measure improvement) and execute the 15-hour training run. This is the core value proposition. State tracking is added here to ensure baseline results feed automatically into post-training comparison.
**Delivers:**
- `utils/pipeline_state.py` — JSON artifact registry for cross-notebook state
- Notebook 03 execution with FP16 model load (not 4-bit) for valid baselines
- Notebook 04 QLoRA training run with checkpoint resume verification
- Comprehensive adapter metadata saved alongside LoRA weights
- Immediate load-and-forward-pass verification after save
**Addresses:** Baseline benchmark (table stakes), QLoRA fine-tuning (table stakes), checkpoint saving (table stakes), three-layer evaluation (differentiator)
**Avoids:** Pitfall 3 (checkpoint resume fragility), Pitfall 4 (UMA OOM), Pitfall 7 (4-bit benchmark skew), Pitfall 11 (missing adapter metadata), Pitfall 12 (inference loading failure)
**Research flag:** MEDIUM — Checkpoint resume behavior with SFTTrainer has known edge cases (data iterator state, LR scheduler restart). A short validation run should confirm resume correctness before the full 15-hour run.

### Phase 3: Evaluation & Hardening (Validation & Polish)
**Rationale:** Post-training evaluation validates that fine-tuning improved (or didn't harm) capabilities. This phase also hardens the pipeline against integration risks and fixes documentation drift.
**Delivers:**
- Notebook 05 with automatic baseline loading from `pipeline_state`
- Unified `format_messages()` utility replacing hardcoded ChatML in `benchmark_utils.py`
- Before/after comparison report (Layer 2 TRIZ + Layer 3 performance)
- README.md correction ("all-linear" → explicit module list)
- Notebook pre-flight checks (paths exist, artifacts present, version compatibility)
- Optional CLI wrapper scripts for headless execution
**Addresses:** Post-training evaluation (table stakes), comparison report (table stakes), TRIZ domain benchmarks (differentiator)
**Avoids:** Pitfall 8 (chat template mismatch), Pitfall 15 (hardcoded baseline score), Pitfall 16 (missing trust_remote_code), Integration Risk 1 (cell execution order), Integration Risk 2 (cross-notebook state pollution)
**Research flag:** LOW — Evaluation patterns are standard; `AutoPeftModelForCausalLM` loading is well-documented. The only uncertainty is whether FP16 inference on DGX Spark has throughput issues, but this is a measurement concern, not a blocker.

### Phase Ordering Rationale

- **Sequential dependency:** Data → Baseline → Training → Evaluation. All phases are strictly ordered. The 20-35 hour total wall time is dominated by training (16-30 hours).
- **Risk mitigation:** Phase 1 fixes the two most dangerous issues (lora_dropout, synthetic data quality) before any GPU time is spent. Phase 2 includes verification steps (checkpoint completeness, adapter load test) that prevent wasted training runs.
- **Minimal disruption:** Each phase builds on existing audited code without refactoring working components. The only new code is `synthetic_pipeline.py`, `pipeline_state.py`, and notebook cells.
- **Fail-fast:** Phase 1 can be validated without GPU (API calls + data validation). Phase 2's baseline (30 min for L2+L3) validates the evaluation pipeline before committing to 15-hour training.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2 (Training Execution):** Checkpoint resume behavior with SFTTrainer — verify data iterator state and LR scheduler continuity on a short test run before the full 15-hour execution.
- **Phase 1 (Synthetic Data):** Moonshot API rate limits and cost for ~6K samples — validate batch size, retry strategy, and total API cost before scaling.

Phases with standard patterns (skip research-phase):
- **Phase 3 (Evaluation):** AutoPeftModel loading, ROUGE scoring, and before/after comparison are well-documented patterns. No novel research needed.
- **Stack updates:** Version pinning and dependency management are standard Python packaging practices.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Direct code inspection of requirements.txt and imports; official release notes for lm-eval 0.4.10; TRL formatting_func deprecation is well-documented. |
| Features | HIGH | All features derived from existing codebase, standard QLoRA/SFT practices, and audit findings. No speculative features. |
| Architecture | HIGH | All findings from direct source code inspection. The notebook-driven model is already implemented; gaps are clearly identified. |
| Pitfalls | HIGH | Based on verified audit findings (CR-001/002/003, MA-001/002/003), NVIDIA forum reports, and community consensus on MoE + dropout. |

**Overall confidence:** HIGH

### Gaps to Address

1. **Moonshot API rate limits and cost:** The 6-stage pipeline in PROJECT.md mentions Moonshot API but no implementation exists. Need to validate: API key availability, rate limits for ~6K calls, cost estimate, and whether batching is supported. Handle during Phase 1 planning.

2. **Checkpoint resume validation:** SFTTrainer's `resume_from_checkpoint` behavior with data iterator state is theoretically risky but untested in this specific codebase. A 100-step test run with intentional interruption and resume should validate before the full training run. Handle during Phase 2 planning.

3. **DGX Spark PyTorch version sweet spot:** Community reports vary on optimal PyTorch for GB10. 2.4.x is the safest validated choice, but 2.5+ may work. If 2.4.x has issues, fallback testing needed. Handle during Phase 1 (Notebook 01 execution).

4. **bf16 vs fp16 on Blackwell:** Config currently sets `bf16=False, fp16=True`. GB10 has excellent bf16 support; switching may improve stability and speed. A short comparison run should decide. Handle during Phase 2 as a hyperparameter tuning task.

5. **Synthetic data quality gates:** Perplexity filtering and diversity scoring are recommended but no implementation exists. Need to define thresholds and validation logic. Handle during Phase 1 implementation.

---

## Sources

### Primary (HIGH confidence)
- Direct source code inspection of `ref/mongoose_ai_dgx/config.py`, `utils/data_utils.py`, `utils/training_utils.py`, `utils/benchmark_utils.py`, `notebooks/01-05.ipynb` — All architecture and feature findings
- Audit Report: `ref/审计报告_猫鼬AI训练方案.md` — CR-001/002/003, MA-001/002/003 findings
- TRL SFTTrainer `formatting_func` evolution (Zenn, 2026) — Version pinning rationale
- lm-eval-harness Releases (EleutherAI, 2024-2025) — 0.4.10 decoupling and `[hf]` extra
- QLoRA Paper (NeurIPS 2023) — NF4 quantization fundamentals
- arXiv:2605.05561 — bitsandbytes NF4 reproducibility across versions
- NVIDIA DGX Spark Performance Tuning Docs — Hardware optimization guidance

### Secondary (MEDIUM confidence)
- NVIDIA Developer Forums: Qwen3.5-35B-A3B bf16 LoRA on DGX Spark — UMA double-allocation, lora_dropout=0, adamw_8bit CUDA 13.2 issue
- GitHub: Fine-Tuning Llama 3.1 70B on DGX Spark — Memory requirements, gradient checkpointing
- Red Hat: 500K+ Evaluations on Quantized LLMs — 4-bit evaluation accuracy gaps
- Meta Llama 3 Technical Report — Synthetic data degradation findings
- arXiv: Clinical Decision Support (MoE dropout=0) — MoE lora_dropout requirement
- Kimi K2.5 API Developer Guide — Moonshot API OpenAI-compatible endpoint

### Tertiary (LOW confidence)
- Apertus.ai DGX Spark Review — Memory bandwidth bottleneck estimates
- iFactory DGX Spark vs RTX PRO 6000 — Throughput comparison benchmarks
- Stack Overflow: SFTTrainer truncation clarification — Packing truncation behavior

---

*Research completed: 2026-05-26*
*Ready for roadmap: yes*
