# Feature Landscape

**Domain:** QLoRA Fine-Tuning Pipeline for TRIZ Domain LLM
**Researched:** 2026-05-26
**Overall confidence:** HIGH (all features derived from existing codebase and standard QLoRA/SFT practices)

---

## Table Stakes

Features users expect from any QLoRA fine-tuning pipeline. Missing any of these makes the pipeline feel broken or incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Synthetic data generation from seed samples** | 548 seeds insufficient for meaningful fine-tuning; ~6K target is standard domain adaptation minimum | Medium | Existing `create_synthetic_data()` uses simple paraphrase templates. Must be replaced/augmented for production quality. |
| **Baseline benchmark before training** | Required to measure delta/improvement from fine-tuning. Without it, "did it work?" is unanswerable. | Medium | Notebook 03 runs Layer 1 (lm-eval) + Layer 2 (TRIZ custom) + Layer 3 (perf). Layer 1 is optional but strongly recommended. |
| **QLoRA fine-tuning execution** | Core value proposition of the project. Must run end-to-end without manual intervention. | Medium | Notebook 04. ~8-15 hours per epoch on DGX Spark. Single most time-consuming step. |
| **Post-training evaluation** | Required to validate that fine-tuning improved (or at least didn't harm) capabilities. | Low | Notebook 05 re-runs same benchmarks as 03 for direct comparison. |
| **Checkpoint saving during training** | Training runs for hours; crashes or OOM must not lose all progress. | Low | `save_steps=200`, `save_total_limit=3` in config. |
| **Train/validation/test split** | Standard ML practice. Validation for early stopping, test for final unbiased evaluation. | Low | 85/10/5 split already implemented in `split_dataset()`. |
| **ChatML format conversion** | Qwen3.6 expects chat-templated input; raw instruction/output pairs won't train correctly. | Low | `convert_to_chatml()` uses `tokenizer.apply_chat_template()` — correct approach per audit CR-003. |
| **Memory-efficient training** | 128GB unified memory is generous but not infinite; 35B model + activations can OOM without care. | Medium | 4-bit NF4 quantization, gradient checkpointing, paged AdamW 8-bit, batch_size=1 + grad_accum=8. |

### Table Stakes — Expected Behavior

**Synthetic Data Generation:**
- Input: 548 seed samples across 6 TRIZ subsets.
- Output: ~6,000 samples (roughly 10x expansion per subset).
- Must preserve domain accuracy: TRIZ principles, contradictions, ARIZ steps must not be corrupted by paraphrasing.
- Must maintain ChatML compatibility: generated samples feed directly into `convert_to_chatml()`.
- Existing implementation (`vary_sample`) is rule-based prefix substitution and output extension. This is **not sufficient** for production — known limitation from audit report.

**Baseline Benchmark (Pre-Training):**
- Load base Qwen3.6-35B-A3B in 4-bit NF4.
- Run Layer 1 (optional, hours): MMLU-Pro, GPQA, HumanEval, MATH, BBH via `lm-eval-harness`.
- Run Layer 2 (required, minutes): TRIZ custom benchmark — principle accuracy, contradiction resolution, case quality, ARIZ completeness.
- Run Layer 3 (required, minutes): Throughput, P50 latency, peak memory.
- Save all results with timestamps for comparison.

**QLoRA Fine-Tuning:**
- Load 4-bit quantized base model (~18-20GB).
- Apply LoRA config: rank=64, alpha=128, dropout=0.05, target_modules = explicit list (NOT `"all-linear"`).
- Use `SFTTrainer` with `formatting_func` (no `data_collator`).
- Train for 2 epochs, cosine scheduler, 5% warmup, effective batch size 8.
- Save LoRA adapter only (~100-200MB), not full merged model.
- Expected peak memory: 60-80GB (well within 128GB).

**Post-Training Evaluation:**
- Load base model + LoRA adapter via `AutoPeftModelForCausalLM`.
- Re-run identical Layer 2 and Layer 3 benchmarks.
- Compute improvement metrics (e.g., TRIZ overall score delta).
- Generate side-by-side comparison report.

---

## Differentiators

Features that set this pipeline apart from a generic QLoRA tutorial. Most are already partially implemented.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Three-layer evaluation** | General + domain + performance gives complete picture; most pipelines only do loss curves. | Medium | Layer 1 (general) is expensive; Layer 2 (TRIZ) is the real differentiator; Layer 3 (perf) ensures deployment viability. |
| **TRIZ domain-specific benchmarks** | No off-the-shelf benchmark exists for TRIZ. Custom evaluator measures what actually matters. | Medium | `TRIZBenchmark` class with principle accuracy, contradiction resolution, case quality, ARIZ completeness. Currently only 5 test questions — needs expansion. |
| **Hybrid architecture target_modules** | Qwen3.6's Gated DeltaNet + Gated Attention + MoE requires explicit module list. Auto-detection is risky. | Low | Explicit list of 12 module names covering all three layer types. Audit confirmed this is correct. |
| **Notebook-driven reproducible workflow** | Each phase is a checkpointed notebook with clear inputs/outputs. Easy to restart from any step. | Low | 01→02→03→04→05 pipeline. Can re-run 03/05 independently for comparison. |
| **Synthetic data pipeline with API integration** | Moonshot API for high-quality paraphrase/variation vs. naive template substitution. | High | Mentioned in PROJECT.md as existing, but code in repo only has naive `vary_sample()`. Likely a gap between intent and implementation. |
| **DGX Spark optimization** | Config tuned for 128GB unified memory, single GPU, FP4 compute. Not generic "works anywhere" settings. | Low | `paged_adamw_8bit`, `fp16=True`, `bf16=False`, `per_device_batch_size=1`. |

### Differentiators — Expected Behavior

**Three-Layer Evaluation:**
- Layer 1 establishes "did we break general capabilities?" Catastrophic forgetting detection.
- Layer 2 establishes "did we learn TRIZ?" Domain expertise measurement.
- Layer 3 establishes "can we deploy this?" Inference efficiency measurement.
- All three layers feed into `aggregate_results()` for a single JSON report.

**TRIZ Domain Benchmarks:**
- Principle accuracy: multiple-choice style — model must identify correct invention principle from description.
- Contradiction resolution: open-ended — model must mention expected keywords (e.g., "composite materials", "porous materials").
- Case quality: generation — model must generate structurally correct TRIZ case with BLEU/ROUGE against reference.
- ARIZ completeness: model must mention all 6 ARIZ steps (problem analysis, model, ideal final result, contradiction analysis, resource analysis, solution evaluation).
- Current test set is only 5 questions. For meaningful evaluation, expand to 50-100 per category.

**Hybrid Architecture Handling:**
- `find_all_linear_names()` can auto-detect modules, but audit recommended explicit list.
- Explicit list covers: Gated Attention (q/k/v/o_proj), Gated DeltaNet (in_proj_qkv/z/b/a, out_proj), MoE MLP (gate/up/down_proj).
- This is a differentiator because most QLoRA guides assume standard Transformer (q/v_proj only) and will silently under-train on hybrid architectures.

---

## Anti-Features

Features to explicitly NOT build. These are common temptations that waste time or cause problems.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Full fine-tuning** | 35B parameters × 2 bytes (FP16) = 70GB just for weights. Optimizer states would exceed 128GB unified memory. | Stick with QLoRA. Adapter is ~100-200MB, trainable parameters ~0.1% of total. |
| **Merged model export** | Merged model is ~140GB. No deployment target exists (out of scope per PROJECT.md). Wastes disk space. | Save adapter only. `AutoPeftModelForCausalLM` can load base+adapter on-the-fly. |
| **Multi-GPU / distributed training** | DGX Spark has one GB10. No additional GPUs available. | Single-device training with gradient accumulation (effective batch=8). |
| **Real-time inference serving** | Out of scope per PROJECT.md. Training focus only. | Notebook 05 does ad-hoc generation tests; no API server needed. |
| **Hardcoded ChatML strings** | Audit CR-003 flagged this. Different models use different chat templates. Hardcoding breaks portability. | Always use `tokenizer.apply_chat_template()`. |
| **Passing `data_collator` to `SFTTrainer`** | Audit CR-001 flagged this. SFTTrainer has internal label-masking logic; external collator conflicts. | Use `formatting_func` parameter only. No `data_collator`. |
| **Using `"all-linear"` for target_modules** | Audit CR-002 flagged this. Qwen3.6 hybrid architecture causes incorrect module inclusion (e.g., `lm_head`). | Use explicit manual module list. |
| **Running Layer 1 benchmarks twice** | Layer 1 (lm-eval) takes hours. Running pre+post is ~2×4-6 hours = significant time sink. | Run Layer 1 once pre-training, optionally spot-check one task post-training. Layer 2 and 3 are fast enough to run twice. |
| **Synthetic data without expert validation** | TRIZ is a precise methodology. LLM-generated variations can introduce subtle errors (wrong principle numbers, incorrect contradiction mappings). | If using API-based generation (Moonshot/GPT-4o), add expert review step or at least automated consistency checks. |

---

## Feature Dependencies

```
Synthetic Data Generation
    → ChatML Format Conversion (generated samples must be convertible)
        → Train/Validation/Test Split
            → QLoRA Fine-Tuning
                → Checkpoint Saving
                    → Post-Training Evaluation
                        → Comparison Report

Baseline Benchmark (Pre-Training)
    → Independent of training, but must use same model loading path
    → Results feed into Comparison Report

Three-Layer Evaluation
    ├── Layer 1 (General) — optional, expensive
    ├── Layer 2 (TRIZ Custom) — requires trained model OR base model
    └── Layer 3 (Performance) — requires loaded model
```

### Dependency Notes

- **Synthetic data → ChatML:** `create_synthetic_data()` output format must match what `convert_to_chatml()` expects (`instruction`, `input`, `output` fields). Current implementation is compatible.
- **ChatML → Split:** `split_dataset()` operates on the `Dataset` object produced by `convert_to_chatml()`. Must be deterministic (seed=42) for reproducibility.
- **Split → Training:** `create_trainer()` takes `train_dataset` and `eval_dataset`. `test_dataset` is held out for final evaluation.
- **Training → Checkpoint:** `save_steps=200` means checkpoints saved mid-training. `save_total_limit=3` prevents disk bloat.
- **Checkpoint → Evaluation:** Notebook 05 loads adapter from `MODELS_DIR / 'meerkat_triz_adapter_v1'`. Path must match training output.
- **Baseline ↔ Post-Training:** Both use same benchmark functions. Baseline results must be persisted (JSON) for comparison.

---

## MVP Recommendation

For the current milestone (v1.0 First Training Run), prioritize:

1. **Synthetic data generation (~6K samples)** — Table stakes. Current 548 seeds won't produce meaningful adaptation. The existing `create_synthetic_data()` is too naive; the PROJECT.md mentions a 6-stage Moonshot API pipeline that may need to be implemented or connected.
2. **Baseline benchmark (Notebook 03)** — Table stakes. Run at least Layer 2 and Layer 3 before training. Layer 1 is optional for MVP but recommended if time permits.
3. **QLoRA fine-tuning run (Notebook 04)** — Table stakes. The core deliverable. 2 epochs, ~15 hours. Must complete without OOM or crash.
4. **Post-training evaluation (Notebook 05)** — Table stakes. Re-run Layer 2 and Layer 3. Generate comparison numbers.

**Defer:**
- **Layer 1 full suite post-training:** Too time-consuming for MVP. Spot-check one task if needed.
- **Merged model export:** Out of scope, wastes disk.
- **Expanded TRIZ test set:** Current 5 questions is minimal but sufficient for a first run. Expand to 50-100 in v1.1.
- **Production synthetic data pipeline:** GPT-4o-based synthesis with expert review is recommended by audit but not required for v1.0.

---

## Complexity Summary

| Feature Area | Complexity | Risk | Time Estimate |
|--------------|------------|------|---------------|
| Synthetic data generation | Medium | HIGH (quality degradation) | 2-4 hours (API calls) |
| Baseline benchmark | Medium | LOW | 30 min (L2+L3) or 4-6 hours (+L1) |
| QLoRA fine-tuning | Medium | MEDIUM (OOM, crash) | 16-30 hours (2 epochs) |
| Post-training evaluation | Low | LOW | 30 min |
| Comparison report | Low | LOW | 15 min |

**Critical path:** Synthetic data → Baseline → Training → Evaluation. All sequential. Total wall time: ~20-35 hours, mostly training.

---

## Sources

- `ref/mongoose_ai_dgx/config.py` — All hyperparameters and hardware configs
- `ref/mongoose_ai_dgx/utils/data_utils.py` — Data loading, ChatML conversion, synthetic generation
- `ref/mongoose_ai_dgx/utils/training_utils.py` — Model loading, QLoRA config, SFTTrainer setup
- `ref/mongoose_ai_dgx/utils/benchmark_utils.py` — Three-layer evaluation implementation
- `ref/mongoose_ai_dgx/notebooks/01-05.ipynb` — Notebook workflow definitions
- `.planning/PROJECT.md` — Milestone scope and validated/active requirements
- `ref/审计报告_猫鼬AI训练方案.md` — Audit findings CR-001 through CR-003, MA-001
