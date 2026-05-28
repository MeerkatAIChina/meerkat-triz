# Milestone v1.0 Requirements

## Validated (from prior work)
- [x] **INFRA-00**: Qwen3.6-35B-A3B base model with QLoRA fine-tuning capability
- [x] **INFRA-01**: 548 seed TRIZ samples across 6 subsets
- [x] **INFRA-02**: Notebook-driven workflow (01–05 + 02b skeleton)
- [x] **INFRA-03**: DGX Spark hardware environment (128GB unified memory)

---

## Milestone v1.0 Active Requirements

### Data Generation (DATA)
- [x] **DATA-01**: Generate ~6K synthetic training samples from 548 seed samples using Moonshot API with true semantic variation (not template paraphrasing)
- [x] **DATA-02**: Implement quality gates for synthetic data: perplexity filtering, diversity scoring, and 20-30% real/human-curated data ratio maintenance
- [x] **DATA-03**: Add token length profiling in Notebook 02 to catch silent truncation before training
- [ ] **DATA-04**: Maintain 85/10/5 train/validation/test split across combined real + synthetic data
- [ ] **DATA-05**: Preserve ChatML format conversion via `tokenizer.apply_chat_template()` for all generated samples

### Baseline Benchmarking (BENCH)
- [x] **BENCH-01**: Execute baseline benchmark (Notebook 03) before any training run
- [x] **BENCH-02**: Load model in FP16 (not 4-bit) for baseline evaluation to avoid quantization skew
- [x] **BENCH-03**: Run Layer 2 TRIZ custom benchmarks (principle accuracy, contradiction resolution, case quality, ARIZ completeness)
- [x] **BENCH-04**: Run Layer 3 performance benchmarks (throughput, P50 latency, peak memory)
- [x] **BENCH-05**: Persist baseline results to pipeline state registry for automatic post-training comparison
- [x] **BENCH-06**: (Optional) Spot-check one Layer 1 task (e.g., MMLU-Pro subset) if time permits

### Training Execution (TRAIN)
- [x] **TRAIN-01**: Execute QLoRA fine-tuning end-to-end without manual intervention (Notebook 04)
- [x] **TRAIN-02**: Use explicit 12-module `target_modules` list (NOT `"all-linear"`) covering Gated Attention, Gated DeltaNet, and MoE MLP layers
- [x] **TRAIN-03**: Set `lora_dropout=0.0` for MoE architecture compatibility
- [x] **TRAIN-04**: Use `SFTTrainer` with `formatting_func` + `packing=True`; never pass `data_collator`
- [x] **TRAIN-05**: Save checkpoints every 200 steps with `save_total_limit=3`
- [x] **TRAIN-06**: Execute 2 epochs with learning rate 2e-4, cosine scheduler, 5% warmup
- [x] **TRAIN-07**: Use memory-efficient config: 4-bit NF4, gradient checkpointing, paged AdamW 8-bit
- [x] **TRAIN-08**: Save comprehensive adapter metadata alongside LoRA weights
- [x] **TRAIN-09**: Verify immediate load-and-forward-pass after checkpoint save
- [x] **TRAIN-10**: Support checkpoint resume with verification (data iterator state, LR scheduler continuity)

### Evaluation (EVAL)
- [ ] **EVAL-01**: Execute post-training evaluation (Notebook 05) with automatic baseline loading from pipeline state
- [ ] **EVAL-02**: Generate before/after comparison report (Layer 2 TRIZ + Layer 3 performance)
- [ ] **EVAL-03**: Use unified `format_messages()` utility replacing hardcoded ChatML in all paths
- [ ] **EVAL-04**: Load adapter via `AutoPeftModelForCausalLM` for evaluation
- [ ] **EVAL-05**: Compute BLEU/ROUGE for TRIZ case quality scoring

### Infrastructure & Hardening (INFRA)
- [x] **INFRA-04**: Update `requirements.txt` with pinned compatible versions (`transformers` 4.45+, `trl` 0.9–0.11, `lm-eval` 0.4.10+, `bitsandbytes` 0.43.x, add `rouge-score` and `openai`)
- [x] **INFRA-05**: Implement `utils/synthetic_pipeline.py` — Moonshot API client with batching, rate limiting, output validation
- [x] **INFRA-06**: Implement `utils/pipeline_state.py` — JSON artifact registry for cross-notebook state tracking
- [ ] **INFRA-07**: Create `02b_synthetic_generation.ipynb` orchestrating ~6K sample generation
- [x] **INFRA-08**: Fix README inaccuracy: replace `"all-linear"` recommendation with explicit module list
- [x] **INFRA-09**: Add notebook pre-flight checks (paths exist, artifacts present, version compatibility)

---

## Future Requirements (deferred to v1.1+)

- **FUTURE-01**: Layer 1 full suite post-training (MMLU-Pro, GPQA, HumanEval, MATH, BBH) — too time-consuming for v1.0
- **FUTURE-02**: Expanded TRIZ test set (50–100 questions vs current 5) — sufficient for first run
- **FUTURE-03**: Production synthetic data pipeline with GPT-4o and expert review — recommended by audit but not required
- **FUTURE-04**: Merged model export — out of scope per PROJECT.md
- **FUTURE-05**: Real-time inference serving — out of scope per PROJECT.md
- **FUTURE-06**: bf16 vs fp16 hyperparameter comparison on GB10 — handle as tuning task

---

## Out of Scope

- Full fine-tuning (resource constraints; QLoRA only)
- Multi-GPU training (single DGX Spark node)
- Deployment/inference serving (training focus only)
- vLLM, Unsloth, Flash Attention, DeepSpeed, or RAGAS integration

---

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Complete |
| DATA-02 | Phase 1 | Complete |
| DATA-03 | Phase 1 | Complete |
| DATA-04 | Phase 1 | Pending |
| DATA-05 | Phase 1 | Pending |
| INFRA-04 | Phase 1 | Complete |
| INFRA-05 | Phase 1 | Complete |
| INFRA-06 | Phase 1 | Complete |
| INFRA-07 | Phase 1 | Pending |
| INFRA-08 | Phase 1 | Complete |
| INFRA-09 | Phase 1 | Complete |
| TRAIN-02 | Phase 1 | Complete |
| TRAIN-03 | Phase 1 | Complete |
| BENCH-01 | Phase 2 | Complete |
| BENCH-02 | Phase 2 | Complete |
| BENCH-03 | Phase 2 | Complete |
| BENCH-04 | Phase 2 | Complete |
| BENCH-05 | Phase 2 | Complete |
| BENCH-06 | Phase 2 | Complete |
| TRAIN-01 | Phase 2 | Complete |
| TRAIN-04 | Phase 2 | Complete |
| TRAIN-05 | Phase 2 | Complete |
| TRAIN-06 | Phase 2 | Complete |
| TRAIN-07 | Phase 2 | Complete |
| TRAIN-08 | Phase 2 | Complete |
| TRAIN-09 | Phase 2 | Complete |
| TRAIN-10 | Phase 2 | Complete |
| EVAL-01 | Phase 3 | Pending |
| EVAL-02 | Phase 3 | Pending |
| EVAL-03 | Phase 3 | Pending |
| EVAL-04 | Phase 3 | Pending |
| EVAL-05 | Phase 3 | Pending |
