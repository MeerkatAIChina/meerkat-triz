# Domain Pitfalls: QLoRA Fine-Tuning Pipeline Execution

**Project:** Meerkat AI (猫鼬AI) — TRIZ domain LLM fine-tuning
**Domain:** QLoRA fine-tuning of Qwen3.6-35B-A3B on DGX Spark (128GB unified memory)
**Researched:** 2026-05-26
**Overall confidence:** HIGH (based on verified audit findings + community reports + official docs)

---

## Critical Pitfalls

Mistakes that cause training failure, silent data corruption, or require full re-runs.

### Pitfall 1: SFTTrainer + DataCollator Conflict (CR-001 Regression Risk)

**What goes wrong:** Re-introducing a `data_collator` into `create_trainer()` causes SFTTrainer's internal label-masking logic to conflict with the collator. The model may learn to predict user tokens (questions) instead of only assistant tokens (answers), or training may crash with shape mismatches.

**Why it happens:** SFTTrainer from `trl` has its own internal tokenization and label-masking pipeline. Passing `data_collator` bypasses or conflicts with this pipeline. The audit (CR-001) already fixed this by using `formatting_func` instead, but any future refactor that adds a collator back will reintroduce the bug.

**Consequences:**
- Silent training corruption: model learns wrong token distribution
- Wasted 8-15 hour training run
- Eval metrics look "okay" but model produces garbage in production

**Prevention:**
- Never pass `data_collator` to `SFTTrainer`
- Always use `formatting_func` + `packing=True` pattern
- Code review checklist: search for `data_collator` in trainer creation code

**Detection:**
- Inspect a batch of training data: verify only assistant tokens have `labels != -100`
- Check `trainer.train_dataset[0]` after trainer initialization
- Compare training loss curve: healthy SFT loss starts high (~2-3) and drops to ~0.5-1.0

**Phase to address:** Notebook 04 (QLoRA fine-tuning) — already fixed, guard against regression

---

### Pitfall 2: target_modules="all-linear" on Hybrid Architecture (CR-002 Regression Risk)

**What goes wrong:** Using `target_modules="all-linear"` with Qwen3.6's hybrid architecture (Gated DeltaNet + Gated Attention + MoE) causes PEFT to incorrectly detect modules, potentially including `lm_head` or missing Gated DeltaNet-specific layers like `in_proj_z`, `in_proj_b`, `in_proj_a`.

**Why it happens:** PEFT's `all-linear` auto-detection relies on `isinstance(module, (nn.Linear, bnb.nn.Linear4bit))` checks. Qwen3.6's Gated DeltaNet layers use custom module types that may not match these checks. The audit (CR-002) fixed this with an explicit manual list.

**Consequences:**
- LoRA adapters miss critical layers → poor training convergence
- `lm_head` incorrectly included → unstable training, NaN loss
- Wasted training run with no meaningful parameter updates

**Prevention:**
- Always use the explicit manual module list:
  ```python
  ["q_proj", "k_proj", "v_proj", "o_proj",
   "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
   "gate_proj", "up_proj", "down_proj"]
  ```
- Run `find_all_linear_names(model)` as a validation step and diff against expected list

**Detection:**
- `model.print_trainable_parameters()` shows unexpectedly low trainable param count
- Compare detected modules vs. expected modules in Notebook 04 cell 4.3

**Phase to address:** Notebook 04 (QLoRA fine-tuning) — already fixed, guard against regression

---

### Pitfall 3: Notebook-Driven Training Without Robust Checkpoint Resume

**What goes wrong:** A 15-hour training run is interrupted (kernel crash, power loss, SSH timeout). Resuming from checkpoint skips data or causes loss explosion because SFTTrainer/Transformers checkpoint resume does not preserve data iterator state or RNG state properly.

**Why it happens:**
- Jupyter kernels are ephemeral; no process supervisor
- `trainer.train(resume_from_checkpoint=...)` restores model/optimizer state but not data loader position
- PyTorch Lightning issue #5325 pattern: resuming mid-epoch skips remaining data for that epoch
- Learning rate scheduler restarts from step 0 instead of continuing

**Consequences:**
- Hours of GPU time lost
- Partial epoch data never seen by model
- Loss curve discontinuity makes comparison impossible
- May need full re-run from scratch

**Prevention:**
- Set `save_steps` to a value that gives reasonable granularity (e.g., every 100 steps ≈ every 30-60 min)
- Before resuming, manually verify checkpoint contains: `optimizer_state_dict`, `lr_scheduler_state_dict`, `rng_state`, `epoch`, `global_step`
- Use `trainer.train(resume_from_checkpoint=latest_checkpoint)` and immediately check `trainer.state.global_step`
- Consider wrapping training in a Python script with `nohup` or `tmux` instead of pure notebook execution for long runs
- Save a metadata file alongside checkpoints: `{"epoch": N, "global_step": M, "timestamp": "..."}`

**Detection:**
- After resume, `trainer.state.global_step` should continue from saved value, not restart
- Training loss should continue smoothly, not spike
- Compare `len(trainer.train_dataloader)` samples processed vs. expected

**Phase to address:** Notebook 04 (QLoRA fine-tuning) — add resume verification cell

---

### Pitfall 4: UMA Double-Allocation OOM on Model Load

**What goes wrong:** Loading Qwen3.6-35B-A3B (~67GB in bf16, ~18-20GB in 4-bit) causes OOM at ~66% progress even though 128GB unified memory should be sufficient.

**Why it happens:** On DGX Spark's unified memory architecture, `from_pretrained()` with `device_map="auto"` causes double allocation:
1. **Page cache**: `mmap`'d model shards (~67GB for bf16, ~20GB for 4-bit)
2. **CUDA tensors**: Materialized model weights in GPU memory (~same size)
Total: ~134GB needed vs. ~119GB actually available (OS overhead), causing OOM.

**Consequences:**
- Cannot load model at all
- Training cannot begin
- Wasted setup time

**Prevention:**
- Use the monkey-patched `_EagerSafeOpen` pattern from the NVIDIA forums: load tensors direct-to-CUDA, eagerly close shards, evict page cache via `posix_fadvise(POSIX_FADV_DONTNEED)`
- Alternatively, preload model with `low_cpu_mem_usage=True` and `torch_dtype=torch.float16`
- For QLoRA: ensure `BitsAndBytesConfig` is passed correctly so model loads directly in 4-bit
- Monitor with `watch -n 1 free -h` during load to detect double-allocation

**Detection:**
- `torch.cuda.memory_allocated()` shows ~20GB but system `free` shows ~40GB+ consumed
- OOM at model load stage, not during training

**Phase to address:** Notebook 01 (download/setup) and 04 (fine-tuning) — verify load pattern

---

### Pitfall 5: lora_dropout > 0 with MoE Architecture

**What goes wrong:** Setting `lora_dropout=0.05` (current config) causes training instability or crashes with MoE models because dropout layers interfere with expert routing parameter wrappers.

**Why it happens:** MoE architectures use custom parameter wrappers around expert layers. Dropout modules inserted by PEFT can conflict with these wrappers, causing gradient computation errors or silent incorrect gradients.

**Consequences:**
- Training may crash with cryptic errors
- Or worse: silently produces incorrect gradients → model does not learn
- Community reports (NVIDIA forums, arXiv 2025 clinical paper): `lora_dropout=0` is "strictly enforced" for MoE compatibility

**Prevention:**
- **Set `lora_dropout=0.0`** for all MoE models including Qwen3.6-35B-A3B
- If regularization is needed, increase `weight_decay` in optimizer instead
- Document this as a hard constraint in config comments

**Detection:**
- Training loss becomes NaN within first 50 steps
- Or loss decreases but eval metrics do not improve (silent gradient corruption)
- Check PEFT issue tracker for MoE + dropout reports

**Phase to address:** Notebook 04 (QLoRA fine-tuning) — config.py change required

---

### Pitfall 6: Synthetic Data Pipeline Produces Low-Quality / Collapsed Distribution

**What goes wrong:** The current `vary_sample()` function uses simple prefix insertion (`从TRIZ角度分析，` etc.) and fixed suffix appending. Scaling this to ~6K samples creates a dataset with:
- High lexical redundancy (same prefixes repeated)
- No semantic diversity (answers unchanged)
- Risk of model collapse: model learns paraphrase patterns instead of TRIZ reasoning

**Why it happens:**
- Template-based variation does not alter the underlying knowledge distribution
- 548 seeds × ~11 variations = ~6K samples, but effective unique information may be <1K
- Meta's Llama 3 405B finding: "training on its own generated data is not helpful (and can even degrade performance)"

**Consequences:**
- Model overfits to paraphrase templates
- Poor generalization on novel TRIZ problems
- Wasted training run: model learns to prepend prefixes, not reason about contradictions
- Eval metrics look good on held-out template variations but fail on real cases

**Prevention:**
- **Do NOT rely solely on template-based synthesis** for production training
- Use Moonshot API (GPT-4o-class) for true semantic variation:
  - Rephrase questions with different problem framings
  - Generate alternative correct answers
  - Create novel contradiction scenarios
- Implement quality gates:
  - Perplexity filtering: discard samples with high model perplexity
  - Diversity scoring: ensure n-gram diversity across generated set
  - Expert review: TRIZ consultant validates 10-20% of samples
- Maintain seed data ratio: at least 20-30% real/human-curated data in final mix

**Detection:**
- Training loss drops too quickly (< 50 steps to < 0.3) → overfitting signal
- BLEU/ROUGE between train and eval is suspiciously high
- Manual inspection: many samples share identical answer text

**Phase to address:** Notebook 02b (synthetic data generation) — requires pipeline redesign

---

### Pitfall 7: Benchmarking with 4-bit Quantized Model (MA-003)

**What goes wrong:** Notebook 03 loads the model in 4-bit quantization for baseline benchmarking. This introduces 1-3% accuracy degradation that is NOT caused by fine-tuning, making pre/post comparison misleading.

**Why it happens:**
- NF4 quantization approximates weights with 4-bit values
- lm-eval-harness tasks (MMLU-Pro, GPQA, MATH) are sensitive to precision
- The audit (MA-003) identified this but it remains in the notebook code

**Consequences:**
- Baseline scores artificially low
- Post-training scores may look like "improvement" when they're just "less quantization loss"
- Cannot attribute changes to fine-tuning vs. quantization artifacts

**Prevention:**
- **Load model in FP16 (not 4-bit) for benchmarking**
- DGX Spark 128GB can hold 35B FP16 (~70GB) + activations
- For QLoRA training, use 4-bit; for evaluation, use FP16
- Document this distinction clearly in notebook comments

**Detection:**
- Compare baseline MMLU score vs. published Qwen3.6-35B-A3B leaderboard score
- If baseline is >2% below published, quantization is likely the cause

**Phase to address:** Notebook 03 (model benchmark) — change model load config

---

### Pitfall 8: Chat Template Mismatch Between Training and Inference

**What goes wrong:** Training uses `tokenizer.apply_chat_template()` with `add_generation_prompt=False`, but inference (Notebook 05, benchmark_utils.py `_build_prompt()`) uses hardcoded ChatML strings. This creates a format mismatch.

**Why it happens:**
- `data_utils.py` correctly uses `apply_chat_template()` for training
- `benchmark_utils.py` `_build_prompt()` hardcodes ChatML format
- Qwen3.6 may have special tokens, thinking mode markers, or whitespace rules that differ from hardcoded format
- The audit (CR-003) fixed training-side but inference-side remains inconsistent

**Consequences:**
- Model trained on one format, evaluated on another
- Eval metrics do not reflect true capability
- Production inference (if using `apply_chat_template()`) may behave differently from eval

**Prevention:**
- **Use `tokenizer.apply_chat_template()` in ALL paths**: training, evaluation, and inference
- Create a single `format_messages()` utility function used everywhere
- Verify by tokenizing the same conversation through both paths and comparing token IDs

**Detection:**
- Token-level diff between training format and inference format
- Model performs better on eval than on interactive chat (format mismatch symptom)

**Phase to address:** Notebook 05 + benchmark_utils.py — unify formatting

---

### Pitfall 9: SFTTrainer Packing with Long Sequences Causes Silent Truncation

**What goes wrong:** With `packing=True` and `max_seq_length=4096`, individual samples that exceed 4096 tokens after `formatting_func` are silently truncated. For TRIZ data with long expert answers, this cuts off the end of responses, teaching the model to produce incomplete outputs.

**Why it happens:**
- SFTTrainer concatenates examples and truncates to `max_seq_length`
- No warning is emitted when truncation occurs
- TRIZ expert answers can be 1000+ tokens; with system prompt + question, easily exceed 4096

**Consequences:**
- Model learns to generate truncated/incomplete responses
- Eval shows poor "ARIZ completeness" because model was trained on incomplete steps
- Silent quality degradation with no error messages

**Prevention:**
- **Profile dataset token lengths BEFORE training**
- Add a data validation step that reports: % of samples > max_seq_length, max token count
- If >10% exceed limit, either:
  - Increase `max_seq_length` (if memory allows)
  - Split long answers into multiple training samples
  - Filter out excessively long samples
- Consider `packing_strategy="longest"` (TRL v1.0+) which splits overflow sequences instead of truncating

**Detection:**
- After tokenization, check `input_ids.shape[1]` for all samples
- Look for abrupt endings in generated text during eval
- Compare average token length in dataset vs. `max_seq_length`

**Phase to address:** Notebook 02 (data preparation) — add token length profiling

---

### Pitfall 10: DGX Spark Memory Bandwidth Bottleneck Causes Unexpectedly Slow Training

**What goes wrong:** Training throughput is far below expectations (~5-10 tokens/s vs. hoped-for 50+). A 15-hour estimate becomes 40+ hours. User assumes configuration error and starts tweaking hyperparameters unnecessarily.

**Why it happens:**
- DGX Spark has 273 GB/s memory bandwidth — much lower than RTX 5090 (1792 GB/s) or even Apple M4 Max (546 GB/s)
- QLoRA requires constant dequantization (4-bit → 16-bit) during forward/backward passes
- This is bandwidth-bound, not compute-bound
- Unified memory architecture means no HBM speedup

**Consequences:**
- Unrealistic timeline expectations
- User may abort training thinking it's "stuck"
- Incorrect conclusions about hyperparameter choices

**Prevention:**
- **Set realistic expectations: 8-15 hours per epoch is normal for this hardware**
- Do NOT compare throughput to cloud A100/H100 benchmarks
- Monitor throughput (steps/sec) in first 30 minutes and extrapolate
- If throughput is < 0.1 steps/sec, investigate; if 0.2-0.5 steps/sec, that's expected

**Detection:**
- `logging_steps=10` shows time per step in TensorBoard logs
- Compare to community reports: Llama 3.1 70B QLoRA on DGX Spark = ~5K tokens/sec (NVIDIA official)
- For 35B model, expect roughly 2-3× faster than 70B

**Phase to address:** Notebook 04 (QLoRA fine-tuning) — add throughput monitoring and expectations

---

### Pitfall 11: Saving Only Adapter Without Base Model Context

**What goes wrong:** The `save_adapter_only()` function saves only LoRA weights (~100-200MB) but the saved directory lacks clear documentation of which base model version, quantization config, and target_modules were used. Months later, the adapter cannot be reproduced or loaded correctly.

**Why it happens:**
- Adapter files (`adapter_config.json`, `adapter_model.safetensors`) don't encode full training context
- Base model revision, transformers version, and PEFT version matter for compatibility
- The `adapter_info.json` only records basic metadata

**Consequences:**
- Cannot reproduce training results
- Adapter may fail to load with newer PEFT/transformers versions
- No audit trail for model lineage

**Prevention:**
- Save comprehensive metadata alongside adapter:
  ```json
  {
    "base_model": "Qwen/Qwen3.6-35B-A3B",
    "base_model_revision": "...",
    "transformers_version": "...",
    "peft_version": "...",
    "trl_version": "...",
    "torch_version": "...",
    "quantization_config": {...},
    "lora_config": {...},
    "training_args": {...},
    "dataset_info": {"total_samples": 6000, "seed_count": 548, "synthetic_method": "moonshot-api"},
    "hardware": "DGX Spark 128GB",
    "training_duration_hours": 15.3,
    "final_eval_loss": 0.87
  }
  ```
- Pin dependency versions in `requirements.txt` with exact versions, not ranges

**Detection:**
- Try loading adapter 3 months later with updated packages
- Check if `adapter_config.json` contains sufficient context

**Phase to address:** Notebook 04 (save step) + `save_adapter_only()` function

---

### Pitfall 12: Inference Loading Fails Due to Missing 4-bit Metadata

**What goes wrong:** After training, loading the saved adapter with `AutoPeftModelForCausalLM.from_pretrained()` fails with `AttributeError: 'Parameter' object has no attribute 'compress_statistics'` or size mismatch errors.

**Why it happens:**
- QLoRA saves adapter weights + a reference to base model
- If base model files are moved, or tokenizer vocab size changed during training, loading fails
- `bnb` 4-bit metadata can be lost if model is saved incorrectly
- Qwen2.5 issue #1218: tokenizer vocab size (151643) vs. model vocab size (151936) mismatch

**Consequences:**
- Cannot evaluate trained model
- Cannot deploy to production
- May need to re-run full training

**Prevention:**
- Always save tokenizer alongside adapter: `tokenizer.save_pretrained(output_dir)`
- Use `AutoPeftModelForCausalLM.from_pretrained(adapter_path)` not manual `PeftModel.from_pretrained(base_model, adapter_path)`
- Verify vocab size consistency before training: `assert len(tokenizer) == model.config.vocab_size`
- Do NOT resize token embeddings during QLoRA training
- Keep base model cached locally and verify SHA checksums

**Detection:**
- Test loading immediately after saving in Notebook 04
- Run a forward pass to verify model produces coherent output

**Phase to address:** Notebook 04 (save) and Notebook 05 (load) — add load verification

---

## Moderate Pitfalls

### Pitfall 13: Gradient Accumulation Without LR Scaling Awareness

**What goes wrong:** With `per_device_train_batch_size=1` and `gradient_accumulation_steps=8`, effective batch size is 8. If user later increases to `gradient_accumulation_steps=16` (effective batch 16) without adjusting learning rate, training may destabilize.

**Why it happens:**
- QLoRA community convention: LR is typically fixed at 2e-4 regardless of effective batch size
- But some practitioners apply linear LR scaling: `new_lr = base_lr × (new_batch / old_batch)`
- Inconsistent practice leads to confusion

**Consequences:**
- NaN loss if LR too high for larger effective batch
- Slower convergence if LR too low

**Prevention:**
- Document the project's LR convention: "LR fixed at 2e-4 for all effective batch sizes 8-32"
- If changing batch size, run a 100-step warmup test and monitor for NaN
- Use `warmup_ratio=0.05` (not 0.03) to give scheduler more time to stabilize

**Phase to address:** Notebook 04 (training args) — document LR policy

---

### Pitfall 14: pad_token Configuration Inconsistency

**What goes wrong:** `training_utils.py` sets `pad_token = eos_token` but some Qwen models have separate pad tokens. During packing, incorrect padding can cause attention to attend to EOS positions.

**Why it happens:**
- Qwen3.6 may define `pad_token` differently from `eos_token`
- Overwriting `pad_token` without checking if one already exists
- SFTTrainer's packing behavior depends on correct pad token

**Consequences:**
- Minor training instability
- Slightly degraded convergence

**Prevention:**
- Check if tokenizer already has pad_token before overwriting:
  ```python
  if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token
  ```
- Verify `tokenizer.pad_token_id != tokenizer.eos_token_id` is handled correctly

**Phase to address:** Notebook 01/04 — token setup

---

### Pitfall 15: Hardcoded Baseline Score in Post-Training Report

**What goes wrong:** Notebook 05 cell 5.5 hardcodes `before_score = 0.35` for baseline TRIZ score. If actual baseline differs, the "improvement" calculation is meaningless.

**Why it happens:**
- Notebook 05 is designed to run standalone, not necessarily after Notebook 03
- User may skip baseline benchmark or use different eval questions
- Hardcoded value is a placeholder from development

**Consequences:**
- Misleading improvement metrics in final report
- Stakeholders make decisions based on fabricated comparisons

**Prevention:**
- Load actual baseline results from `results/` directory
- If baseline missing, display "N/A" instead of hardcoded value
- Store baseline in a JSON file during Notebook 03 execution

**Phase to address:** Notebook 05 — load dynamic baseline

---

## Minor Pitfalls

### Pitfall 16: Missing `trust_remote_code=True` in Some Paths

**What goes wrong:** Qwen3.6 requires `trust_remote_code=True` for both model and tokenizer loading. If omitted in evaluation or merge scripts, loading fails.

**Prevention:**
- Create a wrapper function that always includes `trust_remote_code=True`
- Search codebase for all `from_pretrained` calls and verify flag is present

**Phase to address:** All notebooks — audit `from_pretrained` calls

---

### Pitfall 17: `bf16=False` May Miss Speedups on Blackwell

**What goes wrong:** Config sets `bf16=False, fp16=True`. DGX Spark's GB10 (Blackwell) has excellent bf16 support. Using fp16 instead of bf16 may be slightly slower and less numerically stable.

**Prevention:**
- Test `bf16=True` on a short run and compare throughput
- If stable, switch to bf16 for better numerical range

**Phase to address:** Notebook 04 — hyperparameter tuning

---

### Pitfall 18: `group_by_length=True` with `length_column_name="length"` Mismatch

**What goes wrong:** `TrainingArguments` sets `group_by_length=True` with `length_column_name="length"`, but the dataset's "length" field stores character count, not token count. Grouping by character count does not optimize memory usage as intended.

**Prevention:**
- Add a `token_length` field to the dataset during tokenization
- Use `length_column_name="token_length"` instead
- Or remove `group_by_length` if memory is not constrained

**Phase to address:** Notebook 02/04 — dataset field alignment

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| **Synthetic data gen (02b)** | Template-only variation, no semantic diversity | Use Moonshot API for true paraphrase; add quality gates |
| **Data preparation (02)** | Token length > max_seq_length, silent truncation | Profile token lengths; filter or split long samples |
| **Baseline benchmark (03)** | 4-bit quantization skews baseline scores | Load FP16 for benchmarking; save baseline to JSON |
| **QLoRA config (04)** | `lora_dropout=0.05` incompatible with MoE | Set `lora_dropout=0.0` |
| **Training run (04)** | Notebook kernel death, no resume | Use tmux/nohup; verify checkpoint completeness |
| **Model save (04)** | Missing metadata, future load failure | Save comprehensive metadata; test load immediately |
| **Post-training eval (05)** | Hardcoded baseline, format mismatch | Load dynamic baseline; unify chat template usage |
| **Inference (05)** | `AutoPeftModelForCausalLM` loading errors | Verify vocab size; save tokenizer; test forward pass |

---

## Integration Pitfalls (Adding Execution to Existing System)

When adding the "execute training" capability to the existing notebook framework, these integration-specific risks emerge:

### Integration Risk 1: Notebook Cell Execution Order Violations

Jupyter notebooks allow out-of-order execution. A user might:
- Run Notebook 04 cell 4.5 (create trainer) without running cell 4.2 (load model)
- Re-run cell 4.3 (QLoRA config) after training has already started, creating a second LoRA adapter on top of the first

**Mitigation:** Add state checks at the top of each cell:
```python
assert 'model' in globals(), "Run cell 4.2 first to load model"
assert not hasattr(model, 'peft_config'), "Model already has LoRA adapter attached"
```

### Integration Risk 2: Cross-Notebook State Pollution

Notebook 03 loads model in 4-bit. If user does NOT run the cleanup cell (3.6) before opening Notebook 04, the 4-bit model remains in memory. Notebook 04 then tries to load another model, causing OOM.

**Mitigation:** Add a "clean slate" cell at the top of every notebook:
```python
import gc, torch
for obj in ['model', 'tokenizer', 'trainer']:
    if obj in globals(): del globals()[obj]
gc.collect()
torch.cuda.empty_cache()
```

### Integration Risk 3: requirements.txt Version Drift

The project has `requirements.txt` but notebooks may be run weeks apart. Package updates (especially `transformers`, `trl`, `peft`, `bitsandbytes`) can break compatibility.

**Mitigation:**
- Pin exact versions: `transformers==4.45.0` not `transformers>=4.40.0`
- Add a version check cell at the top of Notebook 01:
  ```python
  import transformers, trl, peft, bitsandbytes
  assert transformers.__version__ == '4.45.0', f"Expected transformers 4.45.0, got {transformers.__version__}"
  ```

### Integration Risk 4: Path Assumptions Across Environments

Notebooks hardcode `/home/meerkat/mongoose_ai`. If the repo is cloned to a different path, all `sys.path.append()` calls fail.

**Mitigation:** Use relative path detection:
```python
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
```

---

## Sources

- [NVIDIA Developer Forums: Qwen3.5-35B-A3B bf16 LoRA on DGX Spark](https://forums.developer.nvidia.com/t/bf16-lora-fine-tuning-of-qwen3-5-35b-a3b-on-dgx-spark-no-quantization-required/363268) — HIGH confidence: UMA double-allocation, `lora_dropout=0`, `adamw_8bit` CUDA 13.2 issue
- [GitHub: Fine-Tuning Llama 3.1 70B on DGX Spark](https://github.com/sanjbasu/Fine-Tuning-Llama-3.1-70B-on-DGX-Spark/blob/main/README.md) — HIGH confidence: memory requirements, gradient checkpointing
- [Red Hat: 500K+ Evaluations on Quantized LLMs](https://developers.redhat.com/articles/2024/10/17/we-ran-over-half-million-evaluations-quantized-llms) — HIGH confidence: 4-bit evaluation accuracy gaps
- [arXiv: Accuracy is Not All You Need](https://arxiv.org/pdf/2407.09141) — HIGH confidence: benchmark accuracy vs. real quality gap
- [Stack Overflow: SFTTrainer truncation clarification](https://stackoverflow.com/questions/78773889/trl-sfttrainer-clarification-on-truncation) — MEDIUM confidence: packing truncation behavior
- [Hugging Face TRL SFTConfig source](https://github.com/huggingface/trl/blob/main/trl/trainer/sft_config.py) — HIGH confidence: packing_strategy parameter
- [PyTorch Lightning Issue #5325](https://github.com/Lightning-AI/pytorch-lightning/issues/5325) — HIGH confidence: checkpoint resume data loss
- [PEFT Issue #1918](https://github.com/huggingface/peft/issues/1918) — HIGH confidence: load_adapter device inference bug
- [PEFT Issue #1083](https://github.com/huggingface/peft/issues/1083) — HIGH confidence: low VRAM merge crash
- [Qwen2.5 Issue #1218](https://github.com/QwenLM/Qwen2.5/issues/1218) — HIGH confidence: vocab size mismatch
- [LlamaFactory Issue #7785](https://github.com/hiyouga/LlamaFactory/issues/7785) — MEDIUM confidence: chat template save/load mismatch
- [TRL Issue #5213](https://github.com/huggingface/trl/issues/5213) — MEDIUM confidence: Qwen3.5 tokenization mismatch
- [arXiv: Clinical Decision Support (MoE dropout=0)](https://arxiv.org/html/2601.03266v2) — MEDIUM confidence: MoE lora_dropout requirement
- [Apertus.ai DGX Spark Review](https://apertus.ai/en/blog/nvidia-dgx-spark-review-vs-ai-box/) — MEDIUM confidence: memory bandwidth bottleneck
- [iFactory DGX Spark vs RTX PRO 6000](https://ifactoryapp.com/sap-integration/on-prem-ai/dgx-spark-vs-rtx-pro-6000-blackwell) — MEDIUM confidence: throughput comparison
- [Audit Report: 猫鼬AI训练方案](ref/审计报告_猫鼬AI训练方案.md) — HIGH confidence: CR-001/002/003, MA-001/002/003 findings
- [Meta Llama 3 Technical Report](https://arxiv.org/pdf/2407.21783v1) — HIGH confidence: synthetic data degradation findings
