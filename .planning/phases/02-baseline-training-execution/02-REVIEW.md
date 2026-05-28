---
phase: 02-baseline-training-execution
reviewed: 2026-05-28T00:00:00Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - ref/mongoose_ai_dgx/utils/training_utils.py
  - ref/mongoose_ai_dgx/utils/__init__.py
  - ref/mongoose_ai_dgx/notebooks/03_model_benchmark.ipynb
  - ref/mongoose_ai_dgx/utils/benchmark_utils.py
  - ref/mongoose_ai_dgx/notebooks/04_qlora_finetune.ipynb
findings:
  critical: 0
  warning: 4
  info: 3
  total: 7
status: issues_found
---

# Phase 02: Code Review Report

**Reviewed:** 2026-05-28
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

## Summary

Reviewed the core training and benchmarking utilities for the Meerkat AI QLoRA fine-tuning pipeline. The code correctly addresses the three critical audit findings (CR-001 through CR-003) by using `formatting_func` with SFTTrainer, explicit `target_modules` lists, and `tokenizer.apply_chat_template()`. However, four warnings and three info-level issues were identified, primarily around training configuration safety, checkpoint callback robustness, benchmark evaluation correctness, and notebook variable scoping.

## Warnings

### WR-01: `load_best_model_at_end=True` without explicit `evaluation_strategy`

**File:** `ref/mongoose_ai_dgx/utils/training_utils.py:369`
**Issue:** `setup_training_arguments()` sets `load_best_model_at_end=True` and `metric_for_best_model="eval_loss"`, but `evaluation_strategy` is only set to `"steps"` (line 374). While this works when `eval_dataset` is provided, if a caller passes `eval_dataset=None` (or the notebook cell is modified to skip eval), `load_best_model_at_end=True` will cause `Trainer.train()` to raise a `ValueError` at the end of training because no evaluation was performed to determine the "best" model. This is a latent crash risk for notebook users who may comment out eval dataset loading.

**Fix:** Add an explicit guard or default:
```python
# In setup_training_arguments, accept an eval_dataset param or add a safety default:
if kwargs.get("eval_dataset") is None and kwargs.get("evaluation_strategy") is None:
    kwargs["load_best_model_at_end"] = False
```
Or, more minimally, document in the docstring that `eval_dataset` is required when `load_best_model_at_end=True`.

---

### WR-02: `CheckpointValidationCallback.on_save` silently skips forward-pass validation when `model` is missing from kwargs

**File:** `ref/mongoose_ai_dgx/utils/training_utils.py:76-91`
**Issue:** The callback checks `model = kwargs.get('model')` on line 76. If the key is absent (e.g., due to a future TRL version change or unexpected callback signature), the forward-pass validation block is silently skipped. The result dict is still appended with no `"status"` key set, leaving the checkpoint marked as unchecked. This is a silent failure mode that could mask corrupted checkpoints.

**Fix:** Add an explicit else-branch to flag the missing model as a failure:
```python
if model:
    try:
        ...
    except Exception as e:
        ...
else:
    result["status"] = "FAILED"
    result["reason"] = "model_not_in_kwargs"
    print(f"[CHECKPOINT] FAILED: model not provided in callback kwargs")
```

---

### WR-03: `TRIZBenchmark.evaluate_principle_accuracy` divides by wrong denominator

**File:** `ref/mongoose_ai_dgx/utils/benchmark_utils.py:168-193`
**Issue:** The method computes `accuracy = correct / total` where `total = len(self.test_questions)` (line 173). However, only questions with `"type" == "multiple_choice"` are actually evaluated in the loop (line 176). If the test question set is expanded to include non-multiple-choice questions, the accuracy denominator will include unevaluated questions, artificially deflating the score. Currently the hardcoded set has 2 multiple_choice out of 5 total, so accuracy is `correct / 5` rather than `correct / 2`.

**Fix:** Count only the evaluated subset:
```python
evaluated_questions = [q for q in self.test_questions if q["type"] == "multiple_choice"]
total = len(evaluated_questions)
for q in evaluated_questions:
    ...
accuracy = correct / total if total > 0 else 0
```

---

### WR-04: `padding_side="right"` comment contradicts the value

**File:** `ref/mongoose_ai_dgx/utils/training_utils.py:122-123`
**Issue:** The code sets `padding_side="right"` but the inline comment says `# 左填充更适合生成` ("left padding is more suitable for generation"). Left padding is indeed the correct choice for causal LM generation (to avoid padding tokens between the actual prompt and the generated continuation), but the code uses right padding. This mismatch between comment and code is confusing and may lead to incorrect generation behavior during inference if the same tokenizer config is reused.

**Fix:** Either change the code to `padding_side="left"` (if generation is the primary use case) or correct the comment to reflect the actual value. Given that training uses SFTTrainer with `formatting_func`, padding side is less critical during training, but for inference consistency `left` is preferred:
```python
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=trust_remote_code,
    padding_side="left",  # left padding for generation
)
```

## Info

### IN-01: `resume_from_checkpoint` does not handle `trainer.optimizer` being None

**File:** `ref/mongoose_ai_dgx/utils/training_utils.py:633-673`
**Issue:** The function accesses `trainer.optimizer.param_groups[0]['lr']` on lines 648 and 657 without checking if `trainer.optimizer` is initialized. In some TRL versions or if `trainer.train()` has not been called yet, `optimizer` may be `None`, causing an `AttributeError`. This is a defensive coding gap.

**Fix:** Use safe attribute access:
```python
initial_lr = trainer.optimizer.param_groups[0]['lr'] if trainer.optimizer else None
```

---

### IN-02: Notebook 04 cell-15 accesses `trainer.state.log_history[-1]` without checking emptiness

**File:** `ref/mongoose_ai_dgx/notebooks/04_qlora_finetune.ipynb` (cell-15, training metadata collection)
**Issue:** The cell builds `training_metadata` with:
```python
'final_loss': trainer.state.log_history[-1].get('loss', 'N/A') if trainer.state.log_history else 'N/A',
```
While there is a ternary guard, `log_history` entries may not contain the `'loss'` key (e.g., eval-only logs), so `.get('loss', 'N/A')` is correct. However, the same pattern is repeated inline for `best_eval_loss` using `hasattr(trainer.state, 'best_metric')` which is fine. No bug here, but the repeated inline conditionals reduce readability and are prone to copy-paste errors.

**Fix:** Extract a small helper in the notebook cell or in `training_utils.py`:
```python
def _get_last_log_value(trainer, key, default='N/A'):
    if not trainer.state.log_history:
        return default
    # Search backwards for the first entry containing the key
    for entry in reversed(trainer.state.log_history):
        if key in entry:
            return entry[key]
    return default
```

---

### IN-03: `__init__.py` imports `find_all_linear_names` which is not used in any reviewed notebook

**File:** `ref/mongoose_ai_dgx/utils/__init__.py:29-30`
**Issue:** `find_all_linear_names` is exported from `__init__.py` and imported in Notebook 04 cell-8, but the cell only uses it for a diagnostic print. The function itself relies on `bitsandbytes` being installed (`import bitsandbytes as bnb` inside the function body). If bnb is not available, the function will raise `ModuleNotFoundError` at call time rather than import time. Since it is only used diagnostically, this is low risk, but it is a latent runtime dependency issue.

**Fix:** Add a soft-import with graceful fallback:
```python
def find_all_linear_names(model) -> List[str]:
    try:
        import bitsandbytes as bnb
        linear_classes = (nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    except ImportError:
        linear_classes = (nn.Linear,)
    ...
```

---

_Reviewed: 2026-05-28_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
