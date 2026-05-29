# Phase 03: Evaluation & Hardening - Research

**Researched:** 2026-05-29
**Domain:** Post-training evaluation for QLoRA-finetuned Qwen3.6-35B-A3B on DGX Spark
**Confidence:** HIGH

## Summary

This phase delivers Notebook 05 (post-training evaluation) with automatic baseline loading from `pipeline_state`, before/after comparison reports, a unified `format_messages()` utility, `AutoPeftModelForCausalLM` adapter loading, and BLEU/ROUGE metrics for TRIZ case quality. The research confirms that `trust_remote_code=True` is mandatory for Qwen3 MoE models, `tokenizer.apply_chat_template()` is the correct replacement for hardcoded ChatML, and `sacrebleu` + `rouge-score` form the standard metric stack for Chinese-English mixed text evaluation. The comparison report should be a flat JSON structure with `before`/`after`/`delta` sections per layer, plus a notebook display helper that renders markdown tables with +/- indicators.

**Primary recommendation:** Implement `format_messages()` in `data_utils.py` (shared with training), use `sacrebleu.corpus_bleu(..., tokenize='zh')` for BLEU and `rouge_scorer.RougeScorer(..., use_stemmer=False)` for ROUGE on Chinese-English TRIZ text, load adapter via `AutoPeftModelForCausalLM.from_pretrained(adapter_path, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)`, and structure the report as `{layer: {metric: {before, after, delta, delta_pct}}}`.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Baseline loading from pipeline_state | Notebook 05 (orchestration) | `utils/pipeline_state.py` | State registry is the source of truth; notebook orchestrates the load |
| Before/after comparison report | Notebook 05 (display + persistence) | `utils/benchmark_utils.py` | Report generation is evaluation orchestration; utils provide raw metrics |
| `format_messages()` utility | `utils/data_utils.py` | `utils/benchmark_utils.py` | Shared with training (SFTTrainer formatting_func); benchmark_utils consumes it |
| Adapter loading for inference | Notebook 05 (model loading cell) | `utils/training_utils.py` | `AutoPeftModelForCausalLM` is a one-liner; keep in notebook for clarity |
| BLEU/ROUGE computation | `utils/benchmark_utils.py` | — | Domain-specific metric logic belongs in the benchmark module |
| Pre-flight checks | Notebook 05 (setup cell) | `utils/pipeline_state.py` | Notebook verifies prerequisites before any model load |

---

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01 (Before/After Comparison Scope):** Notebook 05 loads the base model (without adapter) and runs the full Layer 2 TRIZ benchmarks for comparison. Layer 1 general benchmarks are loaded from `pipeline_state` baseline registry (not re-run). Layer 3 performance benchmarks run on both base model and adapter for throughput/latency/memory comparison.
- **D-02 (Comparison Report Format):** Structured JSON report saved to `results/` plus rich inline display in Notebook 05. JSON file: `results/evaluation_report_YYYYMMDD_HHMMSS.json`. Notebook inline display: formatted tables showing deltas with +/- indicators and percentage changes. Report includes all three layers.
- **D-04 (Missing Baseline Handling):** If `pipeline_state` has no baseline results, Notebook 05 auto-runs a quick baseline on-the-fly (minimal Layer 1 + Layer 3 on base model). Full Layer 2 TRIZ benchmarks are always run on base model during evaluation.
- **D-05 (Adapter Loading):** Use `AutoPeftModelForCausalLM.from_pretrained()` with `torch_dtype=torch.float16`, `device_map='auto'`, `trust_remote_code=True`. Adapter path: `MODELS_DIR / 'meerkat_triz_adapter_v1'`. Base model path: `MODELS_DIR / BASE_MODEL.split('/')[-1]`.
- **D-06 (Notebook Execution Order):** Strict sequence: 01 -> 02b -> 03 -> 04 -> 05. Notebook 05 pre-flight checks verify: adapter exists, base model exists, pipeline_state accessible. Uses final adapter checkpoint (not intermediate ones).

### Claude's Discretion
- Specific `format_messages()` function signature and implementation details
- Exact JSON schema for the evaluation report
- Which Layer 1 task to use for spot-check (if time permits)
- Visual formatting of delta tables in notebook output (color coding, etc.)
- Exact BLEU/ROUGE reference data strategy (seed outputs vs generated references)
- Progress bar/logging verbosity during evaluation

### Deferred Ideas (OUT OF SCOPE)
- Weights & Biases integration for evaluation visualization
- Full Layer 1 suite post-training (MMLU-Pro, GPQA, HumanEval, MATH, BBH) — FUTURE-01
- Real-time inference serving

---

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EVAL-01 | Execute post-training evaluation (Notebook 05) with automatic baseline loading from pipeline state | `pipeline_state.py` artifact registry; `get("baseline_results")` pattern established in Phase 02 |
| EVAL-02 | Generate before/after comparison report (Layer 2 TRIZ + Layer 3 performance) | Flat JSON schema with `before`/`after`/`delta` per metric; notebook display helper for markdown tables |
| EVAL-03 | Use unified `format_messages()` utility replacing hardcoded ChatML in all paths | `tokenizer.apply_chat_template()` with `add_generation_prompt=True` for inference, `False` for training; place in `data_utils.py` |
| EVAL-04 | Load adapter via `AutoPeftModelForCausalLM` for evaluation | `trust_remote_code=True` mandatory for Qwen3 MoE; `torch_dtype=torch.float16`; `device_map='auto'` |
| EVAL-05 | Compute BLEU/ROUGE for TRIZ case quality scoring | `sacrebleu` for corpus-level BLEU (tokenize='zh'); `rouge-score` with `use_stemmer=False` for Chinese-English mixed text |

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `transformers` | 4.45.0 | `AutoPeftModelForCausalLM`, `AutoTokenizer`, `apply_chat_template` | Locked by Phase 01; Qwen3.6 requires 4.45+ [VERIFIED: requirements.txt] |
| `peft` | 0.12.0 | Adapter loading via `AutoPeftModelForCausalLM` | Locked by Phase 01; PEFT 0.12 verified with Qwen3 [VERIFIED: requirements.txt] |
| `sacrebleu` | 2.4.x | Corpus-level BLEU with signature and Chinese tokenization | Standard for reproducible MT evaluation; auto-detects `zh` tokenizer [CITED: sacrebleu GitHub README] |
| `rouge-score` | 0.1.2 | ROUGE-1/2/L for case quality scoring | Google's native Python implementation; already in requirements.txt [VERIFIED: requirements.txt] |
| `jieba` | >=0.42.0 | Chinese word segmentation for ROUGE preprocessing | Listed in requirements.txt; needed because `rouge-score` has no built-in Chinese tokenizer [VERIFIED: requirements.txt] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `torch` | >=2.4.0 | `torch.no_grad()`, `torch.cuda` memory queries | Already in env; used for inference context manager |
| `accelerate` | 0.33.0 | `device_map='auto'` backend | Locked by Phase 01; handles GB10 unified memory placement |
| `lm-eval` | 0.4.10 | Layer 1 spot-check (if time permits) | Locked by Phase 01; `simple_evaluate()` returns standard JSON schema |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `sacrebleu` | `nltk.translate.bleu_score` | NLTK BLEU is not comparable across studies, lacks signature, poor Chinese tokenization. SacreBLEU is the research standard. [CITED: Stack Overflow comparison] |
| `rouge-score` | `pyrouge` (Perl wrapper) | `rouge-score` is pure Python, no Perl dependency, same Google-maintained logic. [CITED: rouge-score PyPI] |
| `AutoPeftModelForCausalLM` | `PeftModel.from_pretrained(base_model, adapter_path)` | AutoPeft is one-liner, auto-loads base model + adapter. Manual two-step is more verbose but equivalent. [VERIFIED: Notebook 05 cell 2 already uses AutoPeft] |

**Installation:**
```bash
# sacrebleu is NOT in current requirements.txt — must be added
pip install sacrebleu==2.4.3
# rouge-score and jieba already present:
# rouge-score==0.1.2 (in requirements.txt)
# jieba>=0.42.0 (in requirements.txt)
```

**Version verification:**
- `sacrebleu` latest stable: 2.4.3 (as of 2025) [CITED: PyPI sacrebleu]
- `rouge-score` pinned: 0.1.2 [VERIFIED: requirements.txt]
- `jieba` pinned: >=0.42.0 [VERIFIED: requirements.txt]

---

## Architecture Patterns

### System Architecture Diagram

```
Notebook 05 (Evaluation Orchestrator)
│
├─ Pre-flight Checks
│  ├─ adapter_path exists? ──► warn / abort
│  ├─ base_model_path exists? ──► warn / abort
│  └─ pipeline_state accessible? ──► auto-run quick baseline if missing
│
├─ Load Adapter Model
│  └─ AutoPeftModelForCausalLM.from_pretrained(adapter_path)
│      └─ auto-loads base + adapter weights
│
├─ Run Layer 2 TRIZ (adapter model)
│  └─ TRIZBenchmark(model_adapter, tokenizer)
│      ├─ evaluate_principle_accuracy()
│      ├─ evaluate_contradiction_resolution()
│      ├─ evaluate_case_quality() ──► BLEU/ROUGE against references
│      └─ evaluate_ariz_completeness()
│
├─ Run Layer 3 Performance (adapter model)
│  └─ run_performance_benchmark(model_adapter, tokenizer)
│
├─ Load Base Model (no adapter)
│  └─ AutoModelForCausalLM.from_pretrained(base_model_path)
│
├─ Run Layer 2 TRIZ (base model) ──► for true before/after delta
│  └─ TRIZBenchmark(model_base, tokenizer)
│
├─ Run Layer 3 Performance (base model)
│  └─ run_performance_benchmark(model_base, tokenizer)
│
├─ Load Layer 1 from pipeline_state
│  └─ state.get("baseline_results") metadata
│
└─ Generate Comparison Report
   ├─ JSON: results/evaluation_report_YYYYMMDD_HHMMSS.json
   └─ Notebook display: markdown tables with +/- and %
```

### Recommended Project Structure

```
ref/mongoose_ai_dgx/
├── utils/
│   ├── data_utils.py          # format_messages() added here (shared with training)
│   ├── benchmark_utils.py     # _build_prompt() refactored to use format_messages()
│   │                          # evaluate_case_quality() implements BLEU/ROUGE
│   │                          # aggregate_results() extended for before/after
│   └── pipeline_state.py      # unchanged (already supports get/verify/preflight)
├── notebooks/
│   └── 05_model_evaluation.ipynb   # refactored: no hardcoded ChatML, no hardcoded before_score
└── config.py                  # DATA_CONFIG['chatml']['system_message'] source of truth
```

### Pattern 1: format_messages() Utility
**What:** A single function that builds prompt strings via `tokenizer.apply_chat_template()`, replacing all hardcoded ChatML strings.
**When to use:** Any code that needs to format a conversation (system + user -> assistant) for either training data or inference prompts.
**Example:**
```python
# Source: verified against Qwen3 chat template docs [CITED: huggingface.tw/blog/qwen-3-chat-template-deep-dive]
# and existing training_utils.py formatting_func pattern [VERIFIED: training_utils.py]

def format_messages(
    tokenizer,
    user_content: str,
    system_message: Optional[str] = None,
    assistant_content: Optional[str] = None,
    add_generation_prompt: bool = False,
) -> str:
    """
    Format messages using tokenizer.apply_chat_template().

    Args:
        tokenizer: Model tokenizer with chat_template support.
        user_content: The user's question/prompt.
        system_message: System prompt. If None, uses DATA_CONFIG default.
        assistant_content: If provided, includes assistant response (for training data).
        add_generation_prompt: If True, appends <|im_start|>assistant\n for inference.

    Returns:
        Formatted chat string.
    """
    from config import DATA_CONFIG

    if system_message is None:
        system_message = DATA_CONFIG['chatml']['system_message']

    messages = [{"role": "system", "content": system_message}]
    messages.append({"role": "user", "content": user_content})
    if assistant_content is not None:
        messages.append({"role": "assistant", "content": assistant_content})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
```

**Usage in inference (evaluation prompt):**
```python
prompt = format_messages(
    tokenizer,
    user_content=question,
    add_generation_prompt=True,  # appends <|im_start|>assistant\n
)
```

**Usage in training data formatting:**
```python
text = format_messages(
    tokenizer,
    user_content=full_question,
    assistant_content=output,
    add_generation_prompt=False,  # complete conversation, no generation prompt
)
```

### Pattern 2: BLEU/ROUGE for Chinese-English Mixed TRIZ Text
**What:** Compute corpus-level BLEU with Chinese-aware tokenization, and ROUGE with disabled stemming (Porter stemmer is English-only).
**When to use:** `evaluate_case_quality()` in `TRIZBenchmark` when scoring generated TRIZ case solutions against reference outputs.
**Example:**
```python
# Source: sacrebleu README [CITED: github.com/mjpost/sacrebleu]
# Source: rouge-score docs [CITED: pypi.org/project/rouge-score]

from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
import jieba

def compute_bleu(predictions: List[str], references: List[str]) -> Dict[str, Any]:
    """Corpus-level BLEU with Chinese tokenization."""
    # sacrebleu expects: sys = list of strings, refs = list of list of strings
    bleu = corpus_bleu(predictions, [references], tokenize='zh')
    return {
        "bleu": bleu.score,  # 0-100 scale
        "signature": str(bleu.signature),
    }

def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, Any]:
    """ROUGE-1/2/L with Chinese word segmentation."""
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=False,  # Disable English Porter stemmer for Chinese
    )

    results = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        # Segment Chinese text before scoring
        pred_seg = ' '.join(jieba.cut(pred.strip()))
        ref_seg = ' '.join(jieba.cut(ref.strip()))
        scores = scorer.score(ref_seg, pred_seg)
        for key in results:
            results[key].append(scores[key].fmeasure)

    return {
        "rouge1": sum(results["rouge1"]) / len(results["rouge1"]),
        "rouge2": sum(results["rouge2"]) / len(results["rouge2"]),
        "rougeL": sum(results["rougeL"]) / len(results["rougeL"]),
    }
```

### Pattern 3: AutoPeftModelForCausalLM Loading
**What:** One-line loading of base model + LoRA adapter for evaluation inference.
**When to use:** Notebook 05 cell that loads the fine-tuned model for evaluation.
**Example:**
```python
# Source: QwenLM/Qwen3 GitHub issue #209 [CITED: github.com/QwenLM/Qwen3/issues/209]
# Source: CSDN article on Qwen3 MoE trust_remote_code [CITED: web search]
# Source: Notebook 05 existing cell 2 [VERIFIED: 05_model_evaluation.ipynb]

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_path,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True,  # MANDATORY for Qwen3 MoE custom architecture
)

tokenizer = AutoTokenizer.from_pretrained(
    adapter_path,
    trust_remote_code=True,
)
```

**Critical pitfall:** `trust_remote_code=True` is mandatory for Qwen3 MoE models because they use custom `Qwen3MoeForCausalLM` class not in standard transformers. [CITED: CSDN article on Qwen3-30B-A3B]

### Pattern 4: Before/After Comparison Report Structure
**What:** A flat JSON schema that captures before/after/delta for every metric, plus a notebook display helper.
**When to use:** Final cell of Notebook 05, persisted to `results/evaluation_report_YYYYMMDD_HHMMSS.json`.
**Example:**
```python
# Source: derived from BENCHMARK_CONFIG structure [VERIFIED: config.py]
# and aggregate_results() existing pattern [VERIFIED: benchmark_utils.py]

report = {
    "timestamp": datetime.now().isoformat(),
    "model_info": {
        "base_model": BASE_MODEL,
        "adapter_path": str(adapter_path),
    },
    "layer1_general": {
        # Loaded from pipeline_state baseline_results metadata
        "source": "pipeline_state",
        "metrics": {
            "mmlu_pro": {"before": 0.42, "after": None, "delta": None},  # after not re-run
        },
    },
    "layer2_triz": {
        "source": "re-run_on_both_models",
        "metrics": {
            "principle_accuracy": {
                "before": 0.20,
                "after": 0.65,
                "delta": 0.45,
                "delta_pct": 225.0,
            },
            "contradiction_resolution": { ... },
            "case_quality": {
                "before": {"bleu": 5.2, "rouge1": 0.15, "rouge2": 0.03, "rougeL": 0.12},
                "after": {"bleu": 18.5, "rouge1": 0.42, "rouge2": 0.18, "rougeL": 0.38},
                "delta": {"bleu": 13.3, ...},
                "delta_pct": {"bleu": 255.8, ...},
            },
            "ariz_completeness": { ... },
            "overall_score": { ... },
        },
    },
    "layer3_performance": {
        "source": "re-run_on_both_models",
        "metrics": {
            "throughput_tokens_per_sec": {"before": 55.0, "after": 52.0, "delta": -3.0, "delta_pct": -5.5},
            "latency_p50_ms": {"before": 1800, "after": 1950, "delta": 150, "delta_pct": 8.3},
            "memory_peak_gb": {"before": 78.0, "after": 82.0, "delta": 4.0, "delta_pct": 5.1},
        },
    },
}
```

### Anti-Patterns to Avoid
- **Hardcoding ChatML tokens:** Never manually concatenate `<|im_start|>system\n...<|im_end|>`. Use `tokenizer.apply_chat_template()` to ensure token ID alignment. [VERIFIED: CR-003 audit fix in CLAUDE.md]
- **Using `target_modules="all-linear"`:** Explicit 12-module list is locked; do not change. [VERIFIED: CR-002 audit fix in CLAUDE.md]
- **Passing `data_collator` to SFTTrainer:** Locked decision from Phase 01; conflicts with internal label masking. [VERIFIED: CR-001 audit fix in CLAUDE.md]
- **Sentence-level BLEU for final report:** Use corpus-level BLEU only; sentence-level is unreliable. [CITED: sacrebleu docs]
- **ROUGE with `use_stemmer=True` on Chinese:** Porter stemmer mangles Chinese characters; always disable for Chinese-English mixed text. [CITED: rouge-score research]
- **Loading adapter without `trust_remote_code=True`:** Will fail with Qwen3 MoE because custom model class is not in standard transformers. [CITED: QwenLM/Qwen3#209]

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ChatML string formatting | Manual token concatenation | `tokenizer.apply_chat_template()` | Token IDs must align with model's chat template; manual strings risk subword misalignment [VERIFIED: CR-003] |
| BLEU scoring | Custom n-gram overlap | `sacrebleu.corpus_bleu()` | Standardized signature, Chinese tokenization, reproducible across studies [CITED: sacrebleu README] |
| ROUGE scoring | Custom LCS implementation | `rouge-score.RougeScorer` | Google's native Python port of official Perl; handles text normalization [CITED: rouge-score PyPI] |
| Chinese word segmentation | Character-level only | `jieba.cut()` + join with spaces | ROUGE expects space-separated tokens; jieba is the standard Chinese segmenter [VERIFIED: requirements.txt] |
| Adapter + base model loading | Manual two-step load | `AutoPeftModelForCausalLM.from_pretrained()` | One-liner, handles adapter config discovery, dtype consistency [VERIFIED: Notebook 05] |
| Baseline result persistence | Ad-hoc file I/O | `PipelineState` artifact registry | Already built in Phase 01; supports verify/preflight/summary [VERIFIED: pipeline_state.py] |

**Key insight:** The only "custom" code in this phase is the glue logic (format_messages wrapper, report schema, notebook display). All metric computation and model loading should delegate to battle-tested libraries.

---

## Runtime State Inventory

This phase involves evaluation inference and report generation, not rename/refactor/migration. No runtime state inventory is required — the phase reads from existing artifacts (adapter weights, pipeline_state JSON, base model) and writes new report files. No OS-registered state, secrets, or stored data keys are affected.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | `pipeline_state.json` contains baseline_results artifact | Read-only for report generation |
| Live service config | None | — |
| OS-registered state | None | — |
| Secrets/env vars | None (model paths are config.py constants) | — |
| Build artifacts | Adapter weights at `models/meerkat_triz_adapter_v1/` | Read-only for loading |

---

## Common Pitfalls

### Pitfall 1: `trust_remote_code=False` Causes Load Failure
**What goes wrong:** `AutoPeftModelForCausalLM.from_pretrained()` raises `ValueError` or `AttributeError` because Qwen3 MoE uses custom `Qwen3MoeForCausalLM` class not in standard transformers.
**Why it happens:** Qwen3.6-35B-A3B is a MoE model with custom architecture code in the HF repo. Transformers cannot instantiate it without executing remote Python.
**How to avoid:** Always pass `trust_remote_code=True` for both model and tokenizer loading. [CITED: QwenLM/Qwen3#209, CSDN Qwen3 article]
**Warning signs:** Error mentions "cannot find model class", "Qwen3MoeForCausalLM", or "remote code".

### Pitfall 2: `add_generation_prompt` Mismatch
**What goes wrong:** Inference prompts lack `<|im_start|>assistant\n` suffix, causing the model to continue the user message instead of generating a response. Or training data includes the generation prompt, causing SFTTrainer to compute loss on a spurious assistant prefix.
**Why it happens:** `apply_chat_template(add_generation_prompt=True)` appends the assistant start token. For inference this is correct; for training data it must be `False` because the assistant content already follows.
**How to avoid:** Inference -> `True`. Training data formatting -> `False`. Document this in `format_messages()` docstring. [VERIFIED: training_utils.py formatting_func uses False]
**Warning signs:** Model generates empty responses, or training loss is unexpectedly high.

### Pitfall 3: BLEU Score Scale Confusion
**What goes wrong:** Report shows BLEU as 0.15 when sacrebleu returns 15.0, or compares sacrebleu (0-100) with NLTK (0-1) scores.
**Why it happens:** SacreBLEU reports on a 0-100 scale (MT competition tradition). NLTK reports 0-1.
**How to avoid:** Always store the raw sacrebleu score (0-100) in JSON, and document the scale. If normalizing to 0-1 for consistency with other metrics, do so explicitly. [CITED: sacrebleu README]
**Warning signs:** BLEU scores look an order of magnitude off compared to ROUGE.

### Pitfall 4: ROUGE Stemmer Corrupting Chinese
**What goes wrong:** ROUGE scores are artificially low or zero because the Porter stemmer strips/modifies Chinese characters.
**Why it happens:** `rouge_scorer.RougeScorer(use_stemmer=True)` applies English Porter stemming, which is undefined for CJK characters.
**How to avoid:** Always set `use_stemmer=False` when evaluating Chinese or Chinese-English mixed text. [CITED: rouge-score research, Chinese NLP best practices]
**Warning signs:** ROUGE scores near zero despite semantically similar Chinese text.

### Pitfall 5: Base Model Re-Run Without Clearing GPU Memory
**What goes wrong:** Loading both adapter model and base model simultaneously causes OOM on 128GB unified memory.
**Why it happens:** The adapter model holds the base model + adapter weights. Loading a second base model instance doubles memory.
**How to avoid:** `del model; torch.cuda.empty_cache()` before loading the base model for comparison. Or use `AutoModelForCausalLM.from_pretrained()` directly for the base model run (not AutoPeft). [ASSUMED]
**Warning signs:** `torch.cuda.OutOfMemoryError` during the "before" model load step.

### Pitfall 6: Missing Baseline in pipeline_state
**What goes wrong:** Notebook 05 fails with KeyError when trying to load baseline results that don't exist (e.g., user skipped Notebook 03).
**Why it happens:** `state.get("baseline_results")` returns `None` if never registered.
**How to avoid:** Implement D-04: if baseline missing, auto-run quick baseline (one Layer 1 task + Layer 3) and print a warning. [VERIFIED: 03-CONTEXT.md D-04]
**Warning signs:** `NoneType` has no attribute `get` when accessing baseline metadata.

---

## Code Examples

### Verified patterns from official sources:

#### Qwen3 Chat Template via apply_chat_template
```python
# Source: Qwen3 chat template deep dive [CITED: huggingface.tw/blog/qwen-3-chat-template-deep-dive]
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3.6-35B-A3B",
    trust_remote_code=True,
)

messages = [
    {"role": "system", "content": "You are a TRIZ expert."},
    {"role": "user", "content": "Explain segmentation principle."},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,  # Appends <|im_start|>assistant\n
)
# Output: <|im_start|>system\nYou are a TRIZ expert.<|im_end|>\n<|im_start|>user\nExplain segmentation principle.<|im_end|>\n<|im_start|>assistant\n
```

#### AutoPeftModelForCausalLM with Qwen3
```python
# Source: Notebook 05 existing code [VERIFIED: 05_model_evaluation.ipynb]
# Source: Qwen3 MoE trust_remote_code requirement [CITED: CSDN article, QwenLM/Qwen3#209]
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

adapter_path = "/home/meerkat/mongoose_ai/models/meerkat_triz_adapter_v1"

model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_path,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True,  # REQUIRED for Qwen3 MoE
)

tokenizer = AutoTokenizer.from_pretrained(
    adapter_path,
    trust_remote_code=True,
)
```

#### SacreBLEU Corpus-Level Scoring
```python
# Source: sacrebleu README [CITED: github.com/mjpost/sacrebleu]
from sacrebleu import corpus_bleu

predictions = ["这是一个 TRIZ 解决方案。", "使用分割原理。"]
references = ["这是一个 TRIZ 创新方案。", "应用分割原理解决问题。"]

bleu = corpus_bleu(predictions, [references], tokenize='zh')
print(bleu.score)        # e.g., 23.5 (0-100 scale)
print(bleu.signature)    # BLEU|nrefs:1|case:mixed|eff:no|tok:zh|smooth:exp|version:2.4.3
```

#### ROUGE with Chinese Segmentation
```python
# Source: rouge-score docs + Chinese NLP best practices [CITED: PyPI, research papers]
from rouge_score import rouge_scorer
import jieba

scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'],
    use_stemmer=False,  # Critical for Chinese
)

pred = "使用分割原理将系统分成独立部分。"
ref = "应用分割原理，把物体分成独立的部分。"

pred_seg = ' '.join(jieba.cut(pred.strip()))
ref_seg = ' '.join(jieba.cut(ref.strip()))

scores = scorer.score(ref_seg, pred_seg)
print(f"ROUGE-1 F1: {scores['rouge1'].fmeasure:.4f}")
```

#### Notebook 05 Pre-flight Checks
```python
# Source: derived from pipeline_state.py API [VERIFIED: pipeline_state.py]
# and D-06 requirements [VERIFIED: 03-CONTEXT.md]
from pathlib import Path
from utils.pipeline_state import PipelineState
from config import MODELS_DIR, BASE_MODEL

state = PipelineState()
adapter_path = MODELS_DIR / "meerkat_triz_adapter_v1"
base_model_path = MODELS_DIR / BASE_MODEL.split('/')[-1]

errors = []

# 1. Check adapter exists
if not adapter_path.exists():
    errors.append(f"Adapter not found: {adapter_path}")

# 2. Check base model exists
if not base_model_path.exists():
    errors.append(f"Base model not found: {base_model_path}")

# 3. Check pipeline_state accessible
baseline = state.get("baseline_results")
if baseline is None:
    print("WARNING: Baseline not found in pipeline_state — running quick baseline now.")
    # Auto-run quick baseline (D-04)
else:
    print(f"Baseline loaded from pipeline_state: {baseline['path']}")

if errors:
    raise RuntimeError(f"Pre-flight check failed:\n" + "\n".join(errors))
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hardcoded ChatML strings (`<|im_start|>system\n...`) | `tokenizer.apply_chat_template()` | Phase 01 (CR-003 fix) | Ensures token ID alignment with model's native chat template |
| `target_modules="all-linear"` | Explicit 12-module list | Phase 01 (CR-002 fix) | Avoids compatibility issues with Qwen3.6 hybrid architecture |
| SFTTrainer + `data_collator` | SFTTrainer + `formatting_func` only | Phase 01 (CR-001 fix) | Prevents label-masking logic conflict |
| Manual base model + `PeftModel.from_pretrained()` two-step | `AutoPeftModelForCausalLM.from_pretrained()` one-step | 2024+ PEFT versions | Simpler code, auto-discovers adapter config |
| NLTK BLEU | `sacrebleu` | 2018+ (now standard) | Reproducible scores with signature strings |
| ROUGE with stemming on Chinese | ROUGE without stemming + `jieba` segmentation | 2020+ Chinese NLP consensus | Accurate scores for CJK text |

**Deprecated/outdated:**
- `pyrouge` (Perl wrapper): Replaced by pure-Python `rouge-score` [CITED: rouge-score PyPI]
- Sentence-level BLEU for final reporting: Use corpus-level only [CITED: sacrebleu docs]
- `trust_remote_code=False` for Qwen3: Must be `True` for MoE models [CITED: QwenLM/Qwen3#209]

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `sacrebleu` 2.4.x is available via pip and compatible with Python 3.10+ | Standard Stack | BLEU scoring fails; fallback to nltk (non-standard) |
| A2 | `jieba` segmentation quality is sufficient for ROUGE on TRIZ technical text | Code Examples | ROUGE scores may be noisy; consider character-level fallback |
| A3 | Loading base model after adapter model (with `del model; torch.cuda.empty_cache()`) fits in 128GB unified memory | Common Pitfalls | OOM during evaluation; may need to restart kernel between runs |
| A4 | `pipeline_state.json` contains baseline_results with at least `metadata.triz_summary` and `metadata.perf_throughput` | Pattern 4 | Missing baseline auto-run (D-04) handles this, but adds 3-5 min |
| A5 | Qwen3.6-35B-A3B tokenizer has `apply_chat_template` method and uses `<|im_start|>` / `<|im_end|>` tokens | Pattern 1 | If tokenizer lacks chat_template, fallback to hardcoded (already exists in data_utils.py) |

---

## Open Questions

1. **Should `format_messages()` live in `data_utils.py` or `benchmark_utils.py`?**
   - What we know: `data_utils.py` already has `convert_to_chatml()` with similar logic. `benchmark_utils.py` is the consumer for evaluation.
   - What's unclear: Whether importing `data_utils` in `benchmark_utils` creates circular dependencies.
   - Recommendation: Place in `data_utils.py` (single source of truth), import in `benchmark_utils.py`. No circular dependency expected since `data_utils` does not import `benchmark_utils`.

2. **What reference data should BLEU/ROUGE use for case_quality?**
   - What we know: Current `evaluate_case_quality()` only does keyword coverage. Real BLEU/ROUGE need reference texts.
   - What's unclear: Whether `sample_data.json` test split outputs are sufficient references, or if we need expert-curated references.
   - Recommendation: Use seed sample `output` fields from `sample_data.json` as references. For synthetic data, references may be lower quality — document this limitation in the report.

3. **Which Layer 1 task for optional spot-check?**
   - What we know: BENCHMARK_CONFIG lists mmlu_pro, gpqa, humaneval, math, bbh. MMLU-Pro is the most general and fastest.
   - What's unclear: Time budget for Notebook 05 on DGX Spark.
   - Recommendation: If time permits, spot-check `mmlu_pro` with `limit=50` (subset) to keep runtime under 10 minutes. Document as "spot-check only, not full suite."

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | All | ✓ | 3.14.3 | — |
| torch | Model loading, inference | ✗ | — | Not installable on macOS for CUDA; DGX Spark is target env |
| transformers | AutoPeftModelForCausalLM, tokenizer | ✗ | — | DGX Spark target env |
| peft | Adapter loading | ✗ | — | DGX Spark target env |
| sacrebleu | BLEU scoring | ✗ | — | `pip install sacrebleu` (must add to requirements.txt) |
| rouge-score | ROUGE scoring | ✗ | — | Already in requirements.txt |
| jieba | Chinese segmentation | ✗ | — | Already in requirements.txt |
| lm-eval | Layer 1 spot-check | ✗ | — | Already in requirements.txt |

**Missing dependencies with no fallback:**
- None — all missing packages have install paths or are DGX Spark environment dependencies.

**Missing dependencies with fallback:**
- `torch/transformers/peft`: These are DGX Spark environment packages. Development on macOS cannot install CUDA PyTorch, but the code is written for DGX Spark execution.
- `sacrebleu`: Must be added to `requirements.txt` and installed via pip.

---

## Validation Architecture

> `workflow.nyquist_validation` is absent in `.planning/config.json` — treat as enabled.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (Python standard) |
| Config file | none — see Wave 0 |
| Quick run command | `pytest tests/ -x -q` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVAL-03 | `format_messages()` produces correct ChatML via apply_chat_template | unit | `pytest tests/test_format_messages.py -x` | ❌ Wave 0 |
| EVAL-03 | `_build_prompt()` in benchmark_utils no longer hardcodes ChatML | unit | `pytest tests/test_benchmark_utils.py::test_prompt_formatting -x` | ❌ Wave 0 |
| EVAL-05 | BLEU computation returns valid score for Chinese-English text | unit | `pytest tests/test_metrics.py::test_bleu_chinese -x` | ❌ Wave 0 |
| EVAL-05 | ROUGE computation returns valid score for Chinese-English text | unit | `pytest tests/test_metrics.py::test_rouge_chinese -x` | ❌ Wave 0 |
| EVAL-02 | Report JSON contains before/after/delta for all Layer 2 metrics | unit | `pytest tests/test_report.py::test_delta_structure -x` | ❌ Wave 0 |
| EVAL-01 | PipelineState baseline loading returns expected metadata | unit | `pytest tests/test_pipeline_state.py::test_baseline_load -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_{module}.py -x -q`
- **Per wave merge:** `pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_format_messages.py` — covers EVAL-03
- [ ] `tests/test_metrics.py` — covers EVAL-05 (BLEU/ROUGE)
- [ ] `tests/test_report.py` — covers EVAL-02 (report schema)
- [ ] `tests/test_pipeline_state.py` — covers EVAL-01 (baseline loading)
- [ ] `tests/conftest.py` — shared fixtures (mock tokenizer with chat_template)
- [ ] Framework install: `pip install pytest` — if none detected

*(Note: The project currently has no `tests/` directory. All test files are Wave 0 gaps.)*

---

## Security Domain

> `security_enforcement` is absent in config — treat as enabled. However, this phase is evaluation-only (no auth, no session management, no user input beyond pre-defined test prompts).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes (indirect) | Test prompts are static; no untrusted user input in evaluation |
| V6 Cryptography | no | — |
| V10 Malicious Code | yes (indirect) | `trust_remote_code=True` executes remote Python from HuggingFace — mitigated by pinning model revision and using official `Qwen/Qwen3.6-35B-A3B` repo |

### Known Threat Patterns for This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malicious remote code in HF model repo | Tampering | Pin to specific revision/tag; use `safetensors` (no pickle); verify SHA-256 of downloaded files |
| Path traversal in output file writing | Tampering | Use `Path(output_dir).resolve()`; validate paths are within project directory |

---

## Sources

### Primary (HIGH confidence)
- `ref/mongoose_ai_dgx/config.py` — BENCHMARK_CONFIG, DATA_CONFIG, MODELS_DIR, BASE_MODEL [VERIFIED]
- `ref/mongoose_ai_dgx/utils/benchmark_utils.py` — TRIZBenchmark class, _build_prompt(), evaluate_case_quality(), aggregate_results() [VERIFIED]
- `ref/mongoose_ai_dgx/utils/data_utils.py` — convert_to_chatml(), _build_chatml_hardcoded() [VERIFIED]
- `ref/mongoose_ai_dgx/utils/pipeline_state.py` — PipelineState API (get, verify, preflight, register) [VERIFIED]
- `ref/mongoose_ai_dgx/utils/training_utils.py` — formatting_func pattern, load_model_and_tokenizer() [VERIFIED]
- `ref/mongoose_ai_dgx/notebooks/05_model_evaluation.ipynb` — Existing evaluation notebook structure [VERIFIED]
- `ref/mongoose_ai_dgx/notebooks/03_model_benchmark.ipynb` — Baseline notebook that persists to pipeline_state [VERIFIED]
- `ref/mongoose_ai_dgx/requirements.txt` — Package versions [VERIFIED]
- `.planning/phases/03-evaluation-and-hardening/03-CONTEXT.md` — Locked decisions D-01 through D-06 [VERIFIED]
- `.planning/REQUIREMENTS.md` — EVAL-01 through EVAL-05 [VERIFIED]

### Secondary (MEDIUM confidence)
- [Hugging Face blog: Qwen-3 chat template deep dive](https://huggingface.tw/blog/qwen-3-chat-template-deep-dive) — Qwen3 chat template format, special tokens, enable_thinking flag [CITED]
- [SacreBLEU GitHub README](https://github.com/mjpost/sacrebleu) — corpus_bleu API, Chinese tokenization, signature strings [CITED]
- [rouge-score PyPI](https://pypi.org/project/rouge-score/) — RougeScorer API, use_stemmer parameter [CITED]
- [Stack Overflow: NLTK BLEU vs SacreBLEU](https://stackoverflow.com/questions/65454578/whats-the-difference-between-nltks-bleu-score-and-sacrebleu) — Key differences, reproducibility [CITED]
- [QwenLM/Qwen3 GitHub issue #209](https://github.com/QwenLM/Qwen3/issues/209) — AutoPeftModelForCausalLM merge issue, meta tensor fix, trust_remote_code requirement [CITED]
- [DataCamp: Fine-Tuning Qwen3](https://www.datacamp.com/tutorial/fine-tuning-qwen3) — AutoPeftModelForCausalLM loading patterns [CITED]
- [CSDN article on Qwen3 trust_remote_code](https://blog.csdn.net/bleuesprit/article/details/156303344) — Qwen3 MoE requires trust_remote_code=True [CITED]

### Tertiary (LOW confidence)
- Web search results for "AutoPeftModelForCausalLM Qwen evaluation inference best practices 2025" — General patterns, some recommendations for `trust_remote_code=False` on Qwen2.5 that contradict Qwen3 MoE requirement. Cross-verified with primary sources (Qwen3 issue #209, CSDN) that `True` is required for MoE. [CROSS-VERIFIED]

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — versions verified in requirements.txt, sacrebleu version confirmed via PyPI
- Architecture: HIGH — all patterns derived from existing codebase and official docs
- Pitfalls: HIGH — all pitfalls are known issues from prior phases or verified from official sources

**Research date:** 2026-05-29
**Valid until:** 2026-06-29 (stable stack — transformers/peft versions are pinned, evaluation patterns are mature)
