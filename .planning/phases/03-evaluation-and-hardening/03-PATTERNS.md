# Phase 03: Evaluation & Hardening - Pattern Map

**Mapped:** 2026-05-29
**Files analyzed:** 7
**Analogs found:** 7 / 7

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `ref/mongoose_ai_dgx/utils/data_utils.py` (modify) | utility | transform | `ref/mongoose_ai_dgx/utils/data_utils.py` (existing `convert_to_chatml`) | exact |
| `ref/mongoose_ai_dgx/utils/benchmark_utils.py` (modify) | service | CRUD + transform | `ref/mongoose_ai_dgx/utils/benchmark_utils.py` (existing `TRIZBenchmark`) | exact |
| `ref/mongoose_ai_dgx/notebooks/05_model_evaluation.ipynb` (modify) | component | request-response | `ref/mongoose_ai_dgx/notebooks/03_model_benchmark.ipynb` | role-match |
| `ref/mongoose_ai_dgx/requirements.txt` (modify) | config | — | `ref/mongoose_ai_dgx/requirements.txt` | exact |
| `tests/test_format_messages.py` (new) | test | transform | No direct analog; pattern from `data_utils.py` validation style | no-analog |
| `tests/test_metrics.py` (new) | test | transform | No direct analog; pattern from existing pytest conventions | no-analog |
| `tests/test_report.py` (new) | test | transform | No direct analog | no-analog |

## Pattern Assignments

### `ref/mongoose_ai_dgx/utils/data_utils.py` (utility, transform)

**Analog:** `ref/mongoose_ai_dgx/utils/data_utils.py` (self — adding `format_messages()` alongside existing `convert_to_chatml`)

**Imports pattern** (lines 1-14):
```python
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datasets import Dataset, DatasetDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

**Core pattern — `convert_to_chatml` uses `tokenizer.apply_chat_template()`** (lines 196-212):
```python
if use_chat_template:
    messages = [
        {"role": "system", "content": sample_system},
        {"role": "user", "content": full_question},
        {"role": "assistant", "content": output},
    ]
    try:
        chatml_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as e:
        logger.warning(f"apply_chat_template失败 ({e})，回退到硬编码格式")
        chatml_text = _build_chatml_hardcoded(sample_system, full_question, output)
```

**Fallback pattern — `_build_chatml_hardcoded`** (lines 233-239):
```python
def _build_chatml_hardcoded(system: str, question: str, answer: str) -> str:
    """硬编码ChatML格式（回退方案）"""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer}<|im_end|>"
    )
```

**System message source pattern** (lines 164-170):
```python
default_system = (
    "You are Meerkat-AI, an expert innovation consultant specializing in TRIZ "
    "(Theory of Inventive Problem Solving). You help users analyze technical contradictions, "
    "recommend invention principles, generate innovative solutions, and guide them through "
    "the ARIZ algorithm. Always provide structured, actionable advice grounded in TRIZ methodology."
)
system_message = system_message or default_system
```

**What to copy for `format_messages()`:**
- Import `logging`, `Optional` from typing.
- Use `logger = logging.getLogger(__name__)`.
- Use `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=...)`.
- Accept `system_message: Optional[str] = None`; if None, pull from `DATA_CONFIG['chatml']['system_message']` (import from `config`).
- Build `messages` list with `system`, `user`, and optionally `assistant` roles.
- Return the formatted string directly (not wrapped in a dataset sample dict).

---

### `ref/mongoose_ai_dgx/utils/benchmark_utils.py` (service, CRUD + transform)

**Analog:** `ref/mongoose_ai_dgx/utils/benchmark_utils.py` (self — modifying existing methods)

**Imports pattern** (lines 1-15):
```python
import json
import time
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

**Core pattern — `_build_prompt` hardcodes ChatML (to be replaced)** (lines 303-309):
```python
def _build_prompt(self, question: str) -> str:
    """构建评测prompt"""
    system_msg = (
        "You are TRIZ-Expert, a specialized AI for TRIZ (Theory of Inventive Problem Solving). "
        "Answer the following question with professional TRIZ knowledge."
    )
    return f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
```

**Core pattern — `_generate_response` inference wrapper** (lines 311-328):
```python
def _generate_response(self, prompt: str, max_new_tokens: int = 512) -> str:
    """生成模型回复"""
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

    with torch.no_grad():
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 移除prompt部分
    response = response[len(prompt):].strip()
    return response
```

**Core pattern — `evaluate_case_quality` (incomplete ROUGE, to be extended)** (lines 218-250):
```python
def evaluate_case_quality(self) -> Dict[str, Any]:
    """评测创新案例生成质量 (使用BLEU/ROUGE)"""
    logger.info("评测: 案例生成质量")

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    except ImportError:
        logger.warning("rouge_score 未安装，跳过ROUGE评测")
        scorer = None

    rouge_scores = []

    for q in self.test_questions:
        if q["type"] == "generation":
            prompt = self._build_prompt(q["question"])
            response = self._generate_response(prompt)

            # 简单的关键词匹配作为质量指标
            keywords = q.get("expected_keywords", [])
            matched = sum(1 for kw in keywords if kw.lower() in response.lower())
            coverage = matched / len(keywords) if keywords else 0

            rouge_scores.append({
                "coverage": coverage,
                "response_length": len(response),
            })

    avg_coverage = sum(s["coverage"] for s in rouge_scores) / len(rouge_scores) if rouge_scores else 0
    return {
        "average_coverage": avg_coverage,
        "details": rouge_scores
    }
```

**Core pattern — `aggregate_results` (single report, to be extended for before/after)** (lines 490-537):
```python
def aggregate_results(
    general_results: Optional[Dict] = None,
    triz_results: Optional[Dict] = None,
    perf_results: Optional[Dict] = None,
    output_dir: str = "./results"
) -> Dict[str, Any]:
    """
    聚合三层评测结果为综合报告

    Returns:
        综合评测报告
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {},
        "layer1_general": general_results or {},
        "layer2_triz": triz_results or {},
        "layer3_performance": perf_results or {},
    }

    # 计算综合评分
    scores = []

    if triz_results:
        triz_score = triz_results.get("overall_score", 0) * 100
        scores.append(triz_score)
        report["summary"]["triz_score"] = f"{triz_score:.1f}/100"

    if perf_results:
        # 性能评分 (吞吐量为主要指标)
        throughput = perf_results.get("throughput_tokens_per_sec", 0)
        perf_score = min(throughput / 2, 100)  # 100 tokens/s = 100分
        scores.append(perf_score)
        report["summary"]["performance_score"] = f"{perf_score:.1f}/100"

    if scores:
        report["summary"]["overall_score"] = f"{sum(scores)/len(scores):.1f}/100"

    # 保存报告
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_file = output_path / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"综合报告已保存: {report_file}")
    return report
```

**What to copy/modify:**
- `_build_prompt`: Replace hardcoded ChatML with call to `format_messages()` from `data_utils.py`.
- `evaluate_case_quality`: Add `sacrebleu.corpus_bleu` and `rouge_scorer` with `use_stemmer=False` + `jieba.cut` segmentation. Keep existing keyword coverage as fallback.
- `aggregate_results`: Extend signature to accept `before_results` and `after_results`, compute deltas and delta_pct per metric, and produce the flat JSON schema `{layer: {metric: {before, after, delta, delta_pct}}}`.
- `_generate_response`: Keep unchanged — it is the established inference pattern.

---

### `ref/mongoose_ai_dgx/notebooks/05_model_evaluation.ipynb` (component, request-response)

**Analog:** `ref/mongoose_ai_dgx/notebooks/03_model_benchmark.ipynb`

**Notebook setup cell pattern** (Notebook 03, cell 1):
```python
import sys
sys.path.append('/home/meerkat/mongoose_ai')

import os
import torch
from datetime import datetime
from utils.training_utils import load_model_and_tokenizer
from utils.pipeline_state import PipelineState
from config import BASE_MODEL, MODELS_DIR, RESULTS_DIR, BENCHMARK_CONFIG

# 确定模型路径
model_path = os.path.join(MODELS_DIR, BASE_MODEL.split('/')[-1])

print(f"评测模型: {model_path}")
print("使用FP16加载模型 (基准评测不使用4-bit量化，避免量化偏差)...")

# 加载模型 (FP16，无量化) — BENCH-02
model, tokenizer = load_model_and_tokenizer(
    model_name_or_path=model_path,
    quantization_config=None,  # 不使用量化
    device_map='auto',
    trust_remote_code=True,
)
```

**PipelineState registration pattern** (Notebook 03, cell 2):
```python
state = PipelineState()

state.register(
    name="baseline_run",
    path=str(RESULTS_DIR),
    artifact_type="benchmark",
    metadata={
        "model_path": model_path,
        "model_dtype": "float16",
        "timestamp": datetime.now().isoformat(),
        "status": "running",
    }
)
```

**PipelineState baseline retrieval pattern** (Notebook 03, cell 10):
```python
state.register(
    name="baseline_results",
    path=str(latest_result) if latest_result else str(RESULTS_DIR),
    artifact_type="benchmark",
    metadata={
        "model_path": model_path,
        "model_dtype": "float16",
        "tasks": tasks if "tasks" in dir() else [],
        "layer2_included": True,
        "triz_summary": triz_summary,
        "perf_throughput": perf_results.get("throughput_tokens_per_sec", "N/A") if "perf_results" in dir() else "N/A",
        "timestamp": datetime.now().isoformat(),
    }
)
```

**GPU memory cleanup pattern** (Notebook 03, cell 12):
```python
del model
del tokenizer
torch.cuda.empty_cache()

print("显存已清理")
print(f"当前显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
```

**Existing Notebook 05 adapter loading pattern** (Notebook 05, cell 2):
```python
from peft import PeftModel, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_path,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    adapter_path,
    trust_remote_code=True,
)
```

**What to copy/modify for Notebook 05:**
- Pre-flight checks: Use `PipelineState.preflight()` pattern (or manual `Path.exists()` checks) for adapter_path, base_model_path, and pipeline_state accessibility.
- Adapter loading: Keep existing `AutoPeftModelForCausalLM.from_pretrained(...)` cell (already correct per EVAL-04).
- Base model loading: Use `AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)` (not AutoPeft) for the "before" run.
- Memory management: Insert `del model; torch.cuda.empty_cache()` between adapter run and base model run (Pitfall 5).
- Baseline loading: Use `state.get("baseline_results")` pattern; if None, auto-run quick baseline (one Layer 1 task + Layer 3 on base model) per D-04.
- Report display: Use markdown tables with `+`/`-` indicators and percentage changes per D-02.
- Remove hardcoded `before_score = 0.35` (cell 10) — replace with dynamic before/after from re-run.
- Replace hardcoded ChatML prompt construction (cell 6) with `format_messages()`.

---

### `ref/mongoose_ai_dgx/requirements.txt` (config, —)

**Analog:** `ref/mongoose_ai_dgx/requirements.txt` (self)

**Existing dependency pattern** (lines 21-25):
```python
# 评测框架
lm-eval==0.4.10           # lm-eval-harness基准测试
evaluate>=0.4.2
scikit-learn>=1.4.0
rouge-score==0.1.2        # BLEU/ROUGE质量评分 (新增)
```

**What to add:**
- Add `sacrebleu==2.4.3` under the evaluation framework section (RESEARCH.md confirms it is missing and must be added).

---

## Shared Patterns

### Authentication / Model Loading Trust
**Source:** `ref/mongoose_ai_dgx/config.py` lines 31-35 and `ref/mongoose_ai_dgx/utils/training_utils.py` lines 118-123
**Apply to:** Notebook 05 (both adapter and base model loading cells)
```python
MODEL_CONFIG = {
    "trust_remote_code": True,
    "torch_dtype": "auto",
    "device_map": "auto",
}
```
Critical: `trust_remote_code=True` is mandatory for Qwen3 MoE models.

### Error Handling — Try/Import with Graceful Degradation
**Source:** `ref/mongoose_ai_dgx/utils/benchmark_utils.py` lines 42-48 and 222-227
**Apply to:** `evaluate_case_quality()` BLEU/ROUGE additions
```python
try:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
except ImportError:
    logger.warning("rouge_score 未安装，跳过ROUGE评测")
    scorer = None
```

### Logging Pattern
**Source:** All utility files
**Apply to:** All modified/new Python files
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

### Path Handling
**Source:** `ref/mongoose_ai_dgx/config.py` lines 10-20
**Apply to:** Notebook 05 path construction, report saving
```python
from pathlib import Path
BASE_DIR = Path("/home/meerkat/mongoose_ai")
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
```

### JSON Report Saving
**Source:** `ref/mongoose_ai_dgx/utils/benchmark_utils.py` lines 529-537
**Apply to:** `aggregate_results()` extended for before/after
```python
output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)

report_file = output_path / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(report_file, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
```

### PipelineState Get/Verify Pattern
**Source:** `ref/mongoose_ai_dgx/utils/pipeline_state.py` lines 68-80
**Apply to:** Notebook 05 baseline loading, pre-flight checks
```python
def get(self, name: str) -> Optional[Dict[str, Any]]:
    """按名称获取工件信息"""
    for a in self.state["artifacts"]:
        if a["name"] == name:
            return a
    return None

def verify(self, name: str) -> bool:
    """验证工件是否存在且路径有效"""
    artifact = self.get(name)
    if not artifact:
        return False
    return Path(artifact["path"]).exists()
```

### SFTTrainer formatting_func Pattern (for `format_messages` training usage)
**Source:** `ref/mongoose_ai_dgx/utils/training_utils.py` lines 448-475
**Apply to:** `format_messages()` docstring and implementation (shows correct `add_generation_prompt=False` for training)
```python
def formatting_func(example):
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": full_question},
        {"role": "assistant", "content": output},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return text
```

## No Analog Found

Files with no close match in the codebase (planner should use RESEARCH.md patterns instead):

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `tests/test_format_messages.py` | test | transform | No `tests/` directory exists in the project; no test file analogs |
| `tests/test_metrics.py` | test | transform | No `tests/` directory exists; no test file analogs |
| `tests/test_report.py` | test | transform | No `tests/` directory exists; no test file analogs |

For these test files, the planner should reference:
- RESEARCH.md "Pattern 1: format_messages() Utility" for expected behavior.
- RESEARCH.md "Pattern 2: BLEU/ROUGE for Chinese-English Mixed TRIZ Text" for metric computation behavior.
- RESEARCH.md "Pattern 4: Before/After Comparison Report Structure" for report schema validation.
- Standard pytest patterns: `assert` statements, `tmp_path` fixture for file I/O, monkeypatch for mocking `tokenizer.apply_chat_template`.

## Metadata

**Analog search scope:** `ref/mongoose_ai_dgx/utils/`, `ref/mongoose_ai_dgx/notebooks/`, `ref/mongoose_ai_dgx/config.py`, `ref/mongoose_ai_dgx/requirements.txt`
**Files scanned:** 8
**Pattern extraction date:** 2026-05-29
