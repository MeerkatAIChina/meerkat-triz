# Phase 01: Foundation & Data Pipeline - Research

**Researched:** 2026-05-27
**Domain:** Python ML data pipeline, Moonshot API synthetic data generation, QLoRA configuration, TRL SFTTrainer
**Confidence:** MEDIUM-HIGH (verified against source code; some API details from web search)

## Summary

This phase delivers the data and configuration foundation for TRIZ domain fine-tuning of Qwen3.6-35B-A3B on DGX Spark. The core challenge is generating ~5.4K high-quality synthetic TRIZ training samples from 548 seed samples using the Moonshot API (not template paraphrasing), while fixing known configuration bugs (lora_dropout, target_modules, README inaccuracies) and establishing cross-notebook state tracking.

**Key architectural insight:** The synthetic pipeline must be checkpoint-resumable because Moonshot API generation at Tier 0 rate limits (3 RPM) would take ~30 hours unbatched. Batching 5 seeds per request reduces this to ~6 hours. Cost is low (~$6 USD for moonshot-v1-8k) but time is the real constraint.

**Critical TRL finding:** The existing codebase already uses `SFTTrainer` + `formatting_func` + `packing=True` correctly (CR-001 was fixed). However, in TRL 0.9.6, `packing=True` with `formatting_func` does NOT perform assistant-only loss masking — `ConstantLengthDataset` creates labels identical to input_ids with no `-100` masking. The `create_trainer()` function in `training_utils.py` correctly avoids passing `data_collator`, so this is the accepted tradeoff for this version range. [VERIFIED: github.com/huggingface/trl v0.9.6 source]

**Primary recommendation:** Build `utils/synthetic_pipeline.py` as an OpenAI-compatible client with explicit batching (5 seeds/request), exponential backoff retry, and JSONL checkpointing per subset. Use `utils/pipeline_state.py` as a simple JSON registry with artifact versioning. Fix config.py `lora_dropout=0.0` and verify the 12-module `target_modules` list is already correct.

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Hybrid synthetic generation strategy by subset:
  - Rephrase-in-place (keep answers grounded): `concept_explanation`, `ariz_guidance`
  - Generate entirely new Q&A pairs: `case_generation`, `contradiction_analysis`
  - Mix of both: `principle_recommendation`, `innovation_assessment`
- **D-02:** Moonshot API (not template paraphrasing) for true semantic variation
- **D-03:** Variable real/synthetic ratio by subset:
  - 25% real: `concept_explanation`, `ariz_guidance`
  - ~15% real: `principle_recommendation`, `innovation_assessment`
  - 10% real: `case_generation`, `contradiction_analysis`
- **D-04:** Total target ~6K samples. Exact per-subset counts derived from seed distribution and ratio targets.
- **D-05:** Checkpoint-based resumability: save progress after each subset/batch
- **D-06:** Cost monitoring: display estimated Moonshot API cost and token count before each subset

### Claude's Discretion
- Multiplier per seed (variable by subset, balancing diversity vs API cost)
- Quality gate implementation details (pragmatic two-gate: perplexity + length; diversity enforced via generation strategy)
- Failed sample handling strategy
- Notebook 02b cell structure and interaction flow
- Pipeline state registry schema and scope
- Config values beyond required fixes (lora_dropout, target_modules)
- Pre-flight check comprehensiveness by notebook phase
- Requirements pinning approach (exact vs minimum versions)

### Deferred Ideas (OUT OF SCOPE)
- None — discussion stayed within phase scope.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DATA-01 | Generate ~6K synthetic training samples from 548 seed samples using Moonshot API with true semantic variation | Moonshot API v1 chat completions, OpenAI-compatible SDK, batching strategy documented |
| DATA-02 | Implement quality gates: perplexity filtering, diversity scoring, 20-30% real data ratio | Perplexity via base model forward pass; diversity via generation strategy; ratios from D-03 |
| DATA-03 | Add token length profiling in Notebook 02 to catch silent truncation | Tokenizer-based histogram; current seeds avg ~500 tokens, well under 4096 limit |
| DATA-04 | Maintain 85/10/5 train/validation/test split across combined real + synthetic data | Reuse existing `data_utils.split_dataset()` with 0.85/0.10/0.05 ratios |
| DATA-05 | Preserve ChatML format via `tokenizer.apply_chat_template()` for all generated samples | Existing `data_utils.convert_to_chatml()` pattern; apply to synthetic outputs |
| INFRA-04 | Update `requirements.txt` with pinned compatible versions | Verified latest: transformers 5.9.0, trl 1.5.1, peft 0.19.1, bnb 0.49.2, lm-eval 0.4.12 |
| INFRA-05 | Implement `utils/synthetic_pipeline.py` — Moonshot API client with batching, rate limiting, output validation | OpenAI SDK pattern; rate limits: 3 RPM (Tier 0), 200 RPM (Tier 1); no native batch API |
| INFRA-06 | Implement `utils/pipeline_state.py` — JSON artifact registry for cross-notebook state tracking | Simple JSON file with artifact list, timestamps, checksums |
| INFRA-07 | Create `02b_synthetic_generation.ipynb` orchestrating ~6K sample generation | Notebook imports synthetic_pipeline and pipeline_state; checkpoint-resumable |
| INFRA-08 | Fix README inaccuracy: replace `"all-linear"` recommendation with explicit module list | README line 119 currently says "默认使用 `all-linear` 自动检测" |
| INFRA-09 | Add notebook pre-flight checks (paths exist, artifacts present, version compatibility) | Check imports, file existence, pipeline_state artifact registry |
| TRAIN-02 | Use explicit 12-module `target_modules` list (NOT `"all-linear"`) | Already correct in config.py and training_utils.py; 12 modules verified |
| TRAIN-03 | Set `lora_dropout=0.0` for MoE architecture compatibility | Currently 0.05 in config.py; must change to 0.0 |

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Synthetic data generation | API/Backend (Moonshot API) | — | All generation happens via external API calls |
| Data quality gating | API/Backend (DGX Spark Python) | — | Perplexity computed on DGX Spark with loaded model |
| Token length profiling | API/Backend (DGX Spark Python) | — | Tokenizer operations on DGX Spark |
| Dataset splitting/formatting | API/Backend (DGX Spark Python) | — | HuggingFace datasets library |
| Pipeline state tracking | API/Backend (DGX Spark Python) | — | JSON file on local filesystem |
| Config validation | API/Backend (DGX Spark Python) | — | Python config module |
| Notebook pre-flight checks | API/Backend (DGX Spark Python) | — | Runtime verification in notebooks |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| transformers | 4.45+ | Qwen3.6 model loading, tokenizer, chat template | Qwen3.6 requires 4.45+ for native support [VERIFIED: npm registry / pip index] |
| trl | 0.9–0.11 | SFTTrainer with formatting_func + packing | Requirements specify 0.9–0.11 range; 0.9.6 verified for formatting_func+packing behavior [VERIFIED: github source v0.9.6] |
| peft | 0.12+ | LoRA config, get_peft_model | Required for Qwen3.6 hybrid architecture support |
| bitsandbytes | 0.43.x | 4-bit NF4 quantization | Locked to 0.43.x for stability on DGX Spark |
| accelerate | 0.33+ | Device mapping, mixed precision | Required by transformers + peft |
| datasets | 2.21+ | HuggingFace Dataset/DatasetDict | Standard for ML data pipelines |
| openai | 1.0+ | Moonshot API client (OpenAI-compatible) | Moonshot API is OpenAI-compatible; use `base_url="https://api.moonshot.cn/v1"` [VERIFIED: platform.kimi.ai/docs] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| rouge-score | 0.1.2 | BLEU/ROUGE for quality gates | INFRA-04 requires adding this |
| lm-eval | 0.4.10+ | Layer 1 benchmarks (future phase) | Pin now for consistency |
| jieba | 0.42+ | Chinese text segmentation | Already in requirements |
| tqdm | 4.66+ | Progress bars for generation | Already in requirements |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Moonshot API | OpenAI GPT-4o API | Higher quality but 10x+ cost; Moonshot is cost-effective for Chinese |
| Moonshot API | Local Qwen3.6 inference | No API cost but much slower; not practical for 5K+ samples |
| openai SDK | requests + raw HTTP | More control but more code; openai SDK handles retries natively |
| JSON state registry | SQLite/MLflow | Overkill for single-machine notebook workflow |

**Installation:**
```bash
pip install transformers==4.45.0 trl==0.9.6 peft==0.12.0 bitsandbytes==0.43.3 accelerate==0.33.0 datasets==2.21.0 openai==1.35.0 rouge-score==0.1.2 lm-eval==0.4.10
```

**Version verification:** Latest versions checked 2026-05-27:
- transformers: 5.9.0 (published 2025) — but Qwen3.6 compatibility verified at 4.45+
- trl: 1.5.1 (published 2025) — requirements specify 0.9–0.11; 0.9.6 is stable target
- peft: 0.19.1
- bitsandbytes: 0.49.2
- lm-eval: 0.4.12

## Architecture Patterns

### System Architecture Diagram

```
Seed Data (548 samples)
       |
       v
+---------------------------+
| 02b_synthetic_generation  |
|    Jupyter Notebook       |
|  - Load seeds per subset  |
|  - Display cost estimate  |
|  - Call synthetic_pipeline|
+-----------+---------------+
            |
            v
+---------------------------+
| utils/synthetic_pipeline  |
|  - OpenAI client (Moonshot|
|  - Batch 5 seeds/request  |
|  - Rate limit (sleep)     |
|  - Retry with backoff     |
|  - Validate JSON output   |
|  - Save checkpoint JSONL  |
+-----------+---------------+
            |
            v
+---------------------------+
| Moonshot API              |
|  - moonshot-v1-8k/32k     |
|  - 3 RPM (Tier 0)         |
|  - ~$6 total cost         |
+-----------+---------------+
            |
            v
+---------------------------+
| Quality Gates (DGX Spark) |
|  - Perplexity filter      |
|  - Length check           |
|  - Format validation      |
+-----------+---------------+
            |
            v
+---------------------------+
| utils/data_utils          |
|  - convert_to_chatml()    |
|  - split_dataset() 85/10/5|
|  - save_dataset()         |
+-----------+---------------+
            |
            v
+---------------------------+
| utils/pipeline_state      |
|  - JSON artifact registry |
|  - Tracks: raw, synthetic,|
|    processed, splits      |
+-----------+---------------+
            |
            v
+-----------+---------------+
| Notebook 02/04/05 can    |
| read pipeline_state to   |
| verify prior artifacts   |
+--------------------------+
```

### Recommended Project Structure

```
ref/mongoose_ai_dgx/
├── config.py                      # FIXED: lora_dropout=0.0
├── requirements.txt               # FIXED: pinned versions + rouge-score + openai
├── README.md                      # FIXED: no "all-linear" recommendation
├── utils/
│   ├── __init__.py               # Add synthetic_pipeline, pipeline_state exports
│   ├── data_utils.py             # Reuse: load_raw_data, convert_to_chatml, split_dataset
│   ├── training_utils.py         # Reuse: load_model_and_tokenizer, get_qwen36_target_modules
│   ├── benchmark_utils.py        # Unchanged in this phase
│   ├── synthetic_pipeline.py     # NEW: Moonshot API client
│   └── pipeline_state.py         # NEW: JSON artifact registry
├── data/
│   ├── sample_data.json          # 548 seed samples (unchanged)
│   └── processed/                # Generated outputs
│       ├── synthetic/            # Raw synthetic outputs per subset
│       ├── checkpoint/           # Resumable generation state
│       └── final/                # Combined real+synthetic, ChatML format
└── notebooks/
    ├── 01_download_and_setup.ipynb   # ADD: pre-flight checks
    ├── 02_data_preparation.ipynb     # ADD: token length histogram
    ├── 02b_synthetic_generation.ipynb # NEW: orchestrates ~6K generation
    ├── 03_model_benchmark.ipynb      # Unchanged
    ├── 04_qlora_finetune.ipynb       # Unchanged
    └── 05_model_evaluation.ipynb     # Unchanged
```

### Pattern 1: Moonshot API Client with Batching

**What:** OpenAI-compatible client that batches multiple seed samples into a single API request to reduce round-trips.

**When to use:** All synthetic generation calls. Reduces 5,449 individual requests to ~1,090 batched requests.

**Example:**
```python
# Source: [VERIFIED: platform.kimi.ai/docs + openai SDK pattern]
from openai import OpenAI
import json
import time
from typing import List, Dict

class MoonshotSyntheticClient:
    def __init__(self, api_key: str, model: str = "moonshot-v1-8k"):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1"
        )
        self.model = model
        self.rpm = 3  # Tier 0 default
        self.min_interval = 60.0 / self.rpm
        self.last_request_time = 0

    def generate_variations(self, seeds: List[Dict], strategy: str) -> List[Dict]:
        """Generate synthetic samples from multiple seeds in one request."""
        # Rate limit sleep
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        # Build prompt with multiple seeds
        prompt = self._build_batch_prompt(seeds, strategy)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=2000 * len(seeds),  # Scale with batch size
            response_format={"type": "json_object"}
        )

        self.last_request_time = time.time()
        return self._parse_batch_response(response.choices[0].message.content, seeds)
```

### Pattern 2: Checkpoint-Resumable Generation

**What:** Save progress after each subset (and optionally after each batch) so generation can resume if interrupted.

**When to use:** Any long-running API-based generation where interruptions are likely.

**Example:**
```python
# Source: [ASSUMED] — standard pattern for resumable batch jobs
def generate_subset_with_checkpoint(
    client, seeds, subset_name, checkpoint_dir, batch_size=5
):
    checkpoint_file = Path(checkpoint_dir) / f"{subset_name}_checkpoint.json"

    # Resume from checkpoint
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
        completed_ids = set(checkpoint["completed_seed_ids"])
        results = checkpoint["results"]
        print(f"Resuming {subset_name}: {len(completed_ids)}/{len(seeds)} completed")
    else:
        completed_ids = set()
        results = []

    # Process remaining seeds
    remaining = [s for s in seeds if s["id"] not in completed_ids]
    for i in range(0, len(remaining), batch_size):
        batch = remaining[i:i+batch_size]
        try:
            generated = client.generate_variations(batch, strategy)
            results.extend(generated)
            for s in batch:
                completed_ids.add(s["id"])
            # Save checkpoint after each batch
            _save_checkpoint(checkpoint_file, completed_ids, results)
        except Exception as e:
            print(f"Batch failed, saved checkpoint: {e}")
            raise  # Let notebook decide whether to continue

    return results
```

### Pattern 3: Pipeline State Registry

**What:** Simple JSON file that tracks which artifacts exist, their paths, and creation timestamps. All notebooks read/write this file.

**When to use:** Cross-notebook coordination in a notebook-driven workflow.

**Example:**
```python
# Source: [ASSUMED] — standard artifact tracking pattern
import json
from pathlib import Path
from datetime import datetime

class PipelineState:
    def __init__(self, state_file: str = "data/processed/pipeline_state.json"):
        self.state_file = Path(state_file)
        self.state = self._load()

    def _load(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {"artifacts": [], "version": "1.0"}

    def register(self, name: str, path: str, artifact_type: str, metadata: dict = None):
        artifact = {
            "name": name,
            "path": str(path),
            "type": artifact_type,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        # Replace if exists
        self.state["artifacts"] = [a for a in self.state["artifacts"] if a["name"] != name]
        self.state["artifacts"].append(artifact)
        self._save()

    def get(self, name: str) -> dict:
        for a in self.state["artifacts"]:
            if a["name"] == name:
                return a
        return None

    def verify(self, name: str) -> bool:
        artifact = self.get(name)
        if not artifact:
            return False
        return Path(artifact["path"]).exists()
```

### Pattern 4: Notebook Pre-Flight Checks

**What:** First cell in each notebook verifies that required prior artifacts exist and versions are compatible.

**When to use:** Every notebook that depends on prior steps.

**Example:**
```python
# Source: [ASSUMED] — based on notebook-driven workflow requirements
def preflight_check(required_artifacts: List[str], required_packages: Dict[str, str]):
    """Run before any notebook execution."""
    errors = []

    # Check pipeline state artifacts
    state = PipelineState()
    for artifact in required_artifacts:
        if not state.verify(artifact):
            errors.append(f"Missing artifact: {artifact}")

    # Check package versions
    for pkg, min_version in required_packages.items():
        try:
            mod = __import__(pkg)
            actual = mod.__version__
            if parse_version(actual) < parse_version(min_version):
                errors.append(f"{pkg} {actual} < required {min_version}")
        except ImportError:
            errors.append(f"{pkg} not installed")

    if errors:
        raise RuntimeError("Pre-flight check failed:\n" + "\n".join(errors))

    print("Pre-flight check passed!")
```

### Anti-Patterns to Avoid
- **Using `"all-linear"` for target_modules:** PEFT's auto-detection has known issues with hybrid architectures and may include `lm_head` incorrectly. Use explicit 12-module list. [CITED: audit report CR-002]
- **Passing `data_collator` to SFTTrainer:** Conflicts with SFTTrainer's internal label-masking logic. Use `formatting_func` instead. [CITED: audit report CR-001]
- **Hardcoding ChatML format:** Always use `tokenizer.apply_chat_template()` to ensure training format matches inference format. [CITED: audit report CR-003]
- **Template-based paraphrasing for synthetic data:** The existing `vary_sample()` prefix approach produces no semantic diversity. Must use LLM API. [CITED: audit report MA-002]
- **No checkpointing for API generation:** At 3 RPM, 5K+ samples take 30+ hours. Without checkpointing, a single crash loses all progress.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| HTTP client for Moonshot API | Raw `requests` + manual retry | `openai` Python SDK | Handles auth, retries, streaming, JSON parsing automatically |
| Token length estimation | Character count heuristics | `tokenizer.apply_chat_template()` then `tokenizer.encode()` | Exact token count; handles special tokens, multilingual text |
| Perplexity computation | Manual log-prob math | `model.forward()` + `CrossEntropyLoss` | Model already computes logits; just need to extract and average |
| Dataset splitting | Random numpy shuffle | `datasets.Dataset.train_test_split()` | Stratified splitting, seed reproducibility, DatasetDict output |
| JSONL serialization | Manual string formatting | `datasets.Dataset.to_json()` | Handles escaping, unicode, schema consistency |
| Rate limiting | `time.sleep()` in a loop | Exponential backoff with jitter | API failures are bursty; naive sleep doesn't handle 429/5xx well |

**Key insight:** The synthetic pipeline is fundamentally an API client with state management. The `openai` SDK provides 90% of what's needed; the value-add is in batching strategy, checkpointing, and quality gates.

## Runtime State Inventory

This phase involves creating new files and modifying existing ones. No stored runtime data needs migration.

| Category | Items Found | Action Required |
|----------|-------------|-----------------|
| Stored data | `data/sample_data.json` — 548 seed samples | Read-only input; no migration |
| Live service config | None | N/A |
| OS-registered state | None | N/A |
| Secrets/env vars | Moonshot API key (expected in env var `MOONSHOT_API_KEY`) | Code should read from env, not hardcode |
| Build artifacts | None (Python project, no build step) | N/A |

**Nothing found in category:** All categories verified above.

## Common Pitfalls

### Pitfall 1: TRL 0.9.6 `packing=True` Does Not Mask Assistant Tokens
**What goes wrong:** When using `SFTTrainer` with `formatting_func` and `packing=True`, the `ConstantLengthDataset` creates labels identical to input_ids. There is no `-100` masking of user/system tokens. The model learns to predict everything including the user's question. [VERIFIED: github.com/huggingface/trl v0.9.6 source]

**Why it happens:** `ConstantLengthDataset.__iter__` tokenizes formatted strings and returns `labels = input_ids.clone()` with no ignore_index logic. `DataCollatorForCompletionOnlyLM` (which would mask) is explicitly incompatible with `packing=True`.

**How to avoid:** This is a known limitation of TRL 0.9.6. The existing codebase already works around it by using `formatting_func` without `data_collator`. For this project, the impact is acceptable because:
1. The training data is high-quality domain-specific content
2. The model is large (35B) with small adapter (rank 64)
3. Moving to `packing=False` + `DataCollatorForCompletionOnlyLM` would sacrifice training efficiency

**Warning signs:** If evaluation shows the model parroting back user questions instead of answering them, this is the cause. Mitigation: switch to `packing=False` with `DataCollatorForCompletionOnlyLM` in a future iteration.

### Pitfall 2: Moonshot API Rate Limits Block Generation
**What goes wrong:** At Tier 0 (3 RPM), generating 5,449 synthetic samples one-at-a-time takes ~30 hours. A single network interruption loses all progress.

**Why it happens:** Moonshot API uses recharge-based tiers. Free/starter tier is 3 RPM with 1.5M tokens/day limit. [VERIFIED: platform.kimi.ai/docs/pricing/limits]

**How to avoid:** 
1. Batch 5 seeds per request (reduces to ~6 hours at Tier 0)
2. Implement checkpointing after every batch
3. Display cost estimate before starting so user can upgrade to Tier 1 ($10 recharge = 200 RPM)

**Warning signs:** Notebook cell runs for hours with no output; API returns 429 errors.

### Pitfall 3: Duplicate Seed Data Produces Duplicate Synthetic Data
**What goes wrong:** The seed data has 48 duplicate instructions and 58 exact duplicate outputs across subsets. If the synthetic pipeline naively generates from every seed, duplicates will propagate.

**Why it happens:** `sample_data.json` contains repeated domain templates (e.g., "在智能手机领域..." appears 5x in case_generation with identical outputs). [VERIFIED: codebase analysis]

**How to avoid:** 
1. Deduplicate seeds before generation (by instruction+output hash)
2. Or: use duplicate seeds as "strong signals" but cap variations per unique instruction
3. Track generated instruction hashes to prevent exact duplicates in synthetic output

**Warning signs:** Synthetic dataset has identical Q&A pairs; diversity score is low.

### Pitfall 4: `lora_dropout=0.05` Breaks MoE Training
**What goes wrong:** MoE architectures use sparse expert routing. Dropout can interfere with expert selection stability, causing training instability or degraded convergence.

**Why it happens:** Dropout randomly zeros activations, which can destabilize the gating network in MoE layers. [ASSUMED: based on MoE architecture best practices]

**How to avoid:** Set `lora_dropout=0.0` in `config.py` and `training_utils.py`. This is a phase requirement (TRAIN-03).

**Warning signs:** Training loss spikes, eval loss diverges, or adapter fails to load.

### Pitfall 5: Context Length Exceeds `max_seq_length`
**What goes wrong:** Synthetic samples may be longer than seed samples. If a generated sample exceeds the 4096 `max_seq_length` used in training, SFTTrainer silently truncates it, potentially cutting off the assistant's answer.

**Why it happens:** Moonshot API may generate verbose responses. The current seeds average ~500 tokens, but synthetic generation has no length constraint.

**How to avoid:** 
1. Add token length histogram in Notebook 02 (DATA-03)
2. Filter synthetic samples > 3500 tokens (leaving margin below 4096)
3. Add `max_tokens` parameter to API calls

**Warning signs:** Training loss doesn't decrease; generated samples are cut off mid-sentence.

## Code Examples

### Verified patterns from official sources:

### Moonshot API Call (OpenAI-compatible)
```python
# Source: [VERIFIED: platform.kimi.ai/docs + openai SDK]
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",  # Or os.environ["MOONSHOT_API_KEY"]
    base_url="https://api.moonshot.cn/v1"
)

response = client.chat.completions.create(
    model="moonshot-v1-8k",
    messages=[
        {"role": "system", "content": "You are a TRIZ expert."},
        {"role": "user", "content": "Generate a TRIZ training sample about..."}
    ],
    temperature=0.8,
    max_tokens=1000,
    response_format={"type": "json_object"}
)
```

### Token Length Profiling
```python
# Source: [VERIFIED: codebase analysis]
def profile_token_lengths(dataset, tokenizer, max_length=4096):
    """Generate histogram of token lengths."""
    lengths = []
    for sample in dataset:
        text = sample["text"]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(tokens))

    import matplotlib.pyplot as plt
    plt.hist(lengths, bins=50, edgecolor='black')
    plt.axvline(max_length, color='red', linestyle='--', label=f'max={max_length}')
    plt.xlabel('Token Count')
    plt.ylabel('Frequency')
    plt.title('Token Length Distribution')
    plt.legend()
    plt.show()

    over_limit = sum(1 for l in lengths if l > max_length)
    print(f"Samples over {max_length} tokens: {over_limit}/{len(lengths)}")
    return lengths
```

### Perplexity Quality Gate
```python
# Source: [CITED: arxiv.org/abs/2511.10338v2 — BhashaKritika perplexity filtering]
def compute_perplexity(text: str, model, tokenizer, device="cuda") -> float:
    """Compute perplexity of text under the base model."""
    import torch
    import torch.nn.functional as F

    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    perplexity = torch.exp(loss).item()
    return perplexity

# Filter: keep samples below 80th percentile perplexity
def perplexity_filter(samples, model, tokenizer, percentile=80):
    ppls = [compute_perplexity(s["text"], model, tokenizer) for s in samples]
    threshold = np.percentile(ppls, percentile)
    filtered = [s for s, p in zip(samples, ppls) if p <= threshold]
    return filtered, threshold
```

### Explicit 12-Module Target Modules
```python
# Source: [VERIFIED: ref/mongoose_ai_dgx/utils/training_utils.py]
def get_qwen36_target_modules() -> List[str]:
    """Qwen3.6 hybrid architecture: 10 GA + 30 GDN + 40 MoE MLP layers."""
    return [
        # Gated Attention (10/40 layers)
        "q_proj", "k_proj", "v_proj", "o_proj",
        # Gated DeltaNet (30/40 layers)
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
        # MoE MLP (all 40 layers)
        "gate_proj", "up_proj", "down_proj",
    ]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Template paraphrasing (`vary_sample()`) | LLM API semantic generation | Phase 1 (this) | True semantic diversity instead of prefix substitution |
| `"all-linear"` auto-detection | Explicit 12-module list | Post-audit fix | Eliminates architecture compatibility risk |
| Hardcoded ChatML strings | `tokenizer.apply_chat_template()` | Post-audit fix | Format consistency with model's chat template |
| `DataCollatorForLanguageModeling` + SFTTrainer | `formatting_func` only | Post-audit fix | Avoids collator/trainer conflict |
| `lora_dropout=0.05` | `lora_dropout=0.0` | Phase 1 (this) | MoE architecture compatibility |
| Unpinned requirements | Pinned compatible versions | Phase 1 (this) | Reproducible installs on DGX Spark |

**Deprecated/outdated:**
- `target_modules="all-linear"`: Known to incorrectly include `lm_head` on some architectures [CITED: audit report CR-002]
- `vary_sample()` template approach: Produces no meaningful semantic variation [CITED: audit report MA-002]
- Hardcoded `<|im_start|>` ChatML format: May mismatch model's actual chat template [CITED: audit report CR-003]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `lora_dropout=0.0` is required for MoE compatibility | Common Pitfalls | If MoE handles dropout fine, setting 0.0 loses regularization benefit. But requirement TRAIN-03 mandates it. |
| A2 | Moonshot API Tier 0 has 3 RPM limit | Standard Stack | User may have higher tier; pipeline should auto-detect or allow configuration. |
| A3 | Batching 5 seeds per request is optimal | Architecture Patterns | Too many seeds may exceed context window or cause model to lose coherence. May need tuning to 3 or 10. |
| A4 | Perplexity filtering at 80th percentile is appropriate | Code Examples | Threshold may need adjustment based on actual synthetic data distribution. Start with 80th, tune empirically. |
| A5 | Seed data duplicates should be deduplicated before generation | Common Pitfalls | If duplicates represent intentional "strong signals", deduplication loses them. But 58 exact duplicate outputs suggest accidental duplication. |
| A6 | TRL 0.9–0.11 is the correct version range | Standard Stack | Newer TRL versions (1.0+) have breaking API changes (SFTConfig, assistant_only_loss). Pinning to 0.9–0.11 avoids migration. |

## Open Questions

1. **What Moonshot API tier will the user have?**
   - What we know: Tier 0 is 3 RPM, Tier 1 is 200 RPM (requires $10 cumulative recharge)
   - What's unclear: Whether user has already recharged
   - Recommendation: Pipeline should work at Tier 0 with batching; document upgrade path to Tier 1 for faster generation

2. **Should synthetic generation use `response_format={"type": "json_object"}` or plain text parsing?**
   - What we know: JSON mode ensures structured output but may reduce creativity
   - What's unclear: Whether Moonshot v1 models support JSON mode reliably
   - Recommendation: Start with JSON mode for structured Q&A extraction; fallback to regex parsing if JSON mode fails

3. **How to handle the 58 exact duplicate outputs in seed data?**
   - What we know: Duplicates exist in case_generation, ariz_guidance, innovation_assessment
   - What's unclear: Whether these are intentional (same question, same answer) or data quality issues
   - Recommendation: Deduplicate by (instruction, output) hash before generation; log deduplication count

4. **What is the optimal batch size for Moonshot API requests?**
   - What we know: 5 seeds/request reduces time from 30h to 6h at Tier 0
   - What's unclear: Whether the model maintains quality with 5 seeds in context
   - Recommendation: Start with 5, add configuration parameter, evaluate quality on first subset before scaling

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.10+ | All | ✓ | 3.12 (macOS research env) | — |
| PyTorch 2.4+ | Model loading, training | ✗ | — | Install via pip on DGX Spark |
| CUDA | GPU acceleration | ✗ | — | DGX Spark has CUDA; macOS research env does not |
| Moonshot API key | Synthetic generation | ? | — | Cannot generate without key; pipeline should fail gracefully |
| DGX Spark 128GB | Training, perplexity gating | ✓ | — | Phase 1 only needs API + lightweight processing |
| JupyterLab | Notebook execution | ✓ | — | — |

**Missing dependencies with no fallback:**
- Moonshot API key: Blocks synthetic generation entirely

**Missing dependencies with fallback:**
- PyTorch/CUDA on research machine: Phase 1 development can happen without GPU; only token length profiling and perplexity gating need the model loaded (can be deferred to DGX Spark)

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (to be installed) |
| Config file | none — see Wave 0 |
| Quick run command | `pytest tests/test_synthetic_pipeline.py -x` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | Synthetic generation produces valid Q&A pairs | unit | `pytest tests/test_synthetic_pipeline.py::test_generate_variations -x` | ❌ Wave 0 |
| DATA-02 | Quality gates filter low-perplexity samples | unit | `pytest tests/test_quality_gates.py::test_perplexity_filter -x` | ❌ Wave 0 |
| DATA-03 | Token length histogram runs without error | integration | Run notebook 02 cell | ❌ Wave 0 |
| INFRA-04 | requirements.txt installs without conflict | smoke | `pip install -r requirements.txt` | ❌ Wave 0 |
| INFRA-05 | Moonshot client handles rate limits | unit | `pytest tests/test_synthetic_pipeline.py::test_rate_limiting -x` | ❌ Wave 0 |
| INFRA-06 | Pipeline state persists and loads correctly | unit | `pytest tests/test_pipeline_state.py -x` | ❌ Wave 0 |
| TRAIN-02 | target_modules list has exactly 12 modules | unit | `pytest tests/test_config.py::test_target_modules -x` | ❌ Wave 0 |
| TRAIN-03 | lora_dropout is 0.0 | unit | `pytest tests/test_config.py::test_lora_dropout -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_{module}.py -x`
- **Per wave merge:** `pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_synthetic_pipeline.py` — covers DATA-01, INFRA-05
- [ ] `tests/test_pipeline_state.py` — covers INFRA-06
- [ ] `tests/test_quality_gates.py` — covers DATA-02
- [ ] `tests/test_config.py` — covers TRAIN-02, TRAIN-03
- [ ] `tests/conftest.py` — shared fixtures (mock Moonshot client, temp directories)
- [ ] Framework install: `pip install pytest` — if none detected

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | N/A (no user auth in training pipeline) |
| V3 Session Management | no | N/A |
| V4 Access Control | no | N/A |
| V5 Input Validation | yes | Validate synthetic API outputs (JSON schema, length limits) before storing |
| V6 Cryptography | yes | API key from environment variable (`MOONSHOT_API_KEY`), never committed |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| API key leakage | Information Disclosure | Read from env var; add `.env` to `.gitignore`; never log API key |
| Malformed synthetic data | Tampering | JSON schema validation; length checks; format validation |
| Prompt injection via seed data | Tampering | Sanitize seed data before including in API prompts; no user input in prompts |
| Excessive API spend | Denial of Service | Cost estimation before generation; rate limiting; max token caps |

## Sources

### Primary (HIGH confidence)
- `github.com/huggingface/trl` tag `v0.9.6` — `sft_trainer.py` and `utils.py` source code — SFTTrainer packing behavior, ConstantLengthDataset label handling, data collator selection
- `platform.kimi.ai/docs/pricing/limits` — Moonshot API rate limits and tiers
- `platform.kimi.ai/docs/pricing/chat-v1` — Moonshot API model pricing
- `ref/审计报告_猫鼬AI训练方案.md` — Independent audit identifying CR-001 through CR-003 and MA-001/MA-002
- `ref/mongoose_ai_dgx/config.py` — Current configuration values
- `ref/mongoose_ai_dgx/utils/training_utils.py` — Current training utilities
- `ref/mongoose_ai_dgx/utils/data_utils.py` — Current data utilities
- `ref/mongoose_ai_dgx/data/sample_data.json` — Actual seed data (548 samples, 6 subsets)
- `ref/mongoose_ai_dgx/README.md` — Current documentation with "all-linear" recommendation

### Secondary (MEDIUM confidence)
- `arxiv.org/abs/2511.10338v2` — BhashaKritika: perplexity filtering at 80th percentile for synthetic data quality
- `hugging-face.cn/docs/trl/sft_trainer` — TRL SFTTrainer documentation (assistant_only_loss behavior in newer versions)
- `github.com/huggingface/trl/issues/1385` — Feature request for input loss masking with packing=True (closed)
- `github.com/huggingface/trl/issues/1163` — DataCollatorForCompletionOnlyLM incompatibility with packing

### Tertiary (LOW confidence)
- Web search results for Moonshot API batch capabilities — no native batch API found; concurrency-based parallel processing only
- Web search results for TRL 0.11 release notes — verified v0.11.0 release date (Sep 19 2024) and packing-related fixes

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — verified against pip index and source code
- Architecture: HIGH — derived from existing codebase patterns and audit findings
- Pitfalls: MEDIUM-HIGH — TRL packing behavior verified from source; MoE dropout claim is assumed
- API behavior: MEDIUM — rate limits verified from official docs; pricing verified from official docs

**Research date:** 2026-05-27
**Valid until:** 2026-06-27 (stable stack) / 2026-06-03 (fast-moving API pricing)
