---
license: apache-2.0
base_model: Qwen3.8-27B
library_name: peft
pipeline_tag: text-generation
tags:
  - lora
  - triz
  - inventive-problem-solving
  - chinese
  - evaluation-harness
language:
  - zh
---

> 🇺🇸 English | 🇨🇳 [中文](README.zh-CN.md)

# Meerkat-TRIZ-v1-Qwen3.8-27B

A TRIZ (Theory of Inventive Problem Solving) domain LoRA adapter on the
**Qwen3.8-27B** base model (qwen3_5 hybrid linear-attention multimodal
architecture, Apache-2.0), for Chinese TRIZ question answering across six
task types: inventive-principle recommendation, contradiction analysis,
ARIZ guidance, innovation assessment, concept explanation, and case
generation.

This is the first Meerkat-TRIZ adapter on the **Qwen3.8-27B** base — a
base-model migration from the earlier
[Meerkat-TRIZ-v1](https://huggingface.co/Meerkat-AI/Meerkat-TRIZ-v1)
(which targets Qwen3.6-35B-A3B). The focus of this release, as always, is
**evaluation transparency**: every claim is produced by a rigorously gated
dual-track harness ([GitHub: meerkat-triz](https://github.com/coidea-sys/meerkat-triz)),
with paired statistics, confidence intervals, and a cross-family external-judge
final review.

> ⚠️ **Base model availability**: this adapter is a LoRA delta on top of
> **Qwen3.8-27B**. The base weights are **not hosted in this repo** and, at the
> time of release, have **no public HuggingFace model ID** — obtain the
> Qwen3.8-27B base separately (see `adapter_config.json →
> base_model_name_or_path`) and point `PeftModel.from_pretrained` at it, or
> use the local-path loading shown below.

## Quick start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Qwen3.8-27B base — replace with the local path or your own copy of the base
BASE = "/path/to/Qwen3.8-27B"

base = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = PeftModel.from_pretrained(base, "Meerkat-AI/Meerkat-TRIZ-v1-Qwen3.8-27B").eval()

# IMPORTANT: keep the empty think block — required for train/eval format consistency
prompt = tok.apply_chat_template(
    [{"role": "system", "content": "你是 TRIZ 创新方法论专家助手, 用中文专业回答用户关于 TRIZ 理论、发明原理、矛盾分析、ARIZ 算法等方面的问题。"},
     {"role": "user", "content": "请解释技术矛盾与物理矛盾的区别, 并各给一例。"}],
    tokenize=False, add_generation_prompt=True)
out = model.generate(**tok(prompt, return_tensors="pt").to("cuda"),
                     max_new_tokens=2048, do_sample=False,
                     pad_token_id=tok.eos_token_id)
print(tok.decode(out[0][tok(prompt, return_tensors="pt")["input_ids"].shape[1]:],
                 skip_special_tokens=True))
```

## Evaluation results

v5 gold set (300 items), dual-track scoring (keyword hit rate + LLM judge 0–4;
judge pinned to moonshot-v1-32k, T=0). All comparisons are per-item paired
(paired bootstrap, 10000 draws, 95% CI). The judge shares a family with the
training-data generator (weak same-origin); the lineage is declared and was
**quantified by a cross-family external-judge review** (below).

### Same-family (Moonshot) reading

| Metric | base (Qwen3.8-27B) | **adapter** | Paired diff [95% CI] |
|---|---|---|---|
| Keyword-track mean | 0.6245 | **0.6236** | −0.0009 [−0.019, +0.017] n.s. |
| Judge Arm-A mean | 2.9300 | **3.5333** | **+0.6033 [+0.497, +0.713] sig.** |
| Judge pass rate | 0.787 [0.737, 0.829] | 0.960 [0.931, 0.977] | McNemar p=3.1e-12 |

Quality gates passed 300/300 (0 invalid); overrefusal 0/300, passed.

### Cross-family external-judge review (300 paired items, verbatim Arm-A protocol)

Three external judges, run after training, on the **same** generated responses:

| Judge | Paired diff [95% CI] | Sig. (raw) |
|---|---|---|
| claude-sonnet-4-6 (Anthropic) | +0.031 [−0.071, +0.129] | no |
| gpt-5.4 (OpenAI) | +0.094 [+0.010, +0.175] | yes (marginal) |
| gemini-3.5-flash (Google) | +0.107 [+0.013, +0.205] | yes (marginal) |

**After noise accounting** (the tensoris-gateway external judges have measured
T=0 flip rates 0.18 / 0.68 / 0.78; per-question rerun variance propagated into
the CI): **none of the three cross-family judges remains statistically
significant** (gpt −0.004…+0.192, gemini −0.015…+0.229, claude n.s.).

### Honest summary

- The same-family (Moonshot) judge reading of **+0.60 is inflated ~6×** by
  judge-family effects. The cross-family reading is **+0.03 to +0.11 — a
  positive direction, but not statistically significant** once judge
  non-determinism is accounted for.
- The keyword track is statistically tied with the base (−0.0009).
- **Net: on this base-model migration, the adapter is statistically tied with
  the Qwen3.8-27B base on the judge track under cross-family judges.** This
  release is provided for transparency and reproducibility; it does **not**
  establish a quality advantage over the base.
- Cross-family judges agree with the same-family judge at per-item Spearman
  0.28–0.30 only — a scoring-system difference, not mere noise.

## Training

- Method: LoRA SFT, r=64, alpha=128, dropout=0, targeting all linear
  projections (incl. DeltaNet `in_proj_*`); BF16 (no quantization).
- 4 epochs configured; **early-stopping triggered at step 1600** (1.15 epochs);
  best checkpoint at step 1300, eval loss **1.5058** (completion-only).
  Trajectory: 1.618 → 1.587 → 1.554 → 1.506 (min) → 1.538 (overfit tail).
- Data: ~11k Chinese TRIZ SFT samples (`v5_train_v5a.jsonl`, 11,096 train /
  1,050 val). **Training set is not public**: derived from third-party
  copyrighted TRIZ textbooks.
- Optimizer: adamw_torch, LR 2e-4, cosine horizon 2774 steps (2-epoch design).
- Frozen hyperparameters: see `configs/train_v6_qwen38.json` in the GitHub repo.

## Limitations

- Validated only for Chinese TRIZ-domain QA; out-of-domain ability is
  unevaluated (general benchmarks not yet run — a known gap).
- Long answers (>2048 tokens) may be truncated; greedy decoding recommended.
- The primary judge is a single vendor (Moonshot), weak same-origin; the
  cross-family review above shows the improvement is **not significant** after
  noise accounting — do not cite +0.60 as a family-independent effect.
- The base is a multimodal (vision-tower) model; this adapter was trained and
  validated on the text (language_model) channel only.
- Base-model availability: see the warning at the top; the Qwen3.8-27B base
  weights are not distributed in this repo.

## Citation

```bibtex
@misc{meerkat-triz-v1-qwen38-2026,
  title  = {Meerkat-TRIZ-v1-Qwen3.8-27B: A TRIZ-Domain LoRA Fine-tune of Qwen3.8-27B},
  author = {Meerkat AI},
  year   = {2026},
  url    = {https://huggingface.co/Meerkat-AI/Meerkat-TRIZ-v1-Qwen3.8-27B}
}
```

Please also cite the Qwen3.8-27B base model per its own license.
