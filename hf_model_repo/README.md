---
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
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

# Meerkat-TRIZ-v1

A TRIZ (Theory of Inventive Problem Solving) domain LoRA adapter on
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (Apache-2.0),
for Chinese TRIZ question answering across six task types: inventive-principle
recommendation, contradiction analysis, ARIZ guidance, innovation assessment,
concept explanation, and case generation.

The focus of this release is **evaluation transparency**: every claim is
produced by a rigorously gated dual-track harness
([GitHub: meerkat-triz](https://github.com/coidea-sys/meerkat-triz)),
with paired statistics and confidence intervals — ties are reported as ties.

## Quick start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.6-35B-A3B", dtype=torch.bfloat16,
    device_map="cuda", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3.6-35B-A3B", trust_remote_code=True)
model = PeftModel.from_pretrained(base, "Meerkat-AI/Meerkat-TRIZ-v1").eval()

# IMPORTANT: keep the empty think block — required for train/eval format consistency (E0 rule)
prompt = tok.apply_chat_template(
    [{"role": "system", "content": "你是 TRIZ 创新方法论专家助手, 用中文专业回答用户关于 TRIZ 理论、发明原理、矛盾分析、ARIZ 算法等方面的问题。"},
     {"role": "user", "content": "请解释技术矛盾与物理矛盾的区别, 并各给一例。"}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False)
out = model.generate(**tok(prompt, return_tensors="pt").to("cuda"),
                     max_new_tokens=2048, do_sample=False,
                     pad_token_id=tok.eos_token_id)
print(tok.decode(out[0][tok(prompt, return_tensors="pt")["input_ids"].shape[1]:],
                 skip_special_tokens=True))
```

> ⚠️ **Prompt-format sensitive**: this model was trained and evaluated with the
> empty think block (`<think>\n\n</think>`) retained in the prompt. If your
> template strips it, output quality may degrade (we measured judge-score
> drops on the order of −0.2 from this mismatch alone). The base is
> thinking-native: with `enable_thinking=False` the template still emits the
> empty block — keep it.

## Evaluation results

Two gold sets, dual-track scoring (keyword hit rate + LLM judge 0–4; judge
pinned to moonshot-v1-32k, T=0). All comparisons are per-item paired
(paired bootstrap, 10000 draws, 95% CI + McNemar). The judge shares a family
with the training-data generator (weak same-origin); the lineage is declared —
and was **quantified by a post-release external-judge final review** (see
below): direction confirmed, magnitude ~4× smaller than the same-family
reading.

### v5 gold (300 items, in-distribution eval protocol)

Judge-track numbers in this table are **same-family (Moonshot) readings**.

| Metric | base | **Meerkat-TRIZ-v1** | Paired diff [95% CI] |
|---|---|---|---|
| Keyword-track mean | 0.6384 | **0.6383** | −0.0001 [−0.017, +0.017] n.s. |
| Judge Arm-A mean | 3.0300 | **3.4233** | **+0.3933 [+0.297, +0.490] sig.** |
| Judge pass rate | 0.843 [0.798, 0.880] | 0.947 [0.915, 0.967] | McNemar p=1.5e-05 |
| Keyword pass rate | 0.737 [0.684, 0.783] | 0.747 [0.695, 0.793] | — |

Judge per-subset diffs: concept_explanation **+0.733** [+0.467, +1.022],
innovation_assessment +0.600, contradiction_analysis +0.467,
case_generation +0.378, ariz_guidance +0.317 — all significantly positive;
principle_recommendation +0.050 n.s.
Quality gates passed 299/300 (1 length-gate invalid, counted);
overrefusal 0/300, passed.

### External-judge final review (post-release, 299 paired items)

The Arm-A protocol re-run verbatim with three external judges
([full report](https://github.com/coidea-sys/meerkat-triz/blob/main/docs/EXTERNAL_JUDGE_REVIEW.md)):

| Judge | Paired diff [95% CI] | Sig. |
|---|---|---|
| claude-sonnet-4-6 (Anthropic) | **+0.094 [+0.020, +0.167]** | yes |
| gpt-5.4 (OpenAI) | **+0.104 [+0.020, +0.184]** | yes |
| gemini-3.5-flash (Google) | −0.048 [−0.144, +0.045] | no |

- The defensible headline: **+0.09 ~ +0.10, significant under two of three
  external judges**; the same-family +0.39 is inflated ~4× by judge-family
  effects.
- External judges agree with each other at per-item Spearman 0.63–0.75, but
  only 0.27–0.31 with the same-family judge — a scoring-system difference,
  not judge noise.
- **concept_explanation is the strongest consensus**: +0.24 / +0.31 / +0.29,
  significant under all three external judges.
- Watch item: gemini scores principle_recommendation significantly lower
  (−0.31); single-judge support — logged as an open signal, not an
  established regression.

### v4 gold (100 items, cross-protocol six-way comparison)

All historical candidates on the same items, judge, and harness:

| Model | Keyword track | Judge track | Judge pass rate |
|---|---|---|---|
| base (legacy anchor, think-contaminated, reference only) | 0.3661 | 1.5700 | 0.120 |
| v2 (clean anchor) | 0.5483 | 2.5800 | 0.620 |
| v3 | 0.5716 | 2.2800 | 0.430 |
| v4 | 0.5568 | 2.5700 | 0.630 |
| v4.1 | 0.5289 | 2.6200 | 0.630 |
| **Meerkat-TRIZ-v1 (v5a)** | 0.5391 | 2.3600 | 0.600 |

Paired vs the v2 clean anchor: keyword −0.0093 [−0.062, +0.039] n.s.;
judge −0.2200 [−0.470, +0.010] n.s. — **statistically tied with v2 on the v4
protocol**. Note that the v4 harness strips the empty think block (a format
mismatch with this model's training format), which explains part of the
negative judge direction (a lower-bound estimate). Keyword per-subset:
contradiction_analysis +0.101 [+0.024, +0.180] significantly positive;
case_generation −0.137 [−0.270, −0.030] significantly negative.

### Claim discipline

- ✅ Claimable: on the v5 protocol, the judge track improves over the clean
  base — **+0.09 ~ +0.10, significant under two of three external judges**
  (Claude, GPT; Gemini n.s.), externally verified post-release. The
  same-family judge reads +0.39; do not extrapolate it across judge families.
- ✅ Claimable: the targeted concept_explanation repair is significant under
  all three external judges (+0.24 / +0.31 / +0.29).
- ✅ Claimable: on the v4 protocol, tied with the strongest internal
  baseline v2 (paired CIs include 0).
- ❌ Not claimable: superiority over any external model (no external
  comparison was run); keyword-track improvement over base (−0.0001, none);
  the same-family +0.39 as a judge-family-independent effect size.

## Training

- Method: LoRA SFT, r=64, alpha=128, dropout=0, targeting all linear
  projections (incl. DeltaNet `in_proj_*`); BF16; full run of 4 epochs /
  5548 steps; final train loss 0.604, eval loss 1.626 (trajectory in
  `adapter_info.json`).
- Data: ~11k Chinese TRIZ ChatML SFT samples (dual-style / dual-decomposition
  / grouped split; empty think block retained). **Training set is not public**:
  derived from third-party copyrighted TRIZ textbooks.
- Contamination check: 3-gram Jaccard ≥0.5 scan of the training set against
  both gold sets — **0 hits**.
- Frozen hyperparameters: see `configs/train_v5a.json` in the GitHub repo.

## Limitations

- Validated only for Chinese TRIZ-domain QA; out-of-domain ability is
  unevaluated. Not for high-stakes decisions.
- Long answers (>2048 tokens) may be truncated; greedy decoding is recommended
  (training and evaluation used greedy).
- The primary judge is a single vendor (Moonshot), weak same-origin. The
  post-release external review (above) confirms the improvement's direction
  under two of three external judges at ~1/4 the magnitude; same-family
  absolute scores should not be extrapolated across judge families. One
  external judge detects a significant principle_recommendation decrease —
  an open signal under investigation, single-judge support only.
- The base is a vision-language MoE; this adapter was validated on the text
  channel only.

## Citation

```bibtex
@misc{meerkat-triz-v1-2026,
  title  = {Meerkat-TRIZ-v1: A TRIZ-Domain LoRA Fine-tune of Qwen3.6-35B-A3B},
  author = {Meerkat AI},
  year   = {2026},
  url    = {https://huggingface.co/Meerkat-AI/Meerkat-TRIZ-v1}
}
```

Please also cite the base model Qwen3.6-35B-A3B (Apache-2.0,
[license](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE)).
