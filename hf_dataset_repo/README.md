---
license: cc-by-nc-4.0
task_categories:
  - text-generation
language:
  - zh
tags:
  - triz
  - evaluation
  - benchmark
  - inventive-problem-solving
size_categories:
  - n<1K
---

> 🇺🇸 English | 🇨🇳 [中文](README.zh-CN.md)

# triz-gold-benchmark

A Chinese TRIZ (Theory of Inventive Problem Solving) evaluation benchmark,
companion to [Meerkat-TRIZ-v1](https://huggingface.co/Meerkat-AI/Meerkat-TRIZ-v1)
and the [meerkat-triz](https://github.com/coidea-sys/meerkat-triz) evaluation
harness.

## Contents

| File | Items | Protocol |
|---|---|---|
| `triz_gold_v4_public.jsonl` | 100 | v4 evaluation protocol (six-way comparison) |
| `triz_gold_v5_public.jsonl` | 300 | v5 evaluation protocol (official release eval) |

One JSON object per line:

```json
{"id": "v5_gold_000", "subset": "ariz_guidance", "question": "...", "keywords": ["..."]}
```

**Six task subsets** (v4 / v5 item counts):

| subset | v4 | v5 |
|---|---|---|
| ariz_guidance | 20 | 60 |
| case_generation | 15 | 45 |
| concept_explanation | 15 | 45 |
| contradiction_analysis | 20 | 60 |
| innovation_assessment | 10 | 30 |
| principle_recommendation | 20 | 60 |

## Important: this public release contains no reference answers

The questions and expected keywords were LLM-generated from third-party TRIZ
textbooks and course materials; the reference answers are derivative rewrites
of copyrighted content. To control copyright risk, **this public release only
publishes question + keywords + subset** — no `reference_answer` field.

Implications:

- **The keyword track is fully reproducible** (the harness keyword track only
  needs question/keywords).
- **The judge track loses its reference-answer anchor**: in the judge prompt
  the reference answer is the quality anchor; without it, absolute scores are
  not directly comparable to the official reports (paired diffs are less
  affected).
- For the full version for academic research, please contact the authors via
  a repo issue.

## Usage

```bash
meerkat-eval --config configs/eval_v5.json \
    --adapter-path <adapter> --tag my_run \
    --eval-file triz_gold_v5_public.jsonl \
    --baseline-results <base result.json>
```

## Leakage statement

This benchmark never entered the training distribution of Meerkat-TRIZ-v1:
a 3-gram Jaccard ≥0.5 scan of the training set against both gold sets found
**0 hits**.

## License

CC-BY-NC-4.0 (non-commercial research use). The question texts are derived
from third-party copyrighted TRIZ materials; contact the authors before any
commercial use.
