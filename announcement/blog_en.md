# Meerkat-TRIZ-v1: The First TRIZ-Domain LLM Fine-tune — and the Evaluation Infrastructure That Kept It Honest

*Meerkat AI · July 2026*

## What is TRIZ?

TRIZ (Russian acronym for the *Theory of Inventive Problem Solving*) is a
methodology distilled by Genrich Altshuller and his colleagues starting in
1946, from the analysis of hundreds of thousands of patents. Its core insight:
**innovation is not random genius** — the same technical problems recur
across industries, and their solutions follow reusable patterns. A problem
solved in aerospace in 1975 may be the same contradiction a battery engineer
faces today.

Its toolbox is unusually structured for a "creativity" method: **40 numbered
inventive principles** and the contradiction matrix for resolving technical
contradictions (improving one parameter without degrading another);
**separation principles** for physical contradictions; **ARIZ**, a full
step-by-step algorithm for reframing and solving hard problems; and the laws
of technical-system evolution for roadmap thinking. Samsung, Siemens, GE and
Boeing have trained thousands of engineers on it; in China, TRIZ is part of
the national innovation-method promotion program.

The catch: this methodology lives almost entirely inside expensive training
courses and certified consultants.

## Why fine-tune for TRIZ at all?

TRIZ is not scattered knowledge — it's an operating system for innovation:
40 numbered inventive principles, a standardized contradiction-analysis
workflow, and a full algorithm (ARIZ). Today this methodology is distributed
almost entirely through certified training and consulting: a company's TRIZ
capability roughly equals the number of certified experts it can afford, and
those experts number in the low thousands worldwide while the engineers who
could use the methodology number in the millions.

Prior LLM+TRIZ work was all prompt engineering. Fine-tuning buys three things
prompts can't: **terminology discipline** (correct principle numbering and
analysis steps, baked into weights rather than begged for in prompts),
**behavioral style** (direct, executable answers instead of verbose
chain-of-thought — our fine-tunes compress base responses to ~1/10 length at
parity coverage), and **internalized capability** (the difference between
consulting a book and knowing the method, most visible on generation tasks).

A fine-tuned model turns methodology application from a per-day-billed service
into infrastructure with zero marginal cost. That is the bet of this release.

## What a TRIZ model means for enterprise innovation

The bottleneck in enterprise innovation is rarely ideas — it is **problem
formulation**. Most engineering teams get stuck long before ideation: they
can't name the contradiction they're actually facing. That is precisely what
TRIZ is for, and precisely where a resident model helps:

- **At the moment of need.** Methodology decays after training courses; a
  model on every engineer's desk answers "which principles apply to my
  contradiction?" at 2am before the design review, not during next quarter's
  workshop.
- **At the entry level, at scale.** Our strongest task types are
  principle_recommendation and concept_explanation — exactly the onboarding
  tier. The cost of "getting every engineer methodologically literate" drops
  from a training-program budget line to an API call.
- **As the drafting layer for experts.** Contradiction-analysis and ARIZ
  reports drafted by the model, reviewed by consultants — expert time shifts
  from production to judgment.

To be explicit about the boundary: this is an assistant, not a replacement.
Complex innovation programs still need human consultants. What changes is the
floor — every engineer gets a methodology-literate colleague, and experts get
leverage.

Today we're releasing three things:

- 🤖 **[Meerkat-TRIZ-v1](https://huggingface.co/Meerkat-AI/Meerkat-TRIZ-v1)** — a LoRA adapter (181MB, 0.24% trainable params) that fine-tunes Qwen3.6-35B-A3B for Chinese TRIZ (Theory of Inventive Problem Solving) question answering. To our knowledge, the first published LLM fine-tune for the TRIZ domain.
- 📊 **[triz-gold-benchmark](https://huggingface.co/datasets/Meerkat-AI/triz-gold-benchmark)** — a dual gold-set benchmark (100 + 300 items) across six TRIZ task types.
- 🛠️ **[meerkat-triz](https://github.com/coidea-sys/meerkat-triz)** — the evaluation harness we built along the way, with every lesson below enforced as a machine-checked assertion. Plus a [bilingual whitepaper](https://github.com/coidea-sys/meerkat-triz/blob/main/docs/WHITEPAPER.md) ([中文](https://github.com/coidea-sys/meerkat-triz/blob/main/docs/WHITEPAPER.zh-CN.md)).

## The results — all of them, including the tie

On its native v5 gold protocol (300 items, paired statistics):

- **Judge track: +0.09 ~ +0.10 over base — significant under two of three external judges** (Claude Sonnet +0.094 [+0.020, +0.167], GPT +0.104 [+0.020, +0.184]; Gemini −0.048 n.s.), from a post-release final review that re-ran our judge protocol verbatim with three external vendors. Our same-family judge reads **+0.39** [+0.30, +0.49] — roughly three quarters of that headline is judge-family effect and must not be extrapolated.
- **The biggest gainer is the one we targeted: concept explanation, +0.24 / +0.31 / +0.29 — significant under all three external judges.** (Same-family reading: +0.73.)
- **Keyword track: −0.0001 — dead parity.** The model covers exactly the same expected terminology as base.
- On the older v4 protocol (100 items), Meerkat-TRIZ-v1 is **statistically tied** with our strongest internal baseline. We say that on the model card, in the whitepaper, and here.

We ran no external-model comparisons and claim none. The external review above varies the *judge*, not the compared models — [full report](https://github.com/coidea-sys/meerkat-triz/blob/main/docs/EXTERNAL_JUDGE_REVIEW.md).

## Why you should read the evaluation story, not just the model card

The more useful contribution of this project is what the evaluation caught while we were iterating. Three failures, each capable of reversing a conclusion:

**1. A baseline that was lying (+1.00 → −0.30).** For three versions, our reports showed the fine-tunes beating base by roughly a full judge point. False. Our harness stripped the empty think block that the thinking-native base model needs; deprived of its "thinking has ended" anchor, the base emitted unterminated English reasoning drafts on 91/100 gold items. Both scoring tracks were grading drafts. Fixed, the v4-vs-base difference reversed sign to **−0.30**. If you evaluate thinking-native models, check your prompt rendering before you check anything else.

**2. Judge position bias at 2× the literature amplitude.** Our pairwise judge picked the second-presented candidate 81% of the time — a 0.87 position-inconsistency rate versus the ~25pp swings reported in the literature. Any single-order pairwise conclusion in this setting is noise. Dual-order merging is now the only valid protocol, enforced in code. (Separately: at temperature 0, our judge was *fully deterministic* across repeats — flip rate 0.000 — which is what makes a pinned-judge, single-run protocol defensible.)

**3. Two tracks voting opposite ways.** Keyword hit-rate and LLM-judge scores share only ~10% variance. During iteration they once disagreed significantly in *opposite directions*: v3 looked better on keywords (+0.056, significant) and worse on the judge (−0.068, significant). It had learned to stuff keywords while semantic quality fell. Rule that survived: **release gates must be dual-track; no single-track verdict ships.**

**4. (Quantified post-release) The judge was family.** We had declared the judge/data-generator family overlap as a limitation — but declarations don't calibrate numbers. So after release we re-ran the entire judge protocol with three external judges. Direction survived (2/3 significant); magnitude did not: the external estimate is roughly one quarter of the same-family +0.39. The three external judges agree with each other at Spearman 0.63–0.75, but only 0.27–0.31 with our in-family judge — a systematic scoring-scale difference, not noise. If your judge shares a family with your data generator, budget for this before you headline a number.

## What fine-tuning actually did

Under the clean anchor, the honest story of the early fine-tunes was behavioral compression: verbose think-style answers (~3,250 chars) became direct answers at ~1/10 the length, at parity keyword coverage, with judge-track quality slightly *below* the verbose base. Meerkat-TRIZ-v1 is the first version to turn that compression into a genuine judge-track gain on its native protocol — while remaining tied with v2 on the older one. Both facts are in the model card.

## Limitations, stated

Single domain (Chinese TRIZ QA). Single base. Single PEFT config. Judge shares a family with the data generator (declared weak same-origin; post-release external review confirmed direction under 2/3 external judges at ~1/4 magnitude — same-family scores are not portable across judge families; one external judge flags a principle_recommendation decrease, single-judge support, under investigation). 100-item subsets are descriptive only. The public benchmark withholds reference answers for copyright reasons — the keyword track is fully reproducible, judge absolute scores are not.

## Get started

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base, "Meerkat-AI/Meerkat-TRIZ-v1")
# ⚠️ Keep the empty think block in your prompt template — the model
# was trained with it retained; stripping it measurably hurts output.
```

- Model: https://huggingface.co/Meerkat-AI/Meerkat-TRIZ-v1
- Benchmark: https://huggingface.co/datasets/Meerkat-AI/triz-gold-benchmark
- Code + harness + CI: https://github.com/coidea-sys/meerkat-triz
- Whitepaper (EN / 中文, PDF available): https://github.com/coidea-sys/meerkat-triz/tree/main/docs

*If your evaluation infrastructure has ever told you a story you wanted to hear — this whitepaper is for you.*
