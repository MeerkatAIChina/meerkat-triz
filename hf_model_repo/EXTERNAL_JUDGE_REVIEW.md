# External-Judge Final Review — Meerkat-TRIZ-v1 (v5a vs base, v5 gold, 300 items)

> 🇺🇸 English | 🇨🇳 [中文](EXTERNAL_JUDGE_REVIEW.zh-CN.md)
> Post-release review · July 2026 · raw data: [external_review_result.json](external_review_result.json)

- Protocol: identical to the v5 harness Arm A — anti-verbosity rubric, untruncated
  input, T=0, batch=5, paired bootstrap n=10,000 seed=42. Only the judge changed.
- Judges: claude-sonnet-4-6 (Anthropic), gpt-5.4 (OpenAI), gemini-3.5-flash
  (Google) — three truly external vendors.

| Judge | n | base mean | v1 mean | Paired diff [95% CI] | Sig. | Per-item ρ vs moonshot |
|---|---|---|---|---|---|---|
| moonshot-v1-32k (same-family reference) | 300 | 3.030 | 3.423 | +0.3933 [+0.2967, +0.4900] | yes | — |
| claude-sonnet-4-6 | 299 | 2.555 | 2.649 | +0.0936 [+0.0201, +0.1672] | yes | 0.3054 |
| gpt-5.4 | 299 | 2.301 | 2.405 | +0.1037 [+0.0201, +0.1839] | yes | 0.3002 |
| gemini-3.5-flash | 299 | 2.871 | 2.823 | −0.0485 [−0.1438, +0.0452] | no | 0.2727 |

## Core findings

### 1. Direction and magnitude of the paired difference (v1 vs base)

- **Same-family reference (moonshot)**: +0.3933 [+0.2967, +0.4900] (highly significant)
- **claude-sonnet-4-6**: +0.0936 [+0.0201, +0.1672] — significant, at **~24%** of the moonshot magnitude
- **gpt-5.4**: +0.1037 [+0.0201, +0.1839] — significant, at **~26%** of the moonshot magnitude
- **gemini-3.5-flash**: −0.0485 [−0.1438, +0.0452] — **not significant**, slightly negative

**Verdict.** Under truly external judges the fine-tune's gain shrinks sharply.
Two external judges (Anthropic, OpenAI) preserve the positive direction with
statistical significance, but at roughly one quarter of the same-family reading;
the Google judge observes no significant advantage. The headline +0.39 produced
by the same-family moonshot judge must not be extrapolated across judge
families; the defensible external estimate is **+0.09 ~ +0.10 under two of
three external judges**, with roughly three quarters of the same-family
headline attributable to judge-family effects.

### 2. Cross-judge per-item Spearman agreement (base / v1 arms)

| Judge pair | base arm ρ | v1 arm ρ |
|---|---|---|
| claude-sonnet-4-6 vs gpt-5.4 | **0.7375** | **0.7458** |
| claude-sonnet-4-6 vs gemini-3.5-flash | **0.6280** | **0.7354** |
| gpt-5.4 vs gemini-3.5-flash | **0.6918** | **0.7517** |

The three external judges agree with each other at **0.63–0.75** — moderate-to-high
consistency — while each agrees with moonshot at only **0.27–0.31**. The
divergence is therefore a systematic scoring-scale difference between judge
families, not judge noise. Absolute scales also differ systematically (gemini
scores high overall, gpt-5.4 low).

### 3. Subset-level signals

- **concept_explanation** — the subset the v4.1 rebalance specifically targeted —
  is the strongest cross-judge consensus: **+0.244 / +0.311 / +0.289, significant
  under all three external judges**. The repair is real and externally verified.
- **principle_recommendation**: gemini scores the release model **significantly
  lower** (−0.308 [−0.525, −0.100]); the other two judges see no significant
  difference (−0.083 n.s., +0.050 n.s.). With single-judge support we log this
  as an open signal for the next data iteration, not an established regression.
- Other subsets (ariz_guidance, innovation_assessment positive-leaning;
  case_generation, contradiction_analysis null) vary by judge without
  cross-judge consensus; see the per-judge tables below.

### 4. Execution notes

- One item (v4_gold_028) was blocked by the provider's content filter (403)
  under all judges even after single-item fallback; hence n=299.
- Batch requests that hit a 403 were automatically degraded to per-item
  requests; only genuinely filtered items were marked `__BLOCKED__` and skipped.

## Per-subset differences — claude-sonnet-4-6

| Subset | Diff | 95% CI |
|---|---|---|
| ariz_guidance | +0.2500 | [+0.1167, +0.3833] |
| case_generation | −0.0227 | [−0.1818, +0.1591] |
| concept_explanation | +0.2444 | [+0.0667, +0.4444] |
| contradiction_analysis | +0.0167 | [−0.1833, +0.2167] |
| innovation_assessment | +0.2333 | [+0.1000, +0.4000] |
| principle_recommendation | −0.0833 | [−0.2333, +0.0667] |

## Per-subset differences — gpt-5.4

| Subset | Diff | 95% CI |
|---|---|---|
| ariz_guidance | +0.2000 | [+0.0500, +0.3500] |
| case_generation | −0.0455 | [−0.2045, +0.1136] |
| concept_explanation | +0.3111 | [+0.0444, +0.5778] |
| contradiction_analysis | −0.0333 | [−0.2500, +0.1833] |
| innovation_assessment | +0.2000 | [−0.0333, +0.4333] |
| principle_recommendation | +0.0500 | [−0.1333, +0.2167] |

## Per-subset differences — gemini-3.5-flash

| Subset | Diff | 95% CI |
|---|---|---|
| ariz_guidance | +0.0333 | [−0.1000, +0.1667] |
| case_generation | −0.0909 | [−0.2955, +0.1136] |
| concept_explanation | +0.2889 | [+0.0222, +0.5778] |
| contradiction_analysis | −0.0833 | [−0.3333, +0.1500] |
| innovation_assessment | −0.0667 | [−0.2667, +0.1333] |
| principle_recommendation | −0.3083 | [−0.5250, −0.1000] |
