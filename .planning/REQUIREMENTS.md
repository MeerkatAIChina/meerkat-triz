# Requirements: v1.1

**Status:** ACTIVE
**Created:** 2026-07-20 (post-retrospective, see `docs/training_retrospective_2026-07-20.md`)
**Supersedes:** `.planning/milestones/v1.0-REQUIREMENTS.md` (archived, SHIPPED)

v1.0 shipped a runnable pipeline but no quantified evidence that training helped. v1.1 requirements are therefore **measurable acceptance criteria**, not "artifact exists" criteria. Every requirement below states a threshold, a measurement method, and the evidence artifact that proves it.

---

## Evaluation Requirements (Effectiveness Evidence)

- [ ] **EVAL-11 — Layer 2 absolute score + delta vs base.** The v1.1 adapter must beat the Qwen3.6-35B-A3B base model on the TRIZ benchmark (currently 40 questions: 30 hardcoded + up to 10 dynamic) by a pre-registered margin: principle-recognition accuracy ≥ base + 15 percentage points, and every other Layer 2 sub-metric ≥ base (no regression). Both runs in the same process, FP16, temperature 0.0, fixed seed. Evidence: `results/eval_<run>_<ts>.json` with base and adapter scores side by side.

- [ ] **EVAL-12 — Base-model control eval_loss measured BEFORE v2 training.** The base model's eval_loss on the same 313-sample validation split used by the 2026-06-19 run must be measured and recorded **before** any v2 training starts, so that adapter eval_loss has a control. Evidence: entry in `results/METRICS_LEDGER.md` + raw JSON in `results/`.

- [ ] **EVAL-13 — Human review rubric (release gate).** N = 20–30 real consulting cases, blind-scored 1–5 by a TRIZ expert (base vs adapter, order randomized). Release requires ≥80% expert approval (score ≥ 4, or adapter preferred in pairwise comparison). No public release or client-facing demo without this gate passing. Evidence: rubric scoresheet archived in `results/`.

- [ ] **EVAL-14 — Layer 1 post-training eval is mandatory.** After every training run, the Layer 1 suite (MMLU-Pro, GPQA, HumanEval, MATH, BBH) must run against the adapter; general-capability loss vs the base-model baseline must be < 3%. A run without post-training Layer 1 numbers is incomplete by definition. Evidence: `results/layer1_<run>_<ts>.json`.

- [ ] **EVAL-15 — Hallucination detection set.** Add a hallucination evaluation set of 20–40 questions (fabricated principles, nonexistent contradiction pairs, trick citations). Hallucination rate must be ≤ 8% (aligned with the technical plan). Evidence: `results/hallucination_<run>_<ts>.json`.

## Data Requirements

- [ ] **DATA-11 — Per-version dataset manifest.** Every training-set version must ship a manifest (see `data/MANIFEST.md` template) recording: sample counts per split, subset distribution, dedup & filtering statistics, generation script + parameters, source-corpus hash, generation date, and the training run that consumed it. No manifest → the dataset must not be trained on.

- [ ] **DATA-12 — V2 quality gates.** The V2 corpus-SFT dataset must pass: enforced subset quotas (no subset starved, e.g. ARIZ), dedup enabled, min-length filter, empty-`<think>`-block cleaning, and ≥ 2% human spot-check of generated samples with the pass rate recorded in the manifest.

## Process Requirements (Hard Rules)

- [ ] **PROC-11 — Deviations require requirement-text revision.** If a requirement cannot be met as written, the requirement text in this file must be revised (with date and rationale) **before** the item is checked off. "Record the deviation and check the box anyway" is forbidden — it voids the meaning of the checklist (precedent: DATA-02's 20–30% real-data target silently shipping at ~8.7% in v1.0).

---

## Traceability

| Requirement | Origin (retrospective section) |
|---|---|
| EVAL-11 / EVAL-12 | §2.4 eval_loss 无对照无阈值；§3.3 训练后评测从未执行 |
| EVAL-13 | §2.1 无人工评审 rubric；§3.3 双评机制未实现 |
| EVAL-14 | §2.1 通用能力损失 <3% 零验证 |
| EVAL-15 | §3.3 缺幻觉检测集；技术方案幻觉率 ≤8% |
| DATA-11 / DATA-12 | §3.1 训练集无 manifest、V1 零质量门 |
| PROC-11 | §2.3 目标静默降级先例 |
