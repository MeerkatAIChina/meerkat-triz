# X/Twitter 长串（英文, 10 推）

1/ 🚀 Releasing Meerkat-TRIZ-v1 — to our knowledge, the first LLM fine-tune for the TRIZ (inventive problem solving) domain.

LoRA on Qwen3.6-35B-A3B (181MB, 0.24% params), Apache-2.0.

But the model is only half the story. The other half: our evaluation caught itself lying. 🧵

2/ What is TRIZ?

Distilled from hundreds of thousands of patents by Altshuller (1946+): innovation isn't random genius — the same technical contradictions recur across industries, and solutions follow reusable patterns.

40 numbered inventive principles. Contradiction matrix. ARIZ algorithm. Used by Samsung, Siemens, GE.

3/ Why fine-tune for TRIZ?

TRIZ is an operating system for innovation — 40 numbered principles, standardized contradiction analysis, the ARIZ algorithm. Today it's distributed via certified consultants (thousands, worldwide) while engineers who need it number in millions.

Fine-tuning turns methodology from a per-day service into zero-marginal-cost infrastructure.

4/ The results — all of them:

✅ Judge track: +0.09~+0.10 over base, significant under 2 of 3 external judges (Claude/GPT; Gemini n.s.) — post-release review, protocol re-run verbatim
⚠️ Same-family judge reads +0.39 — ~4× judge-family inflation, NOT portable across judge families
✅ concept_explanation (the targeted repair): significant under ALL three external judges
➖ Keyword track: −0.0001, dead parity
➖ On the older protocol: statistically TIED with our best internal baseline

No external-model comparisons run, none claimed.

5/ Failure #1: the baseline was lying.

For 3 versions we reported ~+1.00 judge gain over base. False — our harness stripped the empty think block the thinking-native base needs. Base emitted unterminated reasoning drafts on 91/100 items. We were scoring drafts.

Fixed: the gain REVERSED to −0.30.

6/ Failure #2: judge position bias at 2× the literature amplitude.

Our pairwise judge picked whichever candidate came second 81% of the time (0.87 inconsistency rate; lit reports ~25pp swings).

Single-order pairwise = noise. Dual-order merging is now enforced in code.

7/ Failure #3: dual-track divergence.

Keyword hit-rate and LLM-judge share ~10% variance. Once they disagreed significantly in OPPOSITE directions — v3 stuffed keywords while semantic quality fell.

Rule: release gates must be dual-track. No single-track verdict ships.

8/ Bonus finding: at temperature 0, our judge was fully deterministic — 0.000 flip rate across repeats (vs 13.6% reported elsewhere).

That's what makes a pinned-judge, single-run protocol defensible.

9/ Everything is open:

🤖 Model: https://huggingface.co/Meerkat-AI/Meerkat-TRIZ-v1
📊 Benchmark: https://huggingface.co/datasets/Meerkat-AI/triz-gold-benchmark
🛠️ Harness (pip install, CI'd): https://github.com/coidea-sys/meerkat-triz
📄 Whitepaper (EN/中文 + PDF)

10/ The honest summary of what fine-tuning did:

Early versions: verbose think-style → direct answers at 1/10 length, parity keywords, slightly below verbose base on judge.

v1 release: first to turn that compression into a real judge-track gain on its native protocol — a gain that survived re-judging by three external models (2/3 significant) — still tied with v2 on the older one.

Both facts are on the model card.

---

# 中文社交媒体版（即刻/微博/知乎想法, 单条）

发布了 Meerkat-TRIZ-v1——据我们所知 TRIZ（发明问题解决理论）领域首个 LLM 微调模型：Qwen3.6-35B-A3B 上的 LoRA（181MB，0.24% 参数），Apache-2.0。

但模型只是一半的故事。另一半：我们的评测系统抓到了自己说谎——
① 报告的"+1.00 judge 提升"其实是基线被 prompt 渲染坑成了未闭合草稿，修复后反转为 −0.30；
② pairwise 评委 81% 选第二位候选（文献摆动两倍），单序结论全是噪声；
③ 关键词轨和 judge 轨曾显著反向，从此发布门必须双轨。

结果全部如实：judge 轨提升经发布后异源终审确认——三个外部评委（Claude/GPT/Gemini）中两个显著（+0.09~+0.10），同族评委读数 +0.39 约有 3/4 是评委家族效应、不可外推；定向修复的概念讲解子集在三个外部评委下全部显著。关键词轨持平；旧口径与内部最强基线打平。未做外部模型对比，不做任何超额宣称。

模型/基准/harness/双语白皮书全部开源，链接见评论。

---

# LinkedIn 版（英文, 单条, 较正式)

TRIZ — the Theory of Inventive Problem Solving, distilled from hundreds of thousands of patents since 1946 — is the most structured innovation methodology ever built, yet it still reaches engineers almost exclusively through expensive training courses and certified consultants numbering in the low thousands worldwide.

We're releasing Meerkat-TRIZ-v1 — to our knowledge, the first published LLM fine-tune for the TRIZ (Theory of Inventive Problem Solving) domain: a LoRA adapter on Qwen3.6-35B-A3B (181MB, 0.24% trainable parameters, Apache-2.0), for Chinese TRIZ question answering.

The more durable contribution is the evaluation infrastructure. During a five-version iteration campaign, our evaluation caught three measurement failures that each reversed a conclusion:

• A contaminated baseline that inflated the fine-tune's apparent judge gain from +1.00 to a true −0.30 (the harness stripped the empty think block the thinking-native base requires — 91/100 base responses were unterminated drafts).
• Judge position bias at twice the literature amplitude (0.87 inconsistency rate; dual-order merging is now the only valid protocol).
• Systematic divergence between keyword and judge tracks (they once disagreed significantly in opposite directions; release gates are now dual-track by construction).

Results, stated fully: on its native 300-item protocol the model improves judge-track quality over base — confirmed post-release by an external-judge final review that re-ran the protocol verbatim: +0.09 ~ +0.10, significant under two of three external judges (Claude Sonnet, GPT; Gemini n.s.). The same-family judge reads +0.39 [+0.30, +0.49]; roughly three quarters of that figure is attributable to judge-family effects and should not be extrapolated. The targeted concept-explanation repair is significant under all three external judges. Keyword coverage is at parity, and on the older 100-item protocol the model is statistically tied with our strongest internal baseline. No external-model comparisons were run, and none are claimed.

Model, benchmark, evaluation harness (pip-installable, CI-tested), and a bilingual whitepaper are all public:
Model: https://huggingface.co/Meerkat-AI/Meerkat-TRIZ-v1
Code & whitepaper: https://github.com/coidea-sys/meerkat-triz
