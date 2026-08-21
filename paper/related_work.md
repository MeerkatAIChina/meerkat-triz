# 相关工作调研与新颖性定位报告(W2 文献定位员)

> 任务:对论文五个候选贡献点(A–E)逐一做文献核查,给出真实可引用文献(标题/作者/年份/venue 或 arXiv 号/一句话相关性),判定真空白 vs 工程报告,给出投稿定位与候选题目。
> 检索方式:WebSearch/arXiv 检索,2026-07-23 执行;所有引用均经检索结果核实,未编造。个别经二级文献转引的条目已显式标注"转引,引用前请核验原文"。
> 项目事实锚点:`results/METRICS_LEDGER.md`(v1/v2 行,第 11–12 行;基座对照表第 19–28 行);`paper/data/report_20260723_024941.json`(eval2 四方报告本地副本,远端原件 `results/eval2/report_20260723_024941.json`);远端 `results/v4_final_report.md`(v4 金标终报);`paper/data/eval_v4_*_gold_*.json`(金标集各模型原始分)。

---

## 0. 总览:五个贡献点的文献地图结构

每个贡献点给出三栏:
- **支持**:与我们的发现同方向、可直接引用作为背书的文献;
- **相反/张力**:结论不同或提示我们的解释可能不唯一的文献;
- **空白**:检索未发现直接先例的部分(新颖性候选)。

---

## A. 混合架构(Gated DeltaNet + 全注意力 + MoE)上的 LoRA 微调与 target_modules 覆盖

**项目事实**:Qwen3.6-35B-A3B(30 层 Gated DeltaNet + 10 层 Gated Attention + MoE);35B 总参数仅 0.24% 可训练;routed expert 主体未被 LoRA 覆盖(背景材料;训练配置见 `pipeline_v4_remote/README.md`)。

### A1 支持
1. **Hu, E. J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022, arXiv:2106.09685.** — LoRA 原始论文;target_modules 选择(只接注意力 vs 接全部投影层)本身就是其消融对象,是我们讨论"覆盖面"的基线引用。
2. **Manzoni, A., et al. "MoE-Sieve: routing-guided expert selection for LoRA fine-tuning of MoE models." arXiv:2603.24044 (2026).** — 直接对口:对 MoE 做 LoRA 时按路由热度只接每层 top-25% 的专家,效果与全量 LoRA 相当(±1pp),参数省 70–73%;且随机选专家差 ~2.5pp。支持"routed expert 的覆盖并非越多越好、路由信号重要"的论点;反向看也说明"完全不接 expert"是文献中的极端基线,值得报告。
3. **"Dynamic Rank LoRA for Fine-Tuning Mixture-of-Experts Models." arXiv:2601.04823 (2026).** — MoE-LoRA 的秩分配问题:不同 expert 需要不同适配容量;支持"MoE 上做 PEFT 不能照搬稠密模型的默认配置"。
4. **Wu, X., Wang, S., Hall, D., Rostamizadeh, N., Rusu, A. "Mixture of LoRA Experts (MoLE)." ICLR 2024, arXiv:2404.13628.** — 把 LoRA 本身组织成专家混合;MoE 模型上 PEFT 的主流替代路线,相关工作必引。
5. **Li, W., Song, Z., et al. "LoRA-Mixer: Coordinate Modular LoRA Experts Through Serial Attention Routing." arXiv:2507.00029 (2025/2026, 审稿中).** — 明确兼容 Transformer 与 SSM/线性注意力("drop-in compatible with Transformers and SSMs"),是少数触及"线性注意力模型上做 LoRA"的工作。

### A2 相反/张力
6. **Yang, S., Wang, B., et al. "Gated Delta Networks: Improving Mamba2 with Delta Rule." ICLR 2025, arXiv:2412.06464.** — Gated DeltaNet 本身的状态更新机制(delta rule + 门控)决定了其"可学习子空间"与 softmax 注意力不同;文献未回答在 GDN 层接 LoRA(q/k/v/b/a 投影)与在注意力层接 LoRA 的等效性——我们 30/10 层配比下的覆盖问题不能直接由稠密 Transformer 文献推断。
7. **Qwen Team. "Qwen3-Next"(技术博客/报告, 2025, qwenlm.github.io/blog/qwen3_next/).** — 官方架构说明(GDN:Gated Attention = 3:1、超稀疏 MoE 数百专家+共享专家);是我们架构描述的一手来源,但不讨论 PEFT 实践。

### A3 空白(新颖性候选)
- **未见先例**:"Gated DeltaNet × Gated Attention × 超稀疏 MoE" 三元混合架构上的 LoRA target_modules 覆盖报告(哪些层/投影接了、routed expert 全未接意味着什么、0.24% 可训练参数的领域适配后果)。MoE-Sieve(2603.24044)是最接近的工作,但它研究"接多少 expert",且未覆盖线性注意力混合架构。**裁决:有真空白,但体量是"实证报告/分析"级,不是方法创新级**——除非补做 target_modules 消融(接 vs 不接 expert/共享专家/GDN 投影),否则只能作为案例发现陈述。

---

## B. 领域 SFT 的维度不均衡效应:整体上升、特定维度显著回退

**项目事实**:v2 overall 0.589(base 0.529,+0.060)但 ariz_completeness −0.139、v1 ARIZ −0.111、principle_accuracy 0.9→0.6(v1)(`results/METRICS_LEDGER.md` 第 11–12 行);eval2 四方报告:v1/v2 的 principle_accuracy、ariz_step_coverage 显著低于 base(`paper/data/report_20260723_024941.json`)。

### B1 支持
8. **Barnett, S., et al. "Seven Failure Points When Engineering a Retrieval Augmented Generation System." CAIN 2024, arXiv:2401.05856.** — 案例研究式报告:领域适配(含微调选项)在部分子项改善、部分子项恶化的混合结果;与本项目"整体分掩盖维度回退"同构,是"工程案例研究也有发表价值"的直接先例(被引 383 次)。
9. **Lin, J., et al. "SFT Doesn't Always Hurt General Capabilities: Revisiting Domain-Specific Fine-Tuning in LLMs." arXiv:2509.20758 (2025).** — 系统重估"领域 SFT 必然伤害通用能力"的说法,指出退化高度依赖学习率与配置;支持"维度回退不是必然、是可控变量"的框架。
10. **Biderman, D., et al. "LoRA Learns Less and Forgets Less." TMLR 2024, arXiv:2405.09673.** — LoRA 比全参微调学得少也忘得少;提示我们的 v1 大幅回退(principle 0.9→0.6)在 LoRA 设定下相对反常,更可能源于数据质量问题(无质量门)而非 PEFT 本身——可作归因讨论的支点。
11. **Ding, F., et al. "Improved Supervised Fine-Tuning for Large Language Models to Mitigate Catastrophic Forgetting." arXiv:2506.09428 (2025).** — 遗忘缓解方法线,供相关工作完整性引用。
12. **Ovadia, O., et al. "Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs." EMNLP 2024, arXiv:2312.05934.** — 领域知识注入上微调弱于 RAG 的知名负结果(经二级文献转引确认其存在,引用前请核验细节);支撑"小语料 SFT 对知识型维度收益有限"的讨论。

### B2 相反/张力
13. **Lin et al. 2025(同 #9)的另一面**:其核心结论是"SFT 不总是伤通用能力"——与我们 eval2 中 general_probe 基本无显著遗忘(v2 除外)一致,但这反而削弱了"维度回退=遗忘"的叙事:我们观察到的 ARIZ/原理回退是**目标领域内子维度**的回退,不是通用能力遗忘,现有文献框架(遗忘/对齐税)并不直接覆盖。

### B3 空白(新颖性候选)
- **未见先例**:在**同一领域内**将评测拆成多个子维度(原理识别/矛盾求解/案例覆盖/ARIZ 步骤完整性),报告"整体分上升 + 特定子维度显著回退"并做定向补数据干预(v3 ariz boost)的闭环案例。TRIZ/ARIZ 这类**有显式过程结构**的领域(步骤覆盖率可测)为此提供了罕见的可操作维度——通用 benchmark 的子任务分解文献(如 MMLU 子学科)不做干预闭环。**裁决:真空白,但需把故事重心放在"维度级评测如何改变训练决策"而非"维度回退现象本身"。**

---

## C. 关键词/表面匹配指标 vs LLM-as-a-Judge 的结论分歧与同源 judge 风险

**项目事实**:eval2 中 v3 vs v2 关键词轨 +0.0556(显著)而 judge 轨 −0.0682(显著),两轨结论相反(`paper/data/report_20260723_024941.json`);数据由 Moonshot 生成、judge 亦为 moonshot-v1-32k(远端 `results/v4_final_report.md`)。

### C1 支持
14. **Zheng, L., et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." NeurIPS 2023 Datasets & Benchmarks, arXiv:2306.05685.** — 奠基文献:首次系统命名位置偏差、冗长偏差、自我增强偏差;我们 judge 协议设计(与偏差讨论)的必引。
15. **Liu, Y., et al. "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment." EMNLP 2023, arXiv:2303.16634.** — rubric 化 judge 的代表;支持我们用结构化评分维度。
16. **Gu, J., et al. "A Survey on LLM-as-a-Judge." arXiv:2411.15594 (2024)** 与 **Li, H., et al. "From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge." arXiv:2411.16594 (2024)** — 两篇综述,覆盖偏差缓解(多评委、顺序交换、rubric、人类抽检)。
17. **Callison-Burch, C., Osborne, M., Koehn, P. "Re-evaluating the Role of BLEU in Machine Translation Research." EACL 2006.** — 经典先例:n-gram 指标上升既不保证也不需要质量提升——"表面匹配轨与判断轨结论相反"在 MT 史上早有记载,我们的案例是 LLM 时代的新实例。
18. **Panickssery, A., Bowman, S. R., Feng, S. "LLM Evaluators Recognize and Favor Their Own Generations." NeurIPS 2024, arXiv:2404.13076.** — 自我识别→自我偏好的因果证据;支撑我们对"同源 judge"风险的担忧。
19. **Wataoka, K., Takahashi, T., Ri, R. "Self-Preference Bias in LLM-as-a-Judge." arXiv:2410.21819 (2024).** — 把自我偏好归因于困惑度熟悉性(family-level 偏差机制),直接对应"Moonshot 造数据、Moonshot 当评委"。
20. **Saito, K., et al. "Verbosity Bias in Preference Labeling by Large Language Models." arXiv:2310.10076 (2023).** — 冗长偏差专文;v4 回答更长更结构化,judge 分 +1.00 中可能含冗长成分,需引用并缓解。
21. **Shankar, S., et al. "Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences." UIST 2024, arXiv:2404.12272.** — "criteria drift"与评委验证的必要性;支撑 judge 可靠性实验(建议补充实验 E1)。

### C2 相反/张力
22. **"Quantifying and Mitigating Self-Preference Bias of LLM Judges." arXiv:2604.22891 (2025/2026).** — 自我偏好偏差可量化且部分可缓解;但也指出 win-rate 无法区分"能力"与"偏差"——意味着我们的 judge 差分 +1.00 不能简单归因于偏差,需混合评委验证。
23. **Spiliopoulou, E., et al. (2025, 经 2603.04582 转引,引用前请核验原文)** — 发现偏差扩展到"同家族/架构相似系统",说明换用非同族 judge 家族(如 GPT/Qwen/Claude 混合评委)是有效缓解,但并不能完全消除共享文本表面信号偏差。

### C3 空白(新颖性候选)
- 两轨分歧本身(#17 有经典先例)、judge 偏差(#14–21 已饱和)都不是空白。**真正的增量**在于:**同一模型家族既生产训练数据又充当评测 judge 的"同源闭环"在领域微调迭代中的具体量化后果**——以及"当两轨结论相反时,决策门(保留 v2)应听谁的"这一可操作判据。**裁决:无方法级空白;作为案例发现 + 缓解协议(混合评委/位置交换/翻转率)是有价值的实证贡献,但必须以新增实验(E1 judge 可靠性包)支撑,否则只是现象陈述。**

---

## D. 评测驱动开发(Evaluation-Driven Development)与统计严谨性

**项目事实**:100 题金标集;配对 bootstrap CI(judge 差 +1.00 [0.80,1.19])、McNemar p=1.7e-12(远端 `results/v4_final_report.md`);四版本迭代由评测结果直接驱动训练决策(v1 失败→v2 多角度→v3 定向→v4 质量门)。

### D1 支持
24. **Miller, E. "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations." arXiv:2411.00640 (2024, Anthropic;被引 92 次).** — 把评测题视为超总体样本,给出配对差分、聚类 bootstrap、功效分析的完整公式;我们统计协议的理论依据,必引。
25. **Pacchiardi, L., Cheke, L. G., Hernández-Orallo, J. "100 Instances Is All You Need: Predicting LLM Success by Testing on a Few Instances." 2025(经 2509.22506 参考文献确认;venue/arXiv 号引用前请核验).** — 直接回应"100 题金标集够不够":小 n 评测在特定条件下有预测力,但需配合功效声明。
26. **Madaan, L., et al. "Quantifying Variance in Evaluation Benchmarks." NeurIPS 2024 D&B, arXiv:2406.10229.** — 评测方差分解;支撑小 n 噪声讨论。
27. **Dror, R., et al. "The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing." ACL 2018.** — NLP 显著性检验的经典指南;McNemar/bootstrap 使用规范依据。
28. **Dietterich, T. G. "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms." Neural Computation 1998** 与 **McNemar, Q. Psychometrika 1947** — McNemar 检验的方法学出处(本项目 v4 终报直接使用)。
29. **"Eval-Driven Development for LLM Systems." Timeless Research (tmls.nyc/research/eval-driven-development, 2026)** 与 **Husain, H. "Your AI Product Needs Evals" (hamel.dev, 2024 博客)** — EDD 作为工程范式的代表性论述(非同行评审);说明"评测驱动"在工业界已是显学。

### D2 相反/张力
- EDD 在工程圈已高度普及(#29),"我们用了评测驱动迭代"本身不构成学术贡献;学术界更认 #24–27 的统计协议严谨性。

### D3 空白(新颖性候选)
- **空白不在 EDD 概念,而在"学术级 EDD 案例"**:把四个训练版本 × 双轨指标 × 决策门(发布/回滚判据)完整记录、并配齐配对 bootstrap/McNemar/功效分析的**可复现迭代史**,文献中少见(Barnett et al. 2024 是工程失败点案例但无训练迭代统计;Miller 2024 给方法不给领域案例)。**裁决:真空白在"完整案例 + 统计纪律"的组合;单点(EDD 概念、统计方法)均无空白。这是本文最适合的主故事线。**

---

## E. 桌面级硬件(DGX Spark GB10 统一内存)上 35B 级 MoE 的 BF16 LoRA

**项目事实**:DGX Spark GB10,121GB 统一内存;BF16 LoRA(非 4-bit);峰值显存 64.7→65.1 GB,吞吐 15–18 tok/s(`results/METRICS_LEDGER.md` 第 11–12 行及基座对照表)。

### E1 支持
30. **Dettmers, T., et al. "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023, arXiv:2305.14314.** — NF4 + 双重量化使 65B 可在单卡微调;是"低成本适配"的主流路线,我们的 BF16 路线与之对照(统一内存下省量化开销、保数值精度)。
31. **kreuzhofer/dgx-spark-unsloth-qwen3.5-training(GitHub, 2026)** — **直接先例:单台 DGX Spark 上 BF16 LoRA 微调 Qwen3.5-35B-A3B**(同族架构);证明硬件可行性,也说明我们的硬件可行性声明并非首创。
32. **NvMayMay/nvfp4-lora-spark(GitHub, 2026)** — GB10 上 NVFP4-aware LoRA 训练 120B 级 MoE,NVFP4 loss 1.00 vs BF16 0.98;为 BF16-vs-低比特对照提供社区数据点。
33. **awesome-dgx-spark(bidual/awesome-dgx-spark, GitHub)** 与 NVIDIA DGX Spark/GB10 官方论坛容量讨论(单台 128GB:约 210B PEFT(4-bit base)/14B 全参)——社区生态与官方口径的一手材料。
34. **Yang, S., et al. "Delta Networks." arXiv:2406.06484 (2024)** 与 **Raschka, S. "The Big LLM Architecture Comparison"/Gated DeltaNet 图解(2026 博客)** — 混合架构训练特性的背景材料(fla 库 triton kernel 对 GB10 的适配是实践要点)。

### E2 相反/张力
- QLoRA/bnb 路线在消费级 GPU 上的地位牢固(#30);我们的卖点不是"更省显存",而是"统一内存下 BF16 免量化、免 kernel 适配摩擦、可复现"——社区已有同族先例(#31),**纯可行性报告价值有限**。

### E3 空白(新颖性候选)
- **基本无学术空白**:GB10 属新硬件(2025 末发布),学术文献几乎为零,但社区工程报告(#31–33)已覆盖"能跑"。**裁决:工程报告级;作为论文的"可复现性/民主化适配"卖点融入系统描述即可,不宜独立作为贡献点,除非补做 BF16 vs QLoRA/NVFP4 的同任务对照实验。**

---

## 6. 新颖性裁决总表

| 贡献点 | 裁决 | 依据 |
|---|---|---|
| A 混合架构 LoRA 覆盖 | **半空白**:无同架构先例,但属实证报告级;需 target_modules 消融升级 | MoE-Sieve(2603.24044)最接近但未覆盖 GDN 混合架构 |
| B 维度不均衡 + 定向干预闭环 | **真空白(推荐主线之一)**:领域内子维度回退 + 补数据干预的闭环案例未见 | Barnett 2024 有混合结果无干预;Lin 2025 谈通用能力非领域内子维度 |
| C 双轨分歧 + 同源 judge | **无单点空白**;作为"决策门听哪轨"的实证 + 缓解协议有价值,需补 E1 实验 | BLEU 史(#17)、judge 偏差(#14–21)文献饱和 |
| D EDD 完整案例 + 统计纪律 | **真空白(推荐主线)**:"四版本迭代史 × 双轨指标 × 决策门 × 配对 bootstrap/McNemar/功效"的组合案例未见 | Miller 2024 给方法无案例;EDD 工程文献无学术统计 |
| E GB10 边缘训练 | **工程报告**:社区已有同族 BF16 LoRA 先例(kreuzhofer 2026) | 只作系统卖点,不独立成贡献 |

**综合定位:B + D 为论文主贡献,A、C 为支撑性发现,E 为可复现性卖点。**

---

## 7. 推荐投稿定位

1. **首选:ACL/EMNLP/NAACL 的 workshop(如 NLP4DevEval、Eval4NLP、SEAL、ACL 2027 Industry Track)** — 理由:案例研究 + 评测方法学 + 统计纪律的组合正是评测类 workshop 的标准口味;Barnett 2024(CAIN workshop,被引 383)证明此类工程案例有高影响力先例;正文长度与实验规模匹配。
2. **次选:TMLR(Transactions on Machine Learning Research)** — 理由:TMLR 明确欢迎"empirical findings / negative results / reproducibility studies"(Biderman et al. 2024 即发表于此);无需抢占方法新颖性,适合"诚实报告 + 严格统计"的迭代史;审稿周期可控。
3. **中文路线(若需学位/项目考核产出):CCF-B/C 类中文期刊或《中文信息学报》《计算机研究与发展》** — 理由:TRIZ 领域中文语料 + 中文 judge 评测对中文 NLP 社区有独立价值;但需补中文 baseline 对比,否则建议作为英文主线的副产品。
4. **不建议:ACL/EMNLP 主会长文(当前形态)** — 缺方法创新与大规模基准;除非补做 A 的 target_modules 消融 + C 的多评委可靠性包 + B 的干预消融,把案例升级为"方法 + 基准"方可一试。

---

## 8. 候选论文题目(3 个,中英)与故事线

### 题目 1(推荐,D+B 主线)
- 中:**《评测驱动的领域微调:35B 混合架构模型在 TRIZ 领域的四版本迭代、双轨评测与统计纪律》**
- 英:**"Evaluation-Driven Domain Fine-Tuning: A Four-Version Iteration of a 35B Hybrid-Architecture Model on TRIZ with Dual-Track Metrics and Statistical Discipline"**
- 故事线:以"每次训练决策都由带置信区间的评测驱动"为骨架——v1 无质量门整体回退(McNemar/bootstrap 证显著)→ v2 多角度数据整体上升但 ARIZ 维度回退(维度级评测暴露)→ v3 定向补数据出现关键词/judge 两轨结论相反(决策门听哪轨)→ v4 干净管线 judge +1.00 [0.80,1.19] 但 concept_explanation 显著退化,决策门判"保留 v2"。卖点:完整可复现迭代史 + Miller 2024 式统计协议 + 桌面级硬件复现包。

### 题目 2(B 主线,偏现象发现)
- 中:**《整体分的假象:领域 SFT 的子维度回退现象与定向数据干预——来自 TRIZ 领域 35B 模型微调的证据》**
- 英:**"The Aggregate-Score Illusion: Sub-Dimension Regression in Domain SFT and Targeted Data Intervention — Evidence from Fine-Tuning a 35B Model on TRIZ"**
- 故事线:聚合指标掩盖子维度回退(principle 0.9→0.6、ARIZ −0.139 而 overall +0.060);提出"领域内多维评测 + 决策门"的最小协议;用 v3(定向 ARIZ 数据)展示干预的部分成功与意外代价(judge 轨恶化),引出"干预也有维度代价"的开放问题。

### 题目 3(C 主线,偏评测方法学)
- 中:**《当关键词说好、评委说坏:领域微调中双轨评测分歧的案例研究与同源评委风险》**
- 英:**"When Keywords Say Better and the Judge Says Worse: Divergent Evaluation Tracks in Domain Fine-Tuning and the Risk of Same-Family Judges"**
- 故事线:v3 vs v2 两轨显著反向为引子,系统拆解三种解释(指标效度、judge 冗长/位置偏差、Moonshot 同源数据+同源 judge 的自我偏好),给缓解协议(混合评委家族、位置交换、重复翻转率、人类抽检 10%);回扣 BLEU 史(Callison-Burch 2006)说明这是老问题在 LLM 时代的新形态。**注意:此题必须等补充实验 E1(judge 可靠性包)完成后方可投。**

---

## 附录:全部引用清单(核验状态)

已直接核验(arXiv/期刊页面或检索结果中确认标题与编号):#1 arXiv:2106.09685;#2 arXiv:2603.24044;#3 arXiv:2601.04823;#5 arXiv:2507.00029;#6 arXiv:2412.06464;#8 arXiv:2401.05856;#9 arXiv:2509.20758;#10 arXiv:2405.09673;#11 arXiv:2506.09428;#14 arXiv:2306.05685;#15 arXiv:2303.16634;#16 arXiv:2411.15594 / 2411.16594;#17 EACL 2006;#18 arXiv:2404.13076;#19 arXiv:2410.21819;#20 arXiv:2310.10076;#21 arXiv:2404.12272;#22 arXiv:2604.22891;#24 arXiv:2411.00640;#26 arXiv:2406.10229;#28 Dietterich 1998 / McNemar 1947;#30 arXiv:2305.14314;#31–33 GitHub 仓库与 NVIDIA 论坛(非学术引用,标注为社区资料);#34 arXiv:2406.06484。TRIZ 领域:#35 AutoTRIZ: "Artificial Ideation with TRIZ and Large Language Models"(Jiang & Luo 等, arXiv:2403.13002, 2024;期刊版 Computer-Aided Design,引用前核验卷期);#36 TRIZ-GPT(arXiv:2408.05897, 2024);#37 Gusarov et al., "Integrating TRIZ with LLMs… Materials Science at the Molecular Scale"(ChemEngineering, MDPI, 2026, 10(4):54)——**三篇均为"LLM+TRIZ"但均为提示工程/框架集成,无微调先例:TRIZ 微调本身是空白点,建议在各题目引言中引用以确立领域新颖性。**
转引待核验:#4 MoLE(ICLR 2024, arXiv:2404.13628);#12 Ovadia et al.(EMNLP 2024, arXiv:2312.05934);#23 Spiliopoulou et al. 2025;#25 Pacchiardi et al. 2025;#27 Dror et al. ACL 2018;#29 Husain 博客。
