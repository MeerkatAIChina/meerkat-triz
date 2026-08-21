# 论文骨架 — Meerkat-AI TRIZ 领域微调:评测驱动迭代与测量学发现

> 主笔:主笔_Writer ｜ 日期:2026-07-24 ｜ 状态:骨架定稿(数字以干净 base 锚点为准;占位标记见 §8)
> 目标 venue:TMLR 或 ACL/EMNLP 评测类 workshop(依据 `related_work.md` §7)
> 写作纪律:本骨架中所有数字均带来源标注;凡干净 base 臂补跑未完成的数字一律用占位标记,**不得提前编数**。

---

## 1. 候选题目(3 个,中英)与推荐

> 注意:`related_work.md` §8 的三个候选题成稿于 E0 修复**之前**,故事线已随干净锚点结果反转(原"judge +1.00 提升"被证伪)。以下三个题目按 E0 终稿后的诚实叙事重拟。

### 题目 1(★ 推荐,方法学案例主线)
- 中:**《当基线在说谎:思考型模型领域微调评测中的测量污染、评委位置偏差与双轨分歧——一项 TRIZ 领域 35B 模型四版本迭代的案例研究》**
- 英:**"When the Baseline Lies: Measurement Contamination, Judge Position Bias, and Dual-Track Divergence in Evaluating Domain Fine-Tuning — A Four-Version Case Study of a 35B Hybrid-Architecture Model on TRIZ"**
- 理由:三个方法学发现(think 污染 / 极端位置偏差 / 双轨分歧)是本文最强、最新颖、最经得起审稿的部分(E0/E1a/E3 证据链完整);领域微调本身是负结果,不宜作为卖点,但作为案例载体完全合格。契合 TMLR "empirical findings / negative results / reproducibility" 口味(`related_work.md` §7.2)。

### 题目 2(评测驱动迭代主线)
- 中:**《评测驱动的领域微调迭代:四版本、双轨指标与统计纪律——35B 混合架构模型在 TRIZ 领域的完整迭代史》**
- 英:**"Evaluation-Driven Domain Fine-Tuning: Four Versions, Dual-Track Metrics, and Statistical Discipline — A Complete Iteration History of a 35B Hybrid Model on TRIZ"**
- 理由:对应 W2 裁决的真空白 D(学术级 EDD 案例 + 统计纪律,`related_work.md` §6)。风险:E0 修复后,"迭代史"的高潮是"发现自家评测在骗人",故该故事必须包含题目 1 的内容,单独成题反而削弱了最强发现。

### 题目 3(评测方法学专刊向)
- 中:**《中文专业领域 LLM-as-a-Judge 的可靠性解剖:位置偏差、确定性与关键词轨分歧的量化研究》**
- 英:**"Dissecting LLM-as-a-Judge Reliability in a Chinese Technical Domain: Quantifying Position Bias, Determinism, and Keyword-Track Divergence"**
- 理由:把 E1a/E1b/E1c/E3 独立成篇,彻底方法学化;代价是丢弃领域微调案例与 EDD 故事,且训练部分降为"实验材料"。适合 Eval4NLP 类 workshop 短文,不适合承载完整项目。

**推荐:题目 1。** 它同时容纳 B/D 两个真空白贡献点(作为案例背景)与三个方法学发现(作为主体),且对负结果完全诚实。

---

## 2. 摘要(中英双语,各 ≤250 词)

### 中文摘要

领域微调的迭代决策完全依赖评测信号,而评测信号本身可能被污染。本文报告对 Qwen3.6-35B-A3B(Gated DeltaNet + Gated Attention + MoE 混合架构)进行 TRIZ 创新方法领域 LoRA 微调的四版本(v1–v4)评测驱动迭代中,发现并修复的三类测量学问题。其一,**基座锚点污染**:评测 harness 剥离了 thinking-native 基座渲染出的空 think 块,导致基座 91/100 题输出为未闭合英文思考草稿,两轨指标评的都是草稿;修复后,"微调相对基座 judge +1.00 [0.80,1.19] 的提升"反转为 **−0.30 [−0.46,−0.14] 显著为负**,原提升被证实为测量伪影(E0)。其二,**评委位置偏差极端化**:pairwise 评委的位置不一致率达 **0.87**[0.79,0.92],B 位胜率 0.81,摆动幅度为文献报告(约 25pp)的两倍多;双序合并是唯一有效口径(E1a)。其三,**关键词轨与评委轨系统性分歧**:两轨仅共享约 10% 方差,迭代中曾出现两轨显著反向(v3 vs v2:关键词 +0.0556 显著 / judge −0.0682 显著);ARIZ 步骤覆盖的关键词漏判率达 31.7%(E3)。干净锚点下,微调的真实效应是:回答从长篇 think 式作答压缩至约 1/10 长度的直接作答,关键词覆盖持平,judge 轨质量略低于冗长的基座(−0.29~−0.30/4);v4 与 v2 等效,决策门保留 v2。我们给出全部迭代决策、统计协议(配对 bootstrap / McNemar / 功效分析)与修复方案,为思考型模型的领域评测提供一份可复现的避坑案例。

### English Abstract

Iterative domain fine-tuning relies entirely on evaluation signals, yet those signals themselves can be contaminated. We report three measurement problems discovered and repaired during a four-version (v1–v4) evaluation-driven LoRA fine-tuning campaign of Qwen3.6-35B-A3B — a hybrid architecture (Gated DeltaNet + Gated Attention + MoE) — on the TRIZ innovation methodology domain. First, **baseline contamination**: the evaluation harness stripped the empty think block rendered for the thinking-native base model, causing 91/100 base responses to be unterminated English reasoning drafts; both keyword and LLM-judge tracks scored the drafts. After repair, the apparent "+1.00 [0.80, 1.19] judge improvement over base" **reversed to −0.30 [−0.46, −0.14]**, exposing the gain as a measurement artifact (E0). Second, **extreme position bias**: a pairwise judge exhibited a position-inconsistency rate of **0.87** [0.79, 0.92] with the second-position candidate winning 81% — over twice the ~25pp swing reported in the literature; dual-order averaging is the only valid protocol (E1a). Third, **systematic dual-track divergence**: keyword and judge tracks share only ~10% variance; during iteration they once disagreed significantly in opposite directions (v3 vs v2: keyword +0.0556 ✅ / judge −0.0682 ✅), and keyword scoring under-detected ARIZ step coverage at a 31.7% miss rate (E3). Under the clean anchor, fine-tuning's true effect is behavioral: answers compress to ~1/10 length with direct-answer style at parity keyword coverage, while judge-track quality sits slightly below the verbose base (−0.29~−0.30/4); v4 ≈ v2, and the release gate retains v2. We release the full decision history, statistical protocol (paired bootstrap / McNemar / power analysis), and fixes as a reproducible case study for evaluating thinking-native models.

---

## 3. 贡献列表(5 条)

1. **发现并修复一类"思考型模型评测污染"**:harness 在 `apply_chat_template(enable_thinking=False)` 后剥离空 think 块,会使 thinking-native 基座失去"思考已结束"锚点而自吐未闭合英文 think 草稿(100/100 自吐、91/100 未闭合),两轨指标全部失真;给出诊断矩阵(4 种干预策略冒烟)、修复方案(保留空 think 块 + 生成后剥离 + 中文占比校验)与修复前后反号对比(+1.00 → −0.30)。(来源:`experiments/e0/E0_report.md` §1–4)
2. **LLM-judge 位置偏差在中文专业域评测中的极端量化**:位置不一致率 0.87 [0.79, 0.92]、硬翻转率 0.77、B 位总胜率 0.81,远超文献约 25pp 的摆动;确立"pairwise 必须双序平均"为唯一有效口径(双序合并 v4 胜率 0.49 ≈ 平局)。(来源:`experiments/e1/e1_report.md` E1a;文献摆动 25pp 见 `eval_methodology_research.md` §2.2)
3. **关键词轨 vs judge 轨双轨分歧的完整案例史与操作判据**:两轨逐题 Spearman 仅 r≈0.32(共享 ~10% 方差);eval2 中 v3 vs v2 两轨显著反向(kw +0.0556 [0.0010,0.1099] ✅ / judge −0.0682 [−0.1078,−0.0296] ✅);ARIZ 关键词漏判率 31.7%(19/60),rubric 重判后"v2 ARIZ 回退"消失;据此提出"发布决策门必须双轨、单轨结论不独立下判"的操作判据。(来源:`stats_review.md` §3、§5;`evidence_table.md` ②-C;`experiments/e3/e3_report.md`)
4. **一份统计纪律完整的领域微调负结果迭代史**:四版本 × 双轨指标 × 决策门(发布/回滚判据)× 配对 bootstrap(10,000 次)/ McNemar 精确检验 / Wilson CI / 功效分析(MDE 表)全程留痕;干净锚点下微调的真实效应为回答压缩至 ~1/10 长度、kw 持平、judge −0.29~−0.30 显著为负,v4≈v2,决策门保留 v2——诚实报告"微调未带来绝对质量提升"。(来源:`evidence_table.md` ①②;`stats_review.md` §6;`experiments/e0/E0_report.md` §4)
5. **TRIZ 领域首个 LLM 微调报告 + 混合架构 LoRA 实证**:TRIZ 微调本身无先例(既有 LLM+TRIZ 工作均为提示工程/框架集成,`related_work.md` 附录 #35–37);首次报告 Gated DeltaNet × Gated Attention × 超稀疏 MoE 三元混合架构上的 LoRA target_modules 覆盖(12 模块、0.24% 可训练参数、routed expert 未覆盖)与桌面级硬件(DGX Spark GB10 统一内存、BF16 免量化)训练可行性。(来源:`related_work.md` A3/E3 裁决;`evidence_table.md` ①)

---

## 4. 章节骨架(二级标题,含核心论点 / 证据 / 图表计划)

### 1. Introduction
- 1.1 动机:领域微调迭代 = 一连串由评测信号驱动的决策;信号失真则决策全错。
  - 核心论点:本文以 TRIZ 领域四版本迭代为案例,展示三类测量失真如何被发现、量化与修复。
  - 证据:v1→v4 决策链(`evidence_table.md` ①;`docs/training_retrospective_2026-07-20.md`)。
- 1.2 贡献概览(映射 §3 五条)。
- 1.3 路线图。
- 图表:**图 1 迭代时间线**(v1→v4 训练/评测/事故/修复事件轴,含全零 lora_B 事故与 E0 修复;素材:`evidence_table.md` 附表 + §⑤-5)。

### 2. Related Work(组织方案见 §5)
### 3. Setting: System, Data, and Training
- 3.1 基座与硬件:Qwen3.6-35B-A3B 混合架构(30 层 GDN + 10 层 Gated Attention + MoE 40 层 / 256 专家);DGX Spark GB10 121GB 统一内存,BF16 免量化。
  - 证据:`evidence_table.md` ① 末行 + 表体;社区先例 `related_work.md` #31–33。
- 3.2 LoRA 配置与覆盖:12 个 target_modules、84.66M/34.7B=0.24% 可训练、routed expert 未覆盖。
  - 证据:`evidence_table.md` ①;`docs/training_retrospective_2026-07-20.md`。
- 3.3 数据管线演进:v1 零质量门(2,662 条)→ v2 基础门(8,458)→ v3 定向 ARIZ boost(8,963)→ v4 七道质量门漏斗(11,001 → 5,739)。
  - 证据:`evidence_table.md` ④(v4 漏斗逐门计数);v1 缺陷 `docs/training_retrospective_2026-07-20.md` 三.1。
- 3.4 评测体系:三套评测(修复版四连 / eval2 / v4 金标 100 题)+ 双轨指标定义 + 三套体系不可互比的口径声明。
  - 证据:`evidence_table.md` ②、⑤-9;**表 1 版本总表**(数据×质量门×超参×时长,源自 ①)。

### 4. Evaluation Methodology and Statistical Protocol
- 4.1 双轨指标:关键词覆盖(表层词汇对齐)vs rubric 化 LLM-judge(0–4,四维);两轨测不同构念(共享 ~10% 方差)。
  - 证据:`stats_review.md` §3.1。
- 4.2 统计协议:配对 bootstrap 10,000 次(seed=42)、McNemar 精确双侧、Wilson CI;功效分析与 MDE 表(n=100 仅可分辨 ~10pp 级差异;子集 n=15~20 一律描述性)。
  - 证据:`stats_review.md` §1、§6;**表 3 功效/MDE 表**(源自 §6.1–6.3)。
- 4.3 judge 可靠性设计:位置交换双跑、跨评委、重复翻转率——引出 §6。

### 5. Finding 1: The Contaminated Baseline(E0)
- 5.1 现象:base 91/100 未闭合英文 think 草稿;judge 可见文本仅 ~43%。
  - 证据:`stats_review.md` §2.1;`experiments/e0/E0_report.md` §1。
- 5.2 根因:harness 剥空 think 块致 thinking-native 基座失锚;诊断矩阵(4 策略冒烟,仅"保留空 think 块"成功)。
  - 证据:`experiments/e0/E0_report.md` §2。
- 5.3 修复与后果:修复后四方对比;**v4−base judge 差 +1.00 → −0.30 反号**;kw 差 +0.19 → −0.007 ns;McNemar 55/4 → 10/30 反方向显著。
  - 证据:`experiments/e0/E0_report.md` §4.1–4.3;**表 2 干净锚点四方对比**(base_goldfix / v2 / v4 / base_polluted × 双轨均值 + pass 率 Wilson CI,源自 §4.2);**图 4 修复前后差值对比图**(v4−base 两轨差值污染 vs 干净并排)。
- 5.4 讨论:judge 输入截断 1500 字符的不对称在新方向下反而**压低** base 分,base 真实优势可能更大。
  - 证据:`experiments/e0/E0_report.md` §4.5-1。

### 6. Finding 2: Judge Reliability Anatomy(E1a/E1b/E1c)
- 6.1 极端位置偏差(E1a):不一致率 0.87、B 位胜率 0.81、双序合并 0.49 ≈ 平局;单序 pairwise 结论一律无效。
  - 证据:`experiments/e1/e1_report.md` E1a;**图 3 位置偏差图**(AB 序胜率 0.18 vs BA 序 0.80 对比 + 文献 25pp 参考线)。
  - 该实验 base 臂原取自污染缓存;干净 base 臂(E1a',v4 vs base_goldfix)已补跑:双序合并 v4 胜率 **0.260 [0.204, 0.325]**——v4 显著输给干净 base(对照:污染臂合并胜率 0.49 ≈ 平局);位置不一致率 **0.58**、B 位胜率 0.65,第二位锚定机制仍在但幅度减弱(污染臂为 0.87/0.81)。来源:`experiments/e1/e_goldfix_report.json`;`experiments/e0/E0_report.md` §5。
- 6.2 跨评委一致性(E1b):32k vs 8k Spearman ρ=0.945、符号一致率 0.94、排序一致;声明"同族弱异源"局限(kimi-k2 404 兜底)。
  - 证据:`experiments/e1/e1_report.md` E1b;`experiments/STATE.md`。干净 base 臂(E1b 补臂)已回填:base_goldfix 均分 2.87 > v2 2.58 ≈ v4 2.57;v4−base:moonshot-v1-32k **−0.30 [−0.46,−0.14] 显著为负**、moonshot-v1-8k −0.11,符号一致率 0.85;v2−base 32k −0.29 ✅,一致率 0.82;干净锚点下跨评委逐题 Spearman ρ=0.759。
- 6.3 确定性与翻转率(E1c):T=0 翻转率 **0.000**(API 完全确定,独立可发表发现);T=0.7 翻转率 0.015,远低于文献 13.6%;k=3 多数表决一致率 0.988。
  - 证据:`experiments/e1/e1_report.md` E1c;文献值 `eval_methodology_research.md` §2.2;**表 4 judge 可靠性包汇总**(E1a/b/c 三行:指标、数值、CI、文献对照)。
- 6.4 综合:本任务的 judge 纪律清单(T=0 单次即可 / pairwise 必须双序 / 同族评委须声明)。

### 7. Finding 3: Dual-Track Divergence and Metric Validity(eval2 + E3 + E2)
- 7.1 两轨分歧案例:v3 vs v2 关键词 +0.0556 ✅ 与 judge −0.0682 ✅ 结论相反;逐题分歧率 31–41%;v3"仅 kw 过"高达 31 题 → "学会堆关键词、语义质量反降"。
  - 证据:`evidence_table.md` ②-C;`stats_review.md` §3.2;**图 2 双轨分歧图**(eval2 v3−v2 两轨差值森林图 + gold 两轨散点四象限)。
- 7.2 ARIZ 关键词漏判(E3):漏判率 31.7%(19/60)且几乎全为"英文/同义表述被漏判";rubric 轨 v4 0.883 ≈ v2 0.875 > base(草稿)0.675;"v2 ARIZ 回退"系关键词伪影。
  - 证据:`experiments/e3/e3_report.md`(漏判表述清单);`stats_review.md` §5。base 干净臂(E3')已回填:rubric 轨 base_goldfix **0.800** / v2 0.875 / v4 0.883;**v4 vs 干净 base +0.083 [0.033, 0.142] 显著**,v2 vs base +0.075 [−0.025, 0.15] 不显著;干净 base 关键词漏判仅 3/20(草稿臂为 19/60=31.7%)——v4 的 ARIZ 语义优势在干净锚点下依然显著,是最稳健的正面发现。
- 7.3 concept_explanation 退化归因(E2):非噪声也非同义伪影——v4 真缺失工具术语言名(功能分析×3 等 8 词次,synonym 伪影 0 条),但 judge 持平 v2(+0.07 [−0.2,0.33]);cap 门长度漂移显著(MWU p=5.3e-25)但幅度仅 +5%。
  - 证据:`experiments/e2/e2_report.json`(kw_reclass / stats);`experiments/STATE.md` E2。
- 7.4 操作判据:决策门双轨化——单轨显著不独立下判;kw 轨管"术语复诵",judge 轨管"语义质量"。

### 8. What Fine-Tuning Actually Did(诚实主结论)
- 8.1 干净锚点下的真实效应:行为风格转换(think 式长作答 → ~1/10 长度直接作答,均长 3250 → ~300 字符)、kw 覆盖持平、judge −0.29~−0.30 显著为负。
  - 证据:`experiments/e0/E0_report.md` §4.2–4.5。
- 8.2 v4 ≈ v2 与决策门:judge −0.01 [−0.13,+0.11] / kw +0.008 ns;"保留 v2"在干净锚点下依然成立但理由链完全改变。
  - 证据:`experiments/e0/E0_report.md` §4.4。
- 8.3 维度不均衡与定向干预闭环(B 贡献):v2 overall +0.060 但 ARIZ −0.139;v3 定向补数据的得(kw ariz 升)与代价(judge 轨降);v4 质量门管线。
  - 证据:`evidence_table.md` ②-A/②-C;`results/METRICS_LEDGER.md`。
- 8.4 迭代决策复盘表:每一版"评测信号 → 决策 → 事后裁决(信号是否可信)"。
  - **表 5 迭代决策复盘表**(v1→v4 × 触发信号 × 当时决策 × E0 后复核)。

### 9. Limitations
- 9.1 judge 同族:数据生成器与评委均属 Moonshot 家族(kimi-k2 404 致"异源"降级为"同族弱异源");E1b 的高一致不能排除家族级共享偏差;内部溢价模式虽不支持"同源偏袒主导"(v3 溢价反降),但需真异源评委(GPT/Claude/DeepSeek)终审。
  - 证据:`stats_review.md` §7;`experiments/STATE.md`。
- 9.2 子集功效:n=15~20 子集 MDE 0.37–0.77(judge 轨),全部子集结论为描述性;"v4≈v2"应措辞为"未发现显著差异"而非等价(judge 轨 CI 宽于 ±0.13 之外的差异不可分辨)。
  - 证据:`stats_review.md` §6.4。
- 9.3 单域单基座:仅 TRIZ 中文域、单基座、LoRA 单配置;E5(target_modules 消融)未做,A 贡献停留实证报告级。
- 9.4 长度混淆:base 与微调模型回答长度差 ~10 倍,judge 的 completeness 维度天然偏好长答;缺"长度控制后"的公平对比(E0 §6 待办②)。
  - 证据:`experiments/e0/E0_report.md` §4.5、§6。
- 9.5 干净 base 臂补跑已回填(2026-07-23):E1a' 合并胜率 0.260 显著负、E1b v4−base −0.30 ✅、E3' v4 vs base +0.083 ✅;残余风险为 judge 同族与真异源终审未做(见 §8 of 本文档)。

### 10. Conclusion
- 评测基础设施是领域微调的一等公民;三类测量失真的检查清单(think 块完整性 / pairwise 双序 / 双轨决策门)可直接迁移。

---

## 5. Related Work 组织方案(映射 W2 文献)

按"问题 → 既有工作 → 我们的增量"组织为四个小节(文献编号沿用 `related_work.md` 附录):

1. **LLM-as-a-Judge 及其失效模式**(#14 MT-Bench/Chatbot Arena;#15 G-Eval;#16 两篇综述;#18 自我识别;#19 自我偏好归因;#20 冗长偏差;#21 评委验证;#22 自我偏好量化;#23 家族级偏差)→ 我们的增量:中文专业域 0.87 位置不一致率的极端实例 + T=0 完全确定性的反例(对照 Coin Flip Judge 13.6%,引 `eval_methodology_research.md` §2.2 所载 arXiv:2606.13685,**引用前须核验编号**)。
2. **表面匹配指标的局限史**(#17 BLEU 重估 EACL 2006;BERTScore 等,引 `eval_methodology_research.md` §2.1)→ 增量:LLM 时代"关键词 vs judge 双轨显著反向"的新实例 + ARIZ 漏判率 31.7% 的量化。
3. **领域微调的效果异质性**(#8 RAG 失败点;#9 SFT 不总是伤通用;#10 LoRA learns less;#11 遗忘缓解;#12 微调 vs 检索)→ 增量:领域内子维度回退 + 定向干预闭环(B 真空白,W2 §6)。
4. **评测统计纪律与 EDD**(#24 Miller error bars;#25 100 instances;#26 评测方差;#27 Dror 显著性指南;#28 McNemar 出处;#29 工程 EDD 文献)→ 增量:完整案例 + 统计纪律组合(D 真空白,W2 §6)。
5. **混合架构与 PEFT(融入 §3 而非独立 RW 节)**(#1 LoRA;#2 MoE-Sieve;#3 Dynamic Rank LoRA;#4 MoLE;#5 LoRA-Mixer;#6 GDN;#7 Qwen3-Next 官方;#30 QLoRA;#31–33 社区先例)→ 定位:A/E 为实证报告级,不作独立贡献主张。
6. **LLM+TRIZ 先例(#35 AutoTRIZ、#36 TRIZ-GPT、#37 Gusarov et al.)**→ 引言引用,确立"TRIZ 微调无先例"。

---

## 6. 图表总清单

| 编号 | 内容 | 数据来源 |
|---|---|---|
| 图 1 | v1→v4 迭代时间线(训练/评测/事故/修复) | `evidence_table.md` 附表、⑤-5 |
| 图 2 | 双轨分歧:eval2 v3−v2 两轨森林图 + gold 两轨散点 | `stats_review.md` §3、图 B/C 建议 |
| 图 3 | 位置偏差:AB vs BA 胜率对比 + 文献 25pp 参考线 | `experiments/e1/e1_report.md` E1a |
| 图 4 | think 污染修复前后:v4−base 两轨差值反号对比 | `experiments/e0/E0_report.md` §4.3 |
| 表 1 | 版本总表(数据/质量门/超参/时长/硬件) | `evidence_table.md` ① |
| 表 2 | 干净锚点四方对比(base_goldfix/v2/v4/污染 base × 双轨) | `experiments/e0/E0_report.md` §4.2 |
| 表 3 | 功效与 MDE 表 | `stats_review.md` §6 |
| 表 4 | judge 可靠性包汇总(E1a/b/c vs 文献) | `experiments/e1/e1_report.md` |
| 表 5 | 迭代决策复盘表 | 本文档 §4-8.4 |
| 附表 | ARIZ 漏判表述清单(19 条节选) | `experiments/e3/e3_report.md` |

---

## 7. 局限章节要点汇总(对应 §4-9)

① judge 同族(弱异源,需真异源终审);② 子集功效不足(n=15~20 仅描述性;不可宣称等价);③ 单域(TRIZ 中文)单基座(Qwen3.6-35B-A3B)单 PEFT 配置;④ 长度混淆未解(base vs 微调 ~10×);⑤ 干净 base 臂补跑已完成并回填(2026-07-23,见 §8 清单);⑥ 训练数据泄漏未独立检验(harness 有金标去污门,`stats_review.md` §9);⑦ E1a 位置偏差机制发现在干净锚点下依然成立(不一致率 0.58、B 位胜率 0.65),但幅度较污染臂(0.87/0.81)减弱。

---

## 8. 占位标记清单(干净 base 臂补跑回填)

| 占位 | 位置 | 回填结果(最终值) | 状态 |
|---|---|---|---|
| **E1A_CLEAN** | §4-6.1 | E1a' 干净臂(v4 vs base_goldfix):双序合并 v4 胜率 0.260 [0.204, 0.325](显著负,污染臂对照:合并 0.49 平局);位置不一致率 0.58、B 位胜率 0.65(污染臂 0.87/0.81) | ✅ 已回填 |
| **E1B_CLEAN** | §4-6.2 | E1b 干净臂:base_goldfix 2.87 > v2 2.58 ≈ v4 2.57;v4−base 32k −0.30 [−0.46,−0.14] ✅、8k −0.11,符号一致率 0.85;v2−base 32k −0.29 ✅,一致率 0.82;跨评委 ρ=0.759 | ✅ 已回填 |
| **E3_CLEAN** | §4-7.2 | E3' rubric 轨:base_goldfix 0.800 / v2 0.875 / v4 0.883;v4 vs base +0.083 [0.033, 0.142] ✅;v2 vs base +0.075 [−0.025, 0.15] ns;干净 base 漏判仅 3/20(草稿臂 19/60=31.7%) | ✅ 已回填 |

已按干净锚点回填完毕(2026-07-23),来源 `experiments/e1/e_goldfix_report.json`。
