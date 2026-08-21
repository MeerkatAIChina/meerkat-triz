# Sec0 业界实践补充调研(W4)——v5 方案论证的外部证据基座

- 执行人:业界调研员_W4 · 日期:2026-07-24 · 类型:web 调研报告(不写方案,只供证据)
- 检索范围:2023–2026 学界论文(arXiv/ACL/ICLR/TMLR)与业界实践(Scale AI、OpenAI cookbook、Together AI、HuggingFace 生态、ms-swift/Qwen 社区 issue)
- 证据分级:【已核实原文】= 本次调研直接打开原文核对了数字;【文献支持】= 检索到摘要/正文/权威转述,结论可靠但未逐字核对全部数字;【推断】= 本报告基于上述证据的综合判断
- 纪律:所有数字抄录自来源,不改字符;arXiv 号逐条核验过(凡未能核验的明确标注)

---

## §1 SFT 数据配比:真实:合成比例与种子上采样倍数

**R1.1【已核实原文】arXiv 2602.04482 的"等量真实 +11.9%"复核成立,但语境严格限定。**
原文为 ProAgentBench(Tang et al., 2026,arXiv:2602.04482,主动式 agent 基准,28,000+ 真实用户事件)。其 §6.6 实验:LLaMA-3.1-8B-Instruct 分别在 **741 条真实**与 **741 条等量合成**数据上 SFT(lr 2e-5,3 epochs)与 LoRA(r=16,lr 2e-4,3 epochs)。结果:SFT 臂真实数据 Accuracy 74.0% vs 合成 62.1%,**差 +11.9%**;Intention Accuracy 42.1% vs 34.8%(+7.3%)。
- 复核结论:数字与原文一致,"等量真实 +11.9%"可引用。
- **语境限定(引用时必须附带)**:① 这是 agent 时机预测任务(When/How to Assist),非通用指令 SFT;② 每臂仅 741 条;③ 只报了相对提升,未给置信区间;④ 该论文同时报告 LoRA 臂同样"真实 > 合成",方向稳健但幅度未单独摘录。
- 一句话结论:**在小数据(n≈740)设定下,等量真实数据显著优于等量合成数据(+11.9pp Accuracy)——真实数据单价高但有不可被合成复制的信号价值。**

**R1.2【文献支持】真实:合成最优比存在且呈 U 形,实用启发式为 1:1–1:2(真实:合成)。**
学习理论框架研究(arXiv:2510.08095,2025)用算法稳定性导出泛化误差界,预测测试误差随合成占比呈 U 形,并在 CIFAR-10 与脑 MRI 上验证;其 §6 "Insights for Practitioners" 给出明确启发式:**合成质量高时,合成量不超过真实量 2 倍(即真实:合成 1:1 到 1:2)有效;超过后增益递减、分布失配时反而有害**。
- 语境限定:理论验证在视觉/核回归场景,非 LLM SFT;但"存在最优比、过量合成有害"的定性结论与本项目经验(v3 补 674 条合成 ARIZ 数据后 judge 轨显著恶化 −0.0682,见 stats_review.md §3.2)方向一致。

**R1.3【文献支持】种子上采样倍数无普适最优,取决于"查询预算比 q/s"。**
Scale AI 实证研究(Scale.com 技术博客,2024-12,Synthetic Data Generation Strategies for Fine-Tuning LLMs)在三种任务上系统比较合成策略:预算低(q/s 小)时"为既有种子生成新回答(改写/应答增强)"最有效;预算高时"生成新问题(扩展种子覆盖)"最有效;题目改写(question rephrasing)对弱生成模型最鲁棒。
- 对本项目的含义:我们的 ×6/×11/×16 扩展倍数(config.py:162-178)属"固定倍数先验",文献不支持任何固定倍数普适最优;倍数应视为需按子集验证的超参,而非经验默认值。

**R1.4【文献支持】纯合成 SFT 在通用对齐上可成功,但有"模仿上限"。**
- Magpie(Xu et al., 2024,arXiv:2406.08464):纯合成数据 SFT 的 Llama-3-8B 在 AlpacaEval 2 上超过官方指令模型——证明合成数据在通用域可行。
- 反例:Gudibande et al., 2023(arXiv:2305.15717,The False Promise of Imitating Proprietary LLMs):模仿闭源模型的合成数据只学到风格学不到能力,在内容型评测上无增益。
- 综合【推断】:合成数据能教"形式与风格",教不了"分布外的真实信号";本项目 TRIZ 领域属"形式+术语"密集型,合成可行,但 E3 已证实合成措辞与金标关键词的失配会造成关键词轨伪影(stats_review.md §5),v5 数据配比决策须以 rubric/语义轨为准。

---

## §2 指令微调的风格/冗长控制与 verbosity 对 LLM-judge 的影响

**R2.1【文献支持】LLM 评委存在可量化的冗长偏置,且有标准量化方法。**
Saito et al., 2023(Keita Saito 等,arXiv:2310.10076,NeurIPS 2023 Instruction Workshop):GPT-4 在创作类任务中比人类更偏好更长回答;提出基于 accuracy parity 的冗长偏置量化指标(按长短分组比较判对率差)。

**R2.2【文献支持】对齐训练本身会放大长度;长度可被"游戏化"。**
- Singhal et al., 2023(arXiv:2310.03716,ICLR 2024):RLHF 的收益相当部分可由回答长度解释;简单长度惩罚即可大幅纠正偏置。
- Dubois et al., 2024(arXiv:2404.04475,Length-Controlled AlpacaEval):AlpacaEval 胜率可被长度操纵——同一模型仅把 prompt 从 concise 改 verbose,GPT-4 评分大幅摆动;其长度控制胜率(LC)将与人类排序的相关从 0.94 提升到 0.98。**LC 回归建模(长度+基线差)是成熟的长度去偏评测方法。**
- Li et al., 2024(Arena-Hard,arXiv:2406.11939):在 LC 之外进一步引入 style-controlled win rate,同时控制长度与 markdown 风格。
- Zhao et al., 2024(arXiv:2407.01085):胜率可分解为"意愿性(desirability)+信息量(information mass)";AdapAlpaca 通过把基线答案长度对齐到被测模型来消除长度混淆。

**R2.3【文献支持】数据侧控制输出长度/风格的主流手段是训练目标本身,而非事后截断。**
业界共识(综合 OpenAI fine-tuning 文档与 Together AI 指南):SFT 模型的输出长度分布 ≈ 训练集目标长度分布;要得到简洁直接的风格,就用简洁直接的目标文本训练。本项目 v2/v4 正是此机制的实例(训练目标 ~300 字符 → 输出 ~300 字符,base 为 3250 字符;E0_report.md §4.5)。

**R2.4 对本项目的直接相关性【推断】。**
E0 已定量暴露该风险:干净 base(均长 3250 字符)judge 2.87 > v2/v4(~300 字符)2.58/2.57,但 judge 输入截断 1500 字符使 base 仅 ~46% 可见——**长度混淆双向存在**:judge completeness 维度偏好长答,截断又惩罚长答(E0_report.md §4.5 注 1;stats_review.md §2.4)。v5 评测方法学应引入长度控制对比(限长或 LC 式回归),这正是 assessment_report.md 七.5 指出的未解混淆。

---

## §3 领域 SFT 防通用能力退化:replay/通用数据混入的实证

**R3.1【文献支持】LoRA 的"学得少、忘得少"是主文献结论,但存在反例。**
- Biderman et al., 2024(TMLR;arXiv:2405.09673,LoRA Learns Less and Forgets Less):LoRA 在持续预训练与 SFT 设定下遗忘显著少于全参微调,代价是域内学得也少;遗忘随 rank 增大而增加。
- Shuttleworth et al., 2024(arXiv:2410.21228,LoRA vs Full Fine-Tuning: An Illusion of Equivalence):LoRA 解含"侵入维度(intruder dimensions)"——与基座近似正交的大幅奇异向量,会损害无关任务;全参解无此结构。
- 反例(2026):巴西临床指南注入研究(arXiv:2605.01077):LoRA 的 OOD 遗忘反而大于全参(−6.0 vs −1.9,单生成器),最大伤口恰在 IFEval(指令遵循),与 intruder dimensions 机制吻合。
- 综合【推断】:LoRA 防遗忘的结论依赖任务与评测面;**不能用"我们用了 LoRA"替代通用能力回归评测**。本项目 general_probe 30 题(n=30,MDE ~18pp,功效不足,stats_review.md §6.4)与 E6 扩题建议正是对此的正解。

**R3.2【文献支持】通用数据混入(replay)比例:域:通用 1:1 到 1:10 均被实证使用,更高通用占比不一定伤域内。**
- LLM-R(arXiv:2411.04476,2024,领域维护方案生成):系统比较域:通用 = 1:1 / 1:2 / 1:5 / 1:7 / 1:10,**1:10(通用数据最多)在域内 ROUGE/BLEU 与通用指标上均最优**——高通用占比同时保通用又未伤域内。
- SSR(Huang et al., 2024,arXiv:2403.01244,ACL 2024):无原始数据时用模型自合成 rehearsal 样本,效果优于或持平真实 replay,且更省数据。
- 业界工程经验(CallSphere 等 MLOps 实践,2026,权威度低仅供参考):起始 replay 30%,见旧能力回退上调到 50%,新任务学不动下调到 20%。【推断】文献共识区间 ≈ 通用数据占训练混合的 10%–50%,但**比例强任务相关,无迁移保证**(OpenReview 多篇 2025–2026 论文明确指出最优配比对每个域须重新实验)。
- 本项目现状:v1–v4 训练混合中**通用数据占比 0%**(纯 TRIZ);v1 曾 general_probe −0.133 显著退化(evidence_table.md ②-C),v2/v4 未见显著通用退化但 n=30 功效不足。文献与自身证据共同支持 v5 混入少量通用指令数据。

---

## §4 LoRA on MoE:target_modules、expert 覆盖与 rsLoRA

**R4.1【文献支持】rsLoRA(α/√r 标度)在高 rank 下显著优于标准 α/r 标度,可逼近全参。**
- Kalajdzievski, 2023(arXiv:2312.03732):提出 rank-stabilized LoRA,标准 α/r 标度在高 r 下有效学习率塌缩;α/√r 修正后高 rank 可正常训练。
- Shuttleworth et al., 2024(arXiv:2410.21228):实证 rsLoRA 随 rank 增加可接近全参微调性能。
- 业界采用实例:TechING(arXiv:2601.18238,2026,VLM 领域 SFT)与 FirstPass(arXiv:2606.20769,2026,Qwen2.5-7B 审稿判定)均在生产性 LoRA 配置中启用 `use_rslora=True`。
- 本项目:r=64 α=128(标准 α/r=2 标度,12 个 target_modules;config.py:48-66),未用 rsLoRA。r=64 已属"高 rank"区间边缘,文献支持 v5 消融 rsLoRA。

**R4.2【文献支持】target_modules 全覆盖(attention+FFN 全部线性层)优于仅 attention。**
- FirstPass(arXiv:2606.20769):q/k/v/o + gate/up/down 七矩阵全打在预实验中优于 attention-only;并报告 response-only loss masking 是"单一最重要的设计决策"(与本项目 v4 的 completion-only 一致,evidence_table.md ①)。
- LoRA 配置实证指南(综合多篇 2024–2026 消融):域适应场景推荐 r≥32、target 含 FFN;本项目 12 模块配置(含 DeltaNet 的 in_proj_qkv/z/b/a 与 MoE 的 gate/up/down)与文献推荐同向,且比常见 MoE LoRA 实践更全(多数工作只打 up/down 或 attention)。

**R4.3【文献支持】MoE 专家的异质性值得利用:均匀配置不是最优。**
- DR-LoRA(arXiv:2601.04823,2026):OLMoE/Phi-mini-MoE 上按专家路由频率与学习强度动态分配 rank,优于均匀 rank 与 AdaLoRA 剪枝式分配。
- ESFT(Wang et al., 2024,arXiv:2407.01942):只微调与任务相关的专家子集,省参数且保通用能力。
- MoE-LoRA 方法族(MixLoRA arXiv:2404.15159;LoRAMoE arXiv:2312.09985;HydraLoRA arXiv:2404.19245):分别对应 FFN top-K 路由+共享 attention LoRA、专家分组(知识保持组 vs 下游任务组)防遗忘、共享 A 矩阵+专家专属 B 矩阵。
- 本项目:对 256 个专家统一 r=64 打 gate/up/down(等量适配器),未做专家级差异化;【推断】对 121GB 单卡而言,统一配置是合理简化,专家级消融(对应 assessment_report.md 的 E5)是"可选增强"而非"必要修复"。

---

## §5 SFT 后 DPO/偏好对齐的触发条件与低成本偏好对构造

**R5.1【文献支持】DPO 的标准触发条件:质量主观、多个有效输出、风格/语气维度难用 SFT 目标表达。**
- OpenAI fine-tuning cookbook(2025-06,官方):DPO 适用于"响应质量主观、无法客观度量、或语气/风格/得体性等细腻标准重要"的场景;先 SFT 建立任务结构,再从 SFT checkpoint 续 DPO。
- Together AI 技术指南(2025-04):同样推荐 SFT→DPO 两段式;SFT 教格式与内容,DPO 做偏好精炼。
- Zephyr(Tunstall et al., 2023,arXiv:2310.16944):验证了两段式在 7B 上的有效性,SFT(UltraChat)→ DPO(UltraFeedback binarized)无需任何人工标注。
- 低成本偏好对构造的成熟路径【文献支持】:UltraFeedback(Cui et al., 2023,arXiv:2310.01377)模式——同一 prompt 多模型多候选,GPT-4 按维度打分,最高分作 chosen、随机余一作 rejected。本项目已有金标 100 题 × 4 版本响应与 judge 打分缓存,构造 chosen/rejected 的边际成本极低。

**R5.2【文献支持】DPO 的已知风险:长度偏置、OOD 敏感、偏好对质量敏感。**
- Park et al., 2024(arXiv:2403.19159,ICLR 2024):DPO 会放大长度偏置;提出长度正则化 DPO 可解耦长度与质量。
- Xu et al., 2024(arXiv:2404.10719,Is DPO Superior to PPO?):受控比较中 PPO 系统性优于 DPO;DPO 对分布外偏好数据敏感。
- Ivison et al., 2024(arXiv:2406.09279,Unpacking DPO and PPO,Tülu 3 团队):偏好学习最佳实践的关键变量是数据质量与在线性,而非算法选择本身。
- 【推断】对本项目:v5 若上 DPO,触发条件应写成"judge 轨差值在 SFT 后仍显著为负且主要分歧在风格/完整性维度"(可测、可证伪),而非默认动作;偏好对可用现有金标缓存 + 新增 1–2 个候选生成低成本构造,β 取 0.1(Zephyr 常用)起步,并配长度监控(Park et al. 的教训)。

---

## §6 小数据领域微调(n<10K)的 epochs/lr 实证区间

**R6.1【文献支持】LIMA 锚点:1,000 条高质量样本 × 15 epochs 可对齐 65B 模型。**
Zhou et al., 2023(arXiv:2305.11206,LIMA: Less Is More for Alignment):LLaMa-65B 仅用 1,000 条精选样本 SFT(15 epochs,全参),即可获强对齐效果;核心变量是数据质量与多样性而非数量。

**R6.2【文献支持】大规模受控 SFT 实验(1,000+ 模型)的超参扫描区间。**
Harada et al., 2025(arXiv:2506.14681,v2 2025-10):在 ~1,000 条规模的 6 个数据集上扫描 lr ∈ {2e-7, 1e-6, 2e-6, 1e-5, 2e-5, 1e-4} × batch × LoRA/全参 × 10 epochs(每数据集 96 条件、960 候选);发现 **perplexity 稳定预测 SFT 有效性**、中层权重变化与性能增益最相关——支持"看 val loss/perplexity 早停"的实证做法(与本项目 v3/v4 的早停实践一致)。

**R6.3【文献支持】近年域 SFT 论文的实际配置样本点(小数据档)。**
| 来源 | 数据规模 | 方法 | lr | epochs |
|---|---|---|---|---|
| ProAgentBench(arXiv:2602.04482) | 741 | LoRA r=16 | 2e-4 | 3 |
| ProAgentBench(同上) | 741 | 全参 SFT | 2e-5 | 3 |
| Qwen3-14B KG 问答(arXiv:2508.17330) | 未标注(论文级域集) | LoRA r=8 全线性 | 1e-4 | 3 |
| 同上 | — | 全参 | 1e-5 | — |
| FirstPass(arXiv:2606.20769) | 域集(审稿判定) | rsLoRA r=32 α=64 | 未摘录 | — |
| TechING(arXiv:2601.18238) | 域集 | rsLoRA r=32 | 2e-5 | 2 |

- 综合区间【推断】:n<10K 时,**LoRA lr 1e-4–2e-4、全参 lr 1e-5–2e-5、epochs 2–3 为主流;数据越小、质量越高,epochs 可上调(LIMA 的 15 是质量上限锚点)**。本项目 v1–v4 的 lr 2e-4、epochs 2–4 落在区间正中,无需为"超参离群"担忧;v4 早停在 lr 仍高(1.553e-4)处(evidence_table.md ⑤-11)反而是文献视角下更值得修的点(cosine 应按实际步数而非名义 max_steps 排程)。

---

## §7 思考型(thinking-native)模型 SFT 的特殊注意事项

**R7.1【文献支持】Qwen3 官方机制:非思考模式靠"空 think 块前缀"实现——这正是 E0 根因的官方确认。**
Qwen3 Technical Report(Yang et al., 2025,arXiv:2505.09388):混合 thinking/non-thinking 数据 SFT;非思考模式通过在 assistant 起始处前置空思考块 `<think>\n\n</think>` 触发。**即空 think 块是"思考已结束"的结构性锚点**——与本项目 E0 的根因诊断(剥离空 think 块 → base 100/100 自吐未闭合英文 think 草稿,E0_report.md §2)逐字对应。

**R7.2【文献支持】学界对照:显式 flag 比"空 think 块"这类 template cue 更可靠。**
arXiv:2512.13607(2025,推理模式控制研究):明确指出 Qwen3 的双机制(显式 flag + enable_thinking 模板开关)冗余,其实验发现**显式 flag 的模式切换比模板 cue 更可靠**,因而在非思考模式下直接省略空 think 块——从反面印证:空 think 块本身就是一个脆弱的控制信号,任何管线(训练或评测)处理它都必须显式决策,不能默认剥离。

**R7.3【文献支持】训练框架侧已有专门机制处理空 think 块的 loss。**
ms-swift issue #5581(2025-08,ModelScope 官方训练框架):对无思维标注的 SFT 数据微调 Qwen3-Instruct/Thinking 时,提供 `--loss_scale ignore_empty_think` 选项——即**空 think 块是否计入 loss 是一个需要显式选择的训练超参**。QwenLM/Qwen3 discussion #1429(2025-05)与 issue #1625/#1826 记录了社区在 SFT 中关闭 thinking 时的大量不一致行为(/no_think 注入结果不稳定、enable_thinking=False 失效、KV-cache 破坏)——thinking-native 模型的 SFT 模板处理是业界公认的坑位密集区。

**R7.4【文献支持】对 thinking 模型做域 SFT 的副作用实证:输出变短、思考枯竭减少,但基准分可能下降。**
TÜDÜM(arXiv:2607.01927,2026):Qwen3.5-27B(thinking-native)LoRA SFT 后平均响应长度大幅下降、"thinking exhaustion"(思考预算烧光未产答案)大幅减少,但数学基准 accuracy 下降,需后续 RL 挽回——与本项目 v2/v4 的画像(长度 ~1/10、judge 轨 −0.29~−0.30)惊人相似,是"域 SFT 对 thinking 模型的典型效应"的独立第三方复现。

**R7.5【推断】v5 的三条 thinking-native 专属检查项(供方案组采纳):**
① 训练侧:明确空 think 块策略(保留并计 loss / 保留不计 loss / 剔除),与 ms-swift 的显式选项对齐,不许默认行为;② 评测侧:保留渲染模板输出的空 think 块(E0 修复即此),生成后剥离闭合块再评分;③ 验收侧:把"thinking exhaustion 率"(未闭合 think 占比)列为生成质量门指标,TÜDÜM 已示范该指标可量化、可回归。

---

## §8 对照表:业界共识 vs 我们的实践 vs v5 采纳建议

| # | 主题 | 业界/学界共识(出处) | 我们的实践(v1–v4) | v5 采纳建议 |
|---|---|---|---|---|
| 1 | 真实:合成比例 | 真实数据有不可替代信号(+11.9pp,arXiv:2602.04482,R1.1);最优比存在,U 形,启发式 1:1–1:2(R1.2) | 真实占比 6.1%(02b 路径)或 0%(corpus 路径 100% Moonshot 生成)(evidence_table.md ④) | **采纳**:把 385 条真实种子全部保留并显式上采样(如 ×3–×5),使真实占比可测、可声明;不再用"成本优先"作不采真实数据的默认理由 |
| 2 | 种子上采样倍数 | 无普适最优,取决于预算比 q/s(R1.3) | 固定 ×6/×11/×16(config.py:162-178) | **部分采纳**:倍数保留,但按子集记录并在 v5 数据报告中标明该倍数为未消融超参;若预算允许做 1 个倍数消融臂 |
| 3 | 合成数据定位 | 合成教形式/风格,不教分布外真实信号(R1.4) | 100% Moonshot 合成,风格与 Moonshot judge 同族(stats_review.md §7) | **采纳(评测侧)**:v5 新增异源 judge 终审,打破"生成器=评委家族"闭环 |
| 4 | 长度/风格控制 | 输出长度≈训练目标长度分布;风格用数据控制(R2.3) | 已通过 ~300 字符目标实现简洁风格;长度门 2048 token | **采纳(评测侧)**:v5 金标评测增加长度控制对比(限长 base 臂或 LC 式长度回归),解开 assessment_report.md 七.5 的长度混淆 |
| 5 | judge 冗长偏置防护 | verbosity bias 可量化(Saito 2023);LC 胜率去偏(Dubois 2024);双序消位置偏(通行做法) | 双序合并 + 位置不一致率报告(E1a);无长度去偏 | **采纳**:在 pairwise/rubric judge 协议中加"响应长度记录 + 长度-分数相关检验"(成本≈0,用现有缓存即可算) |
| 6 | 防通用退化 | LoRA 不是免遗忘金牌(R3.1 反例);通用数据混入 10%–50%(R3.2) | 通用数据 0%;general_probe 30 题功效不足;v1 曾 −0.133 显著退化 | **采纳**:v5 混入 5%–10% 高质量通用指令数据(约 300–600 条),general_probe 扩至 120 题(即 E6) |
| 7 | replay 数据获取 | 无原始数据时可用 SSR 自合成 rehearsal(R3.2) | 无 replay 机制 | **采纳(低成本版)**:用 base 模型自生成 200–300 条通用问答作 replay 候选,不必外购数据集 |
| 8 | rsLoRA | 高 rank 下 α/√r 显著更优,逼近全参(R4.1) | 标准 α/r,r=64 α=128(标度=2) | **采纳为消融臂**:v5 主配置不动(可比性优先),并行跑 1 个 rsLoRA 臂(r=64,α=64·√64=512 或按 peft `use_rslora=True` 默认)——一次 v4 级训练约 7h,成本可承受 |
| 9 | MoE 专家覆盖 | 专家异质 → 动态/选择性 rank 更优(R4.3) | 256 专家统一 r=64 打 gate/up/down | **不采纳(本轮)**:维持统一配置保可比性;专家级消融(E5)已在 assessment_report.md 判"暂缓",与文献优先级一致 |
| 10 | target_modules | attention+FFN 全打优于仅 attention(R4.2) | 12 模块全打(含 DeltaNet 专属 proj) | **维持现状**:我们的配置已超出文献常见覆盖度,无需变更 |
| 11 | DPO 触发 | 条件:质量主观/风格维度/SFT 已到格式上限(R5.1);风险:长度偏置、OOD 敏感(R5.2) | 未做 DPO | **条件触发,不默认**:仅当 v5 SFT 后 judge 轨仍显著为负且分歧集中在风格/完整性维度时启动;偏好对用现有金标缓存构造(chosen=judge 高分响应),β=0.1,配长度监控 |
| 12 | 小数据 epochs/lr | LoRA 1e-4–2e-4 / 全参 1e-5–2e-5,epochs 2–3,高质量小数据可多 epochs(R6) | lr 2e-4,epochs 2–4,早停 patience=3 | **采纳(修排程)**:cosine 按实际预期步数排程,避免 v4 式"早停时 lr 仍有 1.553e-4"的尾部高 lr 问题(evidence_table.md ⑤-11) |
| 13 | completion-only loss | response-only masking 被独立研究称为"最重要单决策"(R4.2) | v4 已启用(trl 1.5.1) | **维持现状**:已是业界最佳实践 |
| 14 | thinking 模型 SFT | 空 think 块是结构锚点(Qwen3 TR,R7.1);template cue 脆弱(R7.2);loss 处理须显式(R7.3) | E0 已发现根因并修复评测侧;训练数据 think 剥离 0 条(v4 质量门) | **采纳**:训练侧显式声明空 think 策略;评测侧保留 E0 修复;新增"未闭合 think 率"生成质量门(R7.5) |
| 15 | thinking 模型域 SFT 副作用 | 输出变短+基准分下降是第三方复现过的典型效应(TÜDÜM,R7.4) | v2/v4 长度 ~1/10、judge −0.29~−0.30(E0) | **采纳(叙事侧)**:v5 论文/报告中引用 TÜDÜM 作为独立复现证据,把"简洁化代价"定位为已知现象而非本项目孤例 |

---

## §9 调研结论(供 v5 方案组直接引用的 5 句话)

1. "等量真实 +11.9%"复核成立(arXiv:2602.04482 §6.6),但它是 741 条规模的 agent 任务证据,**支撑"真实种子必须保留并可测",不支撑"真实占比必须 >X%"**;文献给出的可辩护区间是真实:合成 ≈ 1:1–1:2,我们 6.1% 的真实占比显著低于文献启发式,是 v5 数据侧最值得修的一项。
2. 本项目最成熟的实践(completion-only loss、双序 judge、12 模块 LoRA、lr 2e-4)全部落在业界共识区间正中;**最落后的两项是:通用 replay 数据 0% 与未做长度去偏评测**——两者修复成本都远低于一次训练 run。
3. rsLoRA 是 v5 唯一有强文献支撑、且硬件预算(7h/run)允许的架构级消融;专家级 LoRA 差异化(DR-LoRA/ESFT)文献方向明确但优先级低,维持暂缓与 assessment_report.md 一致。
4. DPO 不是 v5 的默认项:业界触发条件(主观质量/风格维度/SFT 格式已满)在我们干净锚点证据下**尚未满足**(v4 judge −0.30 的分歧含 completeness 维度,长度混淆未解),应先做长度控制评测再裁决;若裁决通过,低成本偏好对可直接用金标缓存构造。
5. E0 的 think 污染发现不是孤立事故:Qwen3 TR 的机制说明、arXiv:2512.13607 的"显式 flag > template cue"结论、ms-swift 的 `ignore_empty_think` 选项、TÜDÜM 的副作用画像,共同构成学界/业界的完整对照系——**v5 把"thinking-native 处理协议"写成显式检查清单,既有内部证据(E0)又有外部证据(本节),可双线论证**。

---

## 附:引用清单(按首次出现序;arXiv 号均已核验,标注 venue 的以检索结果为准)

1. Tang, H. et al. ProAgentBench: Evaluating LLM Agents for Proactive Assistance with Real-World Data. 2026. arXiv:2602.04482.【已核实原文】
2. (Learning-theoretic synthetic/real ratio). 2025. arXiv:2510.08095.【文献支持】
3. Scale AI. Synthetic Data Generation Strategies for Fine-Tuning LLMs. 2024-12. scale.com 技术博客(配套论文).【文献支持】
4. Xu, Z. et al. Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing. 2024. arXiv:2406.08464.【文献支持】
5. Gudibande, A. et al. The False Promise of Imitating Proprietary LLMs. 2023. arXiv:2305.15717.【文献支持】
6. Saito, K., Wachi, A., Wataoka, K., Akimoto, Y. Verbosity Bias in Preference Labeling by Large Language Models. 2023. arXiv:2310.10076(NeurIPS 2023 Instruction Workshop).【文献支持】
7. Singhal, P., Goyal, T., Xu, J., Durrett, G. A Long Way to Go: Investigating Length Correlations in RLHF. 2023. arXiv:2310.03716(ICLR 2024).【文献支持】
8. Dubois, Y. et al. Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators. 2024. arXiv:2404.04475.【文献支持】
9. Li, T. et al. From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard. 2024. arXiv:2406.11939.【文献支持】
10. Zhao, Y. et al. Rethinking LLM-based Preference Evaluation. 2024. arXiv:2407.01085.【文献支持】
11. Biderman, D. et al. LoRA Learns Less and Forgets Less. 2024. arXiv:2405.09673(TMLR).【文献支持】
12. Shuttleworth, R. et al. LoRA vs Full Fine-Tuning: An Illusion of Equivalence. 2024. arXiv:2410.21228.【文献支持】
13. (Brazilian healthcare knowledge injection). 2026. arXiv:2605.01077.【文献支持】
14. (LLM-R: domain-adaptive maintenance scheme generation). 2024. arXiv:2411.04476.【文献支持】
15. Huang, J. et al. Mitigating Catastrophic Forgetting in Large Language Models with Self-Synthesized Rehearsal. 2024. arXiv:2403.01244(ACL 2024).【文献支持】
16. Kalajdzievski, D. A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA (rsLoRA). 2023. arXiv:2312.03732.【文献支持】
17. (TechING: technical image understanding VLM). 2026. arXiv:2601.18238.【文献支持】
18. (FirstPass: grounding AI scientific judgment). 2026. arXiv:2606.20769.【文献支持】
19. (DR-LoRA: dynamic rank LoRA for MoE adaptation). 2026. arXiv:2601.04823.【文献支持】
20. Wang, X. et al. Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning (ESFT). 2024. arXiv:2407.01942.【文献支持】
21. Li, D. et al. MixLoRA. 2024. arXiv:2404.15159;Dou, S. et al. LoRAMoE. 2023. arXiv:2312.09985;Tian, C. et al. HydraLoRA. 2024. arXiv:2404.19245.【文献支持】
22. OpenAI. Fine-Tuning Techniques: Choosing Between SFT, DPO, and RFT (cookbook). 2025-06.【文献支持】
23. Together AI. Direct Preference Optimization: A Technical Deep Dive. 2025-04.【文献支持】
24. Tunstall, L. et al. Zephyr: Direct Distillation of LM Alignment. 2023. arXiv:2310.16944.【文献支持】
25. Cui, G. et al. UltraFeedback. 2023. arXiv:2310.01377.【文献支持】
26. Park, R., Rafailov, R., Ermon, S., Finn, C. Disentangling Length from Quality in Direct Preference Optimization. 2024. arXiv:2403.19159(ICLR 2024).【文献支持】
27. Xu, S. et al. Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study. 2024. arXiv:2404.10719.【文献支持】
28. Ivison, H. et al. Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback. 2024. arXiv:2406.09279.【文献支持】
29. Zhou, C. et al. LIMA: Less Is More for Alignment. 2023. arXiv:2305.11206.【文献支持】
30. Harada, Y. et al. Massive Supervised Fine-tuning Experiments Reveal How Data, Layer, and Training Factors Shape LLM Alignment Quality. 2025. arXiv:2506.14681.【文献支持】
31. (Qwen3-14B KG QA SFT). 2025. arXiv:2508.17330.【文献支持】
32. Yang, A. et al. Qwen3 Technical Report. 2025. arXiv:2505.09388.【文献支持】
33. (Explicit flag vs template cue for reasoning mode control). 2025. arXiv:2512.13607.【文献支持】
34. ms-swift. Issue #5581: Clarification on --loss_scale ignore_empty_think. 2025-08. GitHub ModelScope/ms-swift.【文献支持】
35. QwenLM/Qwen3. Discussion #1429 / Issues #1625, #1826. 2025. GitHub.【文献支持】
36. (TÜDÜM: Turkish-thinking reasoning pipeline for Qwen3.5-27B). 2026. arXiv:2607.01927.【文献支持】

**诚实声明**:① 条目 2、13、14、17、18、19、31、33、36 的作者名与正式标题未逐字核对(检索结果为 arXiv HTML 正文,标题以编号引用稳妥);② Saito et al. 的"长度差 >20% 时偏好长答 >90%"一类具体阈值数字本次未在原文逐字核实,正文已避免引用该阈值,只保留定性结论;③ 条目 15(SSR)、26(Park et al.)、7(Singhal)、11/12(Biderman/Shuttleworth)的 arXiv 号经交叉来源核验一致。
