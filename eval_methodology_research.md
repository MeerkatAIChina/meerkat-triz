# Meerkat-AI TRIZ 适配器评测方法论研究

日期：2026-07-21（注：远端机器时钟为 2026-07-22）
背景：v2 适配器 overall 0.589 vs base 0.529，但 ariz_completeness 仅 0.250（6 题）；v3 定向补 ARIZ 数据后正在 102 题扩充集上重测。

---

## 1. 评测为什么是这个项目的核心环节

本项目的迭代模式（v2 全量 → 评测发现 ariz 弱项 → v3 定向补数据 → 重测）就是文献中的
**评测驱动开发（Evaluation-Driven Development）**：

- arXiv:2411.13768《Evaluation-Driven Development and Operations of LLM Agents》提出：
  静态基准用作"钉住的回归基线（pinned regression baseline）"，离线评测在受控条件下验证达标，
  再叠加领域专家策划与合成用例持续扩展覆盖。本项目 v1→v2→v3 的固定评测集 + 扩充评测集
  双层结构与此一致。
- 领域微调**并不保证**净收益：Barnett et al. 的 RAG 微调研究发现 Mixtral/Llama2 微调后
  在多数数据集上反而逊于基座。v2 的 ariz 退步（base 0.389 → v2 0.250）正是同类现象——
  没有逐维度评测，这个退步会被 overall 的上升掩盖。
- HELM（Liang et al., TMLR 2023）与 GEM 的建议：永远报告多个互补指标而非单一总分，
  单指标无法反映系统真实质量。

## 2. 与本项目最相关的四类业界研究成果

### 2.1 规则/关键词评测的局限（直接命中本项目现状）

本项目四项指标全部是关键词覆盖/精确匹配，这是文献中问题最充分记录的一类方法：

- BLEU/ROUGE/EM 只数表面词重叠，语义等价的改述会被误判（Papineni 2002; Lin 2004;
  Zhang et al. 2020 BERTScore 论文；arXiv:2503.08542 DAFE）。
- 对本项目的具体影响：**ARIZ 步骤完整性用 6 个步骤名关键词匹配，模型用同义表述
  （如"理想最终结果"写成"IFR/理想解"的其它说法、或步骤合并描述）会被漏判**——
  0.250 vs 0.389 的差距里可能混有"表述差异"而非真实能力差异。
- 缓解：BERTScore（上下文嵌入语义相似度，生成类任务与人类判断相关性 59% vs
  BLEU/ROUGE 的 47-50%，Galileo/Zhang et al.）可作低成本语义补充；
  关键词映射表（本项目已有中英别名映射机制）应持续从误判案例中扩充。

### 2.2 LLM-as-a-Judge（最主流的替代方案）

- Zheng et al. 2023（MT-Bench / Chatbot Arena，NeurIPS）：GPT-4 评委与人类专家
  一致率 >80%，与人类之间的一致率相当——奠定了 LLM 评委的合法性。
- 但 2024-2026 的后续研究量化了它的失效模式（arXiv:2411.15594 综述）：
  - **位置偏差**：交换两个候选顺序可使胜率摆动达 25 个百分点（Wang et al. 2023）
  - **长度/冗长偏差**：控制内容后长回答仍被偏好（Saito et al.）
  - **自我增强偏差**：模型偏好自己家族的输出（Panickssery et al.）
  - **重复试验不可靠**：arXiv:2606.13685《The Coin Flip Judge?》测得同一评委同一输入
    平均翻转率 13.6%，需要约 11 次投票的多数表决才能以 95% 概率复现参考裁决
- 缓解措施（已被验证有效）：位置交换双跑取平均、rubric 锚定评分（每项给定义和示例）、
  reference-guided 评判、温度 0、固定评委模型版本、多次投票。

### 2.3 统计严谨性（本项目当前最大的短板）

- 旧评测集 ariz 只有 **6 题**：单题变化 = 16.7 个百分点，0.250 与 0.389 的差距
  在统计上基本是噪声。
- 扩充到 102 题后：单侧比例的标准误 ≈ √(p(1-p)/102) ≈ 0.043-0.049，
  即 95% 置信区间约 ±9pp；**v3 vs v2 的 ariz 差异若小于约 ±10pp，不能视为显著**。
- 配对设计（v2/v3 跑同一批题）可用配对 bootstrap 或 McNemar 检验提高检验功效，
  比独立样本的区间比较更敏感（Spark-LLM-Eval, arXiv:2603.28769 把 CI 与显著性检验
  内置为评测框架的一等公民）。
- 经验法则：基准题数应让"你关心的最小差异"落在置信区间之外；对 5pp 的分辨率
  需要 n≈400+，对 10pp 需要 n≈100（本项目 102 题恰好可分辨 10pp 级差异）。

### 2.4 领域适配评测的工程实践

- 前后同测（pre/post 同一测试集、同一加载方式）：本项目 apples-to-apples 设计
  （同进程、同 FP16、同 temperature=0）与业界最佳实践一致（Centific 领域微调工作流）。
- 多维加权综合分：本项目的 0.3/0.3/0.2/0.2 加权正是业界通行的 composite metric 模式；
  关键是**权重必须反映业务优先级**，且汇报时永远同时给出分项（本项目已做到）。
- 回归测试：领域微调可能损伤通用能力，建议保留一个小型通用能力探针集
  （如 20-50 题通用指令跟随）作为回归门槛。
- CI/CD 化：Braintrust/Scale AI/Amdocs(NVIDIA NeMo) 等均将评测固化为
  "每次训练后自动跑 + 版本化测试集 + 回归告警"的工程闭环——本项目的
  v3_chain 接力器已是这个模式的雏形。

## 3. 对本项目的具体建议（按优先级）

| # | 建议 | 依据 | 成本 |
|---|---|---|---|
| 1 | 评测报告附 95% 置信区间（配对 bootstrap），ariz 差异 <10pp 不下结论 | §2.3 | 低（纯统计） |
| 2 | ariz_completeness 增加 rubric 化 LLM 评委双轨：6 步骤逐项 0/1 + 判定理由，与关键词分对照 | §2.1/2.2 | 中（需 judge API，远端已有 Moonshot key） |
| 3 | 关键词误判案例回流：每轮人工抽 5-10 题核对，把漏判的同义表述补进 KEYWORD_MAP | §2.1 | 低 |
| 4 | 增加 20-50 题通用能力探针集，作为领域微调的回归门槛 | §2.4 | 低 |
| 5 | 若上 LLM 评委：温度 0、固定版本、pairwise 时位置交换双跑 | §2.2 | 低 |
| 6 | case_quality 可加 BERTScore（中文模型）作语义维度补充，BLEU/ROUGE 维持"仅参考"定位 | §2.1 | 低 |

## 4. 主要参考来源（本次检索实得）

- Zheng et al. 2023, Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena（经多篇转引确认）
- arXiv:2411.15594 — A Survey on LLM-as-a-Judge（偏差分类与缓解）
- arXiv:2606.13685 — The Coin Flip Judge? Reliability and Bias in LLM-as-a-Judge（翻转率 13.6%，11 次投票规则）
- arXiv:2606.22329 — BabelJudge: Measuring LLM-as-a-Judge Reliability（位置偏差 25pp 等量化）
- arXiv:2503.08542 — DAFE（EM/BLEU/ROUGE 局限与 LLM 评委适用边界）
- Zhang et al. 2020 — BERTScore（语义嵌入评测；相关性数据经 Galileo 工程文转引）
- arXiv:2411.13768 — Evaluation-Driven Development and Operations of LLM Agents（过程模型）
- arXiv:2603.28769 — Spark-LLM-Eval（CI 与显著性检验内建）
- arXiv:2411.09539 — A Practical Guide to Fine-tuning LMs with Limited Data（领域适配数据策略）
- Centific / Latitude 工程实践文（pre-post 同测、composite 加权、回归测试、基线对比）
