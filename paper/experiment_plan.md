# 补充实验矩阵与执行卡 — Meerkat-AI TRIZ LoRA 论文

> 作者：W4 实验设计师 ｜ 日期：2026-07-23 ｜ 状态：待 Orchestrator 审定后交 coder 在 DGX(spark-855a) 执行
> 远端项目路径：`/home/meerkat/mongoose_ai`；Python：`venv_v5/bin/python`（环境固定，不可 pip install，无 bert_score）
> 本文所有远端路径在执行前须先 `ls` 验证（W4 无远端读权限，路径依据本地 README 推断）

---

## 0. 设计依据（已核实的约束与事实）

| 事实 | 来源 |
|---|---|
| 评测文献建议 6 条（CI/rubric 双轨/关键词误判回流/通用探针/judge 纪律/BERTScore） | `eval_methodology_research.md:77-85` |
| Coin Flip Judge：同评委同输入平均翻转率 13.6%，约 11 次投票多数表决才能 95% 复现参考裁决 | `eval_methodology_research.md:46-47` |
| 位置偏差可使胜率摆动达 25pp；缓解=位置交换双跑、rubric 锚定、温度 0、固定版本、多次投票 | `eval_methodology_research.md:44,48-49` |
| n=100 时 95%CI 约 ±9pp，只能分辨 10pp 级差异；子集 n=15~20 只能作描述性证据 | `eval_methodology_research.md:55-61` |
| 金标集 100 题构成：principle 20 / contradiction 20 / ariz 20 / case 15 / concept 15 / innovation 10 | `pipeline_v4_remote/README.md:104` |
| judge 探测顺序 `kimi-k2-0711-preview → moonshot-v1-32k → moonshot-v1-8k`；两轨分开报告 | `pipeline_v4_remote/README.md:110-112` |
| v4 数据门第 5 条：concept_explanation / innovation_assessment 各 cap 1500，**cap 内优先保留 output 更长者** | `pipeline_v4_remote/README.md:94` |
| eval2 judge 轨：moonshot-v1-8k、批量 10 条、RPM=3；judge 输入 response 截断前 500 字符 | `eval_pipeline_v2/README.md:38,45` |
| 生成与 judge 打分均有缓存、断点续跑；`--calibrate` 校准模式已存在 | `pipeline_v4_remote/README.md:114-115` |
| 训练实测速度：v1 10.0 s/step（666 步 1h52m）；v2 2,116 步 6h46m（≈11.5 s/step） | `results/METRICS_LEDGER.md:8-9` |
| LoRA 可训练参数 84.66M/34.7B=0.24%，对 MoE routed expert 基本无覆盖；显式 12 个目标模块 | `docs/training_retrospective_2026-07-20.md:65,75` |
| 已有通用探针 30 题（6 类×5：常识/数学/逻辑/写作/代码/指令跟随），关键词评分 | `eval_pipeline_v2/general_probe.json` |
| v4 关键结果：judge overall v4=2.57 vs base=1.57（+1.00 [0.80,1.19]，McNemar p=1.7e-12）；keyword/concept_explanation v4 0.4356 vs base 0.5317 vs v2 0.5187；决策门判"保留 v2" | 远端 `results/v4_final_report.md`（Orchestrator 转述） |
| eval2 两轨结论相反：v3 vs v2 关键词轨 +0.0556 ✅ / judge 轨 −0.0682 ✅ | 远端 `results/eval2/report_20260723_024941.json/.md`（Orchestrator 转述） |

**两档成本分类**：🅰 纯 API/CPU（数小时内，不占 GPU）；🅱 需训练（单 run ≈7h GPU，短 run ≈2–3.5h）。

---

## 1. 实验矩阵总表

| 编号 | 名称 | 科学假设 | 成本档 | 预估耗时 | 优先级 | 可发表性 |
|---|---|---|---|---|---|---|
| E1a | judge 位置交换双跑 | H：v4>base 的 +1.00 judge 差在交换候选顺序后方向不变且幅度稳定（位置偏差 <5pp） | 🅰 | API ≈0.5h | **P0** | 高——直接回应 LLM-judge 评审质疑，方法章节必备 |
| E1b | 多评委交叉（kimi-k2 vs moonshot-v1-32k） | H：两个异源评委对 base/v2/v4 的排序一致（Spearman ρ>0.8），v4 结论不依赖单一评委 | 🅰 | API ≈1.5h | **P0** | 高——评委稳健性是论文可信度前提 |
| E1c | 重复评判翻转率（Coin Flip 复现） | H：本任务翻转率 ≤13.6% 文献均值；若更高，则 judge 轨全部结论需降级为"多数表决后有效" | 🅰 | API ≈1.5h | **P0** | 高——可形成独立方法学小节，复现 arXiv:2606.13685 范式 |
| E2 | v4 concept_explanation 退化归因 | H1（数据）：cap-1500"长答案优先"使 concept 子集训练分布漂移（保留集均值长度↑、风格偏移）导致关键词轨退化；H2（评测）：15 题子集 n 太小，差异在噪声内 | 🅰 | CPU 1h + API 0.5h | **P0** | 高——解释决策门"保留 v2"的核心负结果，是论文诚实性关键 |
| E3 | ARIZ 指标伪影分离（rubric 逐项重判） | H：ariz_step_coverage 的关键词漏判率显著>0；rubric 重判后 v2/base 差距收窄，v4 的 ARIZ 优势（kw 0.7494）在语义轨下依然成立 | 🅰 | API ≈1h | **P0** | 高——量化"表述差异 vs 能力差异"，直接回应关键词法局限 |
| E4 | 数据质量门消融（关近重复去重 / 关冲突丢弃） | H：v4 相对 v2 的 judge 增益主要来自冲突丢弃+近重复去重；关掉任一门后 judge overall 显著回落 | 🅱 | 2 run ≈14h GPU + 评测 2h（短 run 版 ≈7h） | P2 | 中——消融章节加分项；单 GPU 排期贵，且短 run 有步数混淆 |
| E5 | LoRA 模块覆盖消融（仅 Attention 层 vs 仅 DeltaNet 层） | H：混合架构中领域能力主要沉积在 Gated Attention 层（10 层）而非 DeltaNet 层（30 层），或反之 | 🅱 | 2 短 run ≈7h GPU + 评测 2h | P2 | 中高——Gated DeltaNet+MoE 混合架构的 LoRA 放置是新颖点，但非本文主线 |
| E6 | 通用能力回归基准扩题（30→120 题） | H：v2/v4 在扩题后通用探针上相对 base 无显著退化（差异 <10pp 分辨率），支撑"无灾难性遗忘"论断 | 🅰+GPU 推理 | 出题 1h + 推理 ≈2–3h GPU | P1 | 中——回归门槛证据，回应战略 KPI"通用损失 <3%"（retrospective:14） |

> 另注（不在本矩阵）：人工专家盲评 20–30 题（`docs/training_retrospective_2026-07-20.md:119,146` 建议）需 TRIZ 顾问真人参与，无法由 coder 在 DGX 执行，建议 Orchestrator 作为论文后续工作单独立项，不占本次 GPU/API 预算。

**推荐执行顺序**：E1a → E1b → E1c → E3 → E2（全部 🅰，一个工作日内完成）→ E6（P1，GPU 空闲时插队）→ E4/E5（P2，仅当论文审稿风险需要消融时启动）。

---

## 2. P0 实验详细执行卡

### E1a — judge 位置交换双跑（pairwise AB/BA）

**假设**：在 v4 vs base 的成对比较中，交换两个候选的呈现顺序后，v4 胜率方向不变；|胜率(AB) − 胜率(BA)| < 5pp（文献中未控制时可摆动 25pp，`eval_methodology_research.md:44`）。
**自变量**：候选呈现顺序（AB / BA）。**因变量**：pairwise 胜率、位置不一致率（同题两序裁决相反的比例）。
**材料**：复用远端已缓存的 v4 金标评测 responses（base 与 v4 各 100 题），**不重新生成、不占 GPU**。

**执行步骤**（coder 在远端执行）：
```bash
ssh chinux@spark-855a
cd /home/meerkat/mongoose_ai
eval "$(grep '^export MOONSHOT_API_KEY' ~/.bashrc)"
# 0. 验证缓存存在（文件名以实际 ls 为准）
ls results/v4/ results/eval2/ | grep -i response
# 1. 新建脚本 /tmp/e1_position_swap.py，逻辑：
#    - 读取 base/v4 的 100 题 responses（question, response_base, response_v4）
#    - 构造 pairwise prompt：给 question + reference_answer + 候选A + 候选B，
#      要求 judge 输出 {"winner": "A"|"B"|"tie", "reason": "..."}（JSON）
#    - AB 序：A=v4, B=base；BA 序：A=base, B=v4；temperature=0，模型 moonshot-v1-32k
#    - 批量 5 条/请求，遵守 RPM=3（sleep 20s/请求）；结果追加写 /tmp/e1_position_swap.jsonl（断点续跑）
venv_v5/bin/python /tmp/e1_position_swap.py --judge moonshot-v1-32k
# 2. 分析（纯 CPU）：
venv_v5/bin/python /tmp/e1_analyze.py   # 输出：AB 序 v4 胜率、BA 序 v4 胜率、位置不一致率、Wilson 95%CI
```
**资源与时间**：100 题 × 2 序 = 200 次裁决 ÷ 5 条/批 = 40 请求 × 20s ≈ **15–20 分钟**；零 GPU。
**预期可回答的论文问题**：judge 轨 +1.00 的优势是否是位置偏差伪影？（若位置不一致率 >10%，judge 轨所有数字须以双序平均重报。）
**风险**：① moonshot-v1-32k 不可用时按探测顺序回退（`pipeline_v4_remote/README.md:110`）；② 截断 500 字符的旧惯例（`eval_pipeline_v2/README.md:45`）在 pairwise 下会放大位置偏差——本实验 response **不截断或截断至 2000 字符**并记录该决策；③ RPM 实际限制未获取（README 记 RPM=3），脚本须捕获 429 并指数退避。

---

### E1b — 多评委交叉（kimi-k2-0711-preview vs moonshot-v1-32k）

**假设**：两个异源评委对 {base, v2, v4} 三方的金标打分排序一致（子集级 Spearman ρ > 0.8），v4 > base 的结论跨评委成立。
**自变量**：评委模型（2 个）。**因变量**：0–4 rubric 分（沿用 v4 harness 的准确性/完整性/TRIZ 正确性/结构四维，`pipeline_v4_remote/README.md:108-109`）、评委间 Spearman/Pearson 相关、逐题分差分布。
**执行步骤**：
```bash
cd /home/meerkat/mongoose_ai
# 0. 确认 v4 评测缓存中 moonshot-v1-32k 的逐题 judge 明细可直接复用（避免重跑）
ls results/v4/ | grep -i judge
# 1. 仅对 kimi-k2-0711-preview 新跑一轮：复用 pipeline_v4/src/eval_harness.py 的缓存机制
#    若 eval_harness.py 支持 --judge 参数覆盖（需 coder 读源码确认），直接：
venv_v5/bin/python pipeline_v4/src/eval_harness.py --config pipeline_v4/configs/eval_v4.json \
  --tag e1b_kimik2 --judge kimi-k2-0711-preview   # 参数名以源码为准；生成走缓存，仅 judge 新跑
#    若不支持，写 /tmp/e1b_rejudge.py 读取 responses 缓存 + v4_gold.jsonl，按同一 rubric 调 kimi-k2
# 2. 分析：/tmp/e1b_analyze.py —— 逐题配对的 Spearman ρ、均差、v4-base 差值的跨评委一致性（符号一致率）
```
**资源与时间**：3 模型 × 100 题 = 300 裁决 ÷ 5 条/批 = 60 请求 ≈ **25 分钟**（单评委）；若 v2/v4/base responses 缓存不全则按需补（GPU 推理另计，E 卡假设缓存齐）。总计 **≤1.5h**。
**预期可回答的论文问题**：结论的评委稳健性；kimi-k2 与 moonshot 系列是否给出同一故事（若分歧，报告"评委依赖"为局限）。
**风险**：① kimi-k2-0711-preview 可用性未获取——先 `--probe-judge`（`pipeline_v4_remote/README.md:67-70`）冒烟；② 自我增强偏差（数据由 moonshot 生成、评委同族，`eval_methodology_research.md:45`）——kimi-k2 与 moonshot 同属月之暗面家族，严格说非完全异源，报告中须如实标注；③ RPM 限制同上。

---

### E1c — 重复评判翻转率（Coin Flip Judge 范式复现）

**假设**：moonshot-v1-32k 在本任务上的单次翻转率 ≤ 13.6%（文献均值，`eval_methodology_research.md:46-47`）；且 5 次多数表决即可稳定复现最终裁决（文献需 ~11 次，本任务 rubric 锚定后应更低）。
**自变量**：重复次数（1–10 次）。**因变量**：翻转率（第 k 次裁决与多数裁决不一致的比例）、多数表决收敛曲线（k=1,3,5,7,9,11 时与 k=11 裁决的一致率）。
**执行步骤**：
```bash
cd /home/meerkat/mongoose_ai
# 1. 抽样：从 100 题中分层抽 40 题（按 6 个子集等比），pairwise v4 vs base，AB/BA 双序，
#    每序重复 5 次（共 40×2×5=400 裁决，temperature=0 —— 注意：T=0 下若 API 完全确定则翻转率恒 0，
#    故须同时跑 temperature=0.7 一组作对照，共 800 裁决 = 160 请求 ≈ 55min）
venv_v5/bin/python /tmp/e1c_flip.py --judge moonshot-v1-32k --temps 0,0.7 --reps 5 --n-questions 40
# 2. 分析：翻转率、双序×重复 的 20 格一致性矩阵、多数表决收敛曲线；与 13.6% 文献值对比
venv_v5/bin/python /tmp/e1c_analyze.py
```
**资源与时间**：API ≈ **1.5h**；零 GPU。
**预期可回答的论文问题**：单次 judge 裁决的可信度边界；v4 vs base 的 +1.00（McNemar p=1.7e-12）在翻转噪声下是否依然稳健（若翻转率 r，则有效样本量折损，需重算检验功效——与 W3 统计审查衔接）。
**风险**：① T=0 完全确定性的可能（则翻转率≈0，结论变为"确定性 judge 无需多次投票"，同样是可发表发现）；② 80 请求 × 2 温度的成本与 RPM；③ 抽样 40 题后子集层级的翻转率只能描述性报告（n 太小，`eval_methodology_research.md:61`）。

---

### E2 — v4 concept_explanation 退化归因

**背景**：keyword/concept_explanation：v4 0.4356 < base 0.5317 ≈ v2 0.5187（v4 金标终报）；决策门因此判"保留 v2"。concept 子集仅 15 题（`pipeline_v4_remote/README.md:104`），15 题尺度上 0.4356 vs 0.5317 约差 1.4 题——**先要回答"是不是噪声"，再回答"如果是真的，为什么"**。
**假设**：H1（数据漂移）：v4 数据门第 5 条 cap-1500"优先保留 output 更长者"（`pipeline_v4_remote/README.md:94`）改变了 concept_explanation 子集的长度/风格分布，使模型输出偏离金标期望关键词的表述习惯；H2（统计噪声）：15 题配对检验不显著，差异不可与噪声区分。
**自变量**：训练数据版本（v2 原始 vs v4 cap 后）。**因变量**：concept 子集条数、output 长度分布（均值/p50/p95）、v4 被丢弃样本的特征、15 题逐题关键词命中明细、judge 重判分。
**执行步骤**：
```bash
cd /home/meerkat/mongoose_ai
# 0. 读取数据报告（各质量门计数，含 cap 门进/出条数）
cat results/v4_data_report.json | venv_v5/bin/python -m json.tool | head -80
# 1. 数据侧分析（纯 CPU，秒级）：/tmp/e2_data_attr.py
#    - 载入 data/processed/v4_train.jsonl，筛 subset==concept_explanation
#    - 对比 v2 语料中 concept 子集（v4 输入 10327 条，pipeline_v4_remote/README.md:87）：
#      条数、output 长度分布、被 cap 丢弃条数及其长度分布
#    - 检验：保留集 vs 丢弃集长度差（Mann-Whitney U），v4 vs v2 concept 均值长度差
venv_v5/bin/python /tmp/e2_data_attr.py
# 2. 评测侧逐题失败分析：/tmp/e2_item_analysis.py
#    - 从 responses 缓存取 base/v2/v4 在 15 道 concept 题的回答 + 期望关键词
#    - 输出逐题表：哪些关键词 v4 漏而 base/v2 中；漏判词是同义表述（伪影）还是真缺失
#    - 对 15 题 × 3 模型做 judge rubric 重判（0-4），与关键词轨对照（≈10 请求，5min）
venv_v5/bin/python /tmp/e2_item_analysis.py
# 3. 统计：15 题 McNemar / 配对 bootstrap（沿用 eval_harness 的统计函数），给出"v4 vs base 差异是否显著"的直接答案
```
**资源与时间**：CPU ≈1h + API ≈10 分钟；零 GPU。
**预期可回答的论文问题**：v4 的 concept 退化是数据门副作用（→ 论文 limitation + v5 修复方向：cap 内随机保留或按多样性保留）还是统计噪声（→ 决策门"保留 v2"的论据需重写）；同时产出 v5 数据门的具体修复建议。
**风险**：① v2 原始语料 concept 子集的准确条数未获取（须远端实测）；② 若 cap 门进出条数很小（concept 总量 <1500 则 cap 未生效），H1 直接证伪，需转向 H3（其他门：近重复去重误伤同义好样本）；③ 15 题功效不足是结构性限制，结论措辞须保守。

**E2 终判与 v4.1 修复落地（2026-07-27）**：终判见 `paper/experiments/STATE.md`（E2 concept 退化终判）——非噪声也非伪影，v4 真实减少了 TRIZ 工具术语言名复诵（真缺失 8 词次，synonym 伪影 0 条），但 judge 轨与 v2 持平。修复按 `paper/v5_plan/sec1_data.md` §4 回填 v4 为 **v4.1**：`pipeline_v4_remote/configs/data_v4.json` rebalance 弃用 longest_first 改 `term_coverage_random`（术语言表 v1.0 贪心覆盖 60% + 长度三桶分层随机 40%，cap 1500→2500）；`pipeline_v4/configs/train_v4.json` 仅改 run 标识为 v4.1，超参全冻结（唯一自变量 = rebalance 策略）；执行链 `pipeline_v4_remote/run/chain_v4_1.sh`（复用金标与四方评测，评测锚点 = v2 干净锚点，产出 `results/v4_1_final_report.md` 五方对比）。**验证标准：keyword/concept_explanation v4.1 vs v2 差值 CI 回到包含 0（v4 为 -0.083 [-0.148, -0.023]）且 judge 轨不出现反向。**

---

### E3 — ARIZ 指标伪影分离（rubric 化逐项重判）

**背景**：ariz_step_coverage 用 6 个步骤名关键词匹配，同义表述会被漏判（`eval_methodology_research.md:31-33`）；v2 ariz 回退（base 0.389 → v2 0.250，`results/METRICS_LEDGER.md:9`）与 v4 ariz_guidance 四方最高（kw 0.7494 / judge 2.85）之间的故事需要语义轨裁决。
**假设**：H1：关键词轨对 v2/base 的漏判率显著高于 0（即部分"回退"是表述差异伪影）；H2：rubric 语义重判后 v4 > {v2, base} 的方向不变（v4 的 ARIZ 优势是真实能力差异）。
**自变量**：评分方法（关键词 vs rubric-LLM）。**因变量**：6 步骤逐项命中率、漏判率（rubric 判 1 而关键词判 0 的比例）、误判率（反向）、两种评分下模型间差值。
**执行步骤**：
```bash
cd /home/meerkat/mongoose_ai
# 1. 从 100 题金标中取 ariz 20 题；从缓存取 base/v2/v4（+v1 可选）的 responses
# 2. /tmp/e3_ariz_rubric.py：对每题每模型，让 judge 按 ARIZ 6 步骤逐项判 0/1 + 一句理由
#    （rubric 锚定：每步给定义+正例，eval_methodology_research.md:48 建议），
#    输出 JSON：{"step1": 0/1, ..., "step6": 0/1, "rationales": [...]}
#    4 模型 × 20 题 = 80 裁决 ÷ 5 条/批 = 16 请求 ≈ 10 分钟；judge=moonshot-v1-32k, T=0
venv_v5/bin/python /tmp/e3_ariz_rubric.py --judge moonshot-v1-32k
# 3. /tmp/e3_analyze.py：
#    - 逐模型逐步骤：关键词分 vs rubric 分并列，漏判率 + 具体漏判表述清单（回流 KEYWORD_MAP 候选，
#      eval_methodology_research.md:81 建议#3）
#    - 重算模型间对比：rubric 轨下 v4 vs v2 vs base 的配对 bootstrap 95%CI
#    - 判定：v2"ARIZ 回退"在 rubric 轨下是否消失/缩小/维持
```
**资源与时间**：API ≈ **1h** 内；零 GPU。
**预期可回答的论文问题**：① 关键词评测在本任务的测量误差量级（可写成方法学发现）；② v2→v3→v4 的 ARIZ 叙事（"定向补数据有效"）在语义轨下是否成立；③ 产出一版扩充的中英别名映射表（工程副产品）。
**风险**：① rubric 评委自身不可靠 → 依赖 E1 包先行背书（故 E1 排在 E3 前）；② ariz 仅 20 题，步骤级 n=20 只能描述性报告；③ judge 输入截断长度须 ≥ ARIZ 长回答（ARIZ 回答通常很长，500 字符截断不可用于本实验）。

---

## 3. P1/P2 实验概要卡

### E6（P1）— 通用能力回归基准扩题
- **设计**：将 `eval_pipeline_v2/general_probe.json`（30 题，6 类×5）扩至 120 题（6 类×20，沿用同 schema：question + expected_keywords）。由 Moonshot API 批量生成 + 规则校验（答案词必须在参考答案中出现），人工抽检 10%。
- **执行**：出题 ≈1h（API）→ 推理：base/v2/v4（v1/v3 可选）各 120 题，temperature=0，复用 eval2.py 的 generate 框架 ≈2–3h GPU（按 v1 实测 p50 延迟 ~29s/题估算上限，实际开放域短答更快）→ 关键词自动评分 + 配对 bootstrap。
- **回答**：v2/v4 是否存在领域微调导致的通用能力退化（回应"通用损失 <3%"的战略 KPI，`docs/training_retrospective_2026-07-20.md:14`）；30 题 → 120 题把分辨率从 ~18pp 提升到 ~9pp。
- **风险**：生成题质量参差（须抽检）；关键词评分对开放题的固有局限（写作/指令跟随类误判率高，可对这两类加 judge 轨）。

### E4（P2）— 数据质量门消融
- **设计**：以 v4 管线为基准，两个变体各训一版：A) 关闭近重复去重（门 3）；B) 关闭冲突组丢弃（门 2）（门定义见 `pipeline_v4_remote/README.md:91-92`）。配置改 `configs/data_v4.json` 对应开关 + `configs/train_v4.json` 输出目录，其余全同。
- **成本**：完整 run 2×7h GPU + 各 1h 金标评测 ≈ **16h**；短 run 版（1 epoch ≈3.5h×2）≈9h，但短 run 与 v4 完整 run 不可直接比（步数混淆），只能 A vs B vs "v4 同预算短 run 对照"三方比——**若走短 run 版，必须补第三个对照 run，成本升至 ~10.5h+**。
- **回答**：v4 相对 v2 的 judge 增益中，数据门各自的贡献份额。
- **风险**：GPU 排期长；负结果风险（消融后无显著差异 → 结论只能是"v4 增益非单一门贡献"）；建议仅当 E2 显示数据因素重要时启动。

### E5（P2）— LoRA 模块覆盖消融
- **设计**：三个 target_modules 变体（仅 10 层 Gated Attention 的 q/k/v/o/gate 投影；仅 30 层 Gated DeltaNet 的线性注意力投影；全 12 模块=v4 复现对照），同数据同预算短 run（1 epoch）。基座为 Gated DeltaNet 30 层 + Gated Attention 10 层 + MoE 混合架构；已知 LoRA 仅覆盖 0.24% 参数且不触 routed expert（`docs/training_retrospective_2026-07-20.md:75`）。
- **成本**：3 短 run ≈10.5h GPU + 评测 ≈3h。若预算紧张砍对照组（用 v4 既有结果插值，但须注明预算不等价）。
- **回答**：混合架构上 LoRA 放置的架构先验——这是本文潜在新颖点之一（DeltaNet 类线性注意力层的 LoRA 行为文献稀缺）。
- **风险**：模块名清单需 coder 从模型 config/peft 适配器 key 清单实测（未获取）；DeltaNet 层投影维度小可能导致 rank=64 相对过参数化，属已知混淆，报告中注明。

---

## 4. 全局风险与执行纪律

1. **API 预算**：E1+E2+E3 合计 ≈1,500 次裁决（≈300 请求），RPM=3 下纯等待 ≈2.5h；务必全部脚本断点续跑 + 429 退避。
2. **judge 同源问题**：数据与金标均由 moonshot 生成、评委同族——E1b 的"异源"只是弱异源，论文中须如实声明（`eval_methodology_research.md:45` 自我增强偏差）。
3. **统计口径**：子集 n=15~20 的结论一律标注"描述性"；overall n=100 只报 ≥10pp 级差异（`eval_methodology_research.md:56-61`）。
4. **产物归档**：所有脚本与结果落远端 `results/e1|e2|e3|e6_*/`，回传本地 `paper/experiments/`（W3 负责回传通道），并登记 `results/METRICS_LEDGER.md`。
5. **不改环境**：venv_v5 不可 pip install；BERTScore 不可用——E3 的语义轨用 rubric-LLM 替代（与 `eval_pipeline_v2/README.md:46` 的降级决策一致）。
6. **GPU 纪律**：🅱 类实验（E4/E5）串行排队，单 run 期间不并行其他 GPU 任务；E6 推理可在 🅰 类 API 实验等待间隙插入。
