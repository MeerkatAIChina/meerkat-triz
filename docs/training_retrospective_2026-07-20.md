# 猫鼬AI 训练目标与过程复盘报告

> 日期：2026-07-20 ｜ 范围：v1.0 全周期（2026-05 → 2026-07）＋ 唯一完整训练 run（2026-06-19）
> 方法：7 个维度并行调研（目标对齐 / 数据管线 / 训练执行 / 评测体系 / 工程流程 / 战略对齐 / 外部最佳实践），全部结论均有文件级证据

---

## 一、总体判断

**v1.0 交付的是"一条能跑通的训练管线"，而不是"一个被验证有效的模型"。**

- 工程层面（管线、checkpoint、审计闭环、决策留痕）在同类个人/小团队项目中属于上乘；
- 但"训练是否有效"这一根本问题至今没有任何量化证据：训练后评测从未成功执行，唯一的质量信号是 eval_loss 1.3979——一个无对照、无阈值、且因 loss 口径问题而难以解读的孤立数字；
- 战略层定义的量化成功标准（TRIZ 专项 ≥75%、幻觉 ≤8%、通用能力损失 <3%、方案生成准确率 >80%）**零项被验证**。

根本原因是：**顶层目标从未被分解为可衡量的验收标准**，过程管理因此自然退化为"流程完成"导向——v1.0 的 29 条需求和全部 Phase 的 Success Criteria 都是"工件存在、配置正确、流程跑通"型，没有一条是"模型能力提升 ≥ X%"。

---

## 二、对训练目标的反思

### 2.1 目标定义：宏大但不可测

- 顶层目标"world-class TRIZ innovation consultant"（PROJECT.md:9）无任何可验证标准——无目标分数、无对比基线（如"达到 GPT-4o 在 TRIZ 任务的 X%"）、无人工评审 rubric。
- 技术方案文档其实定义了量化标准（TRIZ-Concepts ≥75%、Contradiction ≥75、Product-Innovation ≥70%、Case-Retrieval ≥80%、幻觉率 ≤8%、通用损失 <3%），但这些标准**从未进入项目的执行性需求**——战略文档与执行体系是两张皮。
- 战略报告把本微调定位为 AITRIZ 引擎的能力组件，M12 KPI 是"方案生成准确率 >80%"——当前没有任何测量手段能支撑这个 KPI。

### 2.2 目标演变：两次 pivot 方向正确，但留下未清算的账

1. **合成数据 → 真实语料（2026-07-12 正式决策）**：方向正确，与公司"真实数据资产构筑护城河"的战略一致。但后果是：作为 v1.0 关键路径建成的 ~6K 合成数据管线（02b + synthetic_pipeline.py）**建成后从未被训练使用**，其去留至今无正式裁决。
2. **真实语料 V1 → V2 多角度（进行中）**：`run_corpus_sft_v2.py` 计划 ~11.7K 样本，动机（修正 V1 子集失衡、空 think 块污染）成立，但**在没有先定义质量验收标准的情况下启动**，存在重蹈"先建管线后补目标"的风险。

### 2.3 目标的静默降级：一个危险的先例

DATA-02 要求真实数据占比 20–30%，实际 ~8.7%。处理方式是"记录偏差 + 照常打勾 Complete"，需求文本未改、目标值未重定基线。**这架空了需求勾选制度的含义**——如果偏差不需要修订需求，那需求就只是形式。

### 2.4 eval_loss 1.3979 不能证明目标达成

- 无基座模型在同一 val 集（313 条）上的对照 loss；
- 因 loss 实际对全文（system+user+assistant）计算（见 3.2），数值与常规 completion-only SFT 不可比；
- 无预设阈值，训练前从未定义"eval_loss 多少算成功"。

---

## 三、对训练过程的反思

### 3.1 数据：最大的问题源头

**种子数据（实为 385 条，不是文档里的 548 条）**
- 2026-06-18 提交 `2f72fa6` 移除了 163 条完全重复样本，但 548 的说法至今残留在 `CLAUDE.md:36`、`README.md:233`、`utils/data_utils.py:69`、`PROJECT.md`——文档系统性滞后于代码。
- 残留质量问题：`innovation_assessment` 有 **10 组"同题不同答"**（去重只认完全相同的样本，冲突监督信号会原样进入训练）；答案高度公式化（多条以同一句"综合评估：……建议进一步进行专利检索和竞品分析"收尾）；26 条 output 不足 60 字符。
- 种子由谁编写、是否经 TRIZ 专家审核：**仓库无任何记录**。而技术方案明确要求"TRIZ 顾问人工审核合成数据，通过率 ≥95%"——专家环节被完全绕开。

**实际训练数据（2,662 条 TRIZ-raw corpus SFT）——带着三重已知缺陷训练**
- V1 生成路径**零质量门**：无去重、无最短长度过滤、无 think 块清洗、子集自由分类（`corpus_to_sft.py` 默认参数 `min_output_chars=0, dedup=False`）。V2 代码注释自证 V1 数据存在"子集失衡 + 空 `<think></think>` 块污染 + 未去重"。
- 该数据集**不在本仓库**（仅在 DGX），无 manifest（条数/分布/生成参数/哈希），无法复现"2,662 条到底是什么"。
- 子集失衡：ariz_guidance 种子仅 22 条——ARIZ 恰是 TRIZ 最硬核的能力。
- **Safety-Refusal 子集整体缺失**：技术方案规划 400 条（7%），两轮数据均为零。面向客户交付/私有化部署场景，这是合规与品牌风险。

**真实占比 8.7% 的代价被低估**
- 决策文档只算了 02b 路径的账；实际 corpus 路径下问答 100% 由 Moonshot 生成，"真实"仅体现在基于真实教材片段 grounding——**Moonshot 的表述风格就是模型的天花板**。外部实证：等量真实数据微调一致优于合成（+11.9%，arXiv 2602.04482）。

### 3.2 训练执行：方向健康，但存在三个技术性硬伤

**好的方面**：超参选择（r=64/α=128、lr=2e-4、cosine、dropout=0.0、显式 12 模块）与 2025–2026 年实证最佳实践（VERT 在同代 Qwen3-A3B MoE 上的配方、LoRA 文献）一致；实际训练仅 1 小时 52 分（10.0 s/step × 666 步），成本远低于"15 小时"的陈旧预估。

**硬伤 1：loss 计算范围错误认知**。两个入口都用 `formatting_func` 返回整段对话纯文本，TRL 在该模式下对**全文计 loss**，而 `training_utils.py:411` 注释声称"自动只计算 assistant 回复部分的 loss"——该注释很可能不正确。后果：训练容量被 prompt 复述分走；eval_loss 不可与常规 completion-only SFT 比较；checkpoint 探针 loss 上升（4.87→4.96）与 eval_loss 下降的背离难以解读。

**硬伤 2：max_seq_length 失控（疑似静默截断）**。TRL v1 移除该参数后未在任何入口显式设置；corpus SFT 样本按 2,048-token 目标生成，若 TRL 1.5.1 默认 1024，长回答样本已被静默截断——需在 DGX 上立即验证。

**硬伤 3：明显欠训练**。best eval_loss 出现在最后一步（666），4 次 eval 末次最优，无任何过拟合信号即停训——2 epochs 对 2,662 条数据偏少。

**其他关键发现**：
- **04_worked.ipynb 作为审计工件不可复现**：cell 6 源码含 Unicode 弯引号（Python SyntaxError）却存有成功输出（stale outputs）；逐步 loss 日志因 tqdm HTML widget 未持久化而**永久丢失**。
- **LoRA 对 MoE routed expert 主体基本无覆盖**：可训练参数 84.66M/34.7B = 0.24%，领域知识大量存于 expert；适配器实际挂载位置未验证。
- **4-bit 量化覆盖存疑**：加载显存 62.61GB 远超 ~20GB 预期，可能是 expert/自定义层未被 bnb 量化。
- **发货权重=末步内存态**：无早停、无 best-checkpoint 回载；本次巧合 best=末步，流程不保证下次如此。
- **resume 从未实测**（两条路径并存），且 PEFT v0.19+v5 有 WeightConverter bug 前科。
- **config.py / utils / notebook / script 四方参数漂移**：fp16 在 config 写 True、实际跑 False；`group_by_length` 死配置；`qlora_trtiz_v1` 拼写错误传播。

### 3.3 评测：整个闭环最薄弱的一环

**训练后评测从未成功执行**——Notebook 05 有 4 个关键 cell 语法损坏（换行符丢失）且 0 个执行输出；补写的 `eval_adapter_vs_base.py` 从未跑过；唯一有据可查的基线是 2026-06-04 的一次 Layer 2，但分数文件不在仓库。**"适配器比基座好多少"没有任何数字。**

调研还发现了 3 个此前未知的实现 bug：
1. **原理识别分母 bug**（`benchmark_utils.py:538`）：分母用全部 30–40 题而分子只统计 10 道选择题——即使全对，准确率上限也只有 25–33%。该指标占 overall_score 权重 0.3，2026-06-04 基线分数已被污染。
2. **BLEU/ROUGE 恒被跳过**（`evaluate_case_quality()` 的 predictions/references 长度判断恒为 False）——"看起来有、实际没有"的指标。且 BLEU/ROUGE 对开放式 TRIZ 案例生成在方法论上本就不适用（业界共识）。
3. **`eval_adapter_vs_base.py:55` 键名 bug**：`peak_memory_gb` vs `memory_peak_gb`，峰值内存永远为 None 且被 `(… or 0)` 静默吞掉。

**方法论差距**（对照 ref/DGX_Spark_大模型评测方案 docx）：40 题硬编码 vs 方案 400 题 6 类（缺幻觉检测、Case-Retrieval、Consulting-QA）；关键词匹配 vs 方案"TRIZ 顾问 + GPT-4o 双评（一致性>85%）"；综合评分 40/35/25 加权 + S–D 分级未实现；评测温度 0.7 无 seed，before/after 差值可能只是采样噪声；157 条 held-out test split 从未被使用。

### 3.4 工程与流程：纪律好，但有两个"单点"

**做得好的**：审计驱动闭环真实运转（3 个 P0 全闭环、BLK-01 专项 Phase 3.1）；提交信息规范；决策留痕 20+ 条；SHA-256 双向同步核验；复盘诚实。

**单点 1：测试体系失去信号价值**。全量套件实测 11 failed / 9 errors（`test_benchmark_utils.py` 的 sys.modules 全局污染导致顺序依赖；conftest 与 corpus_builder 测试依赖真实 torch/pymupdf）。`test_metrics.py`/`test_report.py` 是纯 AST 字符串断言的"假测试"。最近只局部跑过 9 个新测试——防回归网形同虚设。

**单点 2：DGX 同步是纯手工流程且已再次漂移**。远程 `.git` 无提交，靠人肉 SHA-256 比对；07-18 刚宣布"完全一致"，07-19 就出现 4 个未入库文件——包括复现唯一成功训练的 `train_qlora.py` 和待合并的 `corpus_to_sft.py.patchwork`（**V2 全量生成 7 小时任务若不打这个 400 错误处理补丁，任何一条 chunk 触发内容过滤就会中断整个任务**）。全部训练产物与评测结果仅存 DGX，故障即丢失。

**流程老问题**：规划文档反复过期（548 vs 385、"5 题" vs 40 题、"13 tests" vs 58 个、15 小时 vs 1.9 小时），RETROSPECTIVE 已两次记录该教训但未形成机制；notebook（干净版/执行版）+ scripts 三份训练逻辑副本并存，改一处其余即腐化。

### 3.5 战略对齐：进度超前，但"效果证据"是解锁一切的总闸门

- 项目比 30 个月战略路线图提前约 4–5 个月启动，卡位正确（TRIZ×LLM 蓝海）；
- 但当前产物（一个未经验证的适配器）距离战略卖点"微调后评测分数作为私有化部署核心卖点"还差三个里程碑：**经验证的效果证据 → 可使用的部署形态 → 可信的公开发布**；
- 部署被划为 Out of Scope，导致适配器没有任何可用形态，内部咨询师无法试用——战略"内部咨询效率提升 20%"无从起步；
- 开源发布的 IP 边界未经确认：TRIZ-raw corpus 是否含专有客户案例，需创始人/法务层面裁决，这与"AITRIZ 是核心 IP 壁垒"的战略定位直接相关。

---

## 四、进一步提升方案

### 阶段 P0：先回答"训练到底有没有用"（1–2 周，解锁一切后续决策）

| # | 行动 | 落点 | 代价 |
|---|------|------|------|
| 1 | **跑通训练后评测**：修复 `eval_adapter_vs_base.py` 键名 bug（:55）与温度参数（0.7→0.0），在 DGX 执行；同时加 `--ppl` 模式用 157 条 held-out test 做基座 vs 适配器困惑度对比（交叉验证）；结果**提交进 git 的 `results/`** | scripts/ | DGX 半天–1 天 |
| 2 | **修复评测三 bug**：分母 bug（benchmark_utils.py:538）、BLEU/ROUGE 对齐（:596-620）、加回归测试"10 题全对 → accuracy == 1.0" | utils/, tests/ | 半天 |
| 3 | **为"世界级 TRIZ 顾问"写下可衡量验收标准**：新建 `.planning/REQUIREMENTS.md`，定义 Layer 2 绝对分阈值 + 适配器 vs 基座 delta 阈值 + 基座对照 eval_loss + 人工评审 rubric（N=20–30 真实案例，专家盲评，≥80% 认可度为发布门禁） | .planning/ | 0.5 天 |
| 4 | **本地审计已训练数据**：在 DGX 统计 2,662 条的空 think 块数、重复率、子集分布、长度分布——形成 v1 适配器的已知缺陷清单与 v2 的对比基线 | DGX 一次性脚本 | 半天 |
| 5 | **合并 .patchwork + 入库关键脚本**：BadRequestError 处理合入 `utils/corpus_to_sft.py`；提交 `train_qlora.py`、`eval_adapter_vs_base.py`、`run_corpus_sft_v2.py` 与 V2 改动；删除 `EOF` 空文件 | repo | 1 小时 |
| 6 | **修复测试套件**：sys.modules 注入改为带 teardown 的 fixture；torch/fitz 依赖加 `pytest.importorskip`；验收标准=裸环境 `pytest tests/ -q` 稳定全绿 | tests/ | 半天 |
| 7 | **解释 train/eval loss 背离**：核实 TRL 1.5.1 下 formatting_func 路径的 loss 掩码行为与默认 max_length；结论写入训练记录 | training_utils.py | 2–4 小时 |

### 阶段 P1：下一轮训练（v2）——把硬伤全部修掉（2–4 周）

1. **数据侧（最大杠杆）**
   - V2 全量生成（~11.7K，先立验收标准再跑：子集配额、去重规则、≥2% 人工抽检、真实占比目标值）；
   - **真实锚定**：385 条种子 2–3 倍上采样进训练集；种子二次清洗（解决 10 组同题不同答、公式化收尾、26 条超短回答）；
   - **质量门升级**：在现有三道规则门后追加困惑度过滤（剔除分布尾部 10–20%，无额外 API 成本）＋ 可选裁判模型 rubric 打分；
   - **补 Safety-Refusal 子集** 200–400 条；
   - **数据 manifest 制度化**：每版训练集落 `data/MANIFEST.md`（条数/分布/参数/哈希/对应 run）；02b/02d 改用时间戳目录禁止互相覆盖。
2. **训练侧**
   - `num_train_epochs: 2 → 4`、`eval_steps: 200 → 100`、加 `EarlyStoppingCallback(patience=3)`（4 epochs ≈ 3.7h）；
   - **completion-only loss**：改用 prompt/completion 格式或 `assistant_only_loss=True`（需验证 chat template 含 `{% generation %}`）；
   - **显式锁定 `max_length=2048`**（用 SFTConfig，按 v2 数据 p95 取值）；
   - **发货 best checkpoint 而非末步**：从磁盘加载 best checkpoint 再 save_adapter_only；
   - **训练日志持久化**：`trainer.state.log_history` 落盘进 git（v1 逐步 loss 已永久丢失，不能再犯）；
   - 超参排序小实验：lr ∈ {1e-4, 2e-4, 5e-4} × rsLoRA ∈ {False, True} 共 6 组各 0.5 epoch（LoRA 对 lr 最敏感；config.py:65 自己注明大 rank 建议开 rsLoRA 但当前关闭）；
   - 核查 LoRA 对 routed expert 的覆盖与 4-bit 量化实际范围（导出适配器 key 清单 + bnb 模块统计）。
3. **评测侧**
   - 评测集扩至 50–100 题 + **新增幻觉检测集 20–40 题**（对 91% 机器生成数据训练的模型最关键）；题库外置为 `data/eval/triz_benchmark_v2.json`；防泄漏检查（与训练集 n-gram 比对）；
   - **引入 LLM-as-judge** 替代 BLEU/ROUGE 作为主信号：4 维 rubric（概念正确性/矛盾分析完整性/原理适用性/案例可行性）、带参考答案、结构化 JSON 输出、成对比较交换位置跑两遍、20–30 条子集上做裁判–人工一致性校准（专业领域零样本裁判不可靠，需校准）；
   - 统一评测精度口径：对比评测一律 FP16 同进程；temperature=0.0 + 固定 seed；
   - 修复 Notebook 05 的 4 个损坏 cell；补跑 Layer 1 训练后评测（验证"通用能力损失 <3%"，即灾难性遗忘检查）；
   - **请 TRIZ 顾问对 20–30 条 base vs adapter 输出盲评**——把公司最大的专家资产接入质量回路。
4. **工程侧**
   - DGX 项目目录初始化为真正的 git 工作副本（pull/push 取代人肉 SHA-256 同步）；
   - 训练逻辑单一事实源：`scripts/train_qlora.py` 为 canonical，notebook 退化为薄封装，`04_worked.ipynb` 归档；
   - 修正 config.py 与实际运行的漂移（fp16、死配置、拼写）；
   - `utils/__init__.py` 瘦身（import 不拉 torch）；
   - 里程碑关闭加"文档新鲜度"硬门槛（数字型事实与代码 diff 一致）。

### 阶段 P2：中期路线（1–3 个月）

1. **Hugging Face 开源发布**（前提：P0 效果证据 + IP 边界裁决）：发布 LoRA 适配器 + 合并权重双 repo；model card 按官方规范（`base_model`、`library_name: peft`、`license`、`datasets`）；许可证三核实——Qwen3.6 基座许可、Moonshot API 生成数据条款、TRIZ 语料版权归属（未澄清前数据集不得公开）。
2. **最小可用部署**：adapter 合并 + 内部 demo 服务，供咨询师试用收集反馈——这是战略 KPI"AI 辅助项目占比 >30%"的测量前提。
3. **DPO/偏好对齐的触发条件**：当 LLM-judge 显示 SFT 在"原理适用性/案例可行性"维度停滞时再上；偏好对可低成本构造（chosen=裁判高分输出，rejected=基座同 prompt 输出），lr 5e-7~1e-6 并混入 SFT loss 防遗忘。**当前阶段不建议做**。
4. **合成管线去留裁决**：v2 corpus 路线若成立，将 `synthetic_pipeline.py` + 02b 标记 archived 或改造为"种子改写扩充"辅助角色。
5. **评测常态化**：每月 Layer 2+3、每季度 Layer 1，落成定时任务，产物写回 `results/` 并登记 pipeline_state；建立跨版本效果台账（`results/METRICS_LEDGER.md`）。
6. **与 AITRIZ 引擎路线图挂钩**：明确本 repo 产物如何接入 LangGraph AITRIZ 引擎，避免模型成为"练完即弃"的孤岛资产。

---

## 五、一句话总结

**项目的 LoRA 配置本身已处于 2025–2026 实证最佳实践区间，工程纪律也在线；最大的提升杠杆不在训练侧，而在评测侧（先证明"有用"）与数据侧（真实锚定 + 质量门 + 专家回路）。在拿到"适配器显著优于基座"的证据之前，不建议投入开源发布或客户侧展示；在效果证据到手之后，v2 训练才有明确的优化靶心。**
