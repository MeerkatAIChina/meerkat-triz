# v5 训练配置方案与论证 —— W2 训练方案支柱

> 作者：方案设计师_W2 ｜ 日期：2026-07-24 ｜ 状态：待 Orchestrator 整合进 v5 总方案
> 范围：仅训练侧（超参 / target_modules / 精度 / loss 口径 / 序列长度 / checkpoint 制度 / 早停 / 训练侧验证路线与成本）。数据门与评测门的 v5 变更属 W1/W3 支柱，本文件只在接口处引用。
> 证据强度标注：**【已证实】**= 本项目实验/文件直接支撑；**【文献支持】**= 外部文献支撑、本项目未实测；**【推断】**= 由证据外推的工程判断。
> 基线事实：单 v4 级训练 run ≈7h 预算（v4 实测早停于 3h04m，`evidence_table.md` ①）；DGX spark-855a GPU 空闲；base/v1–v4 适配器与金标缓存齐备。

---

## 0. v5 训练配置总表（默认值 = 本文件全部决策的落点）

| 项 | v5 值 | 与 v4 差异 | 决策节 |
|---|---|---|---|
| base_model | `models/Qwen3.6-35B-A3B` | 不变 | — |
| LoRA r / α / dropout / bias | 64 / 128 / 0.0 / none | 不变 | §1.1 |
| use_rslora | **False（默认）**，待 §8-P0 六组小扫裁决 | 不变（config.py:65 的注释遗留问题由小扫正式回答） | §1.2 |
| learning_rate | **2e-4（默认）**，待 §8-P0 裁决 | 不变 | §1.1 |
| lr_scheduler | cosine，**horizon 改为 2 epoch**（不再按 4 epoch 设计） | **变更**（修 v4 早停时 lr 仍 1.553e-4 的缺陷） | §1.3 |
| num_train_epochs | 4（仅作安全上限，实际由早停/horizon 决定） | 不变 | §1.3 |
| batch | 1 × grad_accum 8（有效 8） | 不变 | §1.4 |
| warmup_ratio / seed | 0.05 / 42 | 不变 | §1.4 |
| optimizer | HF 默认 adamw_torch（与 v4 实际运行一致，不切换 8bit） | 不变（显式声明） | §1.4 |
| target_modules | 12 模块显式清单，不变 | 不变 | §2 |
| 精度 | 纯 BF16 不量化 | 不变（终裁） | §3 |
| loss | completion-only（prompt/completion 格式 + 启动断言） | 不变 | §4 |
| max_length | **2048 显式锁定 + 启动断言** | 不变（断言为新增硬化） | §5 |
| eval_steps / save_steps | 100 / 100（强制相等） | 不变 | §6 |
| save_total_limit / best 制度 | 8 + BestCheckpointCallback + 磁盘发货 | 不变 | §6 |
| 早停 | patience=3，**min_delta=0.002（新增显式值）** | 参数化补齐 | §7 |

---

## 1. 超参

### 1.1 学习率：默认保持 2e-4，但最终值由六组小扫裁决

**决策**：v5 主 run 的 lr 默认 2e-4；在 §8-P0 的 lr ∈ {1e-4, 2e-4, 5e-4} × rsLoRA ∈ {False, True} 六组 0.5-epoch 小扫完成后，按判据（§8.1）锁定终值。若小扫平局，取 2e-4。

**论证**：
- 【已证实】v1–v4 四轮全部使用 2e-4（`evidence_table.md` ①"学习率/调度"行），在该 lr 下四轮训练均稳定收敛，无发散/梯度爆炸记录；v4 mean_token_accuracy 0.3954→0.6889、v2 0.4813→0.7919（`evidence_table.md` ③ 末注）。改 lr 是本配置空间中唯一没有本项目证据支撑的方向，默认不动。
- 【文献支持】LoRA 对 lr 的敏感度高于其他超参，且 LoRA 所需 lr 显著高于全参微调（Biderman et al. 2024, arXiv:2405.09673，`related_work.md` B1 #10）；领域 SFT 的退化高度依赖学习率与配置（Lin et al. 2025, arXiv:2509.20758，`related_work.md` B1 #9）。项目自身复盘亦指出"LoRA 对 lr 最敏感"，并早在 v2 规划期就开出了 lr ∈ {1e-4, 2e-4, 5e-4} × rsLoRA 六组各 0.5 epoch 的小扫处方（`docs/training_retrospective_2026-07-20.md:139`）——该处方至今未执行，v5 应正式兑现。
- 【已证实】2e-4 并非无代价：v4 的 cosine 按 max_steps=2,872 设计，step 1,000 早停时 lr 仍有 1.553e-4（`evidence_table.md` ⑤-11），即 v4 全程在高 lr 段运行、从未进入退火段——"2e-4 是不是过高"这个问题在 v4 数据上是开放未答的，只能靠小扫回答。

**风险与缓解**：小扫 0.5 epoch 的结论可能不可外推到完整训练（短跑偏好高 lr）。缓解：判据不只看 eval_loss 终点，还要求 top-2 臂过 40 题金标双轨冒烟（§8.1），下游质量与 loss 脱节的教训已发生过一次——v3 eval_loss 1.0705 ≈ v2 1.0673 但 judge 轨显著更差 −0.0682 [−0.1078, −0.0296]（`evidence_table.md` ②-C），故不允许仅凭 loss 锁定 lr。

### 1.2 rsLoRA：默认 False，与 lr 联合小扫后裁决

**决策**：v5 默认 `use_rslora=False`（与 v1–v4 可比）；rsLoRA=True 的三臂（lr ∈ {1e-4, 2e-4, 5e-4}）进入 §8-P0 小扫。若 rsLoRA 臂胜出，v5 采用胜出臂并如实记录与 v4 的不可比性。

**论证**：
- 【已证实】`config.py:65` 自己注明"use_rslora: False，# 大rank时建议开启rsLoRA"——配置作者知道该开但四轮都没开，这是悬而未决的配置债。
- 【文献支持】rsLoRA（Kalajdzievski 2023, arXiv:2312.03732，[arXiv 页面](https://arxiv.org/html/2312.03732v1)）将缩放因子从 α/r 改为 α/√r，解决大 rank 下标准 LoRA 缩放导致的梯度塌缩/高秩适配器利用不足。本项目 r=64、α=128：标准缩放 α/r = 2；rsLoRA 缩放 α/√r = 16——**同 lr 下有效更新幅度放大 8 倍**。这正是 config.py 注释"大 rank 建议开"的理论依据，也意味着 rsLoRA 臂的最优 lr 大概率低于 2e-4，故必须与 lr 做成 3×2 联合网格而非单独开关。
- 【推断】r=64 是否已大到非开 rsLoRA 不可，文献阈值不明确（原文实验多在 r≤64 演示增益）；考虑到四轮 baseline 都是 False，把它当作"候选改进"而非"必修缺陷"对待，小扫数据说话。

**风险与缓解**：rsLoRA 臂在 0.5 epoch 短跑中可能因有效更新大而显得"学得快"，但全程训练中后段过拟合风险也更高。缓解：rsLoRA 臂若胜出，主 run 必须沿用同一早停与 best-checkpoint 制度（§6/§7），且 v5 adapter_info.json 中显式记录 `use_rslora` 与缩放因子，避免与 v1–v4 台账混读（口径纪律先例：`evidence_table.md` ⑤-1 loss 口径不可跨代比较）。

### 1.3 epochs 与调度：cosine horizon 改为 2 epoch，num_train_epochs=4 仅作安全帽

**决策**：`num_train_epochs=4` 保留为早停不触发时的绝对上限；**cosine 退火的 horizon（max_steps）按 2 epoch 的实际步数设置**（v5 训练集规模 D 定后：horizon = ⌈D/8⌉ × 2）。不再沿用 v4"按 4 epoch 设计 cosine"的做法。

**论证**：
- 【已证实】四条 eval_loss 轨迹一致指向"最优点 ≤ 2 epoch"：
  - v1（2,662 条，2 epoch/666 步）：best eval_loss 在**最后一步**，明显欠训练（`docs/training_retrospective_2026-07-20.md` 3.2 硬伤 3）；
  - v2（8,458 条，2 epoch/2,116 步）：best 1.0673376321792603 @ step 2000 ≈ 1.89 epoch，末步 1.0675 已平台化（`evidence_table.md` ①）；
  - v3（8,963 条，4 epoch 设计）：best 1.0705283880233765 @ step 1100 ≈ 0.98 epoch，随后回升至 1.0848 @ step 1400（`evidence_table.md` ①）——过拟合起点在 1 epoch 附近；
  - v4（5,739 条，4 epoch 设计，completion-only 口径）：best 1.5592304468154907 @ eval_step 700 ≈ 0.98 epoch，patience=3 触发于 step 1,000（`evidence_table.md` ①）。
- 【已证实】v4 的设计缺陷：cosine 按 max_steps=2,872 设计但 step 1,000 即早停，**训练在 lr=1.553e-4 的高位戛然而止，模型从未经历低 lr 退火段**（`evidence_table.md` ⑤-11）。按 2 epoch 设计 horizon 后，v4 规模的 run 将在 step ~1,436 完成退火——覆盖 v3/v4 观测到的最优点（~0.98 epoch）与 v2 的最优点（~1.89 epoch），两端都罩住。
- 【推断】轨迹差异的解释：v3/v4 数据经过去重/清洗，有效多样性低于 v2 原始语料，故最优点前移；v5 数据若沿用 v4 门族，最优点更可能落在 ~1 epoch，2-epoch horizon 留出一倍余量是稳妥的。

**风险与缓解**：若 v5 数据显著扩量（如回到 v2 规模 8.5K+）且多样性更高，2-epoch horizon 可能略欠训练。缓解：① 早停基于 eval_loss，若 horizon 末段 eval_loss 仍在显著下降（最后 3 次 eval 持续改善 > min_delta），属可观测信号，复跑时将 horizon 调至 3 epoch 并在 adapter_info 记录理由；② num_train_epochs=4 安全帽保证任何情况下不会无限训练。

### 1.4 batch / warmup / optimizer：全部保持

**决策**：per_device_batch 1 × grad_accum 8（有效 8）；warmup_ratio 0.05；seed 42；optimizer 保持 v4 实际运行的 HF 默认 adamw_torch，**不切换** paged_adamw_8bit。

**论证**：
- 【已证实】有效 batch 8 是 GB10 单卡显存约束下的既定值，四轮一致（`evidence_table.md` ①"批次"行），v4 实测 2.08 samples/s、11.0 s/step（`evidence_table.md` ③），吞吐可接受，无变更理由。
- 【已证实】warmup 0.05 四轮一致且各轮 warmup 段 lr 爬升正常（v2 step10 lr=1.698e-5、v4 step10 lr=1.25e-5，`evidence_table.md` ①），无早期不稳定记录。
- 【推断】`config.py:88` 留有 `paged_adamw_8bit`，但 `pipeline_v4/configs/train_v4.json` 未设 optim 键，v4 实际跑的是 HF 默认 adamw_torch；8bit 优化器是为已不存在的显存压力准备的（§3 BF16 实测余量充足），切换会无谓改变训练轨迹且引入新的不可比性。显式声明此选择，堵住 config.py 与实际运行的又一处漂移（漂移前科见 `evidence_table.md` ⑤-13）。

**风险与缓解**：batch 1 下吞吐受限于 kernel 开销，但这是硬件事实而非配置失误；若未来换卡再重估。optimizer 选择写入 train_v5.json 的 `_notes`，避免下代再漂移。

---

## 2. target_modules：12 模块清单不变；routed expert 不覆盖是"已知且接受"的边界

**决策**：v5 沿用 v1–v4 的显式 12 模块清单（q/k/v/o_proj + in_proj_qkv/z/b/a + out_proj + gate/up/down_proj），不增不减；不接 routed expert。E5 模块消融（仅 Attention vs 仅 DeltaNet）置于 v5 主 run **之后**作为条件性验证（§8.3），不前置、不阻塞 v5。

**论证**：
- 【已证实】12 模块清单是为 Qwen3.6 混合架构手工指定的全线性层覆盖（Gated Attention 10/40 层 + Gated DeltaNet 30/40 层 + MoE MLP 共享/门控投影 40 层），`config.py:51-61` 明确注释"不使用 all-linear（已知兼容性问题）"；v4 用同一清单训练稳定、adapter 验证通过（620 tensors，A=310/B=310 全 BF16 非零，`evidence_table.md` ①）。
- 【已证实】LoRA 仅覆盖 84.66M/34.7B = **0.24%** 参数，routed expert 主体完全未触（`docs/training_retrospective_2026-07-20.md:75`）。这一事实在 E0 之后有了新的解释力：微调的实际效果是**行为风格转换**（回答 3250 → ~300 字符、kw 覆盖持平、judge 轨 −0.30 [−0.46,−0.14] 低于干净 base，`experiments/e0/E0_report.md` §4）而非知识注入——知识大量存于未触的 expert，与"风格/格式可由注意力和共享投影承载"的图景自洽。v5 若在数据侧做 CE 术语保留修复（E2 建议，`assessment_report.md` E2 工程启示⑤），0.24% 的覆盖面足以承载此类表述层调整。
- 【文献支持】MoE 上接 expert 并非"越多越接越好"：MoE-Sieve（arXiv:2603.24044，`related_work.md` A #2）按路由热度只接 top-25% expert 即可与全量 LoRA 相当（±1pp），而"完全不接 expert"是文献中的极端基线；Dynamic Rank LoRA for MoE（arXiv:2601.04823，`related_work.md` A #3）提示 MoE-PEFT 不能照搬稠密默认。这些是 v6+ 的候选方向（路由引导 expert LoRA），但 256 expert 的 LoRA 挂载会使适配器规模与显存占用成倍膨胀、且无任何本项目证据表明必要——v5 不做。
- 【推断】E5 的科学价值（Gated DeltaNet × Gated Attention × MoE 三元混合架构的 LoRA 放置先验，W2 文献裁决为真空白，`related_work.md` A 裁决段）与 v5 的配置决策是解耦的：无论 E5 结果如何，v5 主 run 用 12 模块都是当时的最稳妥选择（全覆盖 = 消融的上界）。故 E5 后置，仅当 v5 主 run 出现未预期的维度退化时才提前为诊断工具。

**风险与缓解**：12 模块全覆盖相对任一子集都有更多参数，在小语料上过拟合风险略高——缓解由早停（§7）与 best-checkpoint（§6）承担。E5 的模块名清单需执行前从模型 config / peft 适配器 key 清单实测核对（`experiment_plan.md` E5 风险①），不得凭文档假设。

---

## 3. 精度终裁：纯 BF16 不量化；更大基座的可扩展性路线

**决策**：v5 = 纯 BF16 加载 + BF16 LoRA，不使用 bitsandbytes 量化（维持 v4 终裁）。同时在 adapter_info 与训练记录中写入**容量规则**：BF16 路线适用于基座 ≤ ~50B 参数；超过则切 NVFP4/QLoRA 路线，且切换时必须执行逐模块量化审计。

**论证**：
- 【已证实】4-bit 路线在本项目有双重事故史：① 名义 QLoRA 4bit 配置（`config.py:40-46`）与实际运行不符，v1–v3 实际 FP16 加载（`evidence_table.md` ⑤-13）；② 加载显存 62.61GB 远超 ~20GB 预期，expert/自定义层疑似未被 bnb 量化（`docs/training_retrospective_2026-07-20.md` 3.2）；③ 4-bit 量化被 v4 管线 README 列为事故链源头之一（事故 #6，`pipeline_v4_remote/README.md`）。v4 改纯 BF16 后训练完成、adapter 620 tensors 全 BF16 非零验证通过（`evidence_table.md` ①）。
- 【已证实】显存账：121GB 统一内存；旧 FP16 加载实测 64.6GB（`pipeline_v4_remote/README.md` 事故 #6）；35B BF16 权重 ~70GB，加 LoRA 优化器状态与梯度检查点激活后仍在容量内（推理评测峰值显存四轮稳定在 64.68–65.22GB，`evidence_table.md` ②/③）。量化省下的显存用不上，量化的精度损失与工程摩擦却是实的。
- 【文献支持】社区对照：NVFP4-aware LoRA 在 GB10 上训 120B 级 MoE，loss 1.00 vs BF16 0.98（`related_work.md` E #32）——低比特在"不得不省显存"时可用且代价已知；同族 BF16 LoRA 先例（kreuzhofer 2026，`related_work.md` E #31）证明本路线非孤例。
- 【推断】可扩展性规则：BF16 下权重大头 ≈ 2 B/参数，50B 基座 ≈ 100GB 权重，已逼近 121GB 上限且训练还需激活/优化器余量，故把规则线划在 ~50B；70B+（~140GB 权重）物理不可行，必须走 NVFP4/QLoRA，且吸取事故 #6 教训——量化后导出 bnb 模块统计逐层核对"量化真实生效"（该审计动作在 `docs/training_retrospective_2026-07-20.md` P1-训练侧末条已有处方）。

**风险与缓解**：BF16 的代价是吞吐（v4 2.08 samples/s），但单 run 7h 预算内可完成（§9），可接受。风险点在未来换更大基座时规则被遗忘——缓解：容量规则写入 `train_v5.json` 的 `_notes.precision`（沿用 v4 在配置里写 `_notes` 的既有惯例）。

---

## 4. completion-only loss：维持，断言机制原样保留

**决策**：v5 维持 prompt/completion 数据格式 + trl 1.5.1 SFTTrainer 自动启用 completion_only_loss + train.py 启动断言（必须为 True 否则退出）+ 禁止传 formatting_func。

**论证**：
- 【已证实】全文计 loss 是 v1–v3 的硬伤 1（`docs/training_retrospective_2026-07-20.md` 3.2）：训练容量被 prompt 复述分走、eval_loss 不可与常规 SFT 比较；v4 修复为 prompt/completion 格式后 trl 1.5.1 自动启用 completion_only_loss（机制经源码确认：`pipeline_v4_remote/README.md` "completion-only loss 在 trl 1.5.1 的实际机制"节，`SFTTrainer.__init__` L1140-1145 按首样本键自动判定，L1452-1488 构造 completion_mask）。
- 【已证实】口径切换的代价已被如实记录：v4 best eval_loss 1.5592 与 v2 1.0673 不可比（`evidence_table.md` ⑤-1）。v5 沿用同口径 = v5 的 eval_loss 可直接与 v4 对比，这是保留该机制的第二重收益（跨代可比性）。
- 【已证实】断言的必要性有事故依据：chat template 无 `{% generation %}` 导致 `assistant_only_loss` 静默失效正是旧管线的缺陷 #5（`pipeline_v4_remote/README.md` 事故表）；自动判定依赖"数据集首样本同时含 prompt/completion 键"，数据构建侧若改 schema 会静默退回全文 loss——启动断言是唯一的硬保障。

**风险与缓解**：trl 版本升级可能改变自动判定逻辑。缓解：① v5 环境固定 venv_v5（peft 0.19.1 / transformers 5.10.1 / trl 1.5.1 / torch 2.12.0，`pipeline_v4_remote/README.md` 头部），与 v4 完全相同；② 在断言之外加一个启动冒烟：decode 首个 batch 的 labels，确认 prompt 区段全为 -100（成本秒级，防"自动判定规则变了但断言也失效"的共模失效）。

---

## 5. max_length=2048：显式锁定 + 启动断言

**决策**：`max_length=2048` 显式写入 train_v5.json 并加启动断言（SFTConfig.max_length == 2048 否则退出）；数据侧保留"prompt+completion > 2048 token 丢弃"门（v4 门 7）。

**论证**：
- 【已证实】静默截断是 v1 时代的疑似硬伤 2：TRL 移除该参数后未显式设置，若默认 1024，长回答样本已被静默截断（`docs/training_retrospective_2026-07-20.md` 3.2）。v4 已在 `train_v4.json:16` 显式设 2048，v5 把它从"配置项"升级为"断言项"，防回归。
- 【已证实】2048 对本数据族充分：v4 数据门 7（>2048 token 丢弃）实际丢弃 **0 条**（`evidence_table.md` ① 质量门行），即 v4 语料 p100 ≤ 2048。
- 【推断】`config.py:140` chatml max_length=4096 是又一处四方漂移；唯一事实源应为 train 配置的 2048，config.py 的 4096 属死配置，v5 不引用。

**风险与缓解**：若 v5 数据侧引入更长答案（如 ARIZ 长流程样本），2048 可能截断。缓解：v5 数据构建报告必须输出 prompt+completion 长度的 p95/p99/p100；若 p99 > 2048，提交 Owner 裁决是否升至 3072（激活显存随序列长增长，121GB 有余量但需实测确认），**默认不升**。此为本文件开放项 #6（§10）。

---

## 6. best-checkpoint 制度：v4 机制全量保留，论证其为最佳实践

**决策**：保留 v4 全套制度：① BestCheckpointCallback——eval_loss 创新低**立即**把对应 checkpoint 复制到 `best/`（不参与轮转）；② `save_total_limit=8`；③ `eval_steps == save_steps == 100` 强制相等（train.py 启动校验）；④ 发货只复制磁盘 `best/` 目录文件，**绝不保存训练内存态**；⑤ `validate_adapter_dir` 发货前断言全部 lora_B 非零（手解 safetensors 逐字节检查）、dtype 全 BF16、sha256 记录；⑥ adapter_info.json 如实记录实际 epoch/步数/早停/best 的 eval step 与 loss/验证结果，FAILED 时退出码非零。

**论证**：
- 【已证实】每一条都对应一个真实事故：save_total_limit=3 曾把真正最优 checkpoint 轮转删除、`find_best_checkpoint` 扫幸存者导致 eval_loss 张冠李戴（事故 #2）；`PeftModel.from_pretrained` 先污染内存再抛异常、fallback 保存了 310 个全零 lora_B（事故 #1，v3 全零 lora_B 发货事故的直接根因，`evidence_table.md` ⑤-5）；训练结束 fallback 保存被污染内存态（事故 #4）（均见 `pipeline_v4_remote/README.md` 事故表）。
- 【已证实】该制度在 v4 实战中按设计运转：best checkpoint 连续晋升 step100→700（`evidence_table.md` 附：v4 全链时间线），最终发货 = best @ eval_step 700 而非早停末步 1,000——若没有该制度，v4 发出的将是一个 eval_loss 已回升 300 步的退化权重。**制度的价值已被一次实战正向验证，不是纸面设计。**
- 【已证实】v1 时代"发货权重=末步内存态、无早停无 best 回载，本次巧合 best=末步，流程不保证下次如此"（`docs/training_retrospective_2026-07-20.md` 3.2）——v4 制度正是该问题的系统性回答，v5 无任何退化理由。

**风险与缓解**：BestCheckpointCallback 的"立即复制"依赖 eval 时 checkpoint 已完整落盘——`eval_steps==save_steps` 强制相等正是为此（`train_v4.json` `_notes`）；v5 保留该校验。磁盘占用：8 轮转 + 1 best，单 checkpoint 为 LoRA 权重级（v4 adapter 169.4MB），成本可忽略。

---

## 7. 早停策略参数化：patience=3，min_delta=0.002，eval_steps=100

**决策**：`EarlyStoppingCallback(patience=3, threshold=0.002)`（HF 参数名 early_stopping_threshold，即本文 min_delta）；eval_steps=100 不变。两个参数显式写入 train_v5.json（v4 只写了 patience，threshold 用了默认 0）。

**论证**：
- 【已证实】patience=3 在 v4 按设计工作：best @ eval_step 700 → 800/900/1,000 三次未改善 → 停于 step 1,000（actual_epochs=1.393，`evidence_table.md` ①/⑤-11）。eval 粒度 100 步 ≈ 18 分钟（11.0 s/step），patience=3 = 容忍 ~55 分钟的无改善训练，粒度与成本匹配合理。
- 【已证实】min_delta 需要非零的证据：v2 末段 1.0673(step2000)→1.0675(step2116)，平台期逐次波动幅度 ~0.0002（`evidence_table.md` ①）——threshold=0 会把噪声级"改善"当作真改善而无限续命。取 0.002 = 观测平台噪声的 10 倍，既滤掉抖动，又远小于有意义的改善速率（v2 后期 1.1576→1.0673，约 0.09/1800 步 ≈ 0.005/100 步），不会误杀真改善。
- 【推断】0.002 的具体取值没有项目直接实证（v4 用默认 0 且顺利触发，因为 v4 早停段是持续回升而非抖动），属参数化补齐：其功能是在"v3 型"轨迹（best 后缓慢波动而非单调回升）下也能正确触发。v3 轨迹 1.0705→1.0848 是单调回升，threshold=0 与 0.002 行为相同；真正的差异场景尚未在本项目出现，故标注【推断】。

**风险与缓解**：min_delta 过大风险 = 把缓慢真改善（<0.002/100 步）判为停滞而提前停。缓解：v2 实测后期真改善速率 ~0.005/100 步 > 0.002，安全边距 2.5 倍；且即便误触发，best-checkpoint 制度（§6）保证发货的仍是 best，损失仅是少训几百步的算力，方向安全。

---

## 8. 训练侧验证路线：执行顺序与判据

原则：**先锁超参（P0 小扫）→ 再跑主 run（P1）→ 条件消融后置（P2）**。每个消融注明"回答什么问题 / 什么结果会改变 v5 配置"。GPU 纪律沿用 `experiment_plan.md` §4-6：串行排队，单 run 期间不并行其他 GPU 任务。

### 8.1 P0：lr × rsLoRA 六组小扫（主 run 前置门，必做）

- **设计**：lr ∈ {1e-4, 2e-4, 5e-4} × use_rslora ∈ {False, True}，各 0.5 epoch（v5 训练集 D 定后 = ⌈D/8⌉/2 步），其余配置与 §0 总表完全一致（同数据、同 BF16、同 completion-only、同 max_length、同 seed=42）；eval_steps=50，completion-only eval_loss 轨迹 + 终点为初级信号。处方来源：`docs/training_retrospective_2026-07-20.md:139`。
- **回答的问题**：在 v5 数据上，2e-4 是否仍最优？r=64 下 rsLoRA 是否兑现 config.py:65 注释的预期？
- **判据（两阶段门）**：
  1. 初筛：终点 eval_loss 最低者；与 2e-4/False 基线差 ≤ 0.01 视为平局（理由：0.5 epoch 短跑的 loss 分辨率有限，且 eval_loss ≠ 下游质量——v3 教训见 §1.1）。
  2. 终判：初筛 top-2 臂各做 **40 题金标双轨冒烟**（从 v4_gold.jsonl 分层抽 40 题，judge=moonshot-v1-32k，T=0，AB/BA 双序合并口径——单序 pairwise 一律无效，`assessment_report.md` 工程启示②）。选 judge 轨更高且 kw 轨无显著退化者；仍平局 → **取 2e-4/False**（与 v1–v4 可比性优先）。
- **什么结果会改变 v5 配置**：任一臂 eval_loss 胜基线 > 0.01 且金标冒烟方向一致 → v5 主 run 采用该臂（lr 与 use_rslora 同步替换，记录不可比性声明）。否则主 run 全按 §0 默认值。
- **成本**：6 run × ~1.1h（D≈5,739 时 0.5 epoch ≈ 360 步 × 11.0 s/step）≈ **6.6h GPU** + top-2 冒烟 ~1h GPU + judge API ~0.5h。若 D 达 v2 规模 ~8.5K，升至 ~9.7h+1h。

### 8.2 P1：v5 主 run + 金标决策门（核心交付）

- **设计**：P0 锁定的配置，cosine horizon=2 epoch，patience=3/min_delta=0.002；训练后走 v4 同款金标双轨评测 + 决策门，对手为 **v2**（现发货版）与 base_goldfix（干净锚点）。
- **判据（沿用工程启示⑦双轨化，`assessment_report.md`）**：v5 替代 v2 需同时满足 ① judge 轨 overall 显著 > v2（paired bootstrap 95%CI 下界 > 0）；② kw 轨无显著退化子集（重点盯 concept_explanation——v4 vs v2 −0.083 [−0.149,−0.023] 的前科，`experiments/e0/E0_report.md` §4.3）；③ 两轨不反向（反向即回滚并启动归因，与 v3 处置一致）。所有 vs base 对比一律用 base_goldfix 缓存，凡非 `base_goldfix` tag 的 base 缓存一律视为受污染（`E0_report.md` §5 通用原则）。
- **成本**：训练 ≤7h 预算帽（按 v3/v4 轨迹预计 2.5–4h 即早停）+ 评测 ~1h GPU + judge API ~0.5h。

### 8.3 P2a：E5 模块消融（仅 Attention vs 仅 DeltaNet）——条件性，主 run 之后

- **回答的问题**：混合架构上领域行为主要沉积在 Gated Attention 层（10 层）还是 Gated DeltaNet 层（30 层）？12 模块全覆盖是否必要？（文献真空白，`related_work.md` A 裁决）
- **设计**：两个 1-epoch 短 run（仅 q/k/v/o_proj；仅 in_proj_qkv/z/b/a+out_proj），数据/超参与 v5 主 run 相同；对照 = v5 主 run 的 1-epoch 截断检查点（若主 run 早停点 ≠ 1 epoch 则需注明预算不等价，或加跑全 12 模块 1-epoch 对照 run +~2.2h）。
- **什么结果会改变 v5 配置**：**不改变 v5**（主 run 已发货）。改变的是 v6：若仅 Attention ≈ 全 12 模块（judge 轨 CI 覆盖 0 且 kw 无显著退化）→ v6 可砍 DeltaNet 侧 5 模块省参防过拟合；若仅 DeltaNet 显著更差 → "Attention 层是主要沉积点"的架构先验成立，可升级为论文贡献 A 的消融证据（`assessment_report.md` §2.1 发现⑤现为"弱-中"正因缺此消融）。仅当 v5 主 run 出现未预期退化时，E5 提前为诊断工具。
- **成本**：2 run × ~2.2h（1 epoch，v4 规模）≈ 4.4h GPU + 评测 ~2h（`experiment_plan.md` E5 估 3 短 run 10.5h 含对照，此处复用 v5 主 run 省 1 run）。
- **启动条件**：v5 主 run 成功发货后，若论文冲 TMLR 或审稿风险需要消融证据则启动；否则按 `assessment_report.md` §5.2 建议暂缓、写入 Limitations。开放项 #4（§10）。

### 8.4 P2b：E4 数据门消融——仅在 v5 数据门发生变更时启动

- **回答的问题**：若 W1 数据支柱采纳 E2 修复建议（CE 样本保留术语枚举式答案 / cap 门改随机保留，`assessment_report.md` E2 节），新门是否带来增益、有无误伤？
- **设计**：新门开/关两个 run（其余全同），金标双轨对比，重点盯 concept_explanation kw 轨。
- **什么结果会改变 v5 配置**：若"关门"版 CE-kw 显著更差而 overall 不变 → 保留新门；若两版 overall 与 CE-kw 均无显著差异 → 新门低价值，v5 数据配置回滚到 v4 门族（简单优先）。
- **成本**：2 run × 7h = 14h GPU + 评测 ~2h（短 run 版需 3 run 含同预算对照 ≈ 10.5h+2h，`experiment_plan.md` E4）。
- **启动条件（关键）**：若 v5 数据完全沿用 v4 门族，E4 **不启动**——`assessment_report.md` §5.2 已判"暂缓：E0 后 v4 相对 base 无增益可解释，消融的科学问题已弱化"。

---

## 9. 成本预算表（GPU 小时；基准：单 run ≤7h，v4 实测 11.0 s/step，1 epoch(v4 规模 5,739 条) ≈ 717 步 ≈ 2.2h）

| 阶段 | 内容 | GPU h | API | 启动条件 | 累计（核心线） |
|---|---|---|---|---|---|
| P0 | lr×rsLoRA 6 组 × 0.5 epoch | ~6.6（D≈5.7K）/ ~9.7（D≈8.5K） | — | 必做（数据 D 定后即跑） | 6.6 |
| P0b | top-2 臂 40 题金标双轨冒烟 | ~1.0 | ~0.5h | 随 P0 | 7.6 |
| P1 | v5 主 run（预算帽 7h，预计 2.5–4h 早停） | ≤7.0 | — | P0 锁定后 | 14.6 |
| P1b | v5 + 对照臂 100 题金标双轨评测 | ~1.0 | ~0.5h | 随 P1 | 15.6 |
| P2a | E5：2 短 run（1 epoch）+ 评测 | ~6.4 | ~0.5h | 条件（§8.3） | — |
| P2b | E4：2 完整 run + 评测 | ~16.0 | ~0.5h | 条件（§8.4） | — |

- **核心线（P0+P1）≈ 15.6h GPU**，分两天排：Day 1 = P0 小扫 + 冒烟（~7.6h）；Day 2 = 主 run + 评测（≤8h）。与远端事实"单 run v5 训练与评测可在 1 天内完成"兼容（主 run 线占 Day 2 一天）。
- 全条件触发上限 ≈ 15.6 + 6.4 + 16.0 = **38h GPU**。
- 非 GPU 成本：judge API 合计 ~1.5h（RPM=3 退避），全部脚本断点续跑（纪律沿用 `experiment_plan.md` §4-1）。

---

## 10. 须 Owner 裁决的开放项（含推荐值）

| # | 开放项 | 推荐值 | 依据 |
|---|---|---|---|
| 1 | cosine horizon 从 4-epoch 设计改为 **2-epoch 设计** | **采纳** | v3/v4 最优点 ~0.98 epoch、v4 早停时 lr 仍 1.553e-4 从未退火（`evidence_table.md` ⑤-11） |
| 2 | 早停 min_delta 从默认 0 改为 **0.002** | **采纳** | v2 平台噪声 ~0.0002 的 10 倍；v2 后期真改善 ~0.005/100 步，边距 2.5× |
| 3 | P0 小扫平局时的默认臂 | **2e-4 / rsLoRA=False** | 与 v1–v4 可比性优先；变更需证据 |
| 4 | E5 模块消融是否执行 | **主 run 成功且冲 TMLR 则做**（2 短 run ~6.4h） | 贡献 A 升级为"架构先验"的唯一路径（`assessment_report.md` §2.1 发现⑤）；否则写 Limitations |
| 5 | E4 数据门消融触发条件 | **仅当 v5 数据门有变更时** | E0 后消融科学问题弱化（`assessment_report.md` §5.2） |
| 6 | 若 v5 数据 p99 > 2048 token，max_length 是否升 3072 | **默认锁 2048**，先报长度分布再裁决 | v4 门 7 丢弃 0 条（`evidence_table.md` ①）；升长需实测显存 |

---

## 附：本文件引用的证据文件

`paper/evidence_table.md`（①版本总表/②指标/③效率/⑤缺陷清单）、`paper/stats_review.md`（§2 污染/§6 功效）、`paper/related_work.md`（A/E 节文献）、`paper/experiment_plan.md`（E4/E5 成本与设计）、`paper/experiments/e0/E0_report.md`（干净锚点）、`paper/assessment_report.md`（工程启示①-⑦、§5.2 取舍）、`docs/training_retrospective_2026-07-20.md`（硬伤 1-3、P1 处方）、`pipeline_v4_remote/README.md`（事故 #1-#7、completion-only 机制）、`pipeline_v4/configs/train_v4.json`、`config.py`（:62/:65/:88/:140）、`results/METRICS_LEDGER.md`。外部文献：Kalajdzievski 2023 arXiv:2312.03732（rsLoRA，已核验）；Biderman et al. 2024 arXiv:2405.09673、Lin et al. 2025 arXiv:2509.20758（`related_work.md` B1 #10/#9）。
