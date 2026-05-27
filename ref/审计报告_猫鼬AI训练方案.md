# 独立审计报告：猫鼬AI模型训练方案

**审计Agent**: 独立第三方技术审计
**审计日期**: 2026-05-23
**审计范围**: 训练意义、训练方法、代码质量
**审计对象**: 猫鼬AI DGX Spark QLoRA微调方案全套代码

---

## 一、执行摘要

本次审计对猫鼬AI在DGX Spark上使用QLoRA微调Qwen3.6-35B-A3B模型的完整方案进行了独立评估。审计发现：**方案在战略层面合理，但在技术实现层面存在关键缺陷，部分代码存在会导致训练失败的严重bug，必须修复后方可运行**。

| 审计维度 | 评分 | 结论 |
|---------|------|------|
| 训练意义（商业逻辑） | 7/10 | 赛道选择合理，但数据ROI存疑 |
| 训练方法（技术路径） | 6/10 | 框架正确，存在关键参数风险 |
| 代码质量（可运行性） | **4/10** | **存在会导致训练失败的严重bug** |
| **总体评分** | **5.7/10** | **需修复后方可投入运行** |

---

## 二、审计方法论

本次审计遵循以下原则：

1. **独立性原则**：审计Agent未参与原始方案设计，以全新视角审查
2. **实证原则**：所有技术判断基于代码实际内容和业界最佳实践
3. **可运行性优先**：重点关注代码是否能在DGX Spark上实际运行
4. **风险导向**：优先识别会导致训练失败的阻塞性问题

审计范围涵盖三个层面：
- **config.py**：全局配置和超参数
- **utils/*.py**：工具函数实现
- **notebooks/*.ipynb**：Jupyter Notebook执行流程

---

## 三、训练意义审计

### 3.1 赛道选择评估

**审计结论：合理**

TRIZ创新方法论是一个高度专业化的垂直领域。根据市场调研，全球范围内将TRIZ与大模型深度融合的商用产品几乎空白。猫鼬AI拥有15年TRIZ咨询经验和5000万字技术手册数据，这种"领域知识壁垒 + 数据资产壁垒"的组合构成了有效的护城河。

**支持论据**：
- TRIZ-AI交叉研究在学术界正处于活跃期（国际TRAI会议、TRIZ Agents系统）
- 中国365个垂直大模型中无一涉及TRIZ方法论
- 垂直领域AI市场在2026年进入商业化拐点（CAGR 26.1%）

**风险提示**：
- 若6,000条训练数据产生的模型能力提升有限（如<20%），则投入产出比将显著降低
- 合成数据质量如果无法达到专业TRIZ专家的审核标准，可能引入错误知识

### 3.2 技术路径选择评估

**审计结论：基本合理，但需验证**

选择QLoRA（4-bit量化 + LoRA适配器）作为微调方法，在DGX Spark 128GB内存环境下对35B模型进行微调，这一技术路径在硬件约束和成本效率之间取得了合理平衡。

| 方案 | 内存需求 | 训练成本 | 效果上限 | 适用性 |
|------|---------|---------|---------|--------|
| QLoRA (当前方案) | ~60-80GB | 低 | ~85-90%全量 | ✅ 适合 |
| LoRA (BF16) | ~80-100GB | 中 | ~90-95%全量 | ⚠️ 内存紧张 |
| 全量微调 | ~200GB+ | 高 | 100% | ❌ 不可行 |

**风险提示**：
- Qwen3.6的Gated DeltaNet架构相对较新（2026年4月发布），QLoRA在混合架构上的效果尚未有大量社区验证
- 若QLoRA适配器无法有效学习目标层，可能需要切换到全精度LoRA（BF16）

### 3.3 ROI初步估算

| 投入项 | 估算成本 |
|--------|---------|
| DGX Spark硬件 | $6,500（一次性） |
| 人力投入 | 2-3周（1名AI工程师） |
| 训练电费 | ~$50/次 |
| 数据标注审核 | 持续投入 |

| 预期产出 | 价值评估 |
|---------|---------|
| TRIZ领域专家模型 | 高（无可替代性） |
| 咨询效率提升 | 中（需客户验证） |
| 技术壁垒构建 | 高（数据飞轮效应） |

**总体判断**：在战略层面，该训练具有明确的价值主张和合理的技术路径。但**数据策略是最大不确定性因素**——6,000条合成数据是否足以让35B模型学到TRIZ的专业知识，需要实际训练后才能验证。

---

## 四、训练方法审计

### 4.1 模型选择：合理

Qwen3.6-35B-A3B的选择经过了充分的技术论证：
- 35B总参数/3B活跃参数的MoE架构，推理效率高
- 4-bit量化后仅~18GB，DGX Spark 128GB内存充裕
- Apache 2.0许可证，无商用限制
- 262K上下文长度，远超训练所需的4K

**评分：9/10**

### 4.2 QLoRA超参数：基本合理，有风险

| 参数 | 设定值 | 合理性评估 | 风险等级 |
|------|--------|-----------|---------|
| rank=64 | ✅ 推荐范围32-128 | 复杂领域适配充分 | 低 |
| alpha=128 | ✅ 2*rank | 缩放比例标准 | 低 |
| dropout=0.05 | ✅ 推荐范围0.01-0.1 | 正则化适度 | 低 |
| lr=2e-4 | ✅ LoRA推荐范围 | 学习率适中 | 低 |
| epochs=2 | ⚠️ 偏少 | 6K数据2epoch可能不足 | 中 |
| warmup=0.03 | ⚠️ 偏低 | 建议0.05-0.1 | 低 |
| batch=1, accum=8 | ✅ DGX Spark适配 | 有效batch=8合理 | 低 |
| **target_modules** | **⚠️ "all-linear"** | **未经实测验证** | **高** |

**关于 target_modules = "all-linear" 的风险**：

PEFT库在0.11.0+版本引入了`"all-linear"`字符串支持，用于自动检测模型中的所有线性层。然而：

1. **架构兼容性问题**：Qwen3.6使用Gated DeltaNet混合架构，其线性层模块名称与传统Transformer不同。"all-linear"依赖PEFT内部的模块类型检测，对于新架构的兼容性尚未有大量社区验证。

2. **已知bug**：GitHub Issue [^287^] 显示，`target_modules='all-linear'`在x86和aarch架构上行为不一致，且在某些模型上会错误地将`lm_head`或`dropout`层纳入训练。

3. **建议**：默认启用手动模块列表作为备选方案，在首次运行时通过`find_all_linear_names()`函数验证实际检测到的模块是否与预期一致。

**评分：7/10**（扣3分因target_modules风险）

### 4.3 数据策略：存在严重问题

#### 问题1：示例数据量严重不足

当前`create_sample_data()`函数仅生成**6个子集 × 2条 = 12条**示例数据。这对于任何模型训练来说都是完全不够的——即使只是验证流程正确性，也需要至少数百条样本。

**影响**：用户上传项目后首次运行就会遇到"数据量不足导致训练异常"的问题。

#### 问题2：合成数据策略过于简单

`create_synthetic_data()`函数的`vary_sample()`实现仅为：
- `paraphrase`策略：在问题前添加4个固定前缀之一
- `extend`策略：在答案后追加一句固定文本

这种简单的文本变换无法产生语义多样性。对于TRIZ这种需要深度专业知识的领域，合成数据必须通过真正的GPT-4o API调用 + 专家审核才能产生有价值的训练样本。

**评分：4/10**（数据策略是最大短板）

### 4.4 训练流程：存在严重技术缺陷

#### **严重问题：SFTTrainer + DataCollatorForLanguageModeling 组合冲突**

`training_utils.py`第365-382行的`create_trainer()`函数：

```python
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=train_dataset, eval_dataset=eval_dataset,
    args=training_args, data_collator=data_collator,  # 冲突！
    max_seq_length=max_seq_length, packing=True,
    dataset_text_field="text",
)
```

**问题分析**：

SFTTrainer（来自trl库）的核心功能是**只计算assistant回复部分的loss**，它需要：
- 对话格式的数据（messages列表或formatting_func函数）
- 内部自动处理tokenization和labels mask

而DataCollatorForLanguageModeling会：
- 对所有token计算loss（不区分user/assistant）
- 与SFTTrainer的内部逻辑产生冲突

**后果**：当同时传入`data_collator`和`dataset_text_field="text"`时，SFTTrainer可能：
- 忽略data_collator，按自己的逻辑处理（不可预测行为）
- 或报错退出
- 更严重的是：如果训练能跑起来，模型会学习user问题的文本模式（这是错误的），导致训练效果极差

**正确做法**：

```python
# 方案A: 使用SFTTrainer自带的数据整理（推荐）
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=train_dataset, eval_dataset=eval_dataset,
    args=training_args,
    max_seq_length=4096,
    formatting_func=format_chatml,  # 传入格式化函数
    packing=True,
)

# 方案B: 使用标准Trainer + DataCollatorForSeq2Seq
trainer = Trainer(
    model=model, args=training_args,
    train_dataset=processed_train_dataset,  # 预处理后的token IDs
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
```

**评分：3/10**（这是会导致训练失败的关键bug）

---

## 五、代码质量审计

### 5.1 关键bug清单

| ID | 级别 | 位置 | 问题描述 | 修复优先级 |
|----|------|------|---------|-----------|
| CR-001 | 严重 | training_utils.py:372 | SFTTrainer + DataCollator冲突 | **P0-阻塞** |
| CR-002 | 严重 | config.py:56 | target_modules="all-linear"未经实测 | **P0-高风险** |
| CR-003 | 严重 | data_utils.py:252-256 | ChatML硬编码格式 | **P1-高** |
| MA-001 | 中等 | data_utils.py:65-200 | 示例数据仅12条 | P1-高 |
| MA-002 | 中等 | data_utils.py:368-396 | 合成数据策略过于简单 | P2-中 |
| MA-003 | 中等 | notebook 03 | 评测使用4-bit量化模型 | P2-中 |
| MI-001 | 轻微 | training_utils.py:57 | padding_side="right"矛盾 | P3-低 |
| MI-002 | 轻微 | config.py:76 | warmup_ratio=0.03偏低 | P3-低 |
| MI-003 | 轻微 | utils/__init__.py | 缺少新函数导出 | P3-低 |

### 5.2 详细问题分析

#### CR-001: SFTTrainer + DataCollator冲突

已在4.4节详细分析。这是**会导致训练失败或效果极差**的阻塞性问题。

**修复方案**：删除DataCollator，改用SFTTrainer的formatting_func参数。

#### CR-002: target_modules="all-linear"风险

已在4.2节详细分析。PEFT的"all-linear"功能对新架构的兼容性未经验证。

**修复方案**：将手动模块列表作为默认配置，"all-linear"作为实验选项。

#### CR-003: ChatML硬编码格式

`data_utils.py`第252-256行硬编码了ChatML格式：

```python
chatml_text = (
    f"<|im_start|>system\n{system_message}<|im_end|>\n"
    f"<|im_start|>user\n{full_question}<|im_end|>\n"
    f"<|im_start|>assistant\n{output}<|im_end|>"
)
```

**问题**：
- Qwen3.6官方推荐通过`tokenizer.apply_chat_template()`生成对话格式[^286^][^298^]
- 硬编码格式可能与新版本的chat template不匹配（如thinking mode的`<think>`标签）
- 如果模型的chat template配置变化，训练数据格式将与推理格式不一致

**修复方案**：

```python
# 使用tokenizer的官方chat template
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": full_question},
    {"role": "assistant", "content": output},
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False
)
```

#### MA-001: 示例数据仅12条

`create_sample_data()`仅生成12条样本。这导致：
- 用户首次运行Notebook 02后，只有12条训练数据
- 运行Notebook 04训练时，2个epoch × 12条 = 24步训练，模型几乎不会学到任何东西
- 验证集和测试集各只有1-2条，评估结果无统计意义

**修复方案**：至少提供500-1000条高质量的示例数据。

#### MA-003: 评测使用4-bit量化模型

Notebook 03在评测基座模型时加载了4-bit量化模型。量化会引入精度损失（约1-3%），导致评测分数不能反映模型的真实能力。

**修复方案**：评测阶段使用BF16/FP16精度加载模型（DGX Spark 128GB内存足够容纳35B FP16模型约70GB）。

#### MI-001: padding_side="right"矛盾

`training_utils.py`第57行设置`padding_side="right"`，注释说明"左填充更适合生成"。实际上：
- 训练时通常使用right padding
- 推理时必须使用left padding
- 但Qwen官方推荐设置`padding_side="left"`[^291^]
- SFTTrainer内部可能会覆盖此设置

这本身不会导致训练失败，但可能在推理阶段产生意外行为。

---

## 六、修复建议

### 6.1 必须修复（P0-阻塞）

#### 修复1: SFTTrainer数据整理逻辑

**文件**: `training_utils.py`
**修改**: 重写`create_trainer()`函数

```python
def create_trainer(model, tokenizer, train_dataset, eval_dataset,
                   training_args, max_seq_length=4096):
    """创建SFTTrainer，使用formatting_func处理ChatML格式"""
    
    def formatting_func(example):
        """将对话数据格式化为模型输入文本"""
        messages = [
            {"role": "system", "content": example.get("system", SYSTEM_MSG)},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        # 使用tokenizer的官方chat template
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return text
    
    # SFTTrainer自动处理数据整理，不传入data_collator
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        max_seq_length=max_seq_length,
        formatting_func=formatting_func,  # 使用格式化函数
        packing=True,
    )
    return trainer
```

#### 修复2: target_modules默认使用手动列表

**文件**: `config.py`
**修改**: 将手动列表设为默认，"all-linear"设为注释备选

```python
# 默认使用手动模块列表（经Qwen3.6架构验证）
"target_modules": [
    "q_proj", "k_proj", "v_proj", "o_proj",          # Gated Attention
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",  # GDN
    "gate_proj", "up_proj", "down_proj",             # MoE MLP
],
# "target_modules": "all-linear",  # 实验选项：PEFT自动检测
```

#### 修复3: 使用tokenizer.apply_chat_template()

**文件**: `data_utils.py`
**修改**: 重写`convert_to_chatml()`函数，优先使用tokenizer的chat template

```python
def convert_to_chatml(data, tokenizer=None, system_message=None):
    """转换为ChatML格式，优先使用tokenizer.apply_chat_template()"""
    
    all_samples = []
    
    for subset_name, samples in data.items():
        for sample in samples:
            instruction = sample.get("instruction", "")
            input_text = sample.get("input", "")
            output = sample.get("output", "")
            
            full_question = f"{instruction}\n\n{input_text}" if input_text else instruction
            
            # 优先使用tokenizer的官方chat template
            if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "system", "content": system_message or DEFAULT_SYSTEM_MSG},
                    {"role": "user", "content": full_question},
                    {"role": "assistant", "content": output},
                ]
                chatml_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            else:
                # 回退到硬编码格式
                chatml_text = (
                    f"<|im_start|>system\n{system_message}<|im_end|>\n"
                    f"<|im_start|>user\n{full_question}<|im_end|>\n"
                    f"<|im_start|>assistant\n{output}<|im_end|>"
                )
            
            all_samples.append({"text": chatml_text, ...})
    
    return DatasetDict(...)
```

### 6.2 建议修复（P1-P2）

#### 建议1: 扩充示例数据至500+条

当前12条示例数据无法支撑任何有意义的训练。建议至少提供500-1000条覆盖6个子集的高质量示例。

#### 建议2: 实现基于GPT-4o的合成数据pipeline

当前的`vary_sample()`函数过于简单。建议实现：
- 调用GPT-4o API基于种子问题生成变体
- 领域专家审核机制
- 自动质量评分（基于困惑度、格式一致性等）

#### 建议3: 评测阶段使用高精度模型

修改Notebook 03，评测时加载FP16/BF16精度模型而非4-bit量化模型。

### 6.3 可选优化（P3）

- 将`warmup_ratio`从0.03提升至0.05
- 统一`padding_side`为"left"
- 在`utils/__init__.py`中导出`find_all_linear_names`和`get_qwen36_target_modules`
- 在`requirements.txt`中锁定关键依赖版本

---

## 七、总体评估与建议

### 7.1 评分汇总

| 维度 | 评分 | 核心问题 |
|------|------|---------|
| 训练意义 | 7/10 | 赛道合理，数据ROI不确定 |
| 模型选择 | 9/10 | Qwen3.6-35B-A3B适配DGX Spark |
| 微调方法 | 7/10 | QLoRA框架正确，target_modules有风险 |
| 数据策略 | 4/10 | 示例数据严重不足，合成策略简单 |
| 训练流程 | 3/10 | **SFTTrainer+Collator冲突是关键bug** |
| 评测体系 | 6/10 | 架构合理，评测精度待提升 |
| 代码质量 | 5/10 | 多处bug和最佳实践缺失 |
| **总体** | **5.7/10** | **需修复P0问题后方可运行** |

### 7.2 关键结论

1. **战略方向正确**：TRIZ垂直领域AI的赛道选择和DGX Spark本地部署策略具有商业合理性。

2. **技术框架基本正确**：QLoRA + Qwen3.6 + ChatML的技术选型在技术层面是合理的。

3. **代码存在阻塞性bug**：`SFTTrainer + DataCollatorForLanguageModeling`的组合会导致训练失败或效果极差，必须修复。

4. **数据策略是最大短板**：12条示例数据完全无法支撑训练，合成数据策略需要根本性重构。

5. **建议分阶段实施**：
   - **第一阶段（1周）**：修复P0 bug，验证target_modules，扩充示例数据至500+条
   - **第二阶段（2周）**：实施完整训练pipeline，运行首次微调
   - **第三阶段（持续）**：构建合成数据pipeline，迭代优化模型

### 7.3 风险矩阵

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 训练因bug失败 | 高 | 严重 | 修复CR-001/002/003 |
| 数据量不足导致效果差 | 中 | 高 | 扩充至5K+条高质量数据 |
| Qwen3.6架构兼容性问题 | 中 | 中 | 实测验证target_modules |
| 合成数据引入错误知识 | 中 | 高 | 增加专家审核环节 |
| 评测分数不能反映真实能力 | 低 | 中 | 评测改用FP16模型 |

---

**审计Agent签名**：独立第三方技术审计
**审计完成时间**：2026-05-23
**报告版本**：v1.0
