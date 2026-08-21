---
license: apache-2.0
base_model: Qwen3.8-27B
library_name: peft
pipeline_tag: text-generation
tags:
  - lora
  - triz
  - inventive-problem-solving
  - chinese
  - evaluation-harness
language:
  - zh
---

> 🇨🇳 中文 | 🇺🇸 [English](README.md)

# Meerkat-TRIZ-v1-Qwen3.8-27B

基于 **Qwen3.8-27B** 基座（qwen3_5 混合线性注意力多模态架构，Apache-2.0）的
TRIZ（发明问题解决理论）领域 LoRA 适配器，面向六类中文 TRIZ 问答任务：发明原理
推荐、矛盾分析、ARIZ 指导、创新评估、概念解释、案例生成。

这是 Meerkat-TRIZ 在 **Qwen3.8-27B** 基座上的首个适配器——是从早前
[Meerkat-TRIZ-v1](https://huggingface.co/Meerkat-AI/Meerkat-TRIZ-v1)
（针对 Qwen3.6-35B-A3B）的一次基座迁移。本版本一如既往强调**评测透明**：所有
结论均由严格门控的双轨 harness（[GitHub: meerkat-triz](https://github.com/coidea-sys/meerkat-triz)）
产出，附配对统计、置信区间与跨族外部评委终审。

> ⚠️ **基座可得性**：本适配器是叠加在 **Qwen3.8-27B** 上的 LoRA 增量。基座权重
> **不包含在本仓库内**，且截至发布时**没有公开的 HuggingFace 模型 ID**——请自行
> 获取 Qwen3.8-27B 基座（见 `adapter_config.json → base_model_name_or_path`），
> 或按下方本地路径方式加载。

## 快速开始

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Qwen3.8-27B 基座 —— 替换为本地路径或你自己的基座副本
BASE = "/path/to/Qwen3.8-27B"

base = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = PeftModel.from_pretrained(base, "Meerkat-AI/Meerkat-TRIZ-v1-Qwen3.8-27B").eval()

prompt = tok.apply_chat_template(
    [{"role": "system", "content": "你是 TRIZ 创新方法论专家助手, 用中文专业回答用户关于 TRIZ 理论、发明原理、矛盾分析、ARIZ 算法等方面的问题。"},
     {"role": "user", "content": "请解释技术矛盾与物理矛盾的区别, 并各给一例。"}],
    tokenize=False, add_generation_prompt=True)
out = model.generate(**tok(prompt, return_tensors="pt").to("cuda"),
                     max_new_tokens=2048, do_sample=False,
                     pad_token_id=tok.eos_token_id)
print(tok.decode(out[0][tok(prompt, return_tensors="pt")["input_ids"].shape[1]:],
                 skip_special_tokens=True))
```

## 评测结果

v5 金标集（300 题），双轨评分（关键词命中率 + LLM 评委 0–4 分；评委钉死
moonshot-v1-32k，T=0）。全部为逐题配对（paired bootstrap，10000 次，95% CI）。
评委与训练数据生成器同族（弱异源），谱系已声明，并经跨族外部评委终审量化。

### 同族（Moonshot）读数

| 指标 | 基座 (Qwen3.8-27B) | **适配器** | 配对差值 [95% CI] |
|---|---|---|---|
| keyword 均值 | 0.6245 | **0.6236** | −0.0009 [−0.019, +0.017] 不显著 |
| judge Arm-A 均值 | 2.9300 | **3.5333** | **+0.6033 [+0.497, +0.713] 显著** |
| judge pass 率 | 0.787 [0.737, 0.829] | 0.960 [0.931, 0.977] | McNemar p=3.1e-12 |

质量门 300/300 全过（0 invalid）；overrefusal 0/300，过门。

### 跨族外部评委终审（300 题配对，臂 A 协议逐字复跑）

| 评委 | 配对差值 [95% CI] | 显著性（原始） |
|---|---|---|
| claude-sonnet-4-6 (Anthropic) | +0.031 [−0.071, +0.129] | 否 |
| gpt-5.4 (OpenAI) | +0.094 [+0.010, +0.175] | 是（贴线） |
| gemini-3.5-flash (Google) | +0.107 [+0.013, +0.205] | 是（贴线） |

**噪声账后**（tensoris 网关外部评委实测 T=0 翻转率 0.18/0.68/0.78，逐题复跑方差
传播进 CI）：**三个跨族评委均不再统计显著**（gpt −0.004…+0.192，gemini
−0.015…+0.229，claude 原本就不显著）。

### 诚实结论

- 同族（Moonshot）judge 读数 **+0.60 被评委家族效应放大约 6 倍**；跨族读数为
  **+0.03~+0.11——方向为正，但计入评委非确定性后不显著**。
- keyword 轨与基座持平（−0.0009）。
- **净结论：本次基座迁移中，适配器在跨族评委下的 judge 轨与 Qwen3.8-27B 基座
  统计打平**。本版本为透明与可复现而发布，**不构成对基座的质量优势证明**。

## 训练

- 方法：LoRA SFT，r=64，alpha=128，dropout=0，覆盖全部线性投影（含 DeltaNet
  `in_proj_*`）；BF16（无量化）。
- 配置 4 epochs；**early-stopping 在 step 1600 触发**（1.15 epochs）；最佳
  checkpoint 在 step 1300，eval loss **1.5058**（completion-only 口径）。
- 数据：~11k 条中文 TRIZ SFT 样本（`v5_train_v5a.jsonl`，11,096 train / 1,050
  val）。**训练集不公开**（源自第三方版权 TRIZ 教材）。
- 优化器 adamw_torch，LR 2e-4，cosine horizon 2774 步。

## 局限

- 仅验证中文 TRIZ 领域问答；领域外能力未评测（通用基准尚未跑，属已知欠账）。
- 长回答（>2048 token）可能被截断；推荐贪心解码。
- 主评委为单一供应商（Moonshot，弱异源）；跨族终审显示提升在噪声账后不显著——
  请勿将 +0.60 当作跨族普适的效应量引用。
- 基座为多模态（含 vision tower）模型；本适配器仅在文本（language_model）通道
  训练与验证。
- 基座可得性：见顶部警告，Qwen3.8-27B 基座权重不随本仓库分发。

## 引用

```bibtex
@misc{meerkat-triz-v1-qwen38-2026,
  title  = {Meerkat-TRIZ-v1-Qwen3.8-27B: A TRIZ-Domain LoRA Fine-tune of Qwen3.8-27B},
  author = {Meerkat AI},
  year   = {2026},
  url    = {https://huggingface.co/Meerkat-AI/Meerkat-TRIZ-v1-Qwen3.8-27B}
}
```

请同时按基座自身的许可证引用 Qwen3.8-27B。
