> 🇺🇸 [English](README.md) | 🇨🇳 中文

# Meerkat-TRIZ-v1

TRIZ（发明问题解决理论）领域 LoRA 适配器，基座为
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)（Apache-2.0）。
面向中文 TRIZ 问答：发明原理推荐、矛盾分析、ARIZ 引导、创新方案评估、
概念讲解与案例生成六类任务。

本模型的发布重点在于**评测方法论的透明**：全部结论由配套的严格门控双轨
harness 产出（[GitHub: meerkat-triz](https://github.com/coidea-sys/meerkat-triz)），
配对统计带置信区间，打平处如实报打平。

## 快速使用

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.6-35B-A3B", dtype=torch.bfloat16,
    device_map="cuda", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3.6-35B-A3B", trust_remote_code=True)
model = PeftModel.from_pretrained(base, "Meerkat-AI/Meerkat-TRIZ-v1").eval()

# 关键: 保留空 think 块 —— 训练/评测格式一致的前提 (E0 纪律)
prompt = tok.apply_chat_template(
    [{"role": "system", "content": "你是 TRIZ 创新方法论专家助手, 用中文专业回答用户关于 TRIZ 理论、发明原理、矛盾分析、ARIZ 算法等方面的问题。"},
     {"role": "user", "content": "请解释技术矛盾与物理矛盾的区别, 并各给一例。"}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False)
out = model.generate(**tok(prompt, return_tensors="pt").to("cuda"),
                     max_new_tokens=2048, do_sample=False,
                     pad_token_id=tok.eos_token_id)
print(tok.decode(out[0][tok(prompt, return_tensors="pt")["input_ids"].shape[1]:],
                 skip_special_tokens=True))
```

> ⚠️ **prompt 格式敏感**：本模型在保留空 think 块（`<think>\n\n</think>`）的
> prompt 上训练与评测。若你的模板剥离了空 think 块，输出质量可能下降
> （我们实测该失配可造成 judge 分数 −0.2 量级的下压）。基座为
> thinking-native 架构，`enable_thinking=False` 下模板仍会产生空块，请保留。

## 评测结果

两套金标集、双轨评分（关键词命中率 + LLM 评委 0–4 分，judge 钉死
moonshot-v1-32k，T=0），全部对比为逐题配对（paired bootstrap 10000 次
95% CI + McNemar）。评委与训练数据生成器同族（弱异源），谱系已声明——
且已由**发布后的异源评委终审**量化（见下）：方向获确认，量级约为同族
读数的 1/4。

### v5 金标（300 题，训练同分布评测口径）

本节 judge 轨数字为**同族（Moonshot）评委读数**。

| 指标 | base | **Meerkat-TRIZ-v1** | 配对差值 [95% CI] |
|---|---|---|---|
| 关键词轨均值 | 0.6384 | **0.6383** | −0.0001 [−0.017, +0.017] 不显著 |
| judge 臂A 均值 | 3.0300 | **3.4233** | **+0.3933 [+0.297, +0.490] 显著** |
| judge pass 率 | 0.843 [0.798, 0.880] | 0.947 [0.915, 0.967] | McNemar p=1.5e-05 |
| 关键词 pass 率 | 0.737 [0.684, 0.783] | 0.747 [0.695, 0.793] | — |

judge 子集差值：concept_explanation **+0.733** [+0.467, +1.022]、
innovation_assessment +0.600、contradiction_analysis +0.467、
case_generation +0.378、ariz_guidance +0.317 均显著为正；
principle_recommendation +0.050 不显著。
质量门 299/300 过门（1 题长度门 invalid，已计入）；overrefusal 0/300 过门。

### 异源评委终审（发布后，299 题配对）

臂 A 协议逐字复跑，换用三个外部评委
（[完整报告](https://github.com/coidea-sys/meerkat-triz/blob/main/docs/EXTERNAL_JUDGE_REVIEW.zh-CN.md)）：

| 评委 | 配对差值 [95% CI] | 显著 |
|---|---|---|
| claude-sonnet-4-6（Anthropic） | **+0.094 [+0.020, +0.167]** | 是 |
| gpt-5.4（OpenAI） | **+0.104 [+0.020, +0.184]** | 是 |
| gemini-3.5-flash（Google） | −0.048 [−0.144, +0.045] | 否 |

- 可辩护的头条结论：**三个外部评委中两个显著（+0.09 ~ +0.10）**；
  同族 +0.39 被评委家族效应放大约 4 倍。
- 外部评委互相逐题 Spearman 0.63–0.75，而与同族评委仅 0.27–0.31——
  这是评分体系差异，不是评委噪声。
- **concept_explanation 是最强共识**：三个外部评委下全部显著
  （+0.24 / +0.31 / +0.29）。
- 观察项：gemini 对 principle_recommendation 给出显著负差（−0.31）；
  单一评委支持，记为开放信号，非已确立的回退。

### v4 金标（100 题，跨口径六方对比）

v4 口径下与历代候选同场（同题集、同 judge、同 harness）：

| 模型 | 关键词轨 | judge 轨 | judge pass 率 |
|---|---|---|---|
| base（旧锚点，think 污染，仅供参考） | 0.3661 | 1.5700 | 0.120 |
| v2（干净锚点） | 0.5483 | 2.5800 | 0.620 |
| v3 | 0.5716 | 2.2800 | 0.430 |
| v4 | 0.5568 | 2.5700 | 0.630 |
| v4.1 | 0.5289 | 2.6200 | 0.630 |
| **Meerkat-TRIZ-v1（v5a）** | 0.5391 | 2.3600 | 0.600 |

vs v2 干净锚点配对：关键词 −0.0093 [−0.062, +0.039] 不显著；
judge −0.2200 [−0.470, +0.010] 不显著 —— **v4 口径下与 v2 统计打平**。
注意 v4 harness 会剥离空 think 块（与本模型训练格式失配），judge 负方向
部分可归因于此（下界估计）；关键词轨子集 contradiction_analysis
+0.101 [+0.024, +0.180] 显著为正，case_generation −0.137 [−0.270, −0.030]
显著为负。

### 结论纪律

- ✅ 可以宣称：v5 口径下 judge 轨相对干净 base 有提升——**三个外部评委
  中两个显著（+0.09 ~ +0.10）**（Claude、GPT；Gemini 不显著），发布后
  已经外部验证。同族评委读数为 +0.39，不得跨评委家族外推。
- ✅ 可以宣称：定向修复的 concept_explanation 在三个外部评委下全部
  显著（+0.24 / +0.31 / +0.29）。
- ✅ 可以宣称：v4 口径下与内部最强基线 v2 打平（配对 CI 含 0）。
- ❌ 不可宣称：超越任何外部模型（未做外部对比）；关键词轨相对 base
  有增量（−0.0001，无差异）；把同族 +0.39 当作与评委家族无关的效应量。

## 训练

- 方法：LoRA SFT，r=64，alpha=128，dropout=0，target 全部线性投影
  （含 DeltaNet `in_proj_*`）；BF16；4 epochs / 5548 steps 完整训练，
  final train loss 0.604，eval loss 1.626（轨迹见 `adapter_info.json`）。
- 数据：~11k 条中文 TRIZ ChatML SFT 样本（双风格/双解耦/分组划分，
  空 think 块保留）。**训练集不公开**：含第三方版权 TRIZ 教材派生内容。
- 污染检查：训练集对 v4/v5 双金标 3-gram Jaccard≥0.5 命中均为 **0 题**。
- 冻结超参配置：见 GitHub 仓 `configs/train_v5a.json`。

## 限制

- 仅验证中文 TRIZ 领域问答；领域外能力未评测，不应用于高风险决策。
- 长回答（>2048 tokens）可能被截断；生成建议使用贪心解码（训练即贪心评测）。
- 主评委单一供应商（Moonshot）同族弱异源。发布后异源终审（见上）在
  2/3 外部评委下确认了提升方向，但量级约为同族读数的 1/4；同族绝对分
  不得跨评委家族外推。一个外部评委检出 principle_recommendation 显著
  负差——单一评委支持的开放信号，正在跟进。
- 基座为视觉-语言 MoE，本适配器只验证文本通道。

## 引用

```bibtex
@misc{meerkat-triz-v1-2026,
  title  = {Meerkat-TRIZ-v1: A TRIZ-Domain LoRA Fine-tune of Qwen3.6-35B-A3B},
  author = {Meerkat AI},
  year   = {2026},
  url    = {https://huggingface.co/Meerkat-AI/Meerkat-TRIZ-v1}
}
```

基座模型请引用 Qwen3.6-35B-A3B（Apache-2.0，
[license](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE)）。
