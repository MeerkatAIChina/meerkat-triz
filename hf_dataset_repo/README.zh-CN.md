> 🇺🇸 [English](README.md) | 🇨🇳 中文

# triz-gold-benchmark

中文 TRIZ（发明问题解决理论）评测金标集，配套
[Meerkat-TRIZ-v1](https://huggingface.co/Meerkat-AI/Meerkat-TRIZ-v1) 与
[meerkat-triz](https://github.com/coidea-sys/meerkat-triz) 评测 harness。

## 内容

| 文件 | 题数 | 来源口径 |
|---|---|---|
| `triz_gold_v4_public.jsonl` | 100 | v4 评测口径（六方对比用） |
| `triz_gold_v5_public.jsonl` | 300 | v5 评测口径（正式发布评测用） |

每行一个 JSON 对象：

```json
{"id": "v5_gold_000", "subset": "ariz_guidance", "question": "...", "keywords": ["..."]}
```

**subset 六类任务**（v4 / v5 题数）：

| subset | v4 | v5 |
|---|---|---|
| ariz_guidance | 20 | 60 |
| case_generation | 15 | 45 |
| concept_explanation | 15 | 45 |
| contradiction_analysis | 20 | 60 |
| innovation_assessment | 10 | 30 |
| principle_recommendation | 20 | 60 |

## 重要说明：本公开版不含参考答案

题目与期望关键词由 LLM 基于第三方 TRIZ 教材与课件生成，参考答案是
对版权材料的派生改写。为控制版权风险，**本公开版只发布
question + keywords + subset**，未发布 reference_answer。

影响：

- **关键词轨可完整复现**（harness 的关键词轨只依赖 question/keywords）。
- **judge 轨的参考答案字段缺失**：judge prompt 中参考答案是质量锚，
  缺失时绝对分不可与官方报告直接对比（配对差值受影响较小）。
- 如需完整版用于学术研究，请在仓库 issue 中联系作者。

## 使用

```bash
meerkat-eval --config configs/eval_v5.json \
    --adapter-path <adapter> --tag my_run \
    --eval-file triz_gold_v5_public.jsonl \
    --baseline-results <base 结果.json>
```

## 泄漏声明

本基准从未进入 Meerkat-TRIZ-v1 的训练分布：训练集对本双金标做
3-gram Jaccard≥0.5 扫描，命中 0 题。

## License

CC-BY-NC-4.0（非商业研究用途）。题目文本派生自第三方版权 TRIZ 材料，
商用前请联系作者确认。
