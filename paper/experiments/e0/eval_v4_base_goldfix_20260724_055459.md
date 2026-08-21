# eval_v4 评测报告 — base_goldfix

- 时间: 20260724_055459
- 适配器: (纯 base)
- 评测集: /home/chinux/jupyterlab/meerkatai/data/processed/v4_gold.jsonl (100 题)
- judge 模型: **moonshot-v1-32k** (探测: kimi-k2-0711-preview=FAIL; moonshot-v1-32k=OK; moonshot-v1-8k=OK)

## keyword 轨

- overall 均值: **0.5642** (n=100)
- pass 率: 0.670 [0.573, 0.754] (Wilson, 67/100)

| 子集 | n | 均值 |
|---|---|---|
| ariz_guidance | 20 | 0.6051 |
| case_generation | 15 | 0.5175 |
| concept_explanation | 15 | 0.5800 |
| contradiction_analysis | 20 | 0.5655 |
| innovation_assessment | 10 | 0.7224 |
| principle_recommendation | 20 | 0.4660 |

## judge 轨

- overall 均值: **2.8700** (n=100)
- pass 率: 0.830 [0.745, 0.891] (Wilson, 83/100)

| 子集 | n | 均值 |
|---|---|---|
| ariz_guidance | 20 | 3.1000 |
| case_generation | 15 | 2.5333 |
| concept_explanation | 15 | 3.2000 |
| contradiction_analysis | 20 | 2.6500 |
| innovation_assessment | 10 | 3.0000 |
| principle_recommendation | 20 | 2.8000 |

## 与基线配对对比 (共同题 n=100)

### keyword 轨
- overall 差值 (this-base): +0.1981 [+0.1492, +0.2495] (paired bootstrap 10000, 显著)
- McNemar: base→this 翻正 33, 翻负 1, p=0.0000

| 子集 | 差值 | 95% CI |
|---|---|---|
| ariz_guidance | +0.1614 | [+0.0571, +0.2700] |
| case_generation | +0.2984 | [+0.1540, +0.4381] |
| concept_explanation | +0.0483 | [+0.0016, +0.0956] |
| contradiction_analysis | +0.2010 | [+0.0788, +0.3377] |
| innovation_assessment | +0.2476 | [+0.0976, +0.4267] |
| principle_recommendation | +0.2442 | [+0.1485, +0.3345] |

### judge 轨
- overall 差值 (this-base): +1.3000 [+1.1100, +1.4900] (paired bootstrap 10000, 显著)
- McNemar: base→this 翻正 73, 翻负 2, p=0.0000

| 子集 | 差值 | 95% CI |
|---|---|---|
| ariz_guidance | +1.4000 | [+1.0500, +1.8000] |
| case_generation | +1.0667 | [+0.4667, +1.6667] |
| concept_explanation | +1.4667 | [+1.0667, +1.8667] |
| contradiction_analysis | +1.6500 | [+1.2000, +2.1000] |
| innovation_assessment | +0.9000 | [+0.6000, +1.2000] |
| principle_recommendation | +1.1000 | [+0.6500, +1.5500] |
