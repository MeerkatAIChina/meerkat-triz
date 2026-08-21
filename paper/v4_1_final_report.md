# pipeline_v4 最终评测汇总

生成时间: 2026-07-29T07:23:33

## 多方对比 (overall)

| 模型 | judge 模型 | 关键词轨均值 | judge 轨均值 | 关键词 pass 率 | judge pass 率 |
|---|---|---|---|---|---|
| base_gold | moonshot-v1-32k | 0.3661 | 1.5700 | 0.350 | 0.120 |
| v2_gold | moonshot-v1-32k | 0.5483 | 2.5800 | 0.630 | 0.620 |
| v3_gold | moonshot-v1-32k | 0.5716 | 2.2800 | 0.640 | 0.430 |
| v4_gold | moonshot-v1-32k | 0.5568 | 2.5700 | 0.650 | 0.630 |
| v4_1_gold | moonshot-v1-32k | 0.5289 | 2.6200 | 0.590 | 0.630 |

## 关键词轨 各子集均值

| 子集 | base_gold | v2_gold | v3_gold | v4_gold | v4_1_gold |
|---|---|---|---|---|---|
| ariz_guidance | 0.4437 | 0.6439 | 0.6938 | 0.7494 | 0.5923 |
| case_generation | 0.2190 | 0.5492 | 0.5683 | 0.5190 | 0.5190 |
| concept_explanation | 0.5317 | 0.5187 | 0.5257 | 0.4356 | 0.4790 |
| contradiction_analysis | 0.3645 | 0.4987 | 0.5401 | 0.5233 | 0.5385 |
| innovation_assessment | 0.4748 | 0.6914 | 0.6748 | 0.7081 | 0.6748 |
| principle_recommendation | 0.2217 | 0.4524 | 0.4661 | 0.4412 | 0.4276 |

## judge 轨 各子集均值

| 子集 | base_gold | v2_gold | v3_gold | v4_gold | v4_1_gold |
|---|---|---|---|---|---|
| ariz_guidance | 1.7000 | 2.6500 | 2.4500 | 2.8500 | 2.9000 |
| case_generation | 1.4667 | 2.4667 | 2.3333 | 2.5333 | 2.4667 |
| concept_explanation | 1.7333 | 2.8000 | 2.5333 | 2.8000 | 2.8000 |
| contradiction_analysis | 1.0000 | 2.5500 | 2.1000 | 2.3500 | 2.7500 |
| innovation_assessment | 2.1000 | 2.8000 | 2.5000 | 2.9000 | 2.9000 |
| principle_recommendation | 1.7000 | 2.3500 | 1.9500 | 2.2000 | 2.0500 |

## 决策门

- 锚点: **v2_gold** (评测时 --baseline-results 指定)
- v4_1_gold-v2_gold judge overall 差值: +0.0400 [-0.0900, +0.1700]
- v4_1_gold-v2_gold 关键词 overall 差值: -0.0195 [-0.0607, +0.0200]
- judge overall 显著优于 v2_gold: 否
- 显著退化子指标: 无
- McNemar (judge pass): p=1.0

### 判定: **保留 v2**

原因: judge 轨 overall 未显著优于 v2_gold
