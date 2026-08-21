# pipeline_v4 最终评测汇总

生成时间: 2026-07-24T01:16:48

## 四方对比 (overall)

| 模型 | judge 模型 | 关键词轨均值 | judge 轨均值 | 关键词 pass 率 | judge pass 率 |
|---|---|---|---|---|---|
| base_gold | moonshot-v1-32k | 0.3661 | 1.5700 | 0.350 | 0.120 |
| v2_gold | moonshot-v1-32k | 0.5483 | 2.5800 | 0.630 | 0.620 |
| v3_gold | moonshot-v1-32k | 0.5716 | 2.2800 | 0.640 | 0.430 |
| v4_gold | moonshot-v1-32k | 0.5568 | 2.5700 | 0.650 | 0.630 |

## 关键词轨 各子集均值

| 子集 | base_gold | v2_gold | v3_gold | v4_gold |
|---|---|---|---|---|
| ariz_guidance | 0.4437 | 0.6439 | 0.6938 | 0.7494 |
| case_generation | 0.2190 | 0.5492 | 0.5683 | 0.5190 |
| concept_explanation | 0.5317 | 0.5187 | 0.5257 | 0.4356 |
| contradiction_analysis | 0.3645 | 0.4987 | 0.5401 | 0.5233 |
| innovation_assessment | 0.4748 | 0.6914 | 0.6748 | 0.7081 |
| principle_recommendation | 0.2217 | 0.4524 | 0.4661 | 0.4412 |

## judge 轨 各子集均值

| 子集 | base_gold | v2_gold | v3_gold | v4_gold |
|---|---|---|---|---|
| ariz_guidance | 1.7000 | 2.6500 | 2.4500 | 2.8500 |
| case_generation | 1.4667 | 2.4667 | 2.3333 | 2.5333 |
| concept_explanation | 1.7333 | 2.8000 | 2.5333 | 2.8000 |
| contradiction_analysis | 1.0000 | 2.5500 | 2.1000 | 2.3500 |
| innovation_assessment | 2.1000 | 2.8000 | 2.5000 | 2.9000 |
| principle_recommendation | 1.7000 | 2.3500 | 1.9500 | 2.2000 |

## 决策门

- v4-base judge overall 差值: +1.0000 [+0.8000, +1.1900]
- v4-base 关键词 overall 差值: +0.1907 [+0.1297, +0.2520]
- judge overall 显著优于 base: 是
- 显著退化子指标: keyword/concept_explanation
- McNemar (judge pass): p=1.6979681549678105e-12

### 判定: **保留 v2**

原因: 存在显著退化: keyword/concept_explanation
