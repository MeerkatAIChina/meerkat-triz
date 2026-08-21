# E0 干净 base 锚点统计报告

## 各模型双轨总览 (n=100)

| 模型 | kw 均值 | kw pass (Wilson) | judge 均值 | judge pass (Wilson) |
|---|---|---|---|---|
| base_goldfix | 0.5642 | 0.670 [0.573, 0.754] | 2.8700 | 0.830 [0.745, 0.891] |
| v2 | 0.5483 | 0.630 [0.532, 0.718] | 2.5800 | 0.620 [0.522, 0.709] |
| v4 | 0.5568 | 0.650 [0.553, 0.736] | 2.5700 | 0.630 [0.532, 0.718] |
| base_polluted | 0.3661 | 0.350 [0.264, 0.447] | 1.5700 | 0.120 [0.070, 0.198] |

## v4_vs_base_goldfix (共同题 n=100)

### keyword 轨
- overall: **-0.0074 [-0.0497, +0.0345]** (不显著)
- McNemar: 翻正 10 / 翻负 12, p=0.8318

| 子集 | 差值 | 95% CI | 显著 |
|---|---|---|---|
| ariz_guidance | +0.1443 | [+0.0360, +0.2513] | ✅ |
| case_generation | +0.0016 | [-0.0921, +0.0857] | — |
| concept_explanation | -0.1444 | [-0.2476, -0.0508] | ✅ |
| contradiction_analysis | -0.0422 | [-0.1221, +0.0362] | — |
| innovation_assessment | -0.0143 | [-0.0643, +0.0357] | — |
| principle_recommendation | -0.0248 | [-0.1143, +0.0595] | — |

### judge 轨
- overall: **-0.3000 [-0.4600, -0.1400]** (显著)
- McNemar: 翻正 10 / 翻负 30, p=0.002221

| 子集 | 差值 | 95% CI | 显著 |
|---|---|---|---|
| ariz_guidance | -0.2500 | [-0.5000, +0.0500] | — |
| case_generation | +0.0000 | [-0.5333, +0.5333] | — |
| concept_explanation | -0.4000 | [-0.8667, +0.0667] | — |
| contradiction_analysis | -0.3000 | [-0.7000, +0.0500] | — |
| innovation_assessment | -0.1000 | [-0.3000, +0.0000] | — |
| principle_recommendation | -0.6000 | [-0.9500, -0.2500] | ✅ |

## v2_vs_base_goldfix (共同题 n=100)

### keyword 轨
- overall: **-0.0158 [-0.0550, +0.0236]** (不显著)
- McNemar: 翻正 9 / 翻负 13, p=0.5235

| 子集 | 差值 | 95% CI | 显著 |
|---|---|---|---|
| ariz_guidance | +0.0388 | [-0.0750, +0.1510] | — |
| case_generation | +0.0317 | [-0.0635, +0.1317] | — |
| concept_explanation | -0.0613 | [-0.1524, +0.0178] | — |
| contradiction_analysis | -0.0668 | [-0.1461, +0.0086] | — |
| innovation_assessment | -0.0310 | [-0.0929, +0.0333] | — |
| principle_recommendation | -0.0136 | [-0.1026, +0.0710] | — |

### judge 轨
- overall: **-0.2900 [-0.4200, -0.1600]** (显著)
- McNemar: 翻正 5 / 翻负 26, p=0.0001922

| 子集 | 差值 | 95% CI | 显著 |
|---|---|---|---|
| ariz_guidance | -0.4500 | [-0.7000, -0.2000] | ✅ |
| case_generation | -0.0667 | [-0.4667, +0.3333] | — |
| concept_explanation | -0.4000 | [-0.7333, -0.0667] | ✅ |
| contradiction_analysis | -0.1000 | [-0.4000, +0.2000] | — |
| innovation_assessment | -0.2000 | [-0.5000, +0.0000] | — |
| principle_recommendation | -0.4500 | [-0.7500, -0.1500] | ✅ |

## v4_vs_v2 (共同题 n=100)

### keyword 轨
- overall: **+0.0084 [-0.0236, +0.0435]** (不显著)
- McNemar: 翻正 8 / 翻负 6, p=0.7905

| 子集 | 差值 | 95% CI | 显著 |
|---|---|---|---|
| ariz_guidance | +0.1055 | [-0.0124, +0.2363] | — |
| case_generation | -0.0302 | [-0.0905, +0.0238] | — |
| concept_explanation | -0.0832 | [-0.1492, -0.0229] | ✅ |
| contradiction_analysis | +0.0246 | [-0.0083, +0.0612] | — |
| innovation_assessment | +0.0167 | [+0.0000, +0.0500] | — |
| principle_recommendation | -0.0112 | [-0.0719, +0.0548] | — |

### judge 轨
- overall: **-0.0100 [-0.1300, +0.1100]** (不显著)
- McNemar: 翻正 11 / 翻负 10, p=1

| 子集 | 差值 | 95% CI | 显著 |
|---|---|---|---|
| ariz_guidance | +0.2000 | [-0.0500, +0.5000] | — |
| case_generation | +0.0667 | [-0.3333, +0.4000] | — |
| concept_explanation | +0.0000 | [-0.2667, +0.2667] | — |
| contradiction_analysis | -0.2000 | [-0.4500, +0.0000] | — |
| innovation_assessment | +0.1000 | [+0.0000, +0.3000] | — |
| principle_recommendation | -0.1500 | [-0.5000, +0.1500] | — |

## v4_vs_base_polluted (共同题 n=100)

### keyword 轨
- overall: **+0.1907 [+0.1297, +0.2520]** (显著)
- McNemar: 翻正 37 / 翻负 7, p=5.3e-06

| 子集 | 差值 | 95% CI | 显著 |
|---|---|---|---|
| ariz_guidance | +0.3057 | [+0.1612, +0.4513] | ✅ |
| case_generation | +0.3000 | [+0.1667, +0.4381] | ✅ |
| concept_explanation | -0.0962 | [-0.2000, -0.0025] | ✅ |
| contradiction_analysis | +0.1588 | [+0.0438, +0.2850] | ✅ |
| innovation_assessment | +0.2333 | [+0.0833, +0.4200] | ✅ |
| principle_recommendation | +0.2195 | [+0.0814, +0.3521] | ✅ |

### judge 轨
- overall: **+1.0000 [+0.8000, +1.1900]** (显著)
- McNemar: 翻正 55 / 翻负 4, p=1.698e-12

| 子集 | 差值 | 95% CI | 显著 |
|---|---|---|---|
| ariz_guidance | +1.1500 | [+0.8000, +1.5000] | ✅ |
| case_generation | +1.0667 | [+0.6667, +1.4667] | ✅ |
| concept_explanation | +1.0667 | [+0.6667, +1.4000] | ✅ |
| contradiction_analysis | +1.3500 | [+0.8500, +1.8500] | ✅ |
| innovation_assessment | +0.8000 | [+0.5000, +1.0000] | ✅ |
| principle_recommendation | +0.5000 | [-0.0500, +1.0000] | — |

## 决策门 (干净 base 锚点)

- 规则: v4 judge overall 显著>base 且两轨所有子集无显著退化 → 建议替代 v2, 否则保留 v2
- judge overall 显著为正: False
- judge 子集显著退化: ['principle_recommendation']
- kw 子集显著退化: ['concept_explanation']
- **判定: 保留 v2**
