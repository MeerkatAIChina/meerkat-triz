# 决策门 2.0 机检报告

## 总判定: **拒绝: 保留 v2** (`keep_v2`)

| 门 | 判定 | 理由 |
|---|---|---|
| G0 | **SKIP** | overrefusal 数据缺失 (仅 base 评测时可缺省) |
| G1 | **FAIL** | 臂A v5−base -0.3000 [-0.4600, -0.1400] (要求 CI 下限 > -0.15); 臂 B 无同桶配对/未跑, 以臂 A 为准 (§6.4) |
| G2 | **FAIL** | judge v5−v2 -0.0100 [-0.1300, +0.1100] (CI 下限须 > -0.05); judge 显著为正=False, kw 显著为正=False (至少一轨显著为正) |
| G3 | **FAIL** | concept_explanation kw -0.0832 [-0.1492,-0.0229] 归因=真缺失 → 退化成立 |
| G4 | **PASS** | 描述性子集: ariz_guidance(n=20), case_generation(n=15), concept_explanation(n=15), contradiction_analysis(n=20), innovation_assessment(n=10), principle_recommendation(n=20) (n<30 不独立否决) |
| G5 | **PASS** | ariz 两轨无分歧 |
| G6 | **SKIP** | 探针数据缺失 (120 题探针未跑; 回溯场景可缺省) |
| G7 | **PASS** | 无两轨显著反向维度 |

- 规则: G1–G7 同时通过才放行 (§6.7); G0 overrefusal 为附加门
