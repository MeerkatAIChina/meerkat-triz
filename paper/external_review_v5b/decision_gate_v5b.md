# 决策门 2.0 机检报告

## 总判定: **拒绝: 保留 v2** (`keep_v2`)

| 门 | 判定 | 理由 |
|---|---|---|
| G0 | **PASS** | 拒答模板命中率 0.00% (阈值 ≤2%) |
| G1 | **PASS** | 臂A v5−base +0.4300 [+0.3233, +0.5367] (要求 CI 下限 > -0.15); 臂 B 无同桶配对/未跑, 以臂 A 为准 (§6.4) |
| G2 | **PASS** | judge v5−v2 +0.7033 [+0.6267, +0.7800] (CI 下限须 > -0.05); judge 显著为正=True, kw 显著为正=True (至少一轨显著为正) |
| G3 | **FREEZE** | principle_recommendation kw CI 上限<0 但未做 E2 式归因 → 冻结 |
| G4 | **PASS** | 全部子集 n≥30 (n<30 不独立否决) |
| G5 | **PASS** | ariz 两轨无分歧 |
| G6 | **SKIP** | 探针数据缺失 (120 题探针未跑; 回溯场景可缺省) |
| G7 | **FAIL** | v5_vs_base/overall: judge +0.4300 vs kw -0.0284 显著反向; v5_vs_v2/principle_recommendation: judge +0.8000 vs kw -0.1353 显著反向 → 默认回滚 + 归因 |
| G8 | **PASS** | v5_vs_base/claude-sonnet-4-6 +0.0669 [-0.0100, +0.1405]; v5_vs_base/gemini-3.5-flash +0.0151 [-0.0870, +0.1154]; v5_vs_base/gpt-5.4 -0.0033 [-0.0836, +0.0803] |

- 规则: G1–G8 同时通过才放行 (§6.7 + G8 异源复核); G0 overrefusal 为附加门
