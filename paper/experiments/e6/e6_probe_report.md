# E6 通用能力探针报告 — Meerkat-TRIZ-v1 (v5a) 无灾难性遗忘验证

- 日期: 2026-07-30 (远端 spark-855a, DGX Spark GB10)
- 探针: `data/processed/v5_data/general_probe_v5.json`, 120 题
  (6 类 × 20: common_sense / math / logic / writing / code / instruction_following)
- 臂: base (Qwen3.6-35B-A3B 纯基座) / v2 / v5a (= Meerkat-TRIZ-v1)
- 生成协议: 与 v5 harness 同源 — BF16、device=cuda、贪心、max_new_tokens=1024、
  空 think 块保留 (E0 冒烟断言全程通过)、compat peft 补丁、中性系统 prompt
- 评分: 纯子串关键词命中率 (无别名表; 别名表为 TRIZ 专用)
- 统计: 逐题配对 bootstrap n=10000, stdlib Random(42); 类级 n=20 一律描述性

## 结果

### Overall (n=120, 三臂全配对, 缺失 0, think 残留 0)

| 臂 | 关键词命中率 |
|---|---|
| base | 84.03% |
| v2 | 86.81% |
| v5a | 85.56% |

| 对比 | 配对差值 [95% CI] | 显著性 |
|---|---|---|
| v5a − base | **+1.53pp [−1.94, +5.28]** | 不显著 |
| v5a − v2 | **−1.25pp [−3.47, +0.28]** | 不显著 |

### G6 裁决 (决策门 2.0, 方案 §7.1)

> **v5a−v2 探针 overall −1.25pp > −5pp → PASS**

G6 从"数据缺失 SKIP"转为"实测 PASS"。决议门 G6 数据缺口已补齐。

### 类级命中率 (描述性, n=20/类, 不作门判据)

| 类别 | base | v2 | v5a |
|---|---|---|---|
| code | 89.2 | 90.0 | 90.0 |
| common_sense | 85.0 | 87.5 | 87.5 |
| instruction_following | 68.3 | 73.3 | 75.0 |
| logic | 95.0 | 100.0 | 95.0 |
| math | 66.7 | 70.0 | 68.3 |
| writing | 100.0 | 100.0 | 97.5 |

## 解读

1. **无灾难性遗忘证据**: v5a 相对 base 点值为正 (+1.53pp), CI 含 0;
   相对 v2 −1.25pp, 远在 G6 下限 (−5pp) 之内。在 120 题分辨率
   (MDE≈4.7pp) 下, 领域微调未造成可分辨的通用能力损失。
2. **分辨率声明** (§6.1 强制): 本探针只能分辨 ≥~5pp 级退化;
   "通用损失 <3%" 级别的微小退化超出本探针分辨能力, 不报"证明无损",
   只报"未检出 ≥5pp 级退化"。
3. **评分口径局限**: 关键词轨对 writing / instruction_following 开放题
   误判率偏高 (E6 设计卡风险条)。本次两类的类级读数仅作归因线索;
   如需升级, 可对这两类加 rubric judge 轨 (后续工作)。
4. **过程事故记录**: 初版生成脚本漏引 `compat` peft 补丁 (v2 臂
   WeightConverter 崩溃) 及 device_map="auto" 触发 CPU offload
   (Triton 崩溃); 修正为 device="cuda" + import compat 后全绿——
   与 v5 harness 加载路径完全一致。

## 产物

- 生成缓存 (远端): `results/e6_probe/gen_{base,v2,v5a}.jsonl` (360 条)
- 本地归档: `paper/experiments/e6/gen_{base,v2,v5a}.jsonl`
- 分析脚本: `paper/experiments/e6/e6_probe_analyze.py`
- 机读结果: `paper/experiments/e6/e6_probe_result.json`
- 生成脚本 (远端+本地): `pipeline_v5/eval/probe_gen_eval_v5.py`
