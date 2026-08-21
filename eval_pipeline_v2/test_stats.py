#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval2 统计函数单元测试（合成数据 + 已知解析解对照）。"""
import sys

sys.path.insert(0, "/tmp/eval_pipeline_v2")
import numpy as np  # noqa: E402
import eval2  # noqa: E402

passed = []


def check(name, cond, detail=""):
    assert cond, f"FAIL {name}: {detail}"
    passed.append(name)
    print(f"PASS {name} {detail}")


# ---- Wilson CI：p=0.5, n=100 的已知解析值 (0.4038, 0.5962) ----
p, lo, hi = eval2.wilson_ci(50, 100)
check("wilson_p", abs(p - 0.5) < 1e-12, f"p={p}")
check("wilson_ci", abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3,
      f"ci=[{lo:.4f},{hi:.4f}]")
p2, lo2, hi2 = eval2.wilson_ci(0, 0)
check("wilson_n0", (p2, lo2, hi2) == (0.0, 0.0, 0.0))
p3, lo3, hi3 = eval2.wilson_ci(10, 10)
check("wilson_bounds", 0.0 <= lo3 <= hi3 <= 1.0 and abs(p3 - 1.0) < 1e-12,
      f"k=n=10 -> ci=[{lo3:.4f},{hi3:.4f}]")

# ---- McNemar 精确检验：b=2, c=9 -> 2*P(X<=2), X~Bin(11,0.5) ----
# P(X<=2) = (C(11,0)+C(11,1)+C(11,2))/2048 = (1+11+55)/2048 = 67/2048
# p = 134/2048 = 0.0654296875
pv = eval2.mcnemar_exact_p(2, 9)
check("mcnemar_hand", abs(pv - 0.0654296875) < 1e-9, f"p={pv}")
check("mcnemar_sym", abs(eval2.mcnemar_exact_p(9, 2) - pv) < 1e-15)
check("mcnemar_zero", eval2.mcnemar_exact_p(0, 0) == 1.0)
# b=0, c=20 -> p = 2/2^20 ≈ 1.907e-6
pv2 = eval2.mcnemar_exact_p(0, 20)
check("mcnemar_extreme", abs(pv2 - 2 / 2 ** 20) < 1e-12, f"p={pv2:.2e}")

# ---- 配对 bootstrap ----
# 常数差：CI 宽度为 0
r = eval2.bootstrap_diff([0.2] * 50, [0.5] * 50, n_boot=2000, seed=1)
check("boot_const", abs(r["diff"] - 0.3) < 1e-12
      and abs(r["ci95"][0] - 0.3) < 1e-12 and abs(r["ci95"][1] - 0.3) < 1e-12,
      f"diff={r['diff']} ci={r['ci95']}")

# 平移数据：diff 恒为 0.5，CI 必含 0.5
rng = np.random.RandomState(7)
x = rng.rand(200)
r = eval2.bootstrap_diff(x.tolist(), (x + 0.5).tolist(), n_boot=5000, seed=42)
check("boot_shift", abs(r["diff"] - 0.5) < 1e-12 and r["ci95"][0] <= 0.5 <= r["ci95"][1],
      f"ci={r['ci95']}")

# 可复现性：同 seed 同结果
a = np.random.RandomState(0).rand(80).tolist()
b = np.random.RandomState(1).rand(80).tolist()
r1 = eval2.bootstrap_diff(a, b, n_boot=2000, seed=42)
r2 = eval2.bootstrap_diff(a, b, n_boot=2000, seed=42)
check("boot_repro", r1 == r2)

# 大样本对照：独立同分布样本均值差的 CI 应接近理论 SE
# diff 的 SD ≈ sd(a-b)/sqrt(n)；用配对样本模拟验证 CI 宽度量级
rng = np.random.RandomState(11)
aa = rng.rand(500)
bb = aa + rng.normal(0, 0.1, 500)  # 正相关对，配对 SE 应较小
r = eval2.bootstrap_diff(aa.tolist(), bb.tolist(), n_boot=5000, seed=42)
se_theory = float(np.std(bb - aa, ddof=1) / np.sqrt(500))
half_width = (r["ci95"][1] - r["ci95"][0]) / 2
check("boot_se_scale", abs(half_width - 1.96 * se_theory) < 0.4 * se_theory,
      f"half={half_width:.5f} vs 1.96*SE={1.96 * se_theory:.5f}")

# 空输入
check("boot_empty", eval2.bootstrap_diff([], []) is None)

# ---- 分层 overall bootstrap ----
# 手工计算: 层1 diff=1-2/3=1/3 (w=0.5), 层2 diff=1-0.5=0.5 (w=0.5) -> 0.4167
r = eval2.bootstrap_overall_diff([[0, 1, 1], [0.5, 0.5]], [[1, 1, 1], [1.0, 1.0]],
                                 [0.5, 0.5], n_boot=2000, seed=42)
check("overall_point", abs(r["diff"] - (0.5 / 3 + 0.25)) < 1e-9, f"diff={r['diff']}")
check("overall_ci_order", r["ci95"][0] <= r["diff"] <= r["ci95"][1])

# ---- 关键词打分逻辑（含映射别名） ----
rec = {"response": "推荐使用分割原理和复合材料原理(#40)，可改善强度与重量矛盾。",
       "expected_keywords": ["分割原理", "强度", "重量"]}
m = eval2.score_item_kw({**rec, "response": rec["response"]}, "contradiction")
check("kw_contra_full", abs(m["contradiction_coverage"] - 1.0) < 1e-12, str(m))
m = eval2.score_item_kw({"response": "用蜂窝结构。", "expected_keywords": ["分割原理", "强度"]},
                        "contradiction")
check("kw_contra_zero", m["contradiction_coverage"] == 0.0, str(m))
m = eval2.score_item_kw({"response": "步骤：问题分析、定义IFR、资源分析、方案评估。",
                         "expected_keywords": []}, "ariz")
check("kw_ariz", 0.4 < m["ariz_step_coverage"] < 0.9, str(m))  # 命中 4/6
m = eval2.score_item_kw({"response": "48", "expected_keywords": ["48"]}, "probe")
check("kw_probe", m["probe_coverage"] == 1.0, str(m))
# principle：全覆盖才记 1
m = eval2.score_item_kw({"response": "推荐分割原理。", "expected_keywords": ["分割原理", "嵌套原理"]},
                        "principle")
check("kw_principle_partial", m["principle_correct"] == 0
      and abs(m["principle_coverage"] - 0.5) < 1e-12, str(m))

# ---- JSON 数组解析 ----
arr = eval2.parse_json_array('```json\n[{"id":"a","steps":{}}]\n```')
check("parse_fence", arr == [{"id": "a", "steps": {}}])
arr = eval2.parse_json_array('以下是结果： [{"id":1}] 完毕')
check("parse_noise", arr == [{"id": 1}])
try:
    eval2.parse_json_array("没有数组")
    check("parse_fail", False)
except ValueError:
    check("parse_fail", True)

# ---- 数据加载 ----
items = eval2.load_items()
n_cat = {}
for it in items:
    n_cat[it["category"]] = n_cat.get(it["category"], 0) + 1
check("load_total", len(items) == 495, f"n={len(items)} (465+30)")
check("load_probe", n_cat.get("general_probe") == 30, str(n_cat.get("general_probe")))
check("load_ids_unique", len({it["id"] for it in items}) == len(items))
pr = [it for it in items if it["category"] == "principle_recommendation"]
check("load_principle_kws", all(it["expected_keywords"] for it in pr),
      f"示例: {pr[0]['expected_keywords']}")
co = [it for it in items if it["category"] == "contradiction_analysis"]
check("load_contra_kws", all(it["expected_keywords"] for it in co),
      f"示例: {co[0]['expected_keywords']}")
check("load_limit", len(eval2.load_items(5)) == 5)

print(f"\nALL {len(passed)} TESTS PASSED")
