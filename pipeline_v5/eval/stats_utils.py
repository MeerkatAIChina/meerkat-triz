#!/usr/bin/env python
"""
pipeline_v5 统计工具(§6.8 统计纪律八条的机检实现)。

纪律:
  ① 配对 bootstrap 10,000 次, seed=42, 钉死 stdlib random.Random(42)(禁 numpy RNG);
  ② McNemar 精确双侧(指纹校验至 p=1.6979681549678105e-12 逐位一致);
  ③ Wilson 95% CI 用于一切比例;
  ⑧ 实现指纹校验: 本模块 import 时自检, 不一致立即抛错(防环境漂移)。

指纹值在本机 Python 3.12 stdlib 下计算并复核:
  mcnemar_exact_p(55, 4) == 1.6979681549678105e-12  (§6.8 给定值, 逐位一致)
  bootstrap_diff([1..8], [2,1,3,5,4,7,6,9], 10000, 42)
      == {"diff": 0.125, "ci95": [-0.5, 0.75], "n": 8}
"""

import math
import random

BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 42


def wilson_ci(k, n, z=1.959963984540054):
    """Wilson score 区间, 返回 (p_hat, lo, hi)。"""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact_p(b, c):
    """McNemar 精确检验 (双侧二项), b/c 为不一致对数。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def bootstrap_diff(arr_a, arr_b, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """配对 bootstrap: B - A 差值均值与 95% 百分位 CI (arr 按同一题序对齐)。

    §6.8-①: 钉死 stdlib random.Random(seed), 禁止 numpy RNG
    (两者千分位 CI 差异已实测, stats_review.md §1)。
    """
    n = len(arr_a)
    if n == 0:
        return None
    if len(arr_b) != n:
        raise ValueError("配对 bootstrap 要求两数组等长")
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            s += arr_b[i] - arr_a[i]
        diffs.append(s / n)
    diffs.sort()
    return {"diff": sum(arr_b[i] - arr_a[i] for i in range(n)) / n,
            "ci95": [diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]],
            "n": n, "n_boot": n_boot, "seed": seed, "rng": "stdlib"}


def is_significant(ci95):
    """95% CI 不跨 0 即显著。"""
    return ci95[0] > 0 or ci95[1] < 0


def mde_note(n, track):
    """§6.8-⑤/⑥: 描述性标注与 MDE 提示。
    n<30 → '描述性'; 返回 (label, note)。"""
    if n < 30:
        return ("描述性", f"描述性(n={n}, 子集结论不作独立否决依据)")
    return ("可推断", f"n={n}")


def self_check():
    """实现指纹校验 (§6.8 执行保障)。任一不符即抛 AssertionError。"""
    p = mcnemar_exact_p(55, 4)
    assert p == 1.6979681549678105e-12, \
        f"McNemar 指纹不符: {p!r} != 1.6979681549678105e-12"
    fp = bootstrap_diff([1, 2, 3, 4, 5, 6, 7, 8],
                        [2, 1, 3, 5, 4, 7, 6, 9], 10000, 42)
    assert fp["diff"] == 0.125 and fp["ci95"] == [-0.5, 0.75] and fp["n"] == 8, \
        f"bootstrap 指纹不符: {fp!r}"
    return True


self_check()  # import 即自检
