#!/usr/bin/env python
"""中文版论文配图: 复用 make_figures.py 的数据, 输出高分辨率 PNG + 新增谱系图。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

setup_plot()

OUT = Path("/Volumes/2nd-HD/claude/Meerkat-AI/paper/arxiv/figures")
CONTEST = Path("/Volumes/2nd-HD/claude/Meerkat-AI/paper/contest")

# ---------- 图1: 谱系图 (替代 tikZ lineage) ----------
fig, ax = plt.subplots(figsize=(9.2, 2.6))
gens = [
    ("v2", "ok", "前代\n(锚点)"),
    ("v3 ×", "rej", "关键词注入\nG7 触发"),
    ("v4 ×", "rej", "数据再平衡\nG3 触发"),
    ("v5", "fix", "harness E0 修复\n+ 语料重建"),
    ("v5a ✓", "ok", "11,096 对\n过门发货 v1"),
    ("v5b ×", "rej", "PR 注入 +408\nG7 触发"),
]
colors = {"ok": "#d5f5e3", "rej": "#fadbd8", "fix": "#d6eaf8"}
edge = {"ok": "#27ae60", "rej": "#c0392b", "fix": "#2e86c1"}
xs = np.arange(len(gens)) * 1.55
for (label, kind, note), x in zip(gens, xs):
    box = mpatches.FancyBboxPatch((x - 0.52, 0.62), 1.04, 0.62,
                                  boxstyle="round,pad=0.06",
                                  fc=colors[kind], ec=edge[kind], lw=1.4)
    ax.add_patch(box)
    ax.text(x, 0.93, label, ha="center", va="center", fontsize=10.5, fontweight="bold")
    ax.text(x, 0.32, note, ha="center", va="center", fontsize=7.8, color="#444")
for x in xs[:-1]:
    ax.annotate("", xy=(x + 1.03, 0.93), xytext=(x + 0.52, 0.93),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#555"))
ax.set_xlim(-0.8, xs[-1] + 0.8)
ax.set_ylim(0, 1.5)
ax.axis("off")
ax.set_title("单变量谱系 v2→v5b（红=决策门否决, 蓝=评测修复, 绿=发货）", fontsize=10)
fig.savefig(OUT / "fig_lineage_zh.png", bbox_inches="tight", dpi=220)
plt.close(fig)

# ---------- 图2: 评委膨胀森林图 ----------
judges = ["Moonshot\n(同族)", "Claude Sonnet 4.6", "GPT-5.4", "Gemini 3.5 Flash"]
diffs = [0.393, 0.094, 0.104, -0.048]
los = [0.297, 0.020, 0.020, -0.144]
his = [0.490, 0.167, 0.184, 0.045]
colors2 = ["#c0392b", "#2e86c1", "#2e86c1", "#2e86c1"]
fig, ax = plt.subplots(figsize=(6.8, 3.0))
y = np.arange(len(judges))[::-1]
for yi, d, lo, hi, c in zip(y, diffs, los, his, colors2):
    ax.plot([lo, hi], [yi, yi], color=c, lw=2.2, zorder=2)
    ax.scatter([d], [yi], color=c, s=46, zorder=3)
    ax.text(hi + 0.012, yi, f"{d:+.3f}", va="center", fontsize=8.5, color=c)
ax.axvline(0, color="gray", lw=0.8, ls="--", zorder=1)
ax.set_yticks(y)
ax.set_yticklabels(judges, fontsize=9)
ax.set_xlabel("judge 轨配对差值 (v5a − base), 95% CI", fontsize=9)
ax.set_xlim(-0.22, 0.62)
ax.tick_params(axis="x", labelsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("同族膨胀：+0.39 在外部评委下缩水为 +0.09/+0.10/−0.05", fontsize=9.5)
fig.savefig(OUT / "fig_judge_inflation_zh.png", bbox_inches="tight", dpi=220)
plt.close(fig)

# ---------- 图3: 评委一致性热图 ----------
def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j < len(v) and v[order[j]] == v[order[i]]:
                j += 1
            for k in range(i, j):
                r[order[k]] = (i + j - 1) / 2 + 1
            i = j
        return r
    n = len(xs)
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    return cov / (vx * vy) ** 0.5

names = ["moonshot", "claude-sonnet-4-6", "gpt-5.4", "gemini-3.5-flash"]
labels = ["Moonshot", "Claude", "GPT", "Gemini"]
caches = {}
for nm in names:
    c = json.load(open(CONTEST / f"contest_cache_{nm}_base.json"))
    caches[nm] = {q: v["overall"] for q, v in c.items()
                  if isinstance(v, dict) and isinstance(v.get("overall"), (int, float))}
M = np.eye(4)
for i, a in enumerate(names):
    for j, b in enumerate(names):
        if i == j:
            continue
        common = sorted(set(caches[a]) & set(caches[b]))
        M[i, j] = spearman([caches[a][q] for q in common],
                           [caches[b][q] for q in common])
fig, ax = plt.subplots(figsize=(4.6, 3.7))
sns.heatmap(M, annot=True, fmt=".2f", cmap="RdYlBu_r", vmin=0, vmax=1,
            xticklabels=labels, yticklabels=labels, ax=ax,
            annot_kws={"size": 9}, cbar_kws={"label": "Spearman ρ"})
ax.tick_params(labelsize=9)
ax.set_title("评委间一致性 (base 臂, n=300)", fontsize=9.5)
fig.savefig(OUT / "fig_judge_agreement_zh.png", bbox_inches="tight", dpi=220)
plt.close(fig)
print("Spearman 矩阵:", [["%.2f" % v for v in row] for row in M])

# ---------- 图4: PR 归因数据构成 ----------
fig, ax = plt.subplots(figsize=(6.4, 3.2))
cats = ["v5a (196 条)", "v5b (521 条)"]
mislabeled = [83, 0]
kept_short = [83, 83]
kept_long = [30, 30]
injected = [0, 408]
x = np.arange(2)
ax.bar(x, mislabeled, 0.5, label="错标（剔除）", color="#c0392b", alpha=0.85)
ax.bar(x, kept_short, 0.5, bottom=mislabeled, label="保留·简洁模式 ≤300 字", color="#95a5a6")
bot = [m + s for m, s in zip(mislabeled, kept_short)]
ax.bar(x, kept_long, 0.5, bottom=bot, label="保留·详细模式", color="#7f8c8d")
bot = [b + l for b, l in zip(bot, kept_long)]
ax.bar(x, injected, 0.5, bottom=bot, label="注入（承诺式, 覆盖全部 40 原理）", color="#2e86c1")
for xi, tot in zip(x, [196, 521]):
    ax.text(xi, tot + 8, f"n={tot}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=9.5)
ax.set_ylabel("训练样本数", fontsize=9)
ax.set_ylim(0, 580)
ax.legend(fontsize=7.5, loc="upper left", frameon=False)
ax.set_title("principle_recommendation 子集：归因 → 注入", fontsize=9.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.savefig(OUT / "fig_pr_attribution_zh.png", bbox_inches="tight", dpi=220)
plt.close(fig)

print("中文版四张图已生成")
