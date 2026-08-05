#!/usr/bin/env python
"""arXiv 论文三张插图 (出版级 PDF, 300dpi 矢量)。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

setup_plot()

OUT = Path("/Volumes/2nd-HD/claude/Meerkat-AI/paper/arxiv/figures")
OUT.mkdir(exist_ok=True)
CONTEST = Path("/Volumes/2nd-HD/claude/Meerkat-AI/paper/contest")

# ---------- Fig 2: 评委膨胀森林图 ----------
judges = ["Moonshot\n(in-family)", "Claude Sonnet 4.6", "GPT-5.4", "Gemini 3.5 Flash"]
diffs = [0.393, 0.094, 0.104, -0.048]
los = [0.297, 0.020, 0.020, -0.144]
his = [0.490, 0.167, 0.184, 0.045]
colors = ["#c0392b", "#2e86c1", "#2e86c1", "#2e86c1"]

fig, ax = plt.subplots(figsize=(6.4, 2.9))
y = np.arange(len(judges))[::-1]
for yi, d, lo, hi, c in zip(y, diffs, los, his, colors):
    ax.plot([lo, hi], [yi, yi], color=c, lw=2.2, zorder=2)
    ax.scatter([d], [yi], color=c, s=46, zorder=3)
    ax.text(hi + 0.012, yi, f"{d:+.3f}", va="center", fontsize=8.5, color=c)
ax.axvline(0, color="gray", lw=0.8, ls="--", zorder=1)
ax.set_yticks(y)
ax.set_yticklabels(judges, fontsize=9)
ax.set_xlabel("Paired judge-track diff (v5a − base), 95% CI", fontsize=9)
ax.set_xlim(-0.22, 0.62)
ax.tick_params(axis="x", labelsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("In-family inflation: +0.39 deflates to +0.09/+0.10/−0.05",
             fontsize=9.5)
fig.savefig(OUT / "fig_judge_inflation.pdf", bbox_inches="tight")
plt.close(fig)

# ---------- Fig 3: 评委一致性热图 (base 臂逐题 Spearman) ----------
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

fig, ax = plt.subplots(figsize=(4.4, 3.6))
sns.heatmap(M, annot=True, fmt=".2f", cmap="RdYlBu_r", vmin=0, vmax=1,
            xticklabels=labels, yticklabels=labels, ax=ax,
            annot_kws={"size": 9}, cbar_kws={"label": "Spearman ρ"})
ax.tick_params(labelsize=9)
ax.set_title("Judge–judge agreement (base arm, n=300)", fontsize=9.5)
fig.savefig(OUT / "fig_judge_agreement.pdf", bbox_inches="tight")
plt.close(fig)
print("Spearman 矩阵:")
for i, a in enumerate(labels):
    print(" ", a, ["%.2f" % M[i, j] for j in range(4)])

# ---------- Fig 4: PR 归因数据构成 ----------
fig, ax = plt.subplots(figsize=(6.2, 3.1))
cats = ["v5a (196)", "v5b (521)"]
mislabeled = [83, 0]
kept_short = [83, 83]
kept_long = [30, 30]
injected = [0, 408]
x = np.arange(2)
b1 = ax.bar(x, mislabeled, 0.5, label="mislabeled (removed)",
            color="#c0392b", alpha=0.85)
b2 = ax.bar(x, kept_short, 0.5, bottom=mislabeled,
            label="legacy kept, concise ≤300ch", color="#95a5a6")
bot = [m + s for m, s in zip(mislabeled, kept_short)]
b3 = ax.bar(x, kept_long, 0.5, bottom=bot,
            label="legacy kept, detailed", color="#7f8c8d")
bot = [b + l for b, l in zip(bot, kept_long)]
b4 = ax.bar(x, injected, 0.5, bottom=bot,
            label="injected (commit-style, 40-principle coverage)",
            color="#2e86c1")
for xi, tot in zip(x, [196, 521]):
    ax.text(xi, tot + 8, f"n={tot}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=9.5)
ax.set_ylabel("training items", fontsize=9)
ax.set_ylim(0, 580)
ax.legend(fontsize=7.5, loc="upper left", frameon=False)
ax.set_title("Principle-recommendation subset: attribution → injection",
             fontsize=9.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.savefig(OUT / "fig_pr_attribution.pdf", bbox_inches="tight")
plt.close(fig)

print("三张图已生成:", [p.name for p in OUT.glob("*.pdf")])
