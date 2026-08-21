#!/usr/bin/env python3
"""Meerkat-TRIZ-v1 发布推文配图（全部数字来自已验证评测缓存）。

输出: meerkat-triz/docs/assets/launch_{eqlen,kw,eff}[_en].png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

setup_plot()

OUT = Path("/Volumes/2nd-HD/claude/Meerkat-AI/meerkat-triz/docs/assets")
OUT.mkdir(parents=True, exist_ok=True)

ACCENT = "#b23c17"
INK = "#1c1a17"
SOFT = "#8a857a"
PAPER = "#faf9f6"
GOLD = "#c98a2d"

plt.rcParams.update({
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "axes.edgecolor": "#e4e0d8",
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": SOFT,
    "ytick.color": SOFT,
    "font.size": 12,
})

TEXT = {
    "zh": {
        "suffix": "",
        "eqlen_title": "等长预算下，外部评委判 v1 显著更好（2/3 显著）",
        "eqlen_x": "v1 − base 配对差值（judge 0–4 分，95% CI）",
        "unconstrained": "无约束（base 以 2.4× 篇幅 sprawling）",
        "matched": "等长（同一篇幅预算）",
        "kw_title": "关键词轨：35B 微调与基座击败全部前沿模型",
        "kw_y": "TRIZ 关键词命中率（300 题金标）",
        "eff_left_title": "篇幅：0.4×",
        "eff_left_y": "回答长度中位数（字符）",
        "eff_right_title": "关键词韧性：base 丢的，v1 保住",
        "eff_right_y": "关键词覆盖差值（等长对照）",
        "chars": "字",
        "base_short": "base 被迫写短",
        "v1_short": "v1（天生简洁）",
    },
    "en": {
        "suffix": "_en",
        "eqlen_title": "At matched length, external judges prefer v1 (2 of 3 significant)",
        "eqlen_x": "Paired difference v1 − base (judge score 0–4, 95% CI)",
        "unconstrained": "Unconstrained (base uses 2.4× length)",
        "matched": "Length-matched (same budget)",
        "kw_title": "Keyword track: the 35B pair beats every frontier model",
        "kw_y": "TRIZ keyword hit rate (300-item gold set)",
        "eff_left_title": "Length: 0.4×",
        "eff_left_y": "Median answer length (characters)",
        "eff_right_title": "Keyword resilience: v1 keeps what base loses",
        "eff_right_y": "Keyword coverage delta (matched length)",
        "chars": "chars",
        "base_short": "base forced short",
        "v1_short": "v1 (naturally concise)",
    },
}


def clean_axes(*axes):
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)


def plot_eqlen(lang):
    t = TEXT[lang]
    judges = ["Claude\nSonnet 4.6", "GPT-5.4", "Gemini\n3.5 Flash"]
    unc = [(0.094, 0.020, 0.167), (0.104, 0.020, 0.184), (-0.048, -0.144, 0.045)]
    mat = [(0.187, 0.110, 0.264), (0.271, 0.177, 0.368), (0.025, -0.074, 0.120)]
    df = pd.DataFrame(
        [(j, "unconstrained", *row) for j, row in zip(judges, unc)]
        + [(j, "matched", *row) for j, row in zip(judges, mat)],
        columns=["judge", "arm", "diff", "lo", "hi"],
    )

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    y = np.arange(len(judges))
    h = 0.32
    for i, judge in enumerate(judges):
        for arm, offset, color, alpha in (
            ("unconstrained", h / 2 + 0.02, SOFT, 0.45),
            ("matched", -h / 2 - 0.02, ACCENT, 0.92),
        ):
            row = df[(df.judge == judge) & (df.arm == arm)].iloc[0]
            diff = row["diff"]
            significant = row["lo"] > 0
            bar_color = color if arm == "unconstrained" or significant else SOFT
            bar_alpha = alpha if arm == "unconstrained" or significant else 0.45
            ax.barh(y[i] + offset, diff, height=h, color=bar_color,
                    alpha=bar_alpha, zorder=2)
            ax.errorbar(diff, y[i] + offset,
                        xerr=[[diff - row["lo"]], [row["hi"] - diff]],
                        color=INK, capsize=3, lw=1.2, zorder=3)
            ax.text(row["hi"] + 0.012, y[i] + offset,
                    f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}",
                    va="center", fontsize=11 if arm == "matched" else 10,
                    color=ACCENT if significant and arm == "matched" else SOFT,
                    fontweight="bold" if arm == "matched" else "normal")
    ax.axvline(0, color=INK, lw=0.8, alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(judges, fontsize=12)
    ax.set_xlim(-0.2, 0.45)
    ax.set_xlabel(t["eqlen_x"])
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, fc=SOFT, alpha=0.45),
            plt.Rectangle((0, 0), 1, 1, fc=ACCENT),
        ],
        labels=[t["unconstrained"], t["matched"]],
        loc="upper right", bbox_to_anchor=(0.98, 0.98),
        frameon=False, fontsize=9.5,
    )
    ax.set_title(t["eqlen_title"], fontsize=14, fontweight="bold", color=INK, pad=12)
    clean_axes(ax)
    fig.savefig(OUT / f"launch_eqlen{t['suffix']}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_kw(lang):
    t = TEXT[lang]
    df = pd.DataFrame({
        "model": ["Meerkat-TRIZ-v1", "base (Qwen3.6-35B)", "GPT-5.4",
                  "Gemini 3.5 Flash", "Claude Sonnet 4.6", "Claude Opus 4.8"],
        "kw": [0.638, 0.638, 0.628, 0.599, 0.576, 0.576],
    })
    colors = [ACCENT, GOLD] + [SOFT] * 4
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    sns.barplot(data=df, x="model", y="kw", ax=ax, palette=colors, width=0.62, zorder=2)
    for i, v in enumerate(df.kw):
        ax.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=11,
                fontweight="bold" if i == 0 else "normal",
                color=ACCENT if i == 0 else INK)
    ax.set_xticklabels(df.model, fontsize=10.5, rotation=12)
    ax.set_ylim(0.5, 0.68)
    ax.set_xlabel("")
    ax.set_ylabel(t["kw_y"])
    ax.set_title(t["kw_title"], fontsize=14, fontweight="bold", pad=12)
    ax.axhline(0.638, color=ACCENT, lw=0.8, ls="--", alpha=0.4)
    clean_axes(ax)
    fig.savefig(OUT / f"launch_kw{t['suffix']}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_eff(lang):
    t = TEXT[lang]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.9),
                                   gridspec_kw={"width_ratios": [1, 1]})
    length_df = pd.DataFrame({"model": ["base", "Meerkat-TRIZ-v1"], "chars": [3242, 1344]})
    sns.barplot(data=length_df, x="model", y="chars", ax=ax1,
                palette=[SOFT, ACCENT], width=0.52, zorder=2)
    for i, v in enumerate(length_df.chars):
        ax1.text(i, v + 60, f"{v:,} {t['chars']}", ha="center", fontsize=11,
                 fontweight="bold", color=INK)
    ax1.text(0.5, 1700, "0.4×", ha="center", fontsize=22, fontweight="bold",
             color=ACCENT)
    ax1.set_xlabel("")
    ax1.set_ylabel(t["eff_left_y"])
    ax1.set_title(t["eff_left_title"], fontsize=13, fontweight="bold")

    resilience_df = pd.DataFrame({
        "model": [t["base_short"], t["v1_short"]],
        "delta": [-0.023, 0.023],
        "lo": [-0.040, 0.005],
        "hi": [-0.005, 0.041],
    })
    sns.barplot(data=resilience_df, x="model", y="delta", ax=ax2,
                palette=[SOFT, ACCENT], width=0.52, zorder=2)
    ax2.errorbar([0, 1], resilience_df.delta,
                 yerr=[resilience_df.delta - resilience_df.lo,
                       resilience_df.hi - resilience_df.delta],
                 fmt="none", ecolor=INK, capsize=4, lw=1.2, zorder=3)
    ax2.axhline(0, color=INK, lw=0.8, alpha=0.5)
    ax2.set_xlabel("")
    ax2.set_ylabel(t["eff_right_y"])
    ax2.set_title(t["eff_right_title"], fontsize=13, fontweight="bold")
    clean_axes(ax1, ax2)
    fig.savefig(OUT / f"launch_eff{t['suffix']}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


for lang in ("zh", "en"):
    plot_eqlen(lang)
    plot_kw(lang)
    plot_eff(lang)

print("saved:", *[str(p) for p in sorted(OUT.glob("launch_*.png"))], sep="\n  ")
