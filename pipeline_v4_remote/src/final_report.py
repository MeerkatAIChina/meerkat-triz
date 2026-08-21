#!/usr/bin/env python
"""
pipeline_v4 最终汇总报告: 多方对比 + 决策门判定。

读取 results/ 中各 tag 最新的 eval_v4_<tag>_*.json, 生成汇总报告:
  - 多方对比表 (关键词轨/judge 轨 overall 均值、pass 率、各子集均值);
  - candidate vs baseline 配对统计摘录 (来自 candidate 结果内嵌的 paired_vs_baseline,
    baseline 锚点由评测时 --baseline-results 决定, 从 meta.baseline_results 反解);
  - 决策门: candidate judge 轨 overall 差值 95% CI 下界 > 0 (显著优于 baseline)
            且所有子指标 (judge 各子集 + 关键词轨 overall 及各子集) 无显著退化
            (差值 CI 上界 >= 0) → "建议替代 v2", 否则 "保留 v2"。

默认行为与 v4 初版一致 (四方对比, candidate=v4_gold, 输出 results/v4_final_report.md)。
v4.1 用法:
  final_report.py --tags base_gold v2_gold v3_gold v4_gold v4_1_gold \
      --candidate v4_1_gold --out results/v4_1_final_report.md
"""

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TAGS = ["base_gold", "v2_gold", "v3_gold", "v4_gold"]


def latest_result(results_dir: Path, tag: str):
    files = sorted(glob.glob(str(results_dir / f"eval_v4_{tag}_*.json")))
    return Path(files[-1]) if files else None


def fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "N/A"


def baseline_tag_of(result: dict) -> str:
    """从 candidate 结果的 meta.baseline_results 文件名反解 baseline tag
    (eval_v4_<tag>_<YYYYMMDD>_<HHMMSS>.json); 反解失败回退 'baseline'。"""
    bl = (result.get("meta") or {}).get("baseline_results")
    if not bl:
        return "baseline"
    stem = Path(bl).name
    if stem.startswith("eval_v4_"):
        stem = stem[len("eval_v4_"):]
    parts = stem.rsplit("_", 2)
    if len(parts) == 3 and parts[1].isdigit():
        return parts[0]
    return stem.replace(".json", "")


def main():
    ap = argparse.ArgumentParser(description="pipeline_v4 最终汇总报告")
    ap.add_argument("--tags", nargs="+", default=DEFAULT_TAGS,
                    help="参与对比的结果标签 (默认 base/v2/v3/v4 四方)")
    ap.add_argument("--candidate", default="v4_gold",
                    help="决策门候选模型标签 (默认 v4_gold; v4.1 用 v4_1_gold)")
    ap.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    ap.add_argument("--out", default=None,
                    help="输出 md 路径 (默认 <results_dir>/v4_final_report.md)")
    args = ap.parse_args()

    tags = args.tags
    results_dir = Path(args.results_dir)
    results = {}
    for tag in tags:
        p = latest_result(results_dir, tag)
        if p:
            with open(p, encoding="utf-8") as f:
                results[tag] = (p.name, json.load(f))

    L = ["# pipeline_v4 最终评测汇总", "",
         f"生成时间: {datetime.now().isoformat(timespec='seconds')}", ""]
    missing = [t for t in tags if t not in results]
    if missing:
        L.append(f"> ⚠️ 缺少结果: {', '.join(missing)}")
        L.append("")

    # ---- 多方对比总表 ----
    L += ["## 多方对比 (overall)", "",
          "| 模型 | judge 模型 | 关键词轨均值 | judge 轨均值 | 关键词 pass 率 | judge pass 率 |",
          "|---|---|---|---|---|---|"]
    for tag in tags:
        if tag not in results:
            L.append(f"| {tag} | — | 缺结果 | — | — | — |")
            continue
        _, r = results[tag]
        jm = r["meta"].get("judge_model", "?")
        warn = " ⚠️同源" if r["meta"].get("judge_same_origin_fallback") else ""
        kw, jd = r["keyword_track"], r["judge_track"]
        L.append(f"| {tag} | {jm}{warn} | {fmt(kw['mean'])} | {fmt(jd['mean'])} | "
                 f"{fmt(kw['pass_rate']['p'], 3)} | {fmt(jd['pass_rate']['p'], 3)} |")
    L.append("")

    # ---- 各子集均值表 ----
    for track_key, track_name in (("keyword_track", "关键词轨"), ("judge_track", "judge 轨")):
        subsets = sorted({s for tag in tags if tag in results
                          for s in results[tag][1][track_key]["per_subset"]})
        L += [f"## {track_name} 各子集均值", "",
              "| 子集 | " + " | ".join(tags) + " |",
              "|---|" + "---|" * len(tags)]
        for s in subsets:
            row = [s]
            for tag in tags:
                if tag in results:
                    d = results[tag][1][track_key]["per_subset"].get(s)
                    row.append(fmt(d["mean"]) if d else "—")
                else:
                    row.append("—")
            L.append("| " + " | ".join(row) + " |")
        L.append("")

    # ---- 决策门 ----
    L += ["## 决策门", ""]
    decision, reasons = "保留 v2", []
    cand = args.candidate
    if cand not in results:
        reasons.append(f"缺 {cand} 结果, 无法判定")
    else:
        _, cr = results[cand]
        anchor = baseline_tag_of(cr)
        comp = cr.get("paired_vs_baseline")
        if not comp or not comp.get("tracks", {}).get("judge", {}).get("overall"):
            reasons.append(f"{cand} 结果中缺与基线的配对统计, 无法判定")
        else:
            jo = comp["tracks"]["judge"]["overall"]
            ko = comp["tracks"]["keyword"]["overall"]
            L.append(f"- 锚点: **{anchor}** (评测时 --baseline-results 指定)")
            L.append(f"- {cand}-{anchor} judge overall 差值: {jo['diff']:+.4f} "
                     f"[{jo['ci95'][0]:+.4f}, {jo['ci95'][1]:+.4f}]")
            L.append(f"- {cand}-{anchor} 关键词 overall 差值: {ko['diff']:+.4f} "
                     f"[{ko['ci95'][0]:+.4f}, {ko['ci95'][1]:+.4f}]")
            significant_win = jo["ci95"][0] > 0
            regressions = []
            if ko["ci95"][1] < 0:
                regressions.append("keyword/overall")
            for s, d in comp["tracks"]["judge"]["per_subset"].items():
                if d and d["ci95"][1] < 0:
                    regressions.append(f"judge/{s}")
            for s, d in comp["tracks"]["keyword"]["per_subset"].items():
                if d and d["ci95"][1] < 0:
                    regressions.append(f"keyword/{s}")
            L.append(f"- judge overall 显著优于 {anchor}: {'是' if significant_win else '否'}")
            L.append(f"- 显著退化子指标: {', '.join(regressions) if regressions else '无'}")
            if significant_win and not regressions:
                decision = "建议替代 v2"
            else:
                if not significant_win:
                    reasons.append(f"judge 轨 overall 未显著优于 {anchor}")
                if regressions:
                    reasons.append(f"存在显著退化: {', '.join(regressions)}")
            mc = comp["tracks"]["judge"].get("mcnemar", {})
            L.append(f"- McNemar (judge pass): p={mc.get('p', 'N/A')}")
    L += ["", f"### 判定: **{decision}**", ""]
    if reasons:
        L.append("原因: " + "; ".join(reasons))
        L.append("")

    out = Path(args.out) if args.out else results_dir / "v4_final_report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"最终报告: {out}\n判定: {decision}")


if __name__ == "__main__":
    main()
