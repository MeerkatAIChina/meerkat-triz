#!/usr/bin/env python
"""汇总 v5 Day2 P0 六组小扫 → sweep_report.json + sweep_report.md。

判据 (sec2_training.md §8.1 双阶段之阶段 1 初筛):
  - 初筛: 终点 eval_loss 最低者; 与 2e-4/False 基线差 ≤ 0.01 视为平局;
  - 推荐 top-2 臂进入阶段 2 (40 题金标双轨冒烟, 由 Orchestrator 另行安排);
  - 若 6 组终点 eval_loss 极差 < 0.01 (噪声阈值) → 明确声明"建议取默认 2e-4/False"。
"""
import glob
import json
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(PROJECT_ROOT, "results")
STATUS_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "sweep_status")
TIE = 0.01
BASELINE = "sweep_lr2e-4_rsFalse"


def fmt(x, nd=6):
    return round(x, nd) if isinstance(x, float) else x


def main() -> int:
    arms = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "run_summary_sweep_*.json"))):
        with open(path) as f:
            s = json.load(f)
        name = s["run_name"]
        failed = os.path.join(STATUS_DIR, f"{name}.failed")
        s["chain_failed"] = os.path.isfile(failed)
        if s["chain_failed"]:
            with open(failed) as f:
                s["failed_info"] = f.read().strip()
        arms.append(s)

    failed_only = []
    for path in sorted(glob.glob(os.path.join(STATUS_DIR, "sweep_*.failed"))):
        name = os.path.basename(path)[:-len(".failed")]
        if not any(a["run_name"] == name for a in arms):
            with open(path) as f:
                failed_only.append({"run_name": name, "failed_info": f.read().strip()})

    ok = [a for a in arms if not a["chain_failed"] and a.get("status") == "PASSED"]
    # 终点 eval_loss = 末步 (0.5 epoch) 评估值; 缺失回退轨迹末值
    for a in ok:
        if a.get("final_eval_loss") is None and a.get("eval_trajectory"):
            a["final_eval_loss"] = a["eval_trajectory"][-1]["eval_loss"]
    ranked = sorted(ok, key=lambda a: a["final_eval_loss"])

    baseline = next((a for a in ok if a["run_name"] == BASELINE), None)
    conclusion = {}
    if len(ranked) >= 2 and baseline is not None:
        spread = max(a["final_eval_loss"] for a in ranked) - min(a["final_eval_loss"] for a in ranked)
        top2 = [a["run_name"] for a in ranked[:2]]
        deltas = {a["run_name"]: round(a["final_eval_loss"] - baseline["final_eval_loss"], 6)
                  for a in ranked}
        if spread < TIE:
            verdict = (f"6 组终点 eval_loss 极差 {spread:.6f} < 噪声阈值 {TIE}, "
                       f"组间无实质差异: 建议取默认 2e-4/False")
            recommend = [BASELINE]
        else:
            tied = [n for n, d in deltas.items() if d <= TIE and d >= -TIE]
            verdict = (f"初筛 top-2 (终点 eval_loss 最低): {top2[0]}, {top2[1]}; "
                       f"与基线 {BASELINE} 差 ≤{TIE} 的平局臂: {tied or '无'}。"
                       f"按判据 top-2 臂进入阶段 2 金标双轨冒烟终判; 冒烟仍平局取 2e-4/False")
            recommend = top2
        conclusion = {
            "spread_final_eval_loss": round(spread, 6),
            "tie_threshold": TIE,
            "baseline": BASELINE,
            "baseline_final_eval_loss": baseline["final_eval_loss"],
            "delta_vs_baseline": deltas,
            "stage1_top2": top2,
            "stage1_recommend": recommend,
            "verdict": verdict,
        }
    else:
        conclusion = {"verdict": f"有效完成组数不足 ({len(ok)}/6), 无法给出初筛结论",
                      "stage1_recommend": []}

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sweep": "P0 lr×rsLoRA 6 组 × 0.5 epoch (669 步), eval/save=100",
        "completed_ok": len(ok),
        "failed": failed_only + [{"run_name": a["run_name"],
                                  "failed_info": a.get("failed_info", "")}
                                 for a in arms if a["chain_failed"]],
        "ranked_by_final_eval_loss": [{
            "rank": i + 1,
            "run_name": a["run_name"],
            "learning_rate": a["learning_rate"],
            "use_rslora": a["use_rslora"],
            "final_eval_loss": fmt(a["final_eval_loss"]),
            "best_eval_loss": fmt(a.get("best_eval_loss")),
            "best_eval_step": a.get("best_eval_step"),
            "actual_steps": a.get("actual_steps"),
            "early_stopped": a.get("early_stopped"),
            "elapsed_hours": a.get("elapsed_hours"),
            "peak_mem_gb": a.get("peak_mem_gb"),
            "eval_trajectory": a.get("eval_trajectory"),
        } for i, a in enumerate(ranked)],
        "stage1_conclusion": conclusion,
    }

    os.makedirs(RESULTS, exist_ok=True)
    jpath = os.path.join(RESULTS, "sweep_report.json")
    with open(jpath, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    interim = len(ok) < 6
    lines = [
        f"# v5 Day2 P0 超参小扫{'中期' if interim else ''}报告 (lr × rsLoRA, 6 组 × 0.5 epoch)",
        "",
        f"> 生成: {report['generated_at']} | 有效完成 {len(ok)}/6 组"
        f"{' (中期, 链仍在运行)' if interim else ''} | "
        f"判据: sec2_training.md §8.1 阶段 1 初筛 (平局阈值 {TIE})",
        "",
        "| 排名 | 臂 | lr | rsLoRA | 终点 eval_loss | best eval_loss | best@step | 步数 | 早停 | 时长 h | 显存峰值 GB |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in report["ranked_by_final_eval_loss"]:
        lines.append(
            f"| {r['rank']} | {r['run_name']} | {r['learning_rate']:.0e} | "
            f"{r['use_rslora']} | {r['final_eval_loss']:.6f} | "
            f"{r['best_eval_loss']:.6f} | {r['best_eval_step']} | "
            f"{r['actual_steps']} | {r['early_stopped']} | "
            f"{r['elapsed_hours']} | {r['peak_mem_gb']} |")
    lines += ["", "## eval_loss 轨迹 (每 100 步 + 终点)", ""]
    for r in report["ranked_by_final_eval_loss"]:
        seen = {}
        for t in r["eval_trajectory"]:  # 末步终点 eval 在 log_history 中出现两次, 去重
            seen[t["step"]] = t["eval_loss"]
        traj = ", ".join(f"{s}:{seen[s]:.4f}" for s in sorted(seen))
        lines.append(f"- **{r['run_name']}**: {traj}")

    # 配对分析: rsLoRA True vs False (同 lr); lr 间对比 (同 rsLoRA)
    by_name = {a["run_name"]: a for a in ok}
    def lr_tag(lr):
        return f"{lr:.0e}".replace("-0", "-")
    lines += ["", "## 配对分析", ""]
    lines.append("**rsLoRA True − False (同 lr, 终点 eval_loss 差; 正 = rsLoRA 更差):**")
    lines.append("")
    for lr in (2e-4, 1e-4, 5e-4):
        f_ = by_name.get(f"sweep_lr{lr_tag(lr)}_rsFalse")
        t_ = by_name.get(f"sweep_lr{lr_tag(lr)}_rsTrue")
        if f_ and t_:
            d = t_["final_eval_loss"] - f_["final_eval_loss"]
            lines.append(f"- lr={lr:.0e}: rsT {t_['final_eval_loss']:.6f} − rsF {f_['final_eval_loss']:.6f} = **{d:+.6f}**")
    lines += ["", "**lr 间对比 (同 rsLoRA, 终点 eval_loss):**", ""]
    for rs in ("False", "True"):
        row = []
        for lr in (1e-4, 2e-4, 5e-4):
            a = by_name.get(f"sweep_lr{lr_tag(lr)}_rs{rs}")
            if a:
                row.append(f"{lr:.0e}:{a['final_eval_loss']:.6f}")
        if len(row) >= 2:
            lines.append(f"- rsLoRA={rs}: " + " | ".join(row))
    lines += ["", "> 注意: 0.5 epoch 短跑系统性偏好高 lr / 高有效更新 (各组末段仍在下降, 均未早停); "
              "初筛结论须经阶段 2 金标双轨冒烟终判 (v3 loss≈v2 但 judge 显著更差的前车之鉴)。",
              "", "## 阶段 1 初筛结论", "", conclusion.get("verdict", "N/A"), ""]
    if report["failed"]:
        lines += ["## 失败组", ""]
        for fr in report["failed"]:
            lines.append(f"- {fr['run_name']}: {fr.get('failed_info', '')}")
        lines.append("")
    lines.append("> 阶段 2 (top-2 臂 40 题金标双轨冒烟) 由 Orchestrator 审查后另行安排。")
    mpath = os.path.join(RESULTS, "sweep_report.md")
    with open(mpath, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"written: {jpath}\nwritten: {mpath}")
    print(conclusion.get("verdict", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
