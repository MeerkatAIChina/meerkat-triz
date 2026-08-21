# E0 STATE — ✅ 任务完成 (2026-07-24 06:00 远端时间)

## 最终状态: 全部完成, 无遗留

- 生成 100/100 (全 mode=direct, 无 think 残留, 无过短; 长度 1246-4080 均 3250 字符)
- judge 100/100 无缺失 (moonshot-v1-32k, T=0, RPM=3 退避)
- harness 汇总 + e0_stats 三组配对统计完成 (watcher 自动, finish.log: E0_FINISH_DONE)
- 产物已全部回传本地 `paper/experiments/e0/`, E0_report.md 终稿完成

## 最终数字 (paired bootstrap 10000, seed=42)

| 对比 | judge 差 [95% CI] | kw 差 [95% CI] |
|---|---|---|
| v4 vs base_goldfix | **−0.30 [−0.46, −0.14] 显著为负** | −0.0074 [−0.0497, +0.0345] ns |
| v2 vs base_goldfix | **−0.29 [−0.42, −0.16] 显著为负** | −0.0158 [−0.0550, +0.0236] ns |
| v4 vs v2 | −0.01 [−0.13, +0.11] ns | +0.008 [−0.024, +0.044] ns |

决策门: **保留 v2** (judge overall 非显著为正; principle_recommendation judge 退化 +
concept_explanation kw 退化)。污染前后对比与论文叙事影响见 E0_report.md §4/§6。

## 远端遗留 (无需清理, 供他人复用)
- tmux 会话 e0basefix/e0judge/e0finish 均已自然结束 (judge 驱动 JUDGE_ALL_DONE 后退出)
- 干净缓存: `results/results/v4_gen_base_goldfix.jsonl` (符号链接→`results/v4_gen_base_goldfix.jsonl`)
  + `results/results/v4_judge_base_goldfix.json` — **后续所有 base 臂实验必须用此锚点**
- 脚本与日志: `results/e0_basefix/`
- 勿动: `p0exp` 会话 (非本任务)

## 移交事项 (E1a/E1c/E3 + 写作组)
见 E0_report.md §5 受影响结论清单: API 包 base 臂需用 base_goldfix 重跑;
论文中全部 "vs base" 数字以 §4 为准。
