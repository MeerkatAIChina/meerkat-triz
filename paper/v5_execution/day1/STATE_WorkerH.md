# STATE_WorkerH.md — 种子扩写 (Owner 裁决方案③)

更新时间: 2026-07-24 17:25 (远端 spark-855a 时间) · 状态: **等待 Worker E 释放 API 额度**

## 任务
把清洗中被长度规则丢弃的 306 条短种子用 Moonshot API 扩写 output 至 150–600 字符
(保留真实 instruction, 半真实锚定), 与 59 条存活合并为 cleaned_seeds_final.jsonl。

## 已完成
1. **候选还原** (无 API): `pipeline_v5/src/seed_expand_reconstruct.py`
   - 385 种子 → R1 冲突组弃 20 → 365; 重放 R2 截句, 还原被弃 306 条
     (truncated_short 196, output_base=截尾后文本 / raw_short 110, output_base=原文)
   - 计数与 Worker A `stage1_report.json` 完全对齐 (196+110, assert 通过)
   - 产物: `data/processed/v5_data/seed_expand_candidates.jsonl` (306 条)
           `data/processed/v5_data/seed_expand_blacklist.json` (R2 黑名单 b1/b2)
2. **扩写脚本**: `pipeline_v5/src/seed_expand_v5.py`
   - kimi-k2.5, batch=5, RPM=3 全局限速, 指数退避, jsonl 追加断点续跑
   - 纪律: 保持原结论与术语, 250–550 字目标 (硬门 150–600), 禁模板收尾
   - 质量门: 长度越界/黑名单命中 → 单条重生成 1 次 → 仍败丢弃计数;
     tail80 频次≥3 拒绝 (Worker A R2 同口径)
3. **终检脚本**: `pipeline_v5/src/seed_expand_finalize.py` (合并+质量门+忠实度抽检 10% md+报告)
4. **监控启动**: tmux `v5gen_H` 运行 `run_workerH_v5.sh` —— 轮询 (≥5 分钟间隔)
   等 E (styleC_long_answers.jsonl ≥3445 行 / 日志"完成:" / 进程退出) 后自动
   扩写 → finalize。

## 待办 / resume 指引
- E 进度查询: `wc -l data/processed/v5_data/styleC_long_answers.jsonl` (目标 3445)
- H 进度查询: `tail data/processed/v5_data/workerH_run.log`;
  `wc -l data/processed/v5_data/seed_expanded.jsonl`
- 若 v5gen_H 意外中断: `tmux new -s v5gen_H` 后重跑
  `bash pipeline_v5/src/run_workerH_v5.sh` (扩写脚本按 group_id 断点续跑)
- 完成后产物 (远端 `data/processed/v5_data/`):
  cleaned_seeds_final.jsonl / seed_expansion_report.json /
  seed_expand_fidelity_review.md / seed_expanded.jsonl / seed_expand_dropped.jsonl
- 回传本地: `scp` 上述产物到 `paper/v5_execution/day1/`

## 关键决策记录
- 扩写目标区间定为 250–550 字 (提示词), 硬校验 150–600, 留安全边距
- 整批 API 失败时逐条单发兜底, 不整批丢弃
- 忠实度抽检: seed=42 随机 10% 队列落 md 人工核对 (不做 API 二次判别, 省额度)
