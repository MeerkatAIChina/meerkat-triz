# STATE_WorkerF — v5 新金标 base/v2/v4 对照生成(Day 1)

- 更新: 2026-07-24 11:12(远端时间)· 状态: **进行中(base 6/100,健康)**
- tmux 会话: `v5gen_F`(注意:机器上另有他人会话 `v5gen`,本会话被挤掉过一次,已改独立名)

## 已完成
- 读 E0_report.md §3 + v5 方案 §6.1,复用 E0 协议写 `results/v5/gen/v5_gen.py`:
  保留空 think 块(`enable_thinking=False` 不剥离)、BF16、贪心、max_new_tokens=2048、
  bad_words_ids 兜底一次、断点续跑;四道质量门(think 残留/非空+中文≥0.3/长度下限
  base≥100·适配器≥50 且≥同题 base 3%/英文草稿检测)。
- 链脚本 `v5_gen_chain.sh`:base→v2→v4 GPU 串行,每段后 5% invalid 门(超门写 chain.abort 停止);
  全完自动生成 `gen_report.json`。
- base 冒烟(前 5 题):5/5 mode=direct,正常中文结构化 TRIZ 作答,无英文 think 草稿 → 全量放行。

## 进行中
- base_v5gold: 6/100,~100s/题,预计 ~13:50 完成;之后 v2、v4(短答,各 ~15-30 min)。
- 产物(远端 `results/v5/gen/`):`responses_{base,v2,v4}_v5gold.jsonl`、`raw_*_v5gold.jsonl`、
  `gen_{base,v2,v4}.log`、`chain.log`、完成后有 `gen_report.json` + `chain.done`。

## 故障史
- 10:19/10:37 两次进程死亡:第一次 NVRM Out of memory + X 会话崩溃;第二次系他人
  tmux 会话同名 `v5gen` 冲突(10:39:46 被创建,疑 kill-session)。改用 `v5gen_F` 后稳定。

## 下一步(续跑者)
1. 等 `results/v5/gen/chain.done` 出现(或 chain.abort → 按质量门失败上报,不硬推)。
2. 核对 `gen_report.json` 三 tag invalid_rate ≤0.05。
3. 回传本地 `paper/v5_execution/day1/`:responses_*、gen_report.json、chain.log。
4. 不做 judge 评分(Day 3)。
