# STATE_WorkerE — Day 1 风格与安全生成(Moonshot API)

> Worker E ｜ 状态:**任务1 完成 ✅;任务2 进行中(tmux 续跑)** ｜ tmux 会话:`v5gen` ｜ 远端脚本:`pipeline_v5/src/safety_gen_v5.py`、`pipeline_v5/src/styleC_gen_v5.py`

## 任务1 Safety-Refusal 300 ✅ 已完成并已回传

- 产物:`safety_refusal_v5.jsonl` 300/300(5 类各 60 ✓)+ `safety_refusal_report.json`,远端 `data/processed/v5_data/` 与本地 `paper/v5_execution/day1/` 各一份。
- 质量门:completion 两段结构(拒答理由+合规替代方向)标记词校验 ✓;长度 150–233 字符(≥150 硬门);**模板控制:tail80 最大频次 1、prefix20 最大频次 2,均远低于频次≥3 黑名单线与 5% 占比线**;生成期拒绝 241 次(completion_short 182 / no_refusal_marker 58 / no_alt_marker 1),拒绝不产出记录。
- 生成参数:moonshot-v1-8k,T=0.9,批量 5 条/请求。

## 任务2 风格 C 长答 3,445 🔄 进行中

- 进度(截至 12:25 远端):**95/3445**,tmux `v5gen` 续跑中,预计 ~6h(实测稳态 ~9-10 条/分钟)。
- 产物:`styleC_long_answers.jsonl`(追加续跑,按 group_id 断点);本地已回传 `styleC_long_answers.partial.jsonl`(95 条)。
- 质量门:长度硬校验 1200–2500(区间外单条重生成 1 次,仍不合格丢弃落 `styleC_long_dropped.jsonl`);末 80 字符频次 ≥3 模板黑名单预防式拒绝;每 25 条刷新 `styleC_long_report.json`;全量完成后自动导出 10% 抽检 `styleC_long_review.md`(seed=42),中途可 `--review-only`。
- 当前计数:batch_empty 3(解析空,自动转单条),零丢弃。

## ⚠️ 关键决策(需 Orchestrator/Owner 知晓)

1. **模型切换**:长答生成由 moonshot-v1-32k 改为 **kimi-k2.5**(同一 Moonshot API/密钥)。实测证据:moonshot-v1 存在 ~600 字/条的批量总预算收敛(batch=5 → 560-670 字/条,batch=2 → 741-802 字/条),无法满足 1200 字硬下限,单条重生成丢弃率曾达 40%;kimi-k2.5 批量 5 条实测 1442–1772 字,试跑 15/15 首过(1427–2013,均值 1721)。k2.x 仅允许 temperature=1,调用不传温度。
2. **输出协议**:JSON 数组改分隔符 `【回答i】...【/回答i】`(长文本 JSON 转义频繁破损);系统提示内嵌 1341 字完整示例锚定长度。
3. **吞吐**:6 线程流水线 + 全局令牌桶(请求启动间隔 ≥20s,RPM=3 纪律不变);k2.5 单请求延迟 ~140s,串行需 ~26h,流水线后 ~6h。
4. 前 13 条为 v1-32k 旧协议产物(已过同一质量门,长度 1204–1471),保留混入;`styleC_long_dropped.jsonl` 中 12 条 v1 时代丢弃记录对应的 group_id 已由新管线重生成覆盖。

## 续跑/恢复

```bash
ssh chinux@spark-855a
tmux attach -t v5gen
# 若中断,直接重跑(断点续跑):
cd /home/meerkat/mongoose_ai && eval "$(grep '^export MOONSHOT_API_KEY' ~/.bashrc)" \
  && venv_v5/bin/python pipeline_v5/src/styleC_gen_v5.py
```

## 待办(resume 时)

1. 监控至 3,445 完成;>500 条时回传一批到本地。
2. 完成后:核对 `styleC_long_report.json`(长度分布/tail80_max_freq/dropped 计数)、确认 `styleC_long_review.md` 已导出;回传全部产物(answers/report/review/dropped)到本地;更新本 STATE 为完成。
