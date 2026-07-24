# STATE_workerC — Day 1 Worker C 金标与探针扩题

> 更新时间: 2026-07-25 (Day 1) ｜ Worker: C ｜ 状态: **已完成** (两项生成 + 校验 + 回传)

## 任务范围
1. 金标扩题 100 题 (方案 §6.5): v4_gold 100 题钉住, 新增 v5_gold_101–200
2. 通用探针扩题 30→120 (方案 §6.6): 6 类 × 20

## 最终结果

### 金标扩题 (v5_gold_new100.jsonl, 100 题)
- 子集: principle 20 / contradiction 20 / ariz 20 / case 15 / concept 15 / innovation 10 ✓
- schema 与 v4_gold 完全一致; id v5_gold_101–200; source chunk 100 个唯一、与 v4 已用 chunk 0 重叠
- reference ≥200 字: 100/100; 关键词全部出现于 reference: 100/100
- 去重: 对 v4 100 题 max 3-gram Jaccard = 0.356, 批内 max = 0.333, 均 < 0.5 (剔除 0 次, 校验拒绝 1 次已补抽)
- 自动初检 FAIL 12 题: 均为关键词数偏离 5–8 目标 (kw=4 ×8, kw=3 ×3, kw=9 ×1),
  均满足 v4 验收规则 (过滤后 ≥3), 已在 v5_gold_review.md 标记优先人工复核; 全部 100 题状态 = 待人工
- sha256(前16): 6ff424a8d3734295

### 探针扩题 (general_probe_v5.json, 120 题)
- 6 类各 20: common_sense / math / logic / writing / code / instruction_following ✓
- 原 30 题逐字节保留在前; 新 90 题 id 沿用原前缀从 06 续号; id 全局唯一
- 去重: 新题 vs 原 30 max J = 0.294, 批内 max J = 0.436, 均 < 0.5 (剔除 0 次, 校验拒绝 2 次已补)
- sha256(前16): 4a80c2863944b5cb

## 产物路径
- 远端 (chinux@spark-855a:/home/meerkat/mongoose_ai/):
  `data/processed/v5_data/{v5_gold_new100.jsonl, v5_gold_review.md, general_probe_v5.json,
  general_probe_v5.report.json, general_probe_v5_new.jsonl, gold_gen_v5.log}`
  脚本/配置: `pipeline_v5/src/{gold_gen_v5.py, probe_gen_v5.py}`,
  `pipeline_v5/configs/{eval_v5_gold.json, eval_v5_probe.json}`
- 本地: `paper/v5_execution/day1/` (同上 6 产物 + 2 脚本 + 2 配置, 哈希与远端一致)

## 执行过程备注
- 金标生成 RPM=3 全程约 35 分钟 (09:22–09:57), 0 次 429, 1 次解析/校验拒绝
- 探针首次前台运行因本地 SSH 300s 超时被切, 缓存断点续跑第二次完成;
  期间缓存出现 probe_if_06–20 双份写入 (两进程各写一份), 已按最终 json 清理缓存 105→90 行,
  最终 json 未受影响 (清理前后 sha256 不变)
- v4_gold.jsonl / general_probe.json 原文件全程只读, 未改动

## 遗留事项
- v5_gold_review.md 100 题待人工抽检 10% (方案 §6.5 要求), 其中 12 题自动初检 FAIL 优先
- 金标新 100 题尚未对 v5 新语料重跑去污门 (属数据构建侧 Worker 职责, 方案 §6.5)
