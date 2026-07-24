# MANIFEST — Meerkat-AI v5 训练数据集 (Day 1 总装)

- 生成时间: 2026-07-24T20:31:08 (远端 spark-855a)
- 构建器: assemble_v5.py (Worker G 总装)
- 方案依据: v5_优化微调方案.md §4.1-C, §4.6, §4.7, §4.8, §11.1
- 独立复核: `pipeline_v5/src/verify_build_v5.py` 全部 PASS (去污命中率 + 划分完整性对账一致)

## ① 输入清单 + sha256

| 输入 | 行数 | sha256 |
|---|---|---|
| gated_corpus | 8613 | `87a728716f97c29f…` |
| styleC_long_answers | 3441 | `82ce4dec1da0c327…` |
| styleC_longanswer_sampling | 3445 | `b226b4629d54e70f…` |
| cleaned_seeds_final | 365 | `ef6e49e5b8ee0528…` |
| safety_refusal_v5 | 300 | `04219f1e5ed41e28…` |
| v4_gold(参照A,只读) | 100 | `ed2eb9a7279fddae…` |
| v5_gold_new100(参照A) | 100 | `6ff424a8d3734295…` |
| sample_data_expanded(参照B,只读) | 465 | `9acdd32ae4f6e44e…` |
| general_probe_v5(参照B) | 120 | `4a80c2863944b5cb…` |

完整 sha256 与绝对路径见 `v5_data_report.json` → `inputs`。

偏差声明: styleC 长答实际 3,441/3,445 条 (4 条未完结, tmux v5gen_E 已退出, 按任务纪律如实记录以实际条数继续); 13 条 completion 带批量 API JSON 包装, 总装时提取 answer 字段修复 (修复 13 条, 0 失败)。

## ② 每门计数

前置门 (Worker A/E/H, 详见各自报告): 种子三规则清洗 385→365; 语料门 1-3+门5 修复 → gated_corpus 8,613; styleC 长答质量门 (长度 1200-2500, tail80 残留=0); 种子扩写 306 条 + 59 存活 = 365; safety 300 条 (5 类×60)。

总装门 (本构建):

| 门 | 计数 |
|---|---|
| 样本池 (双风格合并+种子×3+safety) | 13449 |
| safety 占比断言 ≤5% | 2.23% ✓ |
| 去污剔除·参照集 A (金标 200) | 0 |
| 去污剔除·参照集 B (eval2 465 + probe 120) | 1145 (剔除率 8.51%) |
| 去污双集同中 | 0 |
| 人工审查队列 J∈[0.4,0.5) | A=0 B=63 共 63 → `decon_review_queue.jsonl` |
| 划分交叉检查移回 train | 235 |
| ChatML >2048 token 丢弃 | 0 |

⚠️ 告警记录 (§4.6 风险条款, 不中断): B 集剔除率 8.51% > 3% 阈值, 触发人工复核告警 (按 §4.6 风险条款记录, 不中断构建)

去污算法注: v4 NgramIndex 稀有 token 签名分桶为近似候选, 独立复核实测漏检 12 条 J≥0.5; 总装改用精确 brute-force + size-ratio 剪枝 (对 J≥0.4 无漏检), 复核脚本独立实现同口径算法, 双方数字一致。

## ③ 最终条数 / 子集分布 / 长度分布 / 风格配比

**总计 12,304 条 = train 10,698 + validation 1,050 + test 556**

风格配比: 短答 8,879 : 长答 3,425 = 0.722:0.278 (方案 C 目标 6:4 针对语料两臂; 种子×3 与 safety 为短答-only, 如实记录实际配比)

| split | n | 短答 | 长答 | token mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| train | 10,698 | 7,697 | 3,001 | 412.8 | 178.0 | 1155.0 | 1290.1 | 1521 |
| validation | 1,050 | 766 | 284 | 398.5 | 177.5 | 1141.5 | 1277.0 | 1513 |
| test | 556 | 416 | 140 | 387.7 | 175.5 | 1169.5 | 1309.7 | 1423 |

子集分布 (train): ariz_guidance 1,044; case_generation 2,857; concept_explanation 2,996; contradiction_analysis 313; innovation_assessment 3,037; principle_recommendation 196; safety_refusal 255

## ④ 划分参数与退化声明

- 比例: train 0.85 / validation 0.1 / test 0.05, seed=42
- 分组: union-find(group_id, prefix12) (组数 7,932, 最大组 51); 分层: subset (组多数层)
- 同组同侧保证: 长短双风格版 (同 group_id)、种子 ×3 衍生物、v4 前缀12聚类, 经 union-find 合并为同一划分单元; 划分后 test/validation vs train 3-gram Jaccard≥0.5 交叉检查, 命中整组移回 train
- 退化声明: 样本级无 source/chunk 标识, 沿用 v4 退化: 归一化 instruction 前缀12聚类 (v5 与 group_id 做 union-find 合并, 不弱于 v4)
- 退化声明: 跨 subset 同前缀组按组内多数 subset 归层, 分层为近似分层

ChatML 协议 (E0): apply_chat_template(system+user, add_generation_prompt=True, enable_thinking=False); 空 think 块 **保留** (与 Worker F 生成侧一致; v4 为剥除, 此为有意变更 —— 训练/推理格式一致优先); prompt 尾部 `<|im_start|>assistant\n<think>\n\n</think>\n\n`; EOS 由 TRL 附加。

## ⑤ 输出 sha256 / 行数 / config / git

| 输出 | 行数 | sha256 |
|---|---|---|
| v5_train.jsonl | 10,698 | `c004f21e3f66673a…` |
| v5_validation.jsonl | 1,050 | `e428d21f9520a46d…` |
| v5_test.jsonl | 556 | `5f3f7d872be23fae…` |
| _assembly_sidecar.jsonl | 12,304 | `86f49c8e315d2537…` |
| decon_review_queue.jsonl | 63 | `e5045714f3cb0be5…` |

- 对应 config: `pipeline_v5/configs/data_v5.json` (构建参数快照见 v5_data_report.json)
- 数据产物 git commit: 04b3b41
- 本 MANIFEST 生成: 2026-07-24T20:34:23

## 裁决项 #15 (max_length)

train 集 prompt+completion token p95=1155.0, p99=1290.1, max=1521; 全长尾由 >2048 硬门剔除 0 条 (占比 0.0000%)。p99 ≤ 2048, max_length 锁 2048 不造成静默截断, 裁决项 #15 结论: 安全, 锁定 2048。
