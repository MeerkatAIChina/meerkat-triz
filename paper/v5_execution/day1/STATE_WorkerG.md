# STATE_WorkerG — v5 数据构建最终总装 (Day 1 收官)

- 更新: 2026-07-24 20:40 (远端 spark-855a 时间) · 状态: **完成, VERIFY PASS**

## 产物 (远端 data/processed/v5_data/final/, 已回传本地 day1/final/)
- v5_train.jsonl 10,698 / v5_validation.jsonl 1,050 / v5_test.jsonl 556 = 12,304
- MANIFEST.md (§4.7 五项全) + v5_data_report.json + decon_review_queue.jsonl (63 条) + _assembly_sidecar.jsonl
- 脚本: pipeline_v5/src/{assemble_v5, verify_build_v5, gen_manifest_v5}.py + configs/data_v5.json
- git: 远端 04b3b41 (数据+脚本) + f18ff76 (manifest 回填); 本地 00b5227

## 关键数字
- 样本池 13,449 = gated 8,613 + styleC 3,441 (目标 3,445, 差 4 未完结, v5gen_E 已退出, 如实记录) + 种子 365×3 + safety 300 (占比 2.23% ✓)
- 风格配比 短:长 = 8,879:3,425 = 72.2%:27.8% (种子/safety 短答-only)
- 去污: A (金标 200) 剔除 0 / B (eval2 465+probe 120) 剔除 1,145 (8.51% > 3% **告警已记录**, §4.6 不中断); 审查队列 63; 双中 0
- 划分 85/10/5 seed=42, union-find(group_id + 前缀12), 7,932 组; 交叉检查移回 train 235
- 长度: train token p95=1155 / p99=1290 / max=1521, >2048 丢弃 0 → **裁决项 #15: 锁 2048 安全**
- ChatML: E0 协议保留空 think 块 (v4 为剥除, 有意变更, 与 Worker F 生成侧一致), 12,304/12,304 尾部断言通过

## 重要技术发现 (resume 必读)
1. **v4 NgramIndex 稀有 token 签名分桶是近似候选**, 会漏不共享稀有 token 的高重叠对:
   第一版总装漏 12 条 J≥0.5, 独立复核 (精确 brute-force) 抓出。总装已改精确算法
   (size-ratio 剪枝对 J≥0.4 无漏检), 双方对账一致。**v4 历史去污数字 (门4/门6) 存在同型漏检风险, 训练后复盘应重估。**
2. styleC 长答 13 条 completion 带批量 API JSON 包装 ({"index":1,"answer":"..."}), 总装正则提取修复, 0 失败。
3. B 集高剔除率属预期 (eval2 扩充集与训练语料同源), 已按 §4.6 风险条款告警记录; 审查队列 63 条待人工。

## resume 指引
- 重跑: `venv_v5/bin/python pipeline_v5/src/assemble_v5.py && venv_v5/bin/python pipeline_v5/src/verify_build_v5.py`
- 复核退出码 0=PASS; 任何数字不一致即 FAIL 需排查
- 训练侧接口: prompt/completion/subset schema, system 段已渲染, 训练侧不得二次渲染 (§4.8)
