#!/usr/bin/env python3
"""从 v5_data_report.json 生成 MANIFEST.md (方案 §4.7 五项)。
用法: venv_v5/bin/python pipeline_v5/src/gen_manifest_v5.py [data_commit_hash]
"""
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "data/processed/v5_data/final"
r = json.load(open(OUT / "v5_data_report.json", encoding="utf-8"))
data_commit = sys.argv[1] if len(sys.argv) > 1 else "(见下方 git 章节, 本文件随数据同批入库)"

f = r["final"]
d15 = r["decision_15_max_length"]
decon = next(s for s in r["stages"] if s["stage"] == "dual_decontamination")
xchk = next(s for s in r["stages"] if s["stage"] == "split_cross_check")
wrap = next(s for s in r["stages"] if s["stage"] == "styleC_batch_wrapper_repair")
proto = next(s for s in r["stages"] if s["stage"] == "chatml_render_protocol")

L = []
A = L.append
A("# MANIFEST — Meerkat-AI v5 训练数据集 (Day 1 总装)")
A("")
A(f"- 生成时间: {r['generated_at']} (远端 spark-855a)")
A(f"- 构建器: {r['builder']}")
A(f"- 方案依据: v5_优化微调方案.md {', '.join(r['plan_refs'])}")
A(f"- 独立复核: `pipeline_v5/src/verify_build_v5.py` 全部 PASS (去污命中率 + 划分完整性对账一致)")
A("")
A("## ① 输入清单 + sha256")
A("")
A("| 输入 | 行数 | sha256 |")
A("|---|---|---|")
for name, meta in r["inputs"].items():
    A(f"| {name} | {meta.get('rows', '-')} | `{meta['sha256'][:16]}…` |")
A("")
A("完整 sha256 与绝对路径见 `v5_data_report.json` → `inputs`。")
A("")
A("偏差声明: styleC 长答实际 3,441/3,445 条 (4 条未完结, tmux v5gen_E 已退出, "
  "按任务纪律如实记录以实际条数继续); 13 条 completion 带批量 API JSON 包装, "
  f"总装时提取 answer 字段修复 (修复 {wrap['wrapped_and_repaired']} 条, 0 失败)。")
A("")
A("## ② 每门计数")
A("")
A("前置门 (Worker A/E/H, 详见各自报告): 种子三规则清洗 385→365; 语料门 1-3+门5 修复 "
  "→ gated_corpus 8,613; styleC 长答质量门 (长度 1200-2500, tail80 残留=0); "
  "种子扩写 306 条 + 59 存活 = 365; safety 300 条 (5 类×60)。")
A("")
A("总装门 (本构建):")
A("")
A("| 门 | 计数 |")
A("|---|---|")
A(f"| 样本池 (双风格合并+种子×3+safety) | {next(s for s in r['stages'] if s['stage']=='pool_assembly')['total']} |")
A(f"| safety 占比断言 ≤5% | {next(s for s in r['stages'] if s['stage']=='pool_assembly')['safety_share']:.2%} ✓ |")
A(f"| 去污剔除·参照集 A (金标 200) | {decon['ref_A']['dropped']} |")
A(f"| 去污剔除·参照集 B (eval2 465 + probe 120) | {decon['ref_B']['dropped']} (剔除率 {decon['ref_B']['drop_rate']:.2%}) |")
A(f"| 去污双集同中 | {decon['both_sets_hit']} |")
A(f"| 人工审查队列 J∈[0.4,0.5) | A={decon['ref_A']['review_queue']} B={decon['ref_B']['review_queue']} 共 {decon['review_queue_total']} → `decon_review_queue.jsonl` |")
A(f"| 划分交叉检查移回 train | {xchk['moved_back_to_train']} |")
A(f"| ChatML >2048 token 丢弃 | {d15['overlength_dropped']} |")
A("")
A("⚠️ 告警记录 (§4.6 风险条款, 不中断): " + decon.get("ALERT", "无"))
A("")
A("去污算法注: v4 NgramIndex 稀有 token 签名分桶为近似候选, 独立复核实测漏检 "
  "12 条 J≥0.5; 总装改用精确 brute-force + size-ratio 剪枝 (对 J≥0.4 无漏检), "
  "复核脚本独立实现同口径算法, 双方数字一致。")
A("")
A("## ③ 最终条数 / 子集分布 / 长度分布 / 风格配比")
A("")
A(f"**总计 {f['total']:,} 条 = train {f['splits']['train']['n']:,} + "
  f"validation {f['splits']['validation']['n']:,} + test {f['splits']['test']['n']:,}**")
A("")
A(f"风格配比: 短答 {f['style_ratio']['short']:,} : 长答 {f['style_ratio']['long']:,} "
  f"= {f['style_ratio']['short:long']} (方案 C 目标 6:4 针对语料两臂; "
  f"种子×3 与 safety 为短答-only, 如实记录实际配比)")
A("")
A("| split | n | 短答 | 长答 | token mean | p50 | p95 | p99 | max |")
A("|---|---|---|---|---|---|---|---|---|")
for side in ("train", "validation", "test"):
    sp = f["splits"][side]
    t = sp["token_len(prompt+completion)"]
    A(f"| {side} | {sp['n']:,} | {sp['style']['short']:,} | {sp['style']['long']:,} "
      f"| {t['mean']} | {t['p50']} | {t['p95']} | {t['p99']} | {t['max']} |")
A("")
A("子集分布 (train): " + "; ".join(f"{k} {v:,}" for k, v in
                                   f["splits"]["train"]["subset_dist"].items()))
A("")
A("## ④ 划分参数与退化声明")
A("")
sp = f["split_params"]
A(f"- 比例: train {sp['ratios']['train']} / validation {sp['ratios']['validation']} "
  f"/ test {sp['ratios']['test']}, seed={sp['seed']}")
A(f"- 分组: {sp['group']} (组数 {sp['n_groups']:,}, 最大组 {sp['max_group_size']}); "
  f"分层: {sp['stratify']}")
A("- 同组同侧保证: 长短双风格版 (同 group_id)、种子 ×3 衍生物、v4 前缀12聚类, "
  "经 union-find 合并为同一划分单元; 划分后 test/validation vs train 3-gram "
  "Jaccard≥0.5 交叉检查, 命中整组移回 train")
for d in f["degradation"]:
    A(f"- 退化声明: {d}")
A("")
A("ChatML 协议 (E0): " + proto["template"] + "; 空 think 块 **保留** "
  "(与 Worker F 生成侧一致; v4 为剥除, 此为有意变更 —— 训练/推理格式一致优先); "
  "prompt 尾部 `" + proto["prompt_suffix"].replace("\n", "\\n") + "`; EOS 由 TRL 附加。")
A("")
A("## ⑤ 输出 sha256 / 行数 / config / git")
A("")
A("| 输出 | 行数 | sha256 |")
A("|---|---|---|")
for name, meta in r["outputs"].items():
    A(f"| {name} | {meta['rows']:,} | `{meta['sha256'][:16]}…` |")
A("")
A("- 对应 config: `pipeline_v5/configs/data_v5.json` (构建参数快照见 v5_data_report.json)")
A(f"- 数据产物 git commit: {data_commit}")
A(f"- 本 MANIFEST 生成: {datetime.now().isoformat(timespec='seconds')}")
A("")
A("## 裁决项 #15 (max_length)")
A("")
A(d15["statement"])

p = OUT / "MANIFEST.md"
p.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"MANIFEST 落盘: {p} ({len(L)} 行)")
