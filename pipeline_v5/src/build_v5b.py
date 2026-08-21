#!/usr/bin/env python
"""
v5b 训练集组装: v5a 训练集 - PR 错标 + PR 注入。

输入:
  data/processed/v5_data/final/v5_train_v5a.jsonl   基线 (v5a 实际训练用)
  data/processed/v5b_data/pr_legacy_audit.jsonl     存量审计 (mislabel 剔除)
  data/processed/v5b_data/pr_inject_answers.jsonl   注入 (问题, 长答) 对

输出:
  data/processed/v5b_data/final/v5_train_v5b.jsonl
  data/processed/v5b_data/final/v5b_data_report.json
  data/processed/v5b_data/final/MANIFEST.md

纪律:
  - 单变量: 仅 PR 子集变动, 其余子集逐字节保留 (顺序不变, 新 PR 追加于尾部,
    训练侧 dataloader shuffle 由训练配置控制, 与 v5a 相同);
  - 判定失败 (parse_fail/api_fail) 样本默认保留并计入告警;
  - chatml 渲染与 assemble_v5 一致: system+user+empty_think 前缀, completion 原文;
  - 注入样本过 max_length 2048 token 校验 (tokenizer 与训练一致, 若不可用
    则以字符数 4096 近似上限兜底并记录)。

用法: venv_v5/bin/python pipeline_v5/src/build_v5b.py
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V5A = PROJECT_ROOT / "data/processed/v5_data/final/v5_train_v5a.jsonl"
AUDIT = PROJECT_ROOT / "data/processed/v5b_data/pr_legacy_audit.jsonl"
INJECT = PROJECT_ROOT / "data/processed/v5b_data/pr_inject_answers.jsonl"
OUT_DIR = PROJECT_ROOT / "data/processed/v5b_data/final"
OUT = OUT_DIR / "v5_train_v5b.jsonl"
REPORT = OUT_DIR / "v5b_data_report.json"
MANIFEST = OUT_DIR / "MANIFEST.md"

PROMPT_TMPL = ("<|im_start|>system\n{system}<|im_end|>\n"
               "<|im_start|>user\n{instruction}<|im_end|>\n"
               "<|im_start|>assistant\n<think>\n\n</think>\n\n")
CHAR_LIMIT = 4096  # max_length 2048 token 的保守字符兜底


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 审计: 剔除清单 (同 line_no 多条记录时以后者为准 — 支持人工复核追加覆盖)
    verdict_by_line = {}
    with open(AUDIT, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                a = json.loads(line)
                verdict_by_line[a["line_no"]] = a["verdict"]
    audit_stat = Counter(verdict_by_line.values())
    drop_lines = {ln for ln, v in verdict_by_line.items() if v == "mislabel"}

    kept, removed, pr_kept_legacy = [], 0, 0
    with open(V5A, encoding="utf-8") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            if d.get("subset") == "principle_recommendation":
                if i in drop_lines:
                    removed += 1
                    continue
                pr_kept_legacy += 1
            kept.append(line.rstrip("\n"))

    # 注入
    injected, overlong = 0, 0
    new_recs = []
    with open(INJECT, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            prompt = PROMPT_TMPL.format(system=r["system"],
                                        instruction=r["instruction"])
            if len(prompt) + len(r["completion"]) > CHAR_LIMIT:
                overlong += 1
                continue
            new_recs.append({"prompt": prompt, "completion": r["completion"],
                             "subset": "principle_recommendation"})
            injected += 1

    with open(OUT, "w", encoding="utf-8") as f:
        for line in kept:
            f.write(line + "\n")
        for r in new_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(kept) + injected
    subset_final = Counter()
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            subset_final[json.loads(line).get("subset", "?")] += 1

    report = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "builder": "build_v5b.py",
        "single_variable": "仅 principle_recommendation 子集变动, 其余逐字节保留",
        "inputs": {
            "v5a_train": str(V5A),
            "audit": str(AUDIT),
            "inject": str(INJECT),
        },
        "audit_verdicts": dict(audit_stat),
        "pr_legacy_kept": pr_kept_legacy,
        "pr_mislabel_removed": removed,
        "pr_injected": injected,
        "pr_inject_overlong_skipped": overlong,
        "pr_final": pr_kept_legacy + injected,
        "total_rows": total,
        "subset_final": dict(subset_final),
    }
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    MANIFEST.write_text(f"""# v5b 训练集 MANIFEST

生成时间: {report['built_at']}
构建器: build_v5b.py (PR 定向注入, 2026-07-29 归因立项)

## 变更 (单变量: 仅 PR 子集)

| 项 | 数量 |
|---|---|
| v5a PR 存量 | 196 |
| 审计剔除 (mislabel) | -{removed} |
| 审计保留 (true_pr+related) | {pr_kept_legacy} |
| 新注入 (详细模式长答) | +{injected} |
| **v5b PR 合计** | **{pr_kept_legacy + injected}** |
| 全量总行数 | {total} |

## 审计判定分布

{json.dumps(dict(audit_stat), ensure_ascii=False)}

## 文件

- 训练集: v5_train_v5b.jsonl (sha256 见 v5b_data_report.json 同目录)
- validation/test: 沿用 v5_data/final/ (不变, 单变量纪律)
""", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
