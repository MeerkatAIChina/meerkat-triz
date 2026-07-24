#!/usr/bin/env python3
"""
Worker H 种子扩写 - 步骤1: 还原被弃短种子全集 (无 API)。

复用 pipeline_v5/src/data_build_v5.py 的 Stage1 三规则逻辑 (import, 不重写):
  R1 冲突组整组弃 (385 -> 365) -> 59 条存活于 cleaned_seeds.jsonl
  被弃 306 条 = R2 截尾后 <150 的 196 条 (output_base 用截尾后文本)
              + R3 原始 <150 的 110 条 (output_base 用原文)

输出: data/processed/v5_data/seed_expand_candidates.jsonl
  字段: cand_id(=group_id), instruction, input, output_base, subset,
        drop_reason in {truncated_short, raw_short}, base_len
"""
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "pipeline_v5" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipeline_v4" / "src"))

import data_build as v4db  # noqa: E402
import data_build_v5 as a  # noqa: E402  (复用 load_seeds/截句/常量, 不触发 main)

OUT_DIR = PROJECT_ROOT / "data/processed/v5_data"


def main():
    seeds = a.load_seeds()
    assert len(seeds) == 385, f"种子数 {len(seeds)} != 385"

    # R1: 精确去重 + 冲突组整组弃 (与 Worker A 同函数同口径)
    seeds, dup, cg, cs = v4db.gate_exact_dedup_and_conflict(seeds)
    assert len(seeds) == 365, f"R1 后 {len(seeds)} != 365"

    # R2 黑名单 (数据驱动, 与 Worker A 同口径: tail80 频次>=3 / 末句频次>=3 且长>=8)
    tails = Counter(s["output"][-80:] for s in seeds)
    b1 = {t for t, c in tails.items() if c >= 3}
    sents = Counter(a.last_sentence(s["output"]) for s in seeds)
    b2 = {t for t, c in sents.items() if c >= 3 and len(t) >= 8}

    survived_ids = set()
    with open(OUT_DIR / "cleaned_seeds.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                survived_ids.add(json.loads(line)["group_id"])
    assert len(survived_ids) == 59, f"存活 {len(survived_ids)} != 59"

    candidates = []
    n_trunc_short = n_raw_short = 0
    for s in seeds:
        gid = a.group_id_of(s["instruction"])
        if gid in survived_ids:
            continue  # 59 条存活, 无需扩写
        # 重放 R2 截句 (最多 10 轮, 与 Worker A 一致)
        out, truncated = s["output"], False
        for _ in range(10):
            if out[-80:] in b1 or a.last_sentence(out) in b2:
                out = a.strip_last_sentence(out)
                truncated = True
            else:
                break
        if truncated:
            assert len(out) < a.MIN_OUTPUT_CHARS, \
                f"截尾后 >=150 却不在存活集? gid={gid}"
            reason = "truncated_short"
            n_trunc_short += 1
        else:
            assert len(out) < a.MIN_OUTPUT_CHARS
            reason = "raw_short"
            n_raw_short += 1
        candidates.append({
            "cand_id": gid,
            "group_id": gid,
            "instruction": s["instruction"],
            "input": s.get("input", "") or "",
            "output_base": out,
            "subset": s["subset"],
            "drop_reason": reason,
            "base_len": len(out),
        })

    ids = [c["cand_id"] for c in candidates]
    assert len(set(ids)) == len(ids), "cand_id 存在重复"

    out_path = OUT_DIR / "seed_expand_candidates.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 导出 R2 黑名单, 供扩写脚本收尾校验复用
    (OUT_DIR / "seed_expand_blacklist.json").write_text(
        json.dumps({"b1_tail80": sorted(b1), "b2_last_sentence": sorted(b2)},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    by_subset = Counter(c["subset"] for c in candidates)
    rep = {
        "candidates": len(candidates),
        "truncated_short": n_trunc_short,
        "raw_short": n_raw_short,
        "by_subset": dict(by_subset),
        "base_len_p50": sorted(c["base_len"] for c in candidates)[len(candidates) // 2],
        "output_path": str(out_path),
    }
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    assert n_trunc_short == 196 and n_raw_short == 110, "与 Worker A 报告计数不符"


if __name__ == "__main__":
    main()
