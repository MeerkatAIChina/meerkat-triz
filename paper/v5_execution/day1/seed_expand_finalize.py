#!/usr/bin/env python3
"""
Worker H 种子扩写 - 步骤4: 合并与质量门终检 (无 API)。

合并: cleaned_seeds.jsonl (59 条真实存活) + seed_expanded.jsonl (扩写存活)
  -> cleaned_seeds_final.jsonl  (统一字段 instruction/input/output/subset/group_id,
     扩写条带 expanded=True 标记与 output_base)
质量门 (与 Worker A 同标准):
  - 长度 150-600 硬校验 (扩写条; 存条按 Worker A 已验 >=150)
  - tail80 频次>=3 模板黑名单残留校验 (扩写集内 + 合并全集, 必须=0)
  - R2 黑名单 (b1/b2) 残留校验 (扩写集, 必须=0)
  - 扩写忠实度抽检: 10% 队列落 seed_expand_fidelity_review.md (人工核对
    扩写不得改变原答案技术结论)
产出: seed_expansion_report.json (候选/成功/失败原因分布/最终条数)
"""
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data/processed/v5_data"

LEN_MIN, LEN_MAX = 150, 600
SENT_END = "。!！?？\n"


def last_sentence(text: str) -> str:
    t = (text or "").rstrip()
    if not t:
        return ""
    i = max(t.rfind(c, 0, len(t) - 1) for c in SENT_END)
    return t[i + 1:].strip() if i >= 0 else t


def read_jsonl(p):
    if not p.is_file():
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    survived = read_jsonl(OUT_DIR / "cleaned_seeds.jsonl")
    expanded = read_jsonl(OUT_DIR / "seed_expanded.jsonl")
    dropped = read_jsonl(OUT_DIR / "seed_expand_dropped.jsonl")
    cands = read_jsonl(OUT_DIR / "seed_expand_candidates.jsonl")
    bl = json.loads((OUT_DIR / "seed_expand_blacklist.json").read_text(encoding="utf-8"))
    b1, b2 = set(bl["b1_tail80"]), set(bl["b2_last_sentence"])

    assert len(survived) == 59
    # group_id 不得重叠
    ids_s = {s["group_id"] for s in survived}
    ids_e = [e["group_id"] for e in expanded]
    assert len(set(ids_e)) == len(ids_e), "扩写集 group_id 重复"
    assert not (ids_s & set(ids_e)), "存活集与扩写集 group_id 重叠"

    # --- 质量门 1: 长度硬校验 ---
    len_bad = [e for e in expanded if not (LEN_MIN <= len(e["output"]) <= LEN_MAX)]
    # --- 质量门 2: R2 黑名单残留 ---
    bl_resid = [e for e in expanded
                if e["output"][-80:] in b1 or last_sentence(e["output"]) in b2]
    # --- 质量门 3: tail80 频次 (扩写集内 + 合并全集) ---
    tail_e = Counter(e["output"][-80:] for e in expanded)
    tail_e_bad = {t: c for t, c in tail_e.items() if c >= 3}
    all_out = [s["output"] for s in survived] + [e["output"] for e in expanded]
    tail_all = Counter(o[-80:] for o in all_out)
    tail_all_bad = {t: c for t, c in tail_all.items() if c >= 3}

    # --- 合并输出 ---
    final = []
    for s in survived:
        final.append({"instruction": s["instruction"], "input": s.get("input", ""),
                      "output": s["output"], "subset": s["subset"],
                      "group_id": s["group_id"], "expanded": False})
    for e in expanded:
        final.append({"instruction": e["instruction"], "input": e.get("input", ""),
                      "output": e["output"], "subset": e["subset"],
                      "group_id": e["group_id"], "expanded": True,
                      "output_base": e["output_base"],
                      "drop_reason": e["drop_reason"]})
    p_final = OUT_DIR / "cleaned_seeds_final.jsonl"
    with open(p_final, "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- 忠实度抽检 10% 队列落 md ---
    rng = random.Random(42)
    n_sample = max(1, round(len(expanded) * 0.10))
    sample = rng.sample(expanded, min(n_sample, len(expanded)))
    p_md = OUT_DIR / "seed_expand_fidelity_review.md"
    with open(p_md, "w", encoding="utf-8") as f:
        f.write("# 种子扩写忠实度抽检 (10% 随机队列, seed=42)\n\n")
        f.write(f"扩写存活 {len(expanded)} 条, 抽检 {len(sample)} 条。"
                "核对要点: 扩写是否保持原草稿的技术结论与术语不变, "
                "仅补全推理与结构, 无模板化收尾。\n\n---\n")
        for i, e in enumerate(sample, 1):
            f.write(f"\n## 样本 {i} [{e['subset']}] {e['group_id']}\n\n"
                    f"**问题**: {e['instruction']}\n\n"
                    f"**原草稿 ({len(e['output_base'])} 字符)**:\n\n{e['output_base']}\n\n"
                    f"**扩写后 ({len(e['output'])} 字符)**:\n\n{e['output']}\n\n---\n")

    lens = sorted(len(e["output"]) for e in expanded)
    fail_dist = Counter(d.get("fail_reason", "unknown").split("(")[0] for d in dropped)
    rep = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "owner_decision": "方案③: 保留真实 instruction, API 扩写 output 至 >=150 字符 (半真实锚定)",
        "candidates": {
            "total": len(cands),
            "truncated_short": sum(1 for c in cands if c["drop_reason"] == "truncated_short"),
            "raw_short": sum(1 for c in cands if c["drop_reason"] == "raw_short"),
            "by_subset": dict(Counter(c["subset"] for c in cands)),
        },
        "expansion": {
            "success": len(expanded),
            "success_by_subset": dict(Counter(e["subset"] for e in expanded)),
            "success_by_reason": dict(Counter(e["drop_reason"] for e in expanded)),
            "dropped": len(dropped),
            "fail_reason_distribution": dict(fail_dist),
            "len_min": lens[0] if lens else 0,
            "len_p50": lens[len(lens) // 2] if lens else 0,
            "len_max": lens[-1] if lens else 0,
        },
        "quality_gates": {
            "len_range": [LEN_MIN, LEN_MAX],
            "len_violations": len(len_bad),
            "r2_blacklist_residual": len(bl_resid),
            "tail80_freq_ge3_expanded_set": len(tail_e_bad),
            "tail80_freq_ge3_merged_set": len(tail_all_bad),
            "fidelity_sample_size": len(sample),
            "fidelity_review_md": str(p_md),
        },
        "final": {
            "survived_real": len(survived),
            "expanded": len(expanded),
            "total": len(final),
            "path": str(p_final),
        },
    }
    (OUT_DIR / "seed_expansion_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))

    assert not len_bad, f"长度越界 {len(len_bad)} 条流入最终集"
    assert not bl_resid, f"R2 黑名单残留 {len(bl_resid)} 条"
    assert not tail_e_bad, f"扩写集 tail80 频次>=3 残留 {len(tail_e_bad)} 种"


if __name__ == "__main__":
    main()
