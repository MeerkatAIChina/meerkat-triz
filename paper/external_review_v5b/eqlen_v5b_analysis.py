#!/usr/bin/env python3
"""v5b 的等长对照 (D5 规对论文 §5.4 +0.430 头条的强制伴随实验)。

零新增成本: 全部复用现有缓存 —
  v5b 逐题 judge_overall 来自 eval_v5_v5b_gold_20260731_195036.json
  base 等长臂逐题分来自 contest_cache_moonshot_base_eqlen.json
  base 无约束逐题分来自 paper/contest/contest_cache_moonshot_base.json

注意 (如实披露): 等长臂是按 v5a 长度定的目标 (实际中位 1750 字),
v5b 中位仅 1055 字 — 本对照中 v5b 仍保有残余长度优势 (对照臂更长)。
若差值仍 n.s./负, 结论干净: +0.430 同样不 survive 长度控制;
若显著正, 才需要 v5b 专属等长臂重生成。
"""
import json
import random
import sys
from pathlib import Path

W = Path("/Volumes/2nd-HD/claude/Meerkat-AI/paper/external_review_v5b")
sys.path.insert(0, "/Volumes/2nd-HD/claude/Meerkat-AI/pipeline_v5/eval")


def boot_diff(a, b, n_boot=10000, seed=42):
    n = len(a)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(sum(b[i] - a[i] for i in idx) / n)
    boots.sort()
    diff = sum(b[i] - a[i] for i in range(n)) / n
    return diff, boots[250], boots[9749]


def load_cache(path):
    c = json.load(open(path, encoding="utf-8"))
    return {k: (v["overall"] if isinstance(v, dict) else None)
            for k, v in c.items()}


def main():
    gold = {}
    for l in open(W / "v5_gold.jsonl", encoding="utf-8"):
        r = json.loads(l)
        gold[r["id"]] = r
    ids = sorted(gold)
    subsets = {qid: gold[qid]["subset"] for qid in ids}

    # v5b 逐题 (judge_overall + kw_hit_rate 同一记录内)
    ev = json.load(open(W / "eval_v5_v5b_gold_20260731_195036.json",
                          encoding="utf-8"))
    recs = ev["records"] if isinstance(ev, dict) and "records" in ev else ev
    v5b_judge = {r["id"]: r["judge_overall"] for r in recs
                 if r.get("judge_overall") is not None}
    v5b_kw = {r["id"]: r["kw_hit_rate"] for r in recs
              if r.get("kw_hit_rate") is not None}

    moon_base = load_cache(
        "/Volumes/2nd-HD/claude/Meerkat-AI/paper/contest/contest_cache_moonshot_base.json")
    moon_eqlen = load_cache(W / "contest_cache_moonshot_base_eqlen.json")

    def series(scores, subset=None):
        return [scores[q] for q in ids
                if scores.get(q) is not None
                and (subset is None or subsets[q] == subset)]

    print("== judge 轨 (moonshot 主轨, 0–4) ==")
    for label, sa, sb in (
        ("D1. v5b−base(无约束) [复核+0.430]", moon_base, v5b_judge),
        ("D2. v5b−base(等长@v5a目标)", moon_eqlen, v5b_judge),
    ):
        a, b = series(sa), series(sb)
        d, lo, hi = boot_diff(a, b)
        sig = "显著" if (lo > 0 or hi < 0) else "n.s."
        print(f"{label}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}] {sig} (n={len(a)})")
        for s in ("principle_recommendation", "concept_explanation"):
            a2, b2 = series(sa, s), series(sb, s)
            d2, lo2, hi2 = boot_diff(a2, b2)
            print(f"    {s}: {d2:+.3f} [{lo2:+.3f}, {hi2:+.3f}]")

    print("\n== 均分对照 ==")
    for name, sc in (("base 无约束", moon_base), ("base 等长", moon_eqlen),
                     ("v5b", v5b_judge)):
        s = series(sc)
        print(f"{name}: {sum(s)/len(s):.3f} (n={len(s)})")

    # keyword 轨: v5b kw_hit_rate vs base 等长臂
    print("\n== keyword 轨 ==")
    from keyword_scorer import keyword_score, load_alias_map  # noqa
    kmap = load_alias_map(
        "/Volumes/2nd-HD/claude/Meerkat-AI/pipeline_v5/eval/keyword_map_v5.json")

    def kw_series(gen_path):
        resps = {}
        if str(gen_path).endswith(".jsonl"):
            for l in open(gen_path, encoding="utf-8"):
                if l.strip():
                    r = json.loads(l)
                    resps[r["id"]] = r["response"]
        else:
            d = json.load(open(gen_path, encoding="utf-8"))
            resps = {r["id"]: r["response"] for r in d["records"]}
        return [keyword_score(resps[q], gold[q]["keywords"], kmap)["kw_hit_rate"]
                for q in ids if q in resps]

    kw_eqlen = kw_series(W / "v5_gen_base_eqlen.jsonl")
    kw_base = kw_series(W / "eval_v5_base_goldfix_v5_20260726_234434.json")
    kw_v5b = series(v5b_kw)
    for label, a, b in (("v5b−base(无约束) [复核−0.028]", kw_base, kw_v5b),
                        ("v5b−base(等长)", kw_eqlen, kw_v5b)):
        d, lo, hi = boot_diff(a, b)
        sig = "显著" if (lo > 0 or hi < 0) else "n.s."
        print(f"{label}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}] {sig}")

    # 长度对照
    print("\n== 长度 (字符, 中位) ==")
    import statistics
    eqlen_lens = []
    for l in open(W / "v5_gen_base_eqlen.jsonl", encoding="utf-8"):
        if l.strip():
            eqlen_lens.append(len(json.loads(l)["response"]))
    v5b_lens = [len(r["response"]) for r in recs if r.get("response")]
    print(f"base 等长臂: {statistics.median(eqlen_lens):.0f}  v5b: "
          f"{statistics.median(v5b_lens):.0f}  比值: "
          f"{statistics.median(v5b_lens)/statistics.median(eqlen_lens):.2f}x")


if __name__ == "__main__":
    main()
