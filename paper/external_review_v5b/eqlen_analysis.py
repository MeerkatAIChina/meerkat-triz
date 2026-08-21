#!/usr/bin/env python3
"""等长对照分析 (论文 Limitation ii 的正式回答)。

对比三组配对差值 (paired bootstrap n=10000 seed=42, 与主协议一致):
  A. v5a − base(无约束)   : 既有的 +0.393 主轨头条
  B. v5a − base(等长约束) : 长度基本对齐后的 judge 差
     → 若 B ≈ 0, +0.393 主要为长度伪影; 若 B 仍显著正, 含真实质量差
  C. keyword 轨同步复算 (base 等长后命中面变化)

输入: paper/external_review_v5b/ 下的缓存与生成文件。
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

    moon_v5a = load_cache(Path("/Volumes/2nd-HD/claude/Meerkat-AI/paper/contest/contest_cache_moonshot_v5a.json"))
    moon_base = load_cache(Path("/Volumes/2nd-HD/claude/Meerkat-AI/paper/contest/contest_cache_moonshot_base.json"))
    moon_eqlen = load_cache(W / "contest_cache_moonshot_base_eqlen.json")

    subsets = {qid: gold[qid]["subset"] for qid in ids}

    def series(scores, subset=None):
        return [scores[q] for q in ids
                if scores.get(q) is not None
                and (subset is None or subsets[q] == subset)]

    print("== judge 轨 (moonshot 主轨, 0–4) ==")
    for label, sa, sb in (
        ("A. v5a−base(无约束)", moon_base, moon_v5a),
        ("B. v5a−base(等长)", moon_eqlen, moon_v5a),
        ("C. base(等长)−base(无约束)", moon_base, moon_eqlen),
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
    for name, sc in (("base 无约束", moon_base), ("base 等长", moon_eqlen), ("v5a", moon_v5a)):
        s = series(sc)
        print(f"{name}: {sum(s)/len(s):.3f} (n={len(s)})")

    # keyword 轨 (别名表后)
    print("\n== keyword 轨 (别名表后) ==")
    from keyword_scorer import keyword_score, load_alias_map  # noqa
    kmap = load_alias_map("/Volumes/2nd-HD/claude/Meerkat-AI/pipeline_v5/eval/keyword_map_v5.json")

    def kw_series(gen_path, subset=None):
        resps = {}
        if str(gen_path).endswith(".jsonl"):
            for l in open(gen_path, encoding="utf-8"):
                if l.strip():
                    r = json.loads(l)
                    resps[r["id"]] = r["response"]
        else:
            d = json.load(open(gen_path, encoding="utf-8"))
            resps = {r["id"]: r["response"] for r in d["records"]}
        out = []
        for q in ids:
            if q not in resps:
                continue
            if subset and subsets[q] != subset:
                continue
            sc = keyword_score(resps[q], gold[q]["keywords"], kmap)
            out.append(sc["kw_hit_rate"])
        return out

    kw_base = kw_series(W / "eval_v5_base_goldfix_v5_20260726_234434.json")
    kw_eqlen = kw_series(W / "v5_gen_base_eqlen.jsonl")
    kw_v5a = kw_series(W / "../contest/gen_v5a.jsonl")
    for label, a, b in (("v5a−base(无约束)", kw_base, kw_v5a),
                        ("v5a−base(等长)", kw_eqlen, kw_v5a),
                        ("base(等长)−base(无约束)", kw_base, kw_eqlen)):
        d, lo, hi = boot_diff(a, b)
        sig = "显著" if (lo > 0 or hi < 0) else "n.s."
        print(f"{label}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}] {sig}")


if __name__ == "__main__":
    main()
