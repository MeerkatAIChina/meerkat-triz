#!/usr/bin/env python
"""汇总 v5/gen 下三个 tag 的 raw jsonl → gen_report.json (质量门计数)。"""
import json
import re
import sys
from pathlib import Path

G = Path("/home/meerkat/mongoose_ai/results/v5/gen")
THINK_CLOSED = re.compile(r"<think>.*?</think>", re.DOTALL)
CJK = re.compile(r"[一-鿿]")


def cjk_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if CJK.match(c)) / len(letters)


def gates(clean, min_len):
    return {
        "think_residual": ("<think>" in clean) or ("</think>" in clean),
        "nonempty_zh_fail": not (len(clean) > 0 and cjk_ratio(clean) >= 0.3),
        "length_fail": len(clean) < min_len,
        "english_draft": len(clean) > 0 and cjk_ratio(clean[:300]) < 0.1,
    }


def main():
    base_lens = {}
    resp_base = G / "responses_base_v5gold.jsonl"
    if resp_base.is_file():
        for l in open(resp_base, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                base_lens[r["id"]] = len(r["response"])

    report = {}
    for tag in ["base_v5gold", "v2_v5gold", "v4_v5gold"]:
        raw_f = G / f"raw_{tag}.jsonl"
        resp_f = G / f"responses_{tag}.jsonl"
        if not raw_f.is_file():
            report[tag] = {"status": "missing"}
            continue
        raws = [json.loads(l) for l in open(raw_f, encoding="utf-8") if l.strip()]
        resps = {json.loads(l)["id"]: json.loads(l)["response"]
                 for l in open(resp_f, encoding="utf-8") if l.strip()} \
            if resp_f.is_file() else {}
        modes = {}
        gate_fail = {"think_residual": 0, "nonempty_zh_fail": 0,
                     "length_fail": 0, "english_draft": 0}
        invalid = 0
        lens = []
        for r in raws:
            modes[r["mode"]] = modes.get(r["mode"], 0) + 1
            qid = r["id"]
            clean = resps.get(qid, "")
            ml = 100 if tag == "base_v5gold" else \
                max(50, int(base_lens.get(qid, 0) * 0.03))
            g = gates(clean, ml)
            bad = any(g.values())
            for k, v in g.items():
                if v:
                    gate_fail[k] += 1
            if bad:
                invalid += 1
            lens.append(len(clean))
        n = len(raws)
        report[tag] = {
            "n": n,
            "modes": modes,
            "gate_fail_counts": gate_fail,
            "invalid_total": invalid,
            "invalid_rate": round(invalid / n, 4) if n else None,
            "len_min": min(lens) if lens else None,
            "len_mean": round(sum(lens) / n, 1) if n else None,
            "len_max": max(lens) if lens else None,
            "pass_5pct_gate": (invalid / n <= 0.05) if n else None,
        }
    out = G / "gen_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
