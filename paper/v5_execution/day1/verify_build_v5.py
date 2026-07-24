#!/usr/bin/env python3
"""
pipeline_v5 独立复核脚本 (Worker G 步骤5) —— 与 assemble_v5.py 对账。

独立性: 不 import v4/v5 任何构建代码, tokenize/ngram/Jaccard/包装修复均为
本脚本自包含实现; 仅从原始输入与 final/ 产物重新计算:
  1. 去污命中率对账: 重建预去污样本池 (gated + styleC + 种子×3 + safety),
     独立计算 vs 参照集 A/B 的最大 3-gram Jaccard, 剔除/审查队列计数须与
     v5_data_report.json 完全一致, 不一致即 FAIL。
  2. 划分完整性: 仅从 final 三文件出发 —— 同归一化 instruction 前缀12聚类、
     同 instruction 长短双风格、种子 ×3 副本, 全部必须同侧; 违规即 FAIL。
  3. 终态安全网: final 每条记录 user 文本 vs A/B J<0.5; prompt 以
     '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' 结尾 (E0 空 think 保留);
     completion 非空; prompt+completion ≤2048 token。
  4. 行数/sha256 与报告对账。

退出码: 0=PASS, 1=FAIL。
用法: venv_v5/bin/python pipeline_v5/src/verify_build_v5.py
"""

import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
D = PROJECT_ROOT / "data/processed/v5_data"
OUT = D / "final"

TOKEN_RE = re.compile(r"[a-z0-9]+|[一-鿿]")
WS_RE = re.compile(r"\s+")
J_DROP, J_REVIEW = 0.5, 0.4
SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

failures = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg, flush=True)
    if not cond:
        failures.append(msg)


def tokenize(text):
    return TOKEN_RE.findall((text or "").lower())


def ngrams(tokens, n=3):
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter) if inter else 0.0


def norm_instr(t):
    return WS_RE.sub("", unicodedata.normalize("NFKC", t or "").lower())


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


WRAP_RE = re.compile(r'^\s*\{\s*"index"\s*:\s*\d+\s*,\s*"answer"\s*:\s*"(.*)$', re.DOTALL)


def unwrap(c):
    if not c.lstrip().startswith("{"):
        return c
    m = WRAP_RE.match(c)
    assert m, "包装无法修复"
    return re.sub(r'"\s*\}?\s*$', "", m.group(1))


def max_j(instr_grams_list, ref_grams_list):
    """独立 brute-force + size-ratio 剪枝 (下界 J_REVIEW)。"""
    out = []
    for g in instr_grams_list:
        best = 0.0
        for og in ref_grams_list:
            if g and og and min(len(g), len(og)) / max(len(g), len(og)) < J_REVIEW:
                continue
            j = jaccard(g, og)
            if j > best:
                best = j
        out.append(best)
    return out


def main():
    report = json.load(open(OUT / "v5_data_report.json", encoding="utf-8"))

    # ---- 1. 重建预去污池 (独立) ----
    gated = load_jsonl(D / "gated_corpus.jsonl")
    longs = load_jsonl(D / "styleC_long_answers.jsonl")
    seeds = load_jsonl(D / "cleaned_seeds_final.jsonl")
    safety = load_jsonl(D / "safety_refusal_v5.jsonl")
    pool_instr = ([r["instruction"] for r in gated]
                  + [r["instruction"] for r in longs]
                  + [r["instruction"] for r in seeds for _ in range(3)]
                  + [r["instruction"] for r in safety])
    pool_rep = next(s for s in report["stages"] if s["stage"] == "pool_assembly")
    check(len(pool_instr) == pool_rep["total"],
          f"预去污池规模: 复核 {len(pool_instr)} vs 报告 {pool_rep['total']}")

    gold_v4 = load_jsonl(PROJECT_ROOT / "data/processed/v4_gold.jsonl")
    gold_v5 = load_jsonl(D / "v5_gold_new100.jsonl")
    expanded = json.load(open(PROJECT_ROOT / "data/processed/sample_data_expanded.json", encoding="utf-8"))
    probe = json.load(open(D / "general_probe_v5.json", encoding="utf-8"))
    refs_a = [r["question"] for r in gold_v4] + [r["question"] for r in gold_v5]
    refs_b = ([it["instruction"] for v in expanded.values() for it in v]
              + [q["question"] for q in probe])
    check(len(refs_a) == 200 and len(refs_b) == 585, f"参照集规模 A={len(refs_a)} B={len(refs_b)}")

    print("独立计算 Jaccard (池×A) ...", flush=True)
    ja = max_j([ngrams(tokenize(x)) for x in pool_instr], [ngrams(tokenize(q)) for q in refs_a])
    print("独立计算 Jaccard (池×B) ...", flush=True)
    jb = max_j([ngrams(tokenize(x)) for x in pool_instr], [ngrams(tokenize(q)) for q in refs_b])

    a_drop = sum(1 for j in ja if j >= J_DROP)
    b_drop = sum(1 for j in jb if j >= J_DROP)
    both = sum(1 for x, y in zip(ja, jb) if x >= J_DROP and y >= J_DROP)
    uniq = sum(1 for x, y in zip(ja, jb) if x >= J_DROP or y >= J_DROP)
    a_rev = sum(1 for x, y in zip(ja, jb) if J_REVIEW <= x < J_DROP and y < J_DROP)
    b_rev = sum(1 for x, y in zip(ja, jb) if J_REVIEW <= y < J_DROP and x < J_DROP)
    rev_total = sum(1 for x, y in zip(ja, jb)
                    if (J_REVIEW <= x < J_DROP or J_REVIEW <= y < J_DROP) and x < J_DROP and y < J_DROP)

    decon = next(s for s in report["stages"] if s["stage"] == "dual_decontamination")
    check(a_drop == decon["ref_A"]["dropped"],
          f"A 集剔除: 复核 {a_drop} vs 报告 {decon['ref_A']['dropped']}")
    check(b_drop == decon["ref_B"]["dropped"],
          f"B 集剔除: 复核 {b_drop} vs 报告 {decon['ref_B']['dropped']}")
    check(both == decon["both_sets_hit"], f"双集同中: 复核 {both} vs 报告 {decon['both_sets_hit']}")
    check(uniq == decon["unique_samples_dropped"],
          f"唯一剔除样本: 复核 {uniq} vs 报告 {decon['unique_samples_dropped']}")
    check(a_rev == decon["ref_A"]["review_queue"],
          f"A 审查队列: 复核 {a_rev} vs 报告 {decon['ref_A']['review_queue']}")
    check(b_rev == decon["ref_B"]["review_queue"],
          f"B 审查队列: 复核 {b_rev} vs 报告 {decon['ref_B']['review_queue']}")
    check(rev_total == decon["review_queue_total"],
          f"审查队列总数: 复核 {rev_total} vs 报告 {decon['review_queue_total']}")

    # ---- 2. final 三文件: 行数/sha256/格式/终态去污安全网 ----
    finals = {}
    for side, fname in (("train", "v5_train.jsonl"), ("validation", "v5_validation.jsonl"),
                        ("test", "v5_test.jsonl")):
        p = OUT / fname
        recs = load_jsonl(p)
        finals[side] = recs
        check(len(recs) == report["outputs"][fname]["rows"],
              f"{fname} 行数: {len(recs)} vs 报告 {report['outputs'][fname]['rows']}")
        check(sha256_file(p) == report["outputs"][fname]["sha256"], f"{fname} sha256 对账")

    n_suffix = sum(1 for recs in finals.values() for r in recs if r["prompt"].endswith(SUFFIX))
    n_all = sum(len(v) for v in finals.values())
    check(n_suffix == n_all, f"E0 prompt 尾部 (空 think 保留): {n_suffix}/{n_all}")
    check(all(r["completion"].strip() for recs in finals.values() for r in recs),
          "completion 全部非空")
    check(all(set(r.keys()) == {"prompt", "completion", "subset"}
              for recs in finals.values() for r in recs), "schema = v4 prompt/completion/subset")

    # 终态安全网: user 文本 vs A/B
    user_re = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>\n<\|im_start\|>assistant", re.DOTALL)
    max_seen = 0.0
    leak = 0
    ref_grams = [ngrams(tokenize(q)) for q in refs_a + refs_b]
    for recs in finals.values():
        for r in recs:
            m = user_re.search(r["prompt"])
            g = ngrams(tokenize(m.group(1)))
            for og in ref_grams:
                if g and og and min(len(g), len(og)) / max(len(g), len(og)) < J_DROP:
                    continue
                j = jaccard(g, og)
                max_seen = max(max_seen, j)
                if j >= J_DROP:
                    leak += 1
    check(leak == 0, f"终态去污安全网: final 全量 vs A+B, J>=0.5 泄漏 {leak} 条 (max J={max_seen:.3f})")

    # ---- 3. 划分完整性 (仅从 final 数据) ----
    seen = {}   # norm_prefix12 -> split ; norm_full -> split
    violations = []
    for side, recs in finals.items():
        for r in recs:
            m = user_re.search(r["prompt"])
            user = m.group(1)
            npfx = norm_instr(user)[:12]
            nfull = norm_instr(user)
            if npfx in seen and seen[npfx] != side:
                violations.append(("prefix12", user[:40], seen[npfx], side))
            if nfull in seen and seen[nfull] != side:
                violations.append(("instruction", user[:40], seen[nfull], side))
            seen[npfx] = side
            seen[nfull] = side
    check(not violations, f"划分完整性: 前缀12聚类/同instruction 跨侧违规 {len(violations)} 起")

    # ---- 4. token 长度 ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(PROJECT_ROOT / "models/Qwen3.6-35B-A3B"))
    over = 0
    for recs in finals.values():
        texts = [r["prompt"] + r["completion"] for r in recs]
        for ids in tok(texts, add_special_tokens=False)["input_ids"]:
            if len(ids) > 2048:
                over += 1
    check(over == 0, f"max_length: final 全部 ≤2048 token (超长者 {over})")

    # ---- 结论 ----
    print("=" * 60, flush=True)
    if failures:
        print(f"VERIFY FAIL: {len(failures)} 项不一致", flush=True)
        for f_ in failures:
            print("  - " + f_, flush=True)
        sys.exit(1)
    print("VERIFY PASS: 去污命中率与划分完整性全部对账一致", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
