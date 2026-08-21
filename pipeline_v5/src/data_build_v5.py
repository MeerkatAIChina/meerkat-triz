#!/usr/bin/env python3
"""
pipeline_v5 数据构建 Day1 CPU 主链 (Worker A) —— 不含 API 生成 / 去污 / 划分 / manifest。

严格按 paper/v5_plan/v5_优化微调方案.md §4.2 / §4.3 / §4.5 / §11.1 执行:
  Stage1 种子二次清洗三规则 (§4.2):
    R1 同题不同答冲突组整组丢弃; R2 公式化收尾(末80字符频次>=3 模板串黑名单)截句,
       截后<150字符整条弃, 残留率必须=0; R3 output<150字符整条弃。
  Stage2 语料质量门 v5 版 (§4.3/§4.5):
    沿用 v4 门1-3 (think剥离 / 长度150 / 精确去重+冲突组 / 3-gram Jaccard>=0.7 近重复);
    ariz boost 594 条 (674 - 80 评测预留, 预留集合由 sample_data.json vs
    sample_data_expanded.json 的 ariz_guidance 差集确定性还原, seed=42 与 v3 构建一致);
    门5 修复: concept_explanation/innovation_assessment cap 1500->2500,
       选取 = 术语言表贪心最大覆盖 60% (15词, 每词>=30保底) + 长度三桶分层随机 40%;
    ariz_guidance 占比封顶 15%, 超限随机下采 (优先丢与 v2 语料 Jaccard 高者)。
  Stage3 IP 边界检查 (Owner 裁决项 #2): 样本级无 source/chunk 标识 -> 如实记录
    "无法区分" 并按方案 "如实继承" 条款声明; chunk 级 (triz_corpus.jsonl) 有
    source_path, 但样本->chunk 映射在构建期已丢失, 不编造区分结果。
  Stage4 styleC 长答抽样: 按子集分层随机 40% instruction 清单 (同 instruction
    标记 group_id), 供下一阶段 API 生成长答。

复用: pipeline_v4/src/data_build.py 的归一化 / n-gram / 门函数 (import, 不改 v4 任何文件)。
断点续跑: 每 stage 完成写 data/processed/v5_data/_checkpoints/<stage>.done,
  --resume 时跳过已完成 stage (确定性 seed=42, 重跑结果一致)。

用法:
  venv_v5/bin/python pipeline_v5/src/data_build_v5.py [--resume] [--force]
"""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "pipeline_v4" / "src"))

import data_build as v4db  # noqa: E402  (复用 v4: normalize/tokenize/ngrams/jaccard/NgramIndex/门函数)

# ---------------- 路径 ----------------
P_V2_CORPUS = PROJECT_ROOT / "data/processed/checkpoint_corpus_sft_v2/corpus_sft_checkpoint.json"
P_ARIZ_BOOST = PROJECT_ROOT / "data/processed/corpus_sft_v2_ariz_boost/ariz_guidance.json"
P_SYNTH_DIR = PROJECT_ROOT / "data/processed/synthetic"
P_SAMPLE_DATA = PROJECT_ROOT / "data/sample_data.json"
P_EXPANDED = PROJECT_ROOT / "data/processed/sample_data_expanded.json"
P_GOLD = PROJECT_ROOT / "data/processed/v4_gold.jsonl"
P_PROBE = PROJECT_ROOT / "data/processed/general_probe.json"
P_TRIZ_CORPUS = PROJECT_ROOT / "data/processed/corpus/triz_corpus.jsonl"

OUT_DIR = PROJECT_ROOT / "data/processed/v5_data"
CKPT_DIR = OUT_DIR / "_checkpoints"

SEED = 42
MIN_OUTPUT_CHARS = 150
CAP_V5 = {"concept_explanation": 2500, "innovation_assessment": 2500}
COVERAGE_RATIO = 0.60          # 术语覆盖配额占 cap 的 60%
TERM_FLOOR = 30                # 每词保底条数
ARIZ_SHARE_CAP = 0.15
STYLEC_SAMPLE_RATIO = 0.40

# 术语言表 v1.0 (§4.3, 15 词; "理想解/IFR" 为一词双写)
# 匹配口径: NFKC + 小写 + 去空白 后的子串匹配 (instruction+output 拼接)
TERM_TABLE = [
    ("功能分析", ["功能分析"]),
    ("矛盾矩阵", ["矛盾矩阵"]),
    ("S曲线", ["s曲线", "s-曲线", "s—曲线"]),
    ("40个发明原理", ["40个发明原理", "40大发明原理"]),
    ("技术矛盾", ["技术矛盾"]),
    ("物理矛盾", ["物理矛盾"]),
    ("理想解/IFR", ["理想解", "ifr"]),
    ("资源分析", ["资源分析"]),
    ("物场分析", ["物场分析", "物-场分析", "物—场分析"]),
    ("因果链", ["因果链"]),
    ("Functionality", ["functionality"]),
    ("Cost", ["cost"]),
    ("改善参数", ["改善参数"]),
    ("恶化参数", ["恶化参数"]),
    ("通用工程参数", ["通用工程参数"]),
]

SENT_END = "。!！?？\n"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_text(s: str) -> str:
    """术語匹配用归一化: NFKC + 小写 + 去全部空白 (与 v4 normalize_instruction 同口径)。"""
    return v4db.normalize_instruction(s or "")


def group_id_of(instruction: str) -> str:
    return hashlib.sha1(norm_text(instruction).encode("utf-8")).hexdigest()[:12]


def write_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def done_mark(stage: str):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    (CKPT_DIR / f"{stage}.done").write_text(datetime.now().isoformat(), encoding="utf-8")


def is_done(stage: str) -> bool:
    return (CKPT_DIR / f"{stage}.done").exists()


# ---------------- Stage 1: 种子二次清洗三规则 ----------------

def load_seeds():
    """385 条种子 = synthetic/*_synthetic.json 中 source=='seed' 的条目。"""
    seeds = []
    for fp in sorted(P_SYNTH_DIR.glob("*_synthetic.json")):
        subset = fp.name.replace("_synthetic.json", "")
        for x in json.load(open(fp, encoding="utf-8")):
            if x.get("source") == "seed":
                seeds.append({
                    "instruction": x.get("instruction", ""),
                    "input": x.get("input", "") or "",
                    "output": x.get("output", ""),
                    "subset": x.get("subset", subset),
                })
    return seeds


def last_sentence(text: str) -> str:
    t = (text or "").rstrip()
    if not t:
        return ""
    i = max(t.rfind(c, 0, len(t) - 1) for c in SENT_END)
    return t[i + 1:].strip() if i >= 0 else t


def strip_last_sentence(text: str) -> str:
    """截去最后一个句子 (保留前文及其句末标点); 无句界则返回空串。"""
    t = (text or "").rstrip()
    i = max(t.rfind(c, 0, len(t) - 1) for c in SENT_END)
    return t[: i + 1] if i >= 0 else ""


def stage1_seed_clean():
    seeds = load_seeds()
    n_in = len(seeds)
    rep = {"input": n_in,
           "under_60_chars_raw": sum(1 for s in seeds if len(s["output"]) < 60),
           "under_150_chars_raw": sum(1 for s in seeds if len(s["output"]) < 150),
           "output_len_p50_raw": sorted(len(s["output"]) for s in seeds)[n_in // 2],
           "output_len_max_raw": max(len(s["output"]) for s in seeds)}

    # R1 同题不同答冲突组整组丢弃 (复用 v4 门函数: 精确去重 + 冲突组整组弃)
    seeds, dup, cg, cs = v4db.gate_exact_dedup_and_conflict(seeds)
    rep.update({"rule1_exact_dup_dropped": dup,
                "rule1_conflict_groups": cg,
                "rule1_conflict_samples_dropped": cs,
                "after_rule1": len(seeds)})

    # R2 公式化收尾: 黑名单数据驱动
    #   B1 = 末 80 字符原串频次 >= 3 (方案原文口径)
    #   B2 = 末句原串频次 >= 3 且长度 >= 8 (辅助口径, 防止前 80 字符内混入异文漏检;
    #        截句动作对两者一致, 残留率=0 对两者同时验证)
    tails = Counter(s["output"][-80:] for s in seeds)
    b1 = {t for t, c in tails.items() if c >= 3}
    sents = Counter(last_sentence(s["output"]) for s in seeds)
    b2 = {t for t, c in sents.items() if c >= 3 and len(t) >= 8}
    rep["rule2_blacklist_b1_tail80"] = sorted(b1)
    rep["rule2_blacklist_b2_last_sentence"] = sorted(b2)

    kept, n_trunc, n_drop_after_trunc = [], 0, 0
    for s in seeds:
        out, truncated = s["output"], False
        for _ in range(10):
            if out[-80:] in b1 or last_sentence(out) in b2:
                out = strip_last_sentence(out)
                truncated = True
            else:
                break
        if truncated:
            n_trunc += 1
        if len(out) < MIN_OUTPUT_CHARS:
            if truncated:
                n_drop_after_trunc += 1
            else:
                kept.append(s)  # 不可能是此分支 (未截断且<150 留给 R3 计数)
            continue
        s2 = dict(s)
        s2["output"] = out
        kept.append(s2)
    rep["rule2_truncated"] = n_trunc
    rep["rule2_dropped_short_after_truncation"] = n_drop_after_trunc
    rep["after_rule2"] = len(kept)

    # 残留率必须 = 0 (B1/B2 双口径)
    resid = [s for s in kept if s["output"][-80:] in b1 or last_sentence(s["output"]) in b2]
    rep["rule2_residual_count"] = len(resid)
    assert not resid, f"公式化收尾残留 {len(resid)} 条, 违反 残留率=0 硬约束"

    # R3 output < 150 字符整条弃
    kept3 = [s for s in kept if len(s["output"]) >= MIN_OUTPUT_CHARS]
    rep["rule3_dropped_short"] = len(kept) - len(kept3)
    rep["output_cleaned_seeds"] = len(kept3)

    for s in kept3:
        s["group_id"] = group_id_of(s["instruction"])
    out_path = OUT_DIR / "cleaned_seeds.jsonl"
    write_jsonl(kept3, out_path)
    rep["output_path"] = str(out_path)
    rep["output_sha256"] = sha256_file(out_path)
    (CKPT_DIR / "stage1_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    done_mark("stage1")
    log(f"Stage1 种子清洗: {n_in} -> {len(kept3)} "
        f"(冲突组 {cg} 组/{cs} 条, 截尾 {n_trunc}, 截后弃 {n_drop_after_trunc}, "
        f"R3 弃 {rep['rule3_dropped_short']}, 残留 {len(resid)})")
    return rep


# ---------------- Stage 2: 语料质量门 v5 ----------------

def ariz_reserved_keys():
    """确定性还原 v3 的 80 条评测预留: expanded.ariz_guidance - sample_data.ariz_guidance。"""
    sd = json.load(open(P_SAMPLE_DATA, encoding="utf-8"))
    exp = json.load(open(P_EXPANDED, encoding="utf-8"))
    orig = {v4db.normalize_instruction(x["instruction"]) for x in sd["ariz_guidance"]}
    expd = [v4db.normalize_instruction(x["instruction"]) for x in exp["ariz_guidance"]]
    reserved = set(expd) - orig
    dup_note = len(expd) - len(set(expd))
    return reserved, {"expanded_ariz": len(expd), "original_ariz": len(orig),
                      "reserved_unique": len(reserved), "expanded_internal_dup": dup_note}


def term_set_of(sample) -> set:
    t = norm_text(sample["instruction"] + " " + sample["output"])
    return {name for name, pats in TERM_TABLE if any(p in t for p in pats)}


def gate5_rebalance_v5(samples, caps, rng):
    """cap 2500; 60% 术语言表贪心最大覆盖 (每词>=30保底) + 40% 长度三桶分层随机。"""
    by_subset = defaultdict(list)
    for s in samples:
        by_subset[s["subset"]].append(s)
    kept, detail = [], {}
    for subset, grp in sorted(by_subset.items()):
        cap = caps.get(subset)
        if cap is None or len(grp) <= cap:
            kept.extend(grp)
            if cap is not None:
                detail[subset] = {"before": len(grp), "cap": cap, "kept": len(grp),
                                  "dropped": 0, "selection": "below_cap_keep_all"}
            continue
        quota_cov = int(cap * COVERAGE_RATIO)          # 1500
        quota_rand = cap - quota_cov                   # 1000
        for s in grp:
            s["_terms"] = term_set_of(s)
        # --- 阶段 1: 贪心最大覆盖 ---
        covered = Counter()
        selected, pool = [], list(grp)
        while len(selected) < quota_cov and pool:
            best_gain = max(len(s["_terms"] - set(covered)) for s in pool)
            cands = [s for s in pool if len(s["_terms"] - set(covered)) == best_gain]
            pick = rng.choice(cands)
            pool.remove(pick)
            selected.append(pick)
            covered.update(pick["_terms"])
        # --- 每词 >=30 保底修补 ---
        floor_unmet = []
        for name, _ in TERM_TABLE:
            have = covered.get(name, 0)
            if have >= TERM_FLOOR:
                continue
            donors = [s for s in pool if name in s["_terms"]]
            rng.shuffle(donors)
            for d in donors:
                if covered.get(name, 0) >= TERM_FLOOR:
                    break
                # 牺牲品: 已选样本中不含任何 <=floor 词者, 优先零术语样本
                def protection(x):
                    return sum(1 for t in x["_terms"] if covered.get(t, 0) <= TERM_FLOOR)
                victims = sorted((x for x in selected if protection(x) == 0),
                                 key=lambda x: (-len(x["_terms"])))
                if victims:
                    v = victims[0]
                    selected.remove(v)
                    for t in v["_terms"]:
                        covered[t] -= 1
                    pool.append(v)
                elif len(selected) >= quota_cov:
                    break
                pool.remove(d)
                selected.append(d)
                covered.update(d["_terms"])
            if covered.get(name, 0) < TERM_FLOOR:
                floor_unmet.append({"term": name, "got": covered.get(name, 0),
                                    "pool_available": len(donors)})
        coverage_counts = {name: covered.get(name, 0) for name, _ in TERM_TABLE}
        # --- 阶段 2: 剩余 40% 长度三桶分层随机 ---
        pool_sorted = sorted(pool, key=lambda s: len(s["output"]))
        n_pool = len(pool_sorted)
        b1, b2 = n_pool // 3, 2 * n_pool // 3
        buckets = [pool_sorted[:b1], pool_sorted[b1:b2], pool_sorted[b2:]]
        bucket_sizes = [len(b) for b in buckets]
        rand_pick = []
        remaining = quota_rand
        for bi, bucket in enumerate(buckets):
            k = round(quota_rand * len(bucket) / n_pool) if n_pool else 0
            k = min(k, len(bucket), remaining)
            rand_pick.extend(rng.sample(bucket, k))
            remaining -= k
        # 舍入余量从未被选池补齐
        if remaining > 0:
            rest = [s for s in pool if id(s) not in {id(x) for x in rand_pick}]
            rand_pick.extend(rng.sample(rest, min(remaining, len(rest))))
        final = selected + rand_pick
        for s in final:
            s.pop("_terms", None)
        for s in pool:
            s.pop("_terms", None)
        kept.extend(final)
        detail[subset] = {
            "before": len(grp), "cap": cap, "kept": len(final),
            "dropped": len(grp) - len(final),
            "selection": "term_coverage_greedy_60pct+length3bucket_random_40pct",
            "coverage_quota": quota_cov, "random_quota": quota_rand,
            "coverage_selected": len(selected), "random_selected": len(rand_pick),
            "term_coverage": coverage_counts, "term_floor": TERM_FLOOR,
            "floor_unmet": floor_unmet,
            "length_bucket_sizes_remaining_pool": bucket_sizes,
        }
    return kept, detail


def ariz_share_cap(samples, rng):
    """ariz_guidance 占比封顶 15%; 超限优先丢与 v2 语料 (origin=v2_corpus)
    instruction 3-gram Jaccard 最高的 ariz_boost 来源样本。"""
    total = len(samples)
    ariz = [s for s in samples if s["subset"] == "ariz_guidance"]
    share = len(ariz) / total if total else 0.0
    rep = {"total_before": total, "ariz_before": len(ariz), "share_before": round(share, 4),
           "cap": ARIZ_SHARE_CAP, "downsampled": 0}
    if share <= ARIZ_SHARE_CAP:
        rep["action"] = "below_cap_no_action"
        return samples, rep
    n_drop = len(ariz) - int(ARIZ_SHARE_CAP * total)
    v2_ref = [s for s in samples if s.get("origin") == "v2_corpus"]
    idx = v4db.NgramIndex(sig_k=3, ngram_n=3)
    idx.add_corpus([(f"v{i}", v4db.tokenize(s["instruction"])) for i, s in enumerate(v2_ref)])
    boost = [s for s in ariz if s.get("origin") == "ariz_boost"]
    scored = []
    for s in boost:
        toks = v4db.tokenize(s["instruction"])
        g = v4db.ngrams(toks, 3)
        best = 0.0
        for other in idx.candidates(toks):
            best = max(best, v4db.jaccard(g, idx.grams[other]))
        scored.append((best, s))
    scored.sort(key=lambda x: -x[0])
    drop_ids = {id(s) for _, s in scored[:n_drop]}
    # 若 boost 不够丢, 剩余缺口随机丢 v2 来源 ariz (记录)
    if len(drop_ids) < n_drop:
        rest = [s for s in ariz if id(s) not in drop_ids]
        drop_ids.update(id(s) for s in rng.sample(rest, n_drop - len(drop_ids)))
        rep["fallback_random_from_v2_origin"] = n_drop - len(scored[:n_drop])
    kept = [s for s in samples if id(s) not in drop_ids]
    rep.update({"downsampled": len(drop_ids), "ariz_after": len(kept) and
                sum(1 for s in kept if s["subset"] == "ariz_guidance"),
                "total_after": len(kept),
                "share_after": round(sum(1 for s in kept if s["subset"] == "ariz_guidance")
                                     / len(kept), 4),
                "action": "downsample_high_jaccard_first"})
    return kept, rep


def stage2_corpus_gates():
    rep = {}
    v2 = v4db.load_v2_corpus(P_V2_CORPUS)
    ariz_all = v4db.load_ariz_boost(P_ARIZ_BOOST)
    reserved, rinfo = ariz_reserved_keys()
    ariz_train = [s for s in ariz_all
                  if v4db.normalize_instruction(s["instruction"]) not in reserved]
    rep["ariz_reserve"] = {**rinfo, "boost_total": len(ariz_all),
                           "reserved_matched_in_boost": len(ariz_all) - len(ariz_train),
                           "boost_for_training": len(ariz_train)}
    samples = v2 + ariz_train
    rep["input"] = {"v2_corpus": len(v2), "ariz_boost_train": len(ariz_train),
                    "total": len(samples)}

    samples, n_think = v4db.gate_think_strip(samples)
    rep["gate1_think_strip"] = {"stripped": n_think, "kept": len(samples)}
    samples, n_short = v4db.gate_min_output_chars(samples, MIN_OUTPUT_CHARS)
    rep["gate2_min_output_chars"] = {"dropped": n_short, "kept": len(samples)}
    samples, dup, cg, cs = v4db.gate_exact_dedup_and_conflict(samples)
    rep["gate3_exact_dedup_conflict"] = {"exact_dup_dropped": dup,
                                         "conflict_groups": cg,
                                         "conflict_samples_dropped": cs,
                                         "kept": len(samples)}
    samples, n_near = v4db.gate_near_dedup(samples, threshold=0.7, sig_k=3, ngram_n=3)
    rep["gate4_near_dedup"] = {"threshold": 0.7, "dropped": n_near, "kept": len(samples)}

    rng = random.Random(SEED)
    samples, g5 = gate5_rebalance_v5(samples, CAP_V5, rng)
    rep["gate5_rebalance_v5"] = g5
    rep["gate5_kept_total"] = len(samples)

    samples, acap = ariz_share_cap(samples, rng)
    rep["ariz_share_cap"] = acap
    rep["final"] = {"total": len(samples),
                    "subset_dist": dict(Counter(s["subset"] for s in samples))}

    out_path = OUT_DIR / "gated_corpus.jsonl"
    write_jsonl(samples, out_path)
    rep["output_path"] = str(out_path)
    rep["output_sha256"] = sha256_file(out_path)
    (CKPT_DIR / "stage2_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    done_mark("stage2")
    log(f"Stage2 语料门: {rep['input']['total']} -> {len(samples)} "
        f"(near-dedup 丢 {n_near}, 门5 { {k: v['kept'] for k, v in g5.items()} })")
    return rep


# ---------------- Stage 3: IP 边界检查 ----------------

def stage3_ip_check():
    corpus = read_jsonl(OUT_DIR / "gated_corpus.jsonl")
    keys = set()
    for s in corpus[:2000]:
        keys.update(s.keys())
    id_fields = sorted(k for k in keys if any(w in k.lower()
                       for w in ("source", "chunk", "file", "path", "doc")))
    chunk_dirs = Counter()
    with open(P_TRIZ_CORPUS, encoding="utf-8") as f:
        for line in f:
            m = json.loads(line).get("metadata") or {}
            sp = m.get("source_path", "")
            top = sp.split("/")[1] if sp.startswith("TRIZ-raw/") and "/" in sp else sp
            chunk_dirs[top] += 1
    # 疑似客户案例目录 (名称含公司/项目代号); 仅作记录, 不做样本级剔除
    customer_like = {d: c for d, c in chunk_dirs.items()
                     if any(w in d for w in ("大华", "阳光电源", "大赛", "案例"))}
    rep = {
        "ruling": "Owner 裁决项 #2 推荐值: 仅用公开教材 chunk, 剔除客户案例 chunk",
        "sample_level_identifiers": id_fields,
        "sample_level_distinguishable": bool(id_fields),
        "chunk_level": {"triz_corpus_chunks": sum(chunk_dirs.values()),
                        "has_source_path": True,
                        "top_dirs": dict(chunk_dirs),
                        "customer_like_dirs": customer_like},
        "declaration": (
            "v2 语料样本与 ariz boost 样本仅有 subset/instruction/input/output 字段, "
            "无 source/chunk 标识; 语料构建脚本 (scripts/run_corpus_sft_v2.py + "
            "utils/corpus_to_sft.py) 在生成 SFT 样本时未保留 chunk 映射, 样本->chunk "
            "无法回溯。结论: 样本级【无法区分】公开教材 chunk 与客户案例 chunk 派生样本。"
            "按方案 §4.7 ④ '如实继承' 条款: 本阶段不剔除任何样本, 在报告中如实声明该退化; "
            "chunk 级 source_path 存在 (triz_corpus.jsonl 3914 chunks), 疑似客户案例目录"
            f" {list(customer_like)} 共 {sum(customer_like.values())} chunks, "
            "若后续需要严格执行 IP 剔除, 须重建带 chunk 映射的语料 (超出 Day1 本工作包范围)。"),
    }
    (CKPT_DIR / "stage3_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    done_mark("stage3")
    log("Stage3 IP 检查: 样本级无标识 -> 无法区分, 已如实声明")
    return rep


# ---------------- Stage 4: styleC 长答抽样 ----------------

def stage4_stylec_sampling():
    corpus = read_jsonl(OUT_DIR / "gated_corpus.jsonl")
    rng = random.Random(SEED)
    by_subset = defaultdict(list)
    for s in corpus:
        by_subset[s["subset"]].append(s)
    picked, dist = [], {}
    for subset, grp in sorted(by_subset.items()):
        grp = sorted(grp, key=lambda s: group_id_of(s["instruction"]))
        k = int(round(len(grp) * STYLEC_SAMPLE_RATIO))
        sel = rng.sample(grp, k)
        dist[subset] = {"subset_total": len(grp), "sampled": k,
                        "ratio": round(k / len(grp), 4)}
        for s in sel:
            picked.append({
                "group_id": group_id_of(s["instruction"]),
                "subset": subset,
                "instruction": s["instruction"],
                "input": s.get("input", ""),
                "short_output_chars": len(s["output"]),
                "target_style": "long",
                "target_chars": [1200, 2500],
            })
    out_path = OUT_DIR / "styleC_longanswer_sampling.jsonl"
    write_jsonl(picked, out_path)
    rep = {"ratio_target": STYLEC_SAMPLE_RATIO, "seed": SEED, "per_subset": dist,
           "total_sampled": len(picked), "corpus_total": len(corpus),
           "output_path": str(out_path), "output_sha256": sha256_file(out_path)}
    (CKPT_DIR / "stage4_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    done_mark("stage4")
    log(f"Stage4 styleC 抽样: {len(corpus)} -> {len(picked)} 条长答生成清单")
    return rep


# ---------------- 汇总报告 ----------------

def pct(vals, q):
    vals = sorted(vals)
    if not vals:
        return 0
    i = min(len(vals) - 1, int(q * len(vals)))
    return vals[i]


def final_report(r1, r2, r3, r4):
    corpus = read_jsonl(OUT_DIR / "gated_corpus.jsonl")
    lens = [len(s["output"]) for s in corpus]
    inputs = {}
    for name, p in [("v2_corpus_ckpt", P_V2_CORPUS), ("ariz_boost", P_ARIZ_BOOST),
                    ("sample_data", P_SAMPLE_DATA), ("sample_data_expanded", P_EXPANDED),
                    ("v4_gold", P_GOLD), ("general_probe", P_PROBE),
                    ("triz_corpus", P_TRIZ_CORPUS)]:
        inputs[name] = {"path": str(p), "sha256": sha256_file(p)}
    inputs["seeds_synthetic_dir"] = {"path": str(P_SYNTH_DIR),
                                     "note": "6 个 *_synthetic.json, source=='seed' 合计 385"}
    outputs = {}
    for name in ["gated_corpus.jsonl", "cleaned_seeds.jsonl",
                 "styleC_longanswer_sampling.jsonl"]:
        p = OUT_DIR / name
        outputs[name] = {"path": str(p), "sha256": sha256_file(p),
                         "lines": sum(1 for _ in open(p, encoding="utf-8"))}
    rep = {
        "worker": "WorkerA_day1_data_main_chain",
        "generated_at": datetime.now().isoformat(),
        "spec": "paper/v5_plan/v5_优化微调方案.md §4.2/4.3/4.5 + §11.1",
        "seed": SEED,
        "inputs": inputs,
        "outputs": outputs,
        "seed_cleaning_s4_2": r1,
        "corpus_gates_s4_3_4_5": r2,
        "ip_boundary_ruling2": r3,
        "styleC_sampling": r4,
        "gated_corpus_output_chars": {
            "mean": round(sum(lens) / len(lens), 1), "p50": pct(lens, 0.50),
            "p95": pct(lens, 0.95), "p99": pct(lens, 0.99), "max": max(lens),
            "note": "字符级; token 级 2048 检查在后续 ChatML 渲染阶段 (本工作包不做划分/渲染)"},
        "deferred_stages": ["双重去污(金标100+eval2扩充465+探针30)", "分层分组划分85/10/5",
                            "manifest 制度化", "长答 API 生成", "Safety-Refusal 300 条生成"],
    }
    out = OUT_DIR / "v5_gates_report.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"汇总报告 -> {out}")
    return rep


def load_ckpt_report(stage):
    return json.loads((CKPT_DIR / f"{stage}_report.json").read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if args.force:
        for f in CKPT_DIR.glob("*.done"):
            f.unlink()

    def run(stage, fn):
        if args.resume and is_done(stage):
            log(f"{stage} 已完成, 跳过 (--resume)")
            return load_ckpt_report(stage)
        return fn()

    r1 = run("stage1", stage1_seed_clean)
    r2 = run("stage2", stage2_corpus_gates)
    r3 = run("stage3", stage3_ip_check)
    r4 = run("stage4", stage4_stylec_sampling)
    final_report(r1, r2, r3, r4)
    log("Day1 数据门主链完成")


if __name__ == "__main__":
    main()
