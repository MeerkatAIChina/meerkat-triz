#!/usr/bin/env python
"""
pipeline_v4 数据构建 (TRIZ v4) —— 干净重建版。

输入:
  - v2 语料  data/processed/checkpoint_corpus_sft_v2/corpus_sft_checkpoint.json
             (10327 条, 字段 instruction/input/output/subset, 无 source/chunk 标识)
  - ariz boost data/processed/corpus_sft_v2_ariz_boost/ariz_guidance.json
             (674 条 list, 字段 instruction/input/output, 无 subset → 标记 ariz_guidance)

质量门 (按顺序, 每门计数进报告):
  1. think_strip        去 output 中 <think>...</think> 块
  2. min_output_chars   output < 150 字符丢弃
  3. exact_dedup        归一化 instruction 精确去重; 同 instruction 多答案冲突组整组丢弃
  4. near_dedup         instruction 级 token 3-gram Jaccard >= 0.7 判重 (stdlib 稀有 token
                        签名分桶 + 桶内两两比较, 不依赖 sklearn)
  5. decontamination    与金标集 v4_gold.jsonl 的 question 做 3-gram Jaccard >= 0.5 检查,
                        命中者剔除; 金标集不存在则跳过并在报告注明
  6. rebalance          concept_explanation / innovation_assessment 各 cap, 其余子集全保留。
                        选取策略由 config rebalance.strategy 决定:
                        - longest_first: 按 output 长度降序截取 (v4 初版行为; 已证实致
                          keyword/concept_explanation 退化, 见 E2 归因与 stats_review §4.3)
                        - term_coverage_random (v4.1 默认): TRIZ 术语言表贪心最大覆盖
                          (60% 配额) + 长度三桶分层随机补足 (40%), 见
                          paper/v5_plan/sec1_data.md §4
  7. 分层划分 85/10/5 (seed=42): 样本无 source/chunk 标识 → 退化为按归一化 instruction
     前缀 (12 字符) 聚类分组, 同组同侧, 按子集分层分配; 报告中注明退化
  8. cross_check        划分后 test/validation 与 train 做 3-gram Jaccard >= 0.5 交叉检查,
                        命中者移回 train (报告计数)
  9. max_length_tokens  prompt+completion tokenize 后 > max_length(2048) 丢弃
                        (避免 TRL keep_start 截断产生残缺 completion)

输出 (严格遵守 configs/data_v4.json 的 prompt/completion 契约):
  data/processed/v4_train.jsonl / v4_validation.jsonl / v4_test.jsonl
  每行 {"prompt": <完整 ChatML 前缀, 含 system/user, 以 '<|im_start|>assistant\\n' 结尾>,
        "completion": <目标回答>, "subset": <子集标记>}
  ChatML 渲染: models/Qwen3.6-35B-A3B tokenizer (CPU), enable_thinking=False 后
  剥掉空 think 块 ("<think>\\n\\n</think>\\n\\n") —— 与旧 v2 构建的 think 处理一致,
  train.py/TRL 不应用 chat template, 只做字符串拼接并在 completion 末尾自动附加 EOS。

报告: results/v4_data_report.json (每门输入/剔除/保留计数、最终子集分布、
      划分统计、去污命中数、配置快照)。

用法:
  venv_v5/bin/python pipeline_v4/src/data_build.py --config pipeline_v4/configs/data_v4.json
"""

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = "pipeline_v4/configs/data_v4.json"

# ChatML 渲染时剥掉的空 think 块 (与 scripts/run_corpus_sft_v2.py L43 一致)
EMPTY_THINK = "<think>\n\n</think>\n\n"

# token 化: 拉丁字母/数字连续段为一个 token, 每个 CJK 汉字为一个 token
# (中英文混合语料的 stdlib 折衷; 不用 jieba/sklearn)
TOKEN_RE = re.compile(r"[a-z0-9]+|[一-鿿]")
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
WS_RE = re.compile(r"\s+")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


# ---------------- 归一化与 n-gram ----------------

def normalize_instruction(text: str) -> str:
    """NFKC + 小写 + 去除全部空白, 用于精确去重/冲突检测/分组。"""
    t = unicodedata.normalize("NFKC", text or "").lower()
    return WS_RE.sub("", t)


def normalize_output(text: str) -> str:
    """冲突检测时比较 output 用的归一化 (忽略空白差异)。"""
    t = unicodedata.normalize("NFKC", text or "").lower()
    return WS_RE.sub("", t)


def tokenize(text: str):
    return TOKEN_RE.findall((text or "").lower())


def ngrams(tokens, n=3):
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / (len(a) + len(b) - inter)


class NgramIndex:
    """稀有 token 签名分桶索引: 每文档取 df 最小的 K 个 token 作为桶键,
    候选对只在共享桶键的文档间产生, 再以 size-ratio 剪枝后算精确 Jaccard。"""

    def __init__(self, sig_k=3, ngram_n=3):
        self.sig_k = sig_k
        self.ngram_n = ngram_n
        self.buckets = defaultdict(list)   # token -> [doc_id]
        self.grams = {}                    # doc_id -> 3-gram set
        self._df = Counter()
        self._pending = {}                 # doc_id -> tokens (未入桶, 等 df 统计完)

    def _signature(self, tokens):
        return sorted(set(tokens), key=lambda t: (self._df[t], t))[: self.sig_k]

    def add_corpus(self, doc_ids_tokens):
        """批量建索引: 先统计 df 再统一入桶。"""
        for doc_id, tokens in doc_ids_tokens:
            self.grams[doc_id] = ngrams(tokens, self.ngram_n)
            self._df.update(set(tokens))
            self._pending[doc_id] = tokens
        for doc_id, tokens in self._pending.items():
            for t in self._signature(tokens):
                self.buckets[t].append(doc_id)
        self._pending = {}

    def add_one(self, doc_id, tokens):
        self.grams[doc_id] = ngrams(tokens, self.ngram_n)
        self._df.update(set(tokens))
        for t in self._signature(tokens):
            self.buckets[t].append(doc_id)

    def candidates(self, tokens):
        sig = sorted(set(tokens), key=lambda t: (self._df.get(t, 0), t))[: self.sig_k]
        seen = set()
        for t in sig:
            seen.update(self.buckets.get(t, ()))
        return seen

    def find_match(self, doc_id_exclude, tokens, threshold):
        """在索引中找与 tokens 的 3-gram Jaccard >= threshold 的任一文档 id。"""
        g = ngrams(tokens, self.ngram_n)
        for other in self.candidates(tokens):
            if other == doc_id_exclude:
                continue
            og = self.grams[other]
            # size-ratio 剪枝: Jaccard <= min/max
            if g and og and min(len(g), len(og)) / max(len(g), len(og)) < threshold:
                continue
            if jaccard(g, og) >= threshold:
                return other
        return None


# ---------------- 数据加载 ----------------

def load_v2_corpus(path: Path):
    with open(path, encoding="utf-8") as f:
        ckpt = json.load(f)
    samples = ckpt["samples"]
    out = []
    for s in samples:
        out.append({
            "instruction": s.get("instruction", ""),
            "input": s.get("input", "") or "",
            "output": s.get("output", ""),
            "subset": s.get("subset", "unknown"),
            "origin": "v2_corpus",
        })
    return out


def load_ariz_boost(path: Path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"] if isinstance(data, dict) and "samples" in data else data
    out = []
    for s in samples:
        out.append({
            "instruction": s.get("instruction", ""),
            "input": s.get("input", "") or "",
            "output": s.get("output", ""),
            "subset": "ariz_guidance",   # boost 文件无 subset 字段, 按文件名归属
            "origin": "ariz_boost",
        })
    return out


# ---------------- 质量门 ----------------

def gate_think_strip(samples):
    n = 0
    for s in samples:
        if "<think>" in s["output"]:
            s["output"] = THINK_RE.sub("", s["output"]).strip()
            n += 1
    return samples, n


def gate_min_output_chars(samples, min_chars):
    kept = [s for s in samples if len(s["output"]) >= min_chars]
    return kept, len(samples) - len(kept)


def gate_exact_dedup_and_conflict(samples):
    """归一化 instruction 分组:
    - 组内 output 归一化后全同 → 保留 1 条 (重复计数);
    - 组内 output 不一致 → 冲突组, 整组丢弃 (旧管线同题不同答混入的修复)。"""
    groups = defaultdict(list)
    for s in samples:
        groups[normalize_instruction(s["instruction"])].append(s)
    kept, dup, conflict_groups, conflict_samples = [], 0, 0, 0
    for key, grp in groups.items():
        if not key:
            conflict_groups += 1
            conflict_samples += len(grp)
            continue
        outs = {normalize_output(s["output"]) for s in grp}
        if len(outs) == 1:
            kept.append(grp[0])
            dup += len(grp) - 1
        else:
            conflict_groups += 1
            conflict_samples += len(grp)
    return kept, dup, conflict_groups, conflict_samples


def gate_near_dedup(samples, threshold, sig_k, ngram_n):
    """instruction 级 3-gram Jaccard >= threshold 贪心去重 (按输入顺序保留先见者)。"""
    idx = NgramIndex(sig_k=sig_k, ngram_n=ngram_n)
    kept, dropped = [], 0
    for i, s in enumerate(samples):
        toks = tokenize(s["instruction"])
        if idx.find_match(doc_id_exclude=None, tokens=toks, threshold=threshold) is not None:
            dropped += 1
            continue
        idx.add_one(f"s{i}", toks)
        kept.append(s)
    return kept, dropped


def ngram_filter_against(samples, ref_questions, threshold, sig_k, ngram_n):
    """剔除与任一参考文本 (金标 question 或 train instruction) 3-gram Jaccard
    >= threshold 的样本。返回 (保留, 命中样本列表)。"""
    idx = NgramIndex(sig_k=sig_k, ngram_n=ngram_n)
    idx.add_corpus([(f"r{i}", tokenize(q)) for i, q in enumerate(ref_questions)])
    kept, hits = [], []
    for s in samples:
        toks = tokenize(s["instruction"])
        m = idx.find_match(doc_id_exclude=None, tokens=toks, threshold=threshold)
        if m is not None:
            hits.append({"instruction": s["instruction"][:120], "matched_ref": m})
        else:
            kept.append(s)
    return kept, hits


def _term_coverage_select(grp, cap, coverage_quota, length_buckets, glossary, rng):
    """两阶段选取 (paper/v5_plan/sec1_data.md §4, 回填 v4.1):

    阶段 1: 用 TRIZ 工具术语言表对 output 做贪心最大覆盖选取, 配额
            cap*coverage_quota —— 保底保留术语枚举式答案 (v4 初版"长答案
            优先"把这类答案洗掉了, 致 keyword/concept_explanation 退化:
            v4 0.4356 vs v2 0.5187, E2 真缺失 8 词次)。覆盖饱和后继续
            以 1/(1+doc_count) 加权扶持低频词样本, 直到配额填满或含词
            样本选尽 (对应 v5_plan §4 "每词 >=30 条" 的保底意图)。
    阶段 2: 剩余配额在未选样本中按 output 长度分 length_buckets 个等频桶,
            按桶占比分层随机抽取 (rng 固定种子) —— 消除长度方向选择偏差
            (E2: longest_first 保留集显著更长, MWU p=5.3e-25)。
    """
    n = len(grp)
    term_sets = [{t for t in glossary if t in s["output"]} for s in grp]

    # ---- 阶段 1: 贪心最大覆盖 + 低频词扶持, 填至 quota1 ----
    quota1 = int(cap * coverage_quota)
    covered, chosen = set(), set()
    doc_count = Counter()
    while len(chosen) < quota1:
        best_i, best_key = None, None
        for i in range(n):
            if i in chosen or not term_sets[i]:
                continue
            new_terms = term_sets[i] - covered
            # 主键: 新增覆盖词数; 次键: 低频词加权和 (覆盖饱和后的扶持信号);
            # 再次: 自身覆盖词总数; 末键: -i 保证确定性
            key = (len(new_terms),
                   sum(1.0 / (1 + doc_count[t]) for t in term_sets[i]),
                   len(term_sets[i]), -i)
            if best_key is None or key > best_key:
                best_i, best_key = i, key
        if best_i is None:
            break  # 含词样本已选尽, 余量转阶段 2
        chosen.add(best_i)
        covered |= term_sets[best_i]
        doc_count.update(term_sets[best_i])
    phase1_count = len(chosen)

    # ---- 阶段 2: 长度等频分桶 + 按桶占比分层随机 ----
    quota2 = cap - phase1_count
    pool = [i for i in range(n) if i not in chosen]
    if quota2 >= len(pool):
        chosen.update(pool)
    else:
        pool_sorted = sorted(pool, key=lambda i: len(grp[i]["output"]))
        m = len(pool_sorted)
        buckets = [pool_sorted[k * m // length_buckets:(k + 1) * m // length_buckets]
                   for k in range(length_buckets)]
        raw = [quota2 * len(b) / m for b in buckets]
        take = [int(x) for x in raw]
        # 余项按小数部分从大到小补给各桶, 保证 sum(take)==quota2
        for k in sorted(range(length_buckets),
                        key=lambda k: raw[k] - take[k], reverse=True)[:quota2 - sum(take)]:
            take[k] += 1
        for k, bucket in enumerate(buckets):
            chosen.update(rng.sample(bucket, min(take[k], len(bucket))))

    detail = {
        "phase1_term_coverage": phase1_count,
        "phase2_stratified_random": len(chosen) - phase1_count,
        "covered_terms": sorted(covered),
        "term_doc_count_in_kept": {t: sum(1 for i in chosen if t in term_sets[i])
                                   for t in glossary},
    }
    return [grp[i] for i in chosen], detail


def gate_rebalance(samples, caps, strategy="longest_first", seed=42,
                   coverage_quota=0.6, length_buckets=3, glossary=None):
    """子集再平衡。strategy:
      - longest_first: 按 output 长度降序截取 (v4 初版行为, 保留作回滚/对照)
      - term_coverage_random: 术语覆盖优先 + 长度分层随机 (v4.1 修复,
        见 _term_coverage_select docstring)
    返回 (kept, info); info["dropped_per_subset"] 为各子集丢弃数。
    """
    rng = random.Random(seed)
    by_subset = defaultdict(list)
    for s in samples:
        by_subset[s["subset"]].append(s)
    kept, dropped_per, per_subset = [], {}, {}
    for subset, grp in by_subset.items():
        cap = caps.get(subset)
        if cap is None or len(grp) <= cap:
            kept.extend(grp)
            if cap is not None:
                dropped_per[subset] = 0
            continue
        if strategy == "longest_first":
            chosen = sorted(grp, key=lambda s: len(s["output"]), reverse=True)[:cap]
            detail = {}
        elif strategy == "term_coverage_random":
            chosen, detail = _term_coverage_select(
                grp, cap, coverage_quota, length_buckets, glossary or [], rng)
        else:
            raise ValueError(f"未知 rebalance 策略: {strategy!r}")
        kept.extend(chosen)
        dropped_per[subset] = len(grp) - cap
        per_subset[subset] = detail
    return kept, {"dropped_per_subset": dropped_per,
                  "strategy": strategy, "per_subset": per_subset}


# ---------------- 分层分组划分 ----------------

def group_key(sample, prefix_len):
    """无 source/chunk 标识 → 退化为归一化 instruction 前缀聚类。"""
    return normalize_instruction(sample["instruction"])[:prefix_len]


def stratified_group_split(samples, ratios, seed, prefix_len):
    """按子集分层; 层内按 group_key 分组; 组为单位 shuffle 后按目标比例分配到
    train/validation/test, 保证同组样本落在同一侧。"""
    rng = random.Random(seed)
    by_subset = defaultdict(list)
    for s in samples:
        by_subset[s["subset"]].append(s)
    splits = {"train": [], "validation": [], "test": []}
    group_stats = {"n_groups": 0, "max_group_size": 0}
    for subset in sorted(by_subset):
        grp = by_subset[subset]
        groups = defaultdict(list)
        for s in grp:
            groups[group_key(s, prefix_len)].append(s)
        glist = list(groups.values())
        rng.shuffle(glist)
        group_stats["n_groups"] += len(glist)
        group_stats["max_group_size"] = max(
            group_stats["max_group_size"], max((len(g) for g in glist), default=0))
        n = len(grp)
        n_train = round(n * ratios["train"])
        n_val = round(n * ratios["validation"])
        counts = {"train": 0, "validation": 0, "test": 0}
        for g in glist:
            # 分配给 (当前计数/目标) 缺口最大的一侧, test 兜底
            deficits = {
                "train": n_train - counts["train"],
                "validation": n_val - counts["validation"],
            }
            side = max(deficits, key=deficits.get)
            if deficits[side] <= 0:
                side = "test"
            splits[side].extend(g)
            counts[side] += len(g)
    return splits, group_stats


# ---------------- ChatML 渲染 ----------------

def render_chatml(samples, tokenizer, system_message):
    """prompt = apply_chat_template(system+user, add_generation_prompt=True,
    enable_thinking=False) 后剥空 think 块; completion = output。"""
    stripped = 0
    out = []
    for s in samples:
        user = s["instruction"]
        if s["input"].strip():
            user += "\n" + s["input"]
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if EMPTY_THINK in prompt:
            prompt = prompt.replace(EMPTY_THINK, "")
            stripped += 1
        out.append({"prompt": prompt, "completion": s["output"], "subset": s["subset"]})
    return out, stripped


def filter_max_length(records, tokenizer, max_length):
    """prompt+completion token 数 > max_length 的丢弃 (防 TRL keep_start 截断)。"""
    texts = [r["prompt"] + r["completion"] for r in records]
    enc = tokenizer(texts, add_special_tokens=False)["input_ids"]
    kept, dropped = [], 0
    for r, ids in zip(records, enc):
        if len(ids) <= max_length:
            kept.append(r)
        else:
            dropped += 1
    return kept, dropped


def write_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def subset_dist(samples):
    return dict(Counter(s["subset"] for s in samples))


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="pipeline_v4 数据构建")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    args = ap.parse_args()
    with open(resolve(args.config), encoding="utf-8") as f:
        cfg = json.load(f)
    b = cfg["build"]

    report = {"generated_at": datetime.now().isoformat(timespec="seconds"),
              "config_snapshot": cfg, "gates": [], "notes": []}

    # ---- 1. 加载 ----
    v2 = load_v2_corpus(resolve(b["v2_corpus"]))
    boost = load_ariz_boost(resolve(b["ariz_boost"]))
    samples = v2 + boost
    report["inputs"] = {"v2_corpus": len(v2), "ariz_boost": len(boost),
                        "total": len(samples)}
    log(f"加载: v2={len(v2)} + ariz_boost={len(boost)} = {len(samples)}")

    # ---- 2. think strip ----
    samples, n_think = gate_think_strip(samples)
    report["gates"].append({"gate": "think_strip", "affected": n_think,
                            "kept": len(samples)})

    # ---- 3. output 长度门 ----
    samples, n_short = gate_min_output_chars(samples, b["min_output_chars"])
    report["gates"].append({"gate": "min_output_chars",
                            "threshold": b["min_output_chars"],
                            "dropped": n_short, "kept": len(samples)})
    log(f"think_strip={n_think} | 短 output 剔除={n_short} → {len(samples)}")

    # ---- 4. 精确去重 + 冲突检测 ----
    samples, n_dup, n_cgrp, n_csmp = gate_exact_dedup_and_conflict(samples)
    report["gates"].append({"gate": "exact_dedup_conflict",
                            "duplicates_dropped": n_dup,
                            "conflict_groups_dropped": n_cgrp,
                            "conflict_samples_dropped": n_csmp,
                            "kept": len(samples)})
    log(f"精确去重={n_dup} | 冲突组={n_cgrp} (样本 {n_csmp}) → {len(samples)}")

    # ---- 5. 近重复去重 ----
    samples, n_near = gate_near_dedup(samples, b["near_dedup"]["jaccard_threshold"],
                                      b["near_dedup"]["signature_k"],
                                      b["near_dedup"]["ngram_n"])
    report["gates"].append({"gate": "near_dedup",
                            "jaccard_threshold": b["near_dedup"]["jaccard_threshold"],
                            "dropped": n_near, "kept": len(samples)})
    log(f"近重复剔除={n_near} → {len(samples)}")

    # ---- 6. 去污 (vs 金标集) ----
    gold_path = resolve(b["decontamination"]["gold_file"])
    deco = {"gate": "decontamination",
            "jaccard_threshold": b["decontamination"]["jaccard_threshold"]}
    if gold_path.is_file():
        gold_qs = []
        with open(gold_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    gold_qs.append(json.loads(line)["question"])
        samples, hits = ngram_filter_against(
            samples, gold_qs, b["decontamination"]["jaccard_threshold"],
            b["near_dedup"]["signature_k"], b["near_dedup"]["ngram_n"])
        deco.update({"status": "applied", "gold_questions": len(gold_qs),
                     "dropped": len(hits), "kept": len(samples),
                     "hits": hits[:50]})
        log(f"去污: 金标 {len(gold_qs)} 题, 命中剔除={len(hits)} → {len(samples)}")
    else:
        deco.update({"status": "skipped", "reason": f"金标集不存在: {gold_path}",
                     "dropped": 0, "kept": len(samples)})
        report["notes"].append("金标集不存在, 去污跳过; 金标集生成后需重跑本构建")
        log(f"[注意] 金标集 {gold_path} 不存在, 去污跳过")
    report["gates"].append(deco)

    # ---- 7. 子集再平衡 ----
    dist_before = subset_dist(samples)
    reb = b["rebalance"]
    samples, reb_info = gate_rebalance(
        samples, reb["caps"],
        strategy=reb.get("strategy", "longest_first"),
        seed=reb.get("seed", b["split"]["seed"]),
        coverage_quota=reb.get("coverage_quota", 0.6),
        length_buckets=reb.get("length_buckets", 3),
        glossary=reb.get("glossary"))
    dropped_per = reb_info["dropped_per_subset"]
    gate_entry = {"gate": "rebalance_caps", "strategy": reb_info["strategy"],
                  "caps": reb["caps"],
                  "distribution_before": dist_before,
                  "dropped_per_subset": dropped_per,
                  "dropped_total": sum(dropped_per.values()),
                  "kept": len(samples)}
    if reb.get("glossary"):
        gate_entry["glossary"] = reb["glossary"]  # 术语言表落 manifest (v5_plan §4)
    if any(reb_info["per_subset"].values()):
        gate_entry["per_subset_detail"] = reb_info["per_subset"]
    report["gates"].append(gate_entry)
    log(f"再平衡({reb_info['strategy']}): {dropped_per} → {len(samples)}")

    # ---- 8. 分层分组划分 ----
    splits, group_stats = stratified_group_split(
        samples, b["split"]["ratios"], b["split"]["seed"],
        b["split"]["fallback_prefix_len"])
    split_mode = "instruction_prefix_fallback"
    report["split"] = {
        "mode": split_mode,
        "degraded": True,
        "degraded_reason": "v2 语料与 ariz boost 均无 source/chunk 标识, "
                           "退化为按归一化 instruction 前缀聚类分组",
        "prefix_len": b["split"]["fallback_prefix_len"],
        "seed": b["split"]["seed"], "ratios": b["split"]["ratios"],
        **group_stats,
        "counts": {k: len(v) for k, v in splits.items()},
    }
    log(f"划分(退化模式: 前缀聚类): "
        f"train={len(splits['train'])} val={len(splits['validation'])} "
        f"test={len(splits['test'])}")

    # ---- 9. test/val 与 train 交叉检查, 命中者移回 train ----
    train_instr = [s["instruction"] for s in splits["train"]]
    moved = 0
    for side in ("validation", "test"):
        kept_side, hits = ngram_filter_against(
            splits[side], train_instr, b["split"]["cross_check_jaccard"],
            b["near_dedup"]["signature_k"], b["near_dedup"]["ngram_n"])
        leaked = [s for s in splits[side] if s not in kept_side]
        splits[side] = kept_side
        splits["train"].extend(leaked)
        moved += len(hits)
    report["split"]["cross_check_jaccard"] = b["split"]["cross_check_jaccard"]
    report["split"]["cross_check_moved_back_to_train"] = moved
    log(f"交叉检查: {moved} 条移回 train")

    # ---- 10. ChatML 渲染 (CPU tokenizer) ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(resolve(b["chatml"]["tokenizer_path"])), trust_remote_code=True)
    final = {}
    think_stripped_total = 0
    for side in ("train", "validation", "test"):
        recs, n_strip = render_chatml(splits[side], tok, b["chatml"]["system_message"])
        think_stripped_total += n_strip
        recs, n_long = filter_max_length(recs, tok, b["max_length_tokens"])
        final[side] = recs
        report["gates"].append({"gate": "max_length_tokens", "split": side,
                                "threshold": b["max_length_tokens"],
                                "dropped": n_long, "kept": len(recs)})
        log(f"{side}: 渲染 {len(splits[side])} → 超长剔除 {n_long} → {len(recs)}")
    report["chatml"] = {"empty_think_stripped": think_stripped_total,
                        "tokenizer": b["chatml"]["tokenizer_path"],
                        "enable_thinking": False}

    # ---- 11. 写盘 ----
    write_jsonl(final["train"], resolve(cfg["output_train"]))
    write_jsonl(final["validation"], resolve(cfg["output_validation"]))
    write_jsonl(final["test"], resolve(cfg["output_test"]))
    report["final"] = {
        "counts": {k: len(v) for k, v in final.items()},
        "subset_distribution": {
            side: dict(Counter(r["subset"] for r in recs))
            for side, recs in final.items()},
        "outputs": {"train": cfg["output_train"],
                    "validation": cfg["output_validation"],
                    "test": cfg["output_test"]},
    }

    rep_path = resolve(cfg["report_path"])
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log(f"报告: {rep_path}")
    log(f"完成: train={len(final['train'])} val={len(final['validation'])} "
        f"test={len(final['test'])}")


if __name__ == "__main__":
    main()
