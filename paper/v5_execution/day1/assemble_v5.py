#!/usr/bin/env python3
"""
pipeline_v5 数据构建 Day1 总装 (Worker G) —— 双风格合并 / ChatML 渲染 /
双重去污 / 分层分组划分 / manifest。

严格按 paper/v5_plan/v5_优化微调方案.md §4.1-C / §4.6 / §4.7 / §11.1 执行:
  1. 方案 C 双风格合并: 短答臂 = gated_corpus 全量 (system 简洁模式);
     长答臂 = styleC 长答 (system 详细模式); 同 instruction 长短两版同 group_id。
     种子 365 ×3 上采样注入 (质量门之后、划分之前, 衍生物同 group);
     Safety 300 条并入 (占比 <=5% 断言)。
  2. ChatML 渲染: apply_chat_template(system+user, add_generation_prompt=True,
     enable_thinking=False), **保留空 think 块** (E0 协议, 与 Worker F 生成侧一致;
     对 v4 "剥空 think 块" 的有意变更, 见 MANIFEST 偏差声明)。
     prompt+completion > 2048 token 丢弃并计数。
  3. 双重去污: 参照集 A = v4_gold 100 + v5_gold_new100 100;
     参照集 B = sample_data_expanded 465 + general_probe_v5 120。
     token 3-gram Jaccard: J>=0.5 剔除, J in [0.4,0.5) 人工审查队列;
     分参照集计数; B 集剔除 >3% 告警记录 (不中断)。
  4. 分层分组划分 85/10/5 (seed=42): group_id 与 v4 前缀(12)聚类 union-find
     合并分组, 同组同侧; 划分后 test/validation 与 train 交叉检查, 命中整组移回 train。
  5. MANIFEST.md + v5_data_report.json (§4.7 五项)。

输出: data/processed/v5_data/final/{v5_train,v5_validation,v5_test}.jsonl
      + _assembly_sidecar.jsonl + decon_review_queue.jsonl
      + v5_data_report.json + MANIFEST.md
纪律: 不覆盖任何 v4 及更早产物; 全程 CPU。

用法: venv_v5/bin/python pipeline_v5/src/assemble_v5.py
"""

import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "pipeline_v4" / "src"))

import data_build as v4db  # noqa: E402  复用 v4: tokenize/ngrams/jaccard/NgramIndex/normalize_instruction

from transformers import AutoTokenizer  # noqa: E402

# ---------------- 路径 ----------------
D = PROJECT_ROOT / "data/processed/v5_data"
P_GATED = D / "gated_corpus.jsonl"
P_LONG = D / "styleC_long_answers.jsonl"
P_SAMPLING = D / "styleC_longanswer_sampling.jsonl"
P_SEEDS = D / "cleaned_seeds_final.jsonl"
P_SAFETY = D / "safety_refusal_v5.jsonl"
P_GOLD_V4 = PROJECT_ROOT / "data/processed/v4_gold.jsonl"  # 只读引用, 不复制不覆盖
P_GOLD_V5 = D / "v5_gold_new100.jsonl"
P_EXPANDED = PROJECT_ROOT / "data/processed/sample_data_expanded.json"  # 只读引用
P_PROBE = D / "general_probe_v5.json"
P_TOKENIZER = PROJECT_ROOT / "models/Qwen3.6-35B-A3B"

OUT_DIR = D / "final"

SYS_SHORT = "简洁模式:直接给出结论与要点,不超过 300 字"
SYS_LONG = "详细模式:给出完整的结构化分析,包含术语定义、推理过程与实施建议"
EMPTY_THINK = "<think>\n\n</think>\n\n"

RATIOS = {"train": 0.85, "validation": 0.10, "test": 0.05}
SEED = 42
PREFIX_LEN = 12
MAX_LENGTH = 2048
J_DROP = 0.5
J_REVIEW = 0.4
SEED_UPSAMPLE = 3
B_ALERT_RATE = 0.03


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(p: Path):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def percentile(sorted_vals, q):
    """线性插值百分位, sorted_vals 已升序。"""
    if not sorted_vals:
        return 0
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = (n - 1) * q
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def dist_stats(vals):
    s = sorted(vals)
    return {"n": len(s), "mean": round(sum(s) / len(s), 1) if s else 0,
            "p50": round(percentile(s, 0.50), 1), "p95": round(percentile(s, 0.95), 1),
            "p99": round(percentile(s, 0.99), 1), "max": s[-1] if s else 0}


# ---------------- batch 包装修复 ----------------
WRAP_PREFIX_RE = re.compile(r'^\s*\{\s*"index"\s*:\s*\d+\s*,\s*"answer"\s*:\s*"(.*)$', re.DOTALL)


def repair_wrapped_completion(c: str):
    """Worker E 批量 API 响应 JSON 包装残留修复: 提取 "answer" 字段原文。
    返回 (repaired_text, was_wrapped)。无法修复返回 (None, True)。"""
    if not c.lstrip().startswith("{"):
        return c, False
    m = WRAP_PREFIX_RE.match(c)
    if not m:
        return None, True
    body = m.group(1)
    # 去掉尾部闭合引号与可选的 \n}
    body = re.sub(r'"\s*\}?\s*$', "", body)
    if len(body) < 100:
        return None, True
    return body, True


# ---------------- 去污: 最大 Jaccard (精确) ----------------
def max_jaccard_against(instructions, ref_questions):
    """对每条 instruction 求与参照集的最大 3-gram Jaccard。
    精确 brute-force + size-ratio 剪枝 (下界 J_REVIEW, 剪枝对 J>=J_REVIEW 无漏检:
    J <= min(|a|,|b|)/max(|a|,|b|))。按 instruction 去重缓存。
    注: v4 NgramIndex 稀有 token 签名分桶为近似候选, 可能漏掉不共享稀有 token
    的高重叠对 (独立复核实测漏 12 条 J>=0.5), 总装采用精确算法。"""
    ref_grams = [v4db.ngrams(v4db.tokenize(q), 3) for q in ref_questions]
    cache = {}
    out = []
    for instr in instructions:
        if instr in cache:
            out.append(cache[instr])
            continue
        g = v4db.ngrams(v4db.tokenize(instr), 3)
        best_j, best_r = 0.0, None
        for i, og in enumerate(ref_grams):
            if g and og and min(len(g), len(og)) / max(len(g), len(og)) < J_REVIEW:
                continue
            j = v4db.jaccard(g, og)
            if j > best_j:
                best_j, best_r = j, i
        res = (best_j, best_r)
        cache[instr] = res
        out.append(res)
    return out


# ---------------- union-find ----------------
class UF:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        p = self.parent.setdefault(x, x)
        while p != self.parent[p]:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        self.parent[x] = p
        return p

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def main():
    t0 = datetime.now()
    report = {"generated_at": t0.isoformat(timespec="seconds"),
              "builder": "assemble_v5.py (Worker G 总装)",
              "plan_refs": ["§4.1-C", "§4.6", "§4.7", "§4.8", "§11.1"],
              "stages": [], "notes": [], "warnings": []}

    # ---- 0. 输入清单 + sha256 ----
    input_files = {"gated_corpus": P_GATED, "styleC_long_answers": P_LONG,
                   "styleC_longanswer_sampling": P_SAMPLING, "cleaned_seeds_final": P_SEEDS,
                   "safety_refusal_v5": P_SAFETY, "v4_gold(参照A,只读)": P_GOLD_V4,
                   "v5_gold_new100(参照A)": P_GOLD_V5,
                   "sample_data_expanded(参照B,只读)": P_EXPANDED,
                   "general_probe_v5(参照B)": P_PROBE}
    report["inputs"] = {}
    for name, p in input_files.items():
        assert p.exists(), f"输入缺失: {p}"
        report["inputs"][name] = {"path": str(p), "sha256": sha256_file(p)}
    log("输入 sha256 完成")

    gated = load_jsonl(P_GATED)
    longs = load_jsonl(P_LONG)
    sampling = load_jsonl(P_SAMPLING)
    seeds = load_jsonl(P_SEEDS)
    safety = load_jsonl(P_SAFETY)
    gold_v4 = load_jsonl(P_GOLD_V4)
    gold_v5 = load_jsonl(P_GOLD_V5)
    expanded = json.load(open(P_EXPANDED, encoding="utf-8"))
    probe = json.load(open(P_PROBE, encoding="utf-8"))

    n_expanded = sum(len(v) for v in expanded.values())
    report["inputs"]["gated_corpus"]["rows"] = len(gated)
    report["inputs"]["styleC_long_answers"]["rows"] = len(longs)
    report["inputs"]["styleC_longanswer_sampling"]["rows"] = len(sampling)
    report["inputs"]["cleaned_seeds_final"]["rows"] = len(seeds)
    report["inputs"]["safety_refusal_v5"]["rows"] = len(safety)
    report["inputs"]["v4_gold(参照A,只读)"]["rows"] = len(gold_v4)
    report["inputs"]["v5_gold_new100(参照A)"]["rows"] = len(gold_v5)
    report["inputs"]["sample_data_expanded(参照B,只读)"]["rows"] = n_expanded
    report["inputs"]["general_probe_v5(参照B)"]["rows"] = len(probe)

    assert len(gated) == 8613, f"gated 行数异常: {len(gated)}"
    assert len(seeds) == 365, f"seeds 行数异常: {len(seeds)}"
    assert len(safety) == 300, f"safety 行数异常: {len(safety)}"
    assert len(gold_v4) == 100 and len(gold_v5) == 100
    assert n_expanded == 465 and len(probe) == 120

    # styleC 完结偏差声明
    n_samp = len(sampling)
    if len(longs) != n_samp:
        report["notes"].append(
            f"styleC 长答实际 {len(longs)}/{n_samp} 条 (差 {n_samp - len(longs)} 条未完结; "
            f"tmux v5gen_E 已退出, 按任务纪律如实记录并以实际条数继续)")
    log(f"输入: gated={len(gated)} long={len(longs)}/{n_samp} seeds={len(seeds)} "
        f"safety={len(safety)} refA=200 refB={n_expanded + len(probe)}")

    # ---- 1. 组装样本池 (方案 C) ----
    instr2gid = {r["instruction"]: r["group_id"] for r in sampling}
    gid2instr = {}
    for r in sampling:
        gid2instr.setdefault(r["group_id"], r["instruction"])

    pool = []  # {instruction,input,output,subset,style,group_id,origin}

    for r in gated:
        gid = instr2gid.get(r["instruction"]) or \
            "g_" + hashlib.sha1(v4db.normalize_instruction(r["instruction"]).encode()).hexdigest()[:12]
        pool.append({"instruction": r["instruction"], "input": r.get("input", ""),
                     "output": r["output"], "subset": r["subset"], "style": "short",
                     "group_id": gid, "origin": r.get("origin", "gated_corpus")})

    repaired, repair_fail = 0, []
    for r in longs:
        c, was = repair_wrapped_completion(r["completion"])
        if was:
            if c is None:
                repair_fail.append(r["group_id"])
                continue
            repaired += 1
        assert r["group_id"] in gid2instr, f"长答孤儿 group_id: {r['group_id']}"
        pool.append({"instruction": r["instruction"], "input": r.get("input", ""),
                     "output": c, "subset": r["subset"], "style": "long",
                     "group_id": r["group_id"], "origin": "styleC_long"})
    assert not repair_fail, f"包装修复失败: {repair_fail}"
    report["stages"].append({"stage": "styleC_batch_wrapper_repair",
                             "wrapped_and_repaired": repaired})
    log(f"长答臂: {len(longs)} 条 (batch 包装修复 {repaired} 条)")

    # 种子 ×3 上采样 (质量门后、划分前; 衍生物同 group)
    for r in seeds:
        for k in range(SEED_UPSAMPLE):
            pool.append({"instruction": r["instruction"], "input": r.get("input", ""),
                         "output": r["output"], "subset": r["subset"], "style": "short",
                         "group_id": r["group_id"],
                         "origin": f"seed_x{SEED_UPSAMPLE}#{'expanded' if r.get('expanded') else 'survivor'}"})
    log(f"种子注入: {len(seeds)} ×{SEED_UPSAMPLE} = {len(seeds) * SEED_UPSAMPLE}")

    for r in safety:
        pool.append({"instruction": r["instruction"], "input": "",
                     "output": r["completion"], "subset": "safety_refusal",
                     "style": "short", "group_id": "safety_" + r["id"],
                     "origin": f"safety/{r['category']}"})

    n_total = len(pool)
    safety_share = len(safety) / n_total
    assert safety_share <= 0.05, f"safety 占比超限: {safety_share:.4f}"
    report["stages"].append({"stage": "pool_assembly", "total": n_total,
                             "short": sum(1 for s in pool if s["style"] == "short"),
                             "long": sum(1 for s in pool if s["style"] == "long"),
                             "seed_upsample": SEED_UPSAMPLE,
                             "safety_share": round(safety_share, 4)})
    log(f"样本池: {n_total} (safety 占比 {safety_share:.2%} <=5% ✓)")

    # ---- 2. 双重去污 (§4.6) ----
    refs_a = [r["question"] for r in gold_v4] + [r["question"] for r in gold_v5]
    refs_b = ([it["instruction"] for v in expanded.values() for it in v]
              + [q["question"] for q in probe])
    assert len(refs_a) == 200 and len(refs_b) == 585

    log(f"去污参照: A={len(refs_a)} B={len(refs_b)}; 计算最大 Jaccard ...")
    instrs = [s["instruction"] for s in pool]
    ja = max_jaccard_against(instrs, refs_a)
    jb = max_jaccard_against(instrs, refs_b)
    log("Jaccard 计算完成")

    kept, dropped, review = [], [], []
    cnt = {"A_drop": 0, "B_drop": 0, "both_drop": 0, "A_review": 0, "B_review": 0}
    for s, (j_a, r_a), (j_b, r_b) in zip(pool, ja, jb):
        hit_a, hit_b = j_a >= J_DROP, j_b >= J_DROP
        if hit_a or hit_b:
            dropped.append(s)
            cnt["A_drop"] += hit_a
            cnt["B_drop"] += hit_b
            cnt["both_drop"] += hit_a and hit_b
            continue
        band_a = J_REVIEW <= j_a < J_DROP
        band_b = J_REVIEW <= j_b < J_DROP
        if band_a or band_b:
            cnt["A_review"] += band_a
            cnt["B_review"] += band_b
            entry = {"instruction": s["instruction"], "subset": s["subset"],
                     "style": s["style"], "origin": s["origin"]}
            if band_a:
                entry["A"] = {"j": round(j_a, 4), "ref": refs_a[r_a][:120]}
            if band_b:
                entry["B"] = {"j": round(j_b, 4), "ref": refs_b[r_b][:120]}
            review.append(entry)
        kept.append(s)

    b_drop_rate = cnt["B_drop"] / n_total
    decon_rep = {"rule": "token 3-gram Jaccard; J>=0.5 剔除; J∈[0.4,0.5) 人工审查队列; "
                         "精确 brute-force+size-ratio 剪枝 (v4 NgramIndex 近似桶经独立复核发现漏检, 弃用)",
                 "ref_A": {"size": 200, "dropped": cnt["A_drop"], "review_queue": cnt["A_review"]},
                 "ref_B": {"size": 585, "dropped": cnt["B_drop"], "review_queue": cnt["B_review"],
                           "drop_rate": round(b_drop_rate, 4)},
                 "both_sets_hit": cnt["both_drop"],
                 "unique_samples_dropped": len(dropped),
                 "review_queue_total": len(review)}
    if b_drop_rate > B_ALERT_RATE:
        decon_rep["ALERT"] = (f"B 集剔除率 {b_drop_rate:.2%} > 3% 阈值, 触发人工复核告警 "
                              f"(按 §4.6 风险条款记录, 不中断构建)")
        report["warnings"].append(decon_rep["ALERT"])
        log("!! " + decon_rep["ALERT"])
    report["stages"].append({"stage": "dual_decontamination", **decon_rep})
    log(f"去污: 剔除 {len(dropped)} (A={cnt['A_drop']} B={cnt['B_drop']} 双中={cnt['both_drop']}); "
        f"审查队列 {len(review)} (A={cnt['A_review']} B={cnt['B_review']})")

    # ---- 3. 分组 (union-find: group_id + v4 前缀12聚类) ----
    uf = UF()
    for i, s in enumerate(kept):
        gkey = "gid:" + s["group_id"]
        pkey = "pfx:" + v4db.normalize_instruction(s["instruction"])[:PREFIX_LEN]
        uf.find(gkey)
        uf.union(gkey, pkey)
    for s in kept:
        s["root"] = uf.find("gid:" + s["group_id"])
    report["stages"].append({"stage": "grouping",
                             "rule": f"union-find(group_id, normalize(instruction)[:{PREFIX_LEN}])",
                             "n_groups": len({s['root'] for s in kept})})

    # ---- 4. 分层分组划分 85/10/5 (seed=42) ----
    rng = random.Random(SEED)
    root2members = defaultdict(list)
    for s in kept:
        root2members[s["root"]].append(s)
    # 组的层 = 组内多数 subset (平票取字典序小者); 组绝不拆分
    strata = defaultdict(list)
    for root, members in root2members.items():
        sub = sorted(Counter(m["subset"] for m in members).items(),
                     key=lambda kv: (-kv[1], kv[0]))[0][0]
        strata[sub].append(members)

    splits = {"train": [], "validation": [], "test": []}
    group_stats = {"n_groups": 0, "max_group_size": 0}
    for subset in sorted(strata):
        glist = strata[subset]
        rng.shuffle(glist)
        group_stats["n_groups"] += len(glist)
        group_stats["max_group_size"] = max(group_stats["max_group_size"],
                                            max(len(g) for g in glist))
        n = sum(len(g) for g in glist)
        n_train = round(n * RATIOS["train"])
        n_val = round(n * RATIOS["validation"])
        counts = {"train": 0, "validation": 0, "test": 0}
        for g in glist:
            deficits = {"train": n_train - counts["train"],
                        "validation": n_val - counts["validation"]}
            side = max(deficits, key=deficits.get)
            if deficits[side] <= 0:
                side = "test"
            splits[side].extend(g)
            counts[side] += len(g)

    # ---- 5. 划分交叉检查: test/validation vs train, 命中整组移回 train ----
    # 精确 brute-force + size-ratio 剪枝 (同去污, 不用 v4 近似索引)
    train_grams = {}
    for s in splits["train"]:
        if s["instruction"] not in train_grams:
            train_grams[s["instruction"]] = v4db.ngrams(v4db.tokenize(s["instruction"]), 3)
    tg_list = list(train_grams.values())

    def hits_train(instr):
        g = v4db.ngrams(v4db.tokenize(instr), 3)
        for og in tg_list:
            if g and og and min(len(g), len(og)) / max(len(g), len(og)) < J_DROP:
                continue
            if v4db.jaccard(g, og) >= J_DROP:
                return True
        return False

    moved = 0
    for side in ("validation", "test"):
        hit_roots = set()
        for s in splits[side]:
            if hits_train(s["instruction"]):
                hit_roots.add(s["root"])
        if hit_roots:
            stay = [s for s in splits[side] if s["root"] not in hit_roots]
            moved_recs = [s for s in splits[side] if s["root"] in hit_roots]
            moved += len(moved_recs)
            splits["train"].extend(moved_recs)
            splits[side] = stay
    report["stages"].append({"stage": "split_cross_check",
                             "rule": "test/validation vs train 3-gram Jaccard>=0.5, 命中整组移回 train",
                             "moved_back_to_train": moved})
    log(f"划分交叉检查: 移回 train {moved} 条")

    # ---- 6. ChatML 渲染 (保留空 think 块, E0 协议) ----
    log("加载 tokenizer 渲染 ChatML ...")
    tok = AutoTokenizer.from_pretrained(str(P_TOKENIZER))
    sys_by_style = {"short": SYS_SHORT, "long": SYS_LONG}
    suffix_expected = "<|im_start|>assistant\n" + EMPTY_THINK

    def render(s):
        user = s["instruction"]
        if s["input"].strip():
            user += "\n" + s["input"]
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": sys_by_style[s["style"]]},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert prompt.endswith(suffix_expected), "ChatML 模板尾部不符合 E0 协议"
        return {"prompt": prompt, "completion": s["output"], "subset": s["subset"]}

    rendered = {}
    for side in ("train", "validation", "test"):
        recs = [render(s) for s in splits[side]]
        # max_length 门
        texts = [r["prompt"] + r["completion"] for r in recs]
        enc = tok(texts, add_special_tokens=False)["input_ids"]
        kept_recs, dropped_len = [], 0
        for r, ids, s in zip(recs, enc, splits[side]):
            if len(ids) <= MAX_LENGTH:
                kept_recs.append((r, len(ids), s))
            else:
                dropped_len += 1
        rendered[side] = kept_recs
        report["stages"].append({"stage": f"chatml_render+max_length[{side}]",
                                 "max_length": MAX_LENGTH, "dropped_overlength": dropped_len,
                                 "kept": len(kept_recs)})
        log(f"{side}: 渲染 {len(recs)} 条, >{MAX_LENGTH}token 丢弃 {dropped_len}, 保留 {len(kept_recs)}")

    report["stages"].append({
        "stage": "chatml_render_protocol",
        "template": "apply_chat_template(system+user, add_generation_prompt=True, enable_thinking=False)",
        "empty_think_block": "保留 (E0 协议, 与 Worker F v5_gen.py 生成侧一致; "
                             "v4 为剥除, 此为有意变更, 训练/推理格式一致优先)",
        "prompt_suffix": suffix_expected,
        "eos": "由 TRL 附加 (completion_only_loss 机制, 同 v4)"})

    # ---- 7. 统计 ----
    final = {}
    for side in ("train", "validation", "test"):
        recs = rendered[side]
        lens = [x[1] for x in recs]
        styles = Counter(x[2]["style"] for x in recs)
        subsets = Counter(x[2]["subset"] for x in recs)
        final[side] = {"n": len(recs), "style": dict(styles),
                       "subset_dist": dict(sorted(subsets.items())),
                       "token_len(prompt+completion)": dist_stats(lens)}
    tot = sum(v["n"] for v in final.values())
    style_all = Counter()
    for v in final.values():
        style_all.update(v["style"])
    report["final"] = {"total": tot, "splits": final,
                       "style_ratio": {"short": style_all["short"], "long": style_all["long"],
                                       "short:long": f"{style_all['short'] / tot:.3f}:{style_all['long'] / tot:.3f}"},
                       "split_params": {"ratios": RATIOS, "seed": SEED,
                                        "stratify": "subset (组多数层)",
                                        "group": "union-find(group_id, prefix12)",
                                        **group_stats},
                       "degradation": [
                           "样本级无 source/chunk 标识, 沿用 v4 退化: 归一化 instruction 前缀12聚类 "
                           "(v5 与 group_id 做 union-find 合并, 不弱于 v4)",
                           "跨 subset 同前缀组按组内多数 subset 归层, 分层为近似分层"]}

    # 裁决项 #15: max_length 锁 2048 是否安全
    tr = final["train"]["token_len(prompt+completion)"]
    all_lens = sorted(x[1] for side in rendered for x in rendered[side])
    dropped_len_total = sum(st.get("dropped_overlength", 0) for st in report["stages"]
                            if st["stage"].startswith("chatml_render+"))
    decision15 = {
        "train_p95": tr["p95"], "train_p99": tr["p99"], "train_max": tr["max"],
        "all_splits_p99": round(percentile(all_lens, 0.99), 1),
        "overlength_dropped": dropped_len_total,
        "conclusion": ("安全" if tr["p99"] <= MAX_LENGTH else "不安全"),
        "statement": ""}
    decision15["statement"] = (
        f"train 集 prompt+completion token p95={tr['p95']}, p99={tr['p99']}, max={tr['max']}; "
        f"全长尾由 >2048 硬门剔除 {dropped_len_total} 条 (占比 "
        f"{dropped_len_total / (tot + dropped_len_total):.4%})。"
        + ("p99 ≤ 2048, max_length 锁 2048 不造成静默截断, 裁决项 #15 结论: 安全, 锁定 2048。"
           if decision15["conclusion"] == "安全" else
           "p99 > 2048, max_length 锁 2048 不安全, 需上调或截断策略评审。"))
    report["decision_15_max_length"] = decision15
    log(decision15["statement"])

    # ---- 8. 落盘 ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_files = {}
    for side, fname in (("train", "v5_train.jsonl"), ("validation", "v5_validation.jsonl"),
                        ("test", "v5_test.jsonl")):
        p = OUT_DIR / fname
        with open(p, "w", encoding="utf-8") as f:
            for r, _, _ in rendered[side]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out_files[fname] = {"path": str(p), "rows": len(rendered[side]), "sha256": sha256_file(p)}

    # sidecar (审计: group/style/split 映射, 供独立复核)
    p = OUT_DIR / "_assembly_sidecar.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for side in ("train", "validation", "test"):
            for r, ntok, s in rendered[side]:
                f.write(json.dumps({"split": side, "group_id": s["group_id"], "root": s["root"],
                                    "style": s["style"], "subset": s["subset"],
                                    "origin": s["origin"], "instruction": s["instruction"],
                                    "n_tokens": ntok}, ensure_ascii=False) + "\n")
    out_files["_assembly_sidecar.jsonl"] = {"path": str(p), "rows": tot, "sha256": sha256_file(p)}

    p = OUT_DIR / "decon_review_queue.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for e in review:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    out_files["decon_review_queue.jsonl"] = {"path": str(p), "rows": len(review),
                                             "sha256": sha256_file(p)}

    report["outputs"] = out_files
    report["elapsed_sec"] = round((datetime.now() - t0).total_seconds(), 1)

    p = OUT_DIR / "v5_data_report.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log(f"报告落盘: {p}")

    print(json.dumps({"train": final["train"]["n"], "validation": final["validation"]["n"],
                      "test": final["test"]["n"], "style": report["final"]["style_ratio"],
                      "decon": decon_rep, "decision15": decision15["conclusion"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
