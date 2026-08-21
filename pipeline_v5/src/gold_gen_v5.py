#!/usr/bin/env python
"""
pipeline_v5 金标扩题 (v5_gold_101-200, 新增 100 题)。

方案依据: v5_优化微调方案.md §6.5 —— 原 100 题 (v4_gold_000-099) 钉住不动;
新增 100 题按 6 子集等比: principle +20 / contradiction +20 / ariz +20 /
case +15 / concept +15 / innovation +10; 沿用 v4 gold_gen.py 管线与 schema
(question / reference_answer>=200字 / 5-8 期望关键词必须在 reference 出现 /
source chunk id)。

与 v4 gold_gen.py 的差异:
  1. 输出 v5_gold_new100.jsonl, id 从 v5_gold_101 起编号;
  2. **question 级 3-gram Jaccard >= 0.5 去重**: 与 v4_gold.jsonl 既有 100 题
     及本批已接受题比对, 命中剔除重抽 (tokenize/ngrams/jaccard 逐行抄自
     pipeline_v4/src/data_build.py L63/L91-113, 保证口径一致);
  3. 排除 v4_gold.jsonl 已用过的 source chunk (避免同 chunk 再生近似题);
  4. 抽检队列 v5_gold_review.md 带自动规则初检标记 + 待人工标记。

健壮性同 v4: RPM=3 限速, 指数退避重试, 解析失败丢弃补抽, 逐条追加断点续跑。

用法:
  venv_v5/bin/python pipeline_v5/src/gold_gen_v5.py \
      --config pipeline_v5/configs/eval_v5_gold.json
  # 试跑 (2 题, 产物到 /tmp):
  venv_v5/bin/python pipeline_v5/src/gold_gen_v5.py --config ... --limit 2 \
      --output /tmp/v5_gold_trial.jsonl --review-file /tmp/v5_gold_review_trial.md
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = "pipeline_v5/configs/eval_v5_gold.json"

SUBSET_TASKS = {
    "principle_recommendation":
        "基于材料出 1 道 TRIZ 发明原理推荐题: 给出一个具体技术矛盾情境, "
        "要求回答应推荐哪些发明原理并说明理由。",
    "contradiction_analysis":
        "基于材料出 1 道技术矛盾/物理矛盾分析题: 给出一个工程问题, "
        "要求识别并表述其中的技术矛盾 (改善参数 vs 恶化参数) 或物理矛盾。",
    "ariz_guidance":
        "基于材料出 1 道 ARIZ 算法应用题: 给出一个发明问题, "
        "要求按 ARIZ 关键步骤 (问题分析→矛盾构建→理想解IFR→资源分析→方案评估) 引导求解。",
    "case_generation":
        "基于材料出 1 道创新案例分析题: 针对给定行业/产品情境, "
        "要求运用 TRIZ 方法生成并分析创新解决方案案例。",
    "concept_explanation":
        "基于材料出 1 道 TRIZ 概念解释题: 要求准确解释材料中出现的 TRIZ/创新方法概念 "
        "(如矛盾矩阵、理想度、S曲线、物场分析等) 及其应用。",
    "innovation_assessment":
        "基于材料出 1 道创新方案评估题: 给出一个技术方案, "
        "要求从 TRIZ 视角评估其创新性、可行性与潜在矛盾。",
}

GEN_SYSTEM = (
    "你是 TRIZ 领域评测出题专家。根据给定的 TRIZ 原始材料片段, 生成一道高质量中文评测题。"
    "只输出一个 JSON 对象, 不要输出任何其他文字、解释或 markdown 围栏。JSON 格式:\n"
    '{"question": "题目(中文, 具体明确)", '
    '"reference_answer": "参考答案(中文, >=200字, 专业准确, 结构完整)", '
    '"keywords": ["关键词1", "关键词2", ...]}\n'
    "keywords 为 5-8 个参考答案中必定出现的核心专业术语/短语, 用于自动评分。"
)

# ---- 3-gram Jaccard (逐行抄自 pipeline_v4/src/data_build.py, 口径一致) ----
TOKEN_RE = re.compile(r"[a-z0-9]+|[一-鿿]")


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


def question_grams(text: str):
    return ngrams(tokenize(text), 3)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


_LAST_CALL = [0.0]


def rate_limit_sleep(rpm):
    interval = 60.0 / rpm
    wait = interval - (time.time() - _LAST_CALL[0])
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL[0] = time.time()


def get_client(base_url):
    from openai import OpenAI
    key = os.environ.get("MOONSHOT_API_KEY")
    if not key:
        raise RuntimeError(
            "MOONSHOT_API_KEY 未设置 (非交互 shell 请 eval bashrc 中的 export 行)")
    return OpenAI(api_key=key, base_url=base_url)


def parse_json_object(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"输出中未找到 JSON 对象: {t[:120]}")
    obj = json.loads(t[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("解析结果不是 JSON 对象")
    return obj


def call_gen(client, model, system, user, rpm, max_tokens, temperature, max_retries):
    """限速 + 指数退避重试; 成功返回解析后的 dict, 重试耗尽返回 None。"""
    delay = 5
    for attempt in range(max_retries):
        try:
            rate_limit_sleep(rpm)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_tokens=max_tokens, temperature=temperature)
            return parse_json_object(resp.choices[0].message.content)
        except Exception as e:
            log(f"生成调用失败 (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
    return None


def validate_question(obj, subset, min_ref_chars, min_kw):
    """最低校验 (同 v4): 字段齐全、reference 够长、关键词过滤后仍达标。"""
    q = str(obj.get("question", "")).strip()
    ref = str(obj.get("reference_answer", "")).strip()
    kws = obj.get("keywords", [])
    if not q or not ref or not isinstance(kws, list):
        return False, None
    if len(ref) < min_ref_chars:
        return False, None
    kws = [str(k).strip() for k in kws if str(k).strip()]
    kws = [k for k in kws if k in ref]
    if len(kws) < min_kw:
        return False, None
    return True, {"subset": subset, "question": q, "reference_answer": ref,
                  "keywords": kws}


def load_corpus(path: Path):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                chunks.append({"id": d["id"], "text": d["text"],
                               "category": d.get("metadata", {}).get("category", "unknown")})
    return chunks


def stratified_chunks(chunks, n_candidates, seed):
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for c in chunks:
        by_cat[c["category"]].append(c)
    total = len(chunks)
    picked = []
    for cat in sorted(by_cat):
        grp = by_cat[cat]
        rng.shuffle(grp)
        k = max(2, round(n_candidates * len(grp) / total))
        picked.extend(grp[:k])
    rng.shuffle(picked)
    return picked


def load_existing(path: Path):
    records, per_subset, used = [], Counter(), set()
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    records.append(r)
                    per_subset[r["subset"]] += 1
                    used.add(r["source_chunk_id"])
    return records, per_subset, used


def load_v4_gold(path: Path):
    """读取钉住的 v4 100 题: 返回 (question 3-gram 集列表, 已用 chunk id 集)。"""
    qgrams, used_chunks = [], set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                qgrams.append((r["id"], question_grams(r["question"])))
                used_chunks.add(r["source_chunk_id"])
    return qgrams, used_chunks


def max_jaccard(grams, ref_grams_list):
    """与参照 question 集的最大 Jaccard 及对应 id (短问题 grams 为空时返回 0)。"""
    best_j, best_id = 0.0, None
    if not grams:
        return best_j, best_id
    for rid, rg in ref_grams_list:
        j = jaccard(grams, rg)
        if j > best_j:
            best_j, best_id = j, rid
    return best_j, best_id


def auto_check(rec, max_j_v4, max_j_new):
    """自动规则初检, 返回 (是否全过, 检查明细列表)。"""
    checks = []
    ok = True

    def add(name, passed, detail):
        nonlocal ok
        checks.append((name, passed, detail))
        if not passed:
            ok = False

    add("reference>=200字", len(rec["reference_answer"]) >= 200,
        f"{len(rec['reference_answer'])} 字")
    add("keywords 5-8 个", 5 <= len(rec["keywords"]) <= 8,
        f"{len(rec['keywords'])} 个")
    add("keywords 全部出现于 reference",
        all(k in rec["reference_answer"] for k in rec["keywords"]),
        f"{sum(1 for k in rec['keywords'] if k in rec['reference_answer'])}/{len(rec['keywords'])}")
    add("question 非空且 >=10 字", len(rec["question"]) >= 10,
        f"{len(rec['question'])} 字")
    add("vs v4 金标 Jaccard < 0.5", max_j_v4 < 0.5, f"max J = {max_j_v4:.3f}")
    add("vs 本批新题 Jaccard < 0.5", max_j_new < 0.5, f"max J = {max_j_new:.3f}")
    return ok, checks


def export_review(records, path: Path, v4_qgrams):
    # 导出时统一重算 Jaccard (断点续跑场景也保证口径正确)
    jaccard_meta = {}
    grams_so_far = []
    for r in records:
        qg = question_grams(r["question"])
        mj4, _ = max_jaccard(qg, v4_qgrams)
        mjn, _ = max_jaccard(qg, grams_so_far)
        jaccard_meta[r["id"]] = (mj4, mjn)
        grams_so_far.append((r["id"], qg))
    lines = ["# v5 金标扩题 (v5_gold_101-200) 人工抽检队列", "",
             f"生成时间: {datetime.now().isoformat(timespec='seconds')} | 共 {len(records)} 题", "",
             "说明: 每题已做自动规则初检 (reference 长度 / 关键词数与出现率 / question 长度 / ",
             "对 v4 金标及本批新题的 3-gram Jaccard)。**全部题目状态 = 待人工抽检**,",
             "自动初检 FAIL 的题优先人工复核。", ""]
    n_fail = 0
    for r in records:
        mj4, mjn = jaccard_meta.get(r["id"], (0.0, 0.0))
        ok, checks = auto_check(r, mj4, mjn)
        if not ok:
            n_fail += 1
        lines += [
            f"## {r['id']} [{r['subset']}] (chunk: {r['source_chunk_id']})",
            "",
            f"自动初检: {'PASS' if ok else '**FAIL**'} | 人工状态: 待人工", ""]
        for name, passed, detail in checks:
            lines.append(f"- [{'x' if passed else ' '}] {name} ({detail})")
        lines += ["",
                  f"**Question**: {r['question']}", "",
                  "**Reference**:", "", r["reference_answer"], "",
                  "**Keywords**: " + "、".join(r["keywords"]), "",
                  "---", ""]
    lines.insert(4, f"自动初检 FAIL 题数: {n_fail}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"抽检队列: {path} (自动初检 FAIL {n_fail} 题)")


def main():
    ap = argparse.ArgumentParser(description="pipeline_v5 金标扩题 (新增 100 题)")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--limit", type=int, default=None,
                    help="最多新接受 N 题后停止 (试跑用)")
    ap.add_argument("--output", default=None, help="覆盖输出 jsonl 路径")
    ap.add_argument("--review-file", default=None, help="覆盖抽检 md 路径")
    args = ap.parse_args()
    with open(resolve(args.config), encoding="utf-8") as f:
        cfg = json.load(f)
    g = cfg["gold_gen_v5"]

    out_path = resolve(args.output or g["output_file"])
    review_path = resolve(args.review_file or g["review_file"])
    dedup_threshold = float(g.get("dedup_jaccard_threshold", 0.5))

    chunks = load_corpus(resolve(g["corpus_file"]))
    log(f"语料 chunks: {len(chunks)}")

    v4_qgrams, v4_used_chunks = load_v4_gold(resolve(g["v4_gold_file"]))
    log(f"v4 金标: {len(v4_qgrams)} 题 (question 去重参照 + 排除其已用 "
        f"{len(v4_used_chunks)} 个 chunk)")

    n_candidates = int(g["total_questions"] * g["candidate_chunk_multiplier"])
    candidates = stratified_chunks(chunks, n_candidates, g["seed"])
    log(f"分层候选 chunks: {len(candidates)} (覆盖 "
        f"{len({c['category'] for c in candidates})} 个 category)")

    records, per_subset, used_chunks = load_existing(out_path)
    if records:
        log(f"断点续跑: 已有 {len(records)} 题 {dict(per_subset)}")
    # 本批已接受题的 question grams (批内去重)
    new_qgrams = [(r["id"], question_grams(r["question"])) for r in records]

    quotas = dict(g["quotas"])
    remaining = {s: quotas.get(s, 0) - per_subset.get(s, 0) for s in quotas}
    todo_total = sum(max(0, v) for v in remaining.values())
    log(f"配额: {quotas} | 待生成: {todo_total}")

    if todo_total <= 0:
        log("配额已满, 只导出抽检队列")
        export_review(records, review_path, v4_qgrams)
        return

    client = get_client(g["base_url"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_new, rejected, dedup_rejected = 0, 0, 0
    # 候选池排除: v4 已用 chunk + 本批已用 chunk
    excluded = v4_used_chunks | used_chunks
    chunk_iter = (c for c in candidates if c["id"] not in excluded)

    def next_subset():
        cand = {s: v for s, v in remaining.items() if v > 0}
        if not cand:
            return None
        return max(cand, key=lambda s: (cand[s], s))

    with open(out_path, "a", encoding="utf-8") as fout:
        for chunk in chunk_iter:
            subset = next_subset()
            if subset is None:
                break
            if args.limit is not None and accepted_new >= args.limit:
                break
            text = chunk["text"][: g["chunk_text_max_chars"]]
            user = (f"【任务】{SUBSET_TASKS[subset]}\n\n"
                    f"【材料】\n{text}\n\n"
                    f"【要求】question 必须与材料内容相关; reference_answer 至少 "
                    f"{g['min_reference_chars']} 字; keywords {g['keywords_range'][0]}-"
                    f"{g['keywords_range'][1]} 个且必须出现在 reference_answer 中。")
            obj = call_gen(client, g["model"], GEN_SYSTEM, user, g["rpm"],
                           g["max_tokens"], g["temperature"], g["max_api_retries"])
            ok, clean = (False, None)
            if obj is not None:
                ok, clean = validate_question(obj, subset, g["min_reference_chars"],
                                              g["min_keywords_after_filter"])
            used_chunks.add(chunk["id"])
            if not ok:
                rejected += 1
                log(f"[拒] {subset} chunk={chunk['id']} (解析/校验失败, 补抽)")
                continue
            # question 级 3-gram Jaccard 去重: vs v4 金标 + vs 本批已接受
            qg = question_grams(clean["question"])
            mj4, mj4_id = max_jaccard(qg, v4_qgrams)
            mjn, mjn_id = max_jaccard(qg, new_qgrams)
            if mj4 >= dedup_threshold or mjn >= dedup_threshold:
                dedup_rejected += 1
                log(f"[拒-去重] {subset} chunk={chunk['id']} "
                    f"J_v4={mj4:.3f}({mj4_id}) J_new={mjn:.3f}({mjn_id}) 剔除重抽")
                continue
            rec_id = f"v5_gold_{g['id_start'] + len(records):03d}"
            rec = {"id": rec_id,
                   "source_chunk_id": chunk["id"], "category": chunk["category"],
                   **clean}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            records.append(rec)
            new_qgrams.append((rec_id, qg))
            remaining[subset] -= 1
            accepted_new += 1
            log(f"[{accepted_new}] {rec_id} {subset} chunk={chunk['id']} "
                f"kw={len(clean['keywords'])} J_v4={mj4:.3f} (该子集剩 {remaining[subset]})")

    log(f"本轮: 接受 {accepted_new} 题, 校验拒绝 {rejected} 次, 去重拒绝 {dedup_rejected} 次; "
        f"累计 {len(records)} 题")
    still = {s: v for s, v in remaining.items() if v > 0}
    if still:
        log(f"[警告] 候选 chunk 耗尽或到达 limit, 剩余配额: {still}")
    export_review(records, review_path, v4_qgrams)


if __name__ == "__main__":
    main()
