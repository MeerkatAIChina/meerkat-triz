#!/usr/bin/env python
"""
pipeline_v4 金标评测集生成 (TRIZ v4)。

从 data/processed/corpus/triz_corpus.jsonl (3914 chunks) 按 category 分层随机抽 chunk,
调用 Moonshot API (默认 moonshot-v1-8k —— 生成器与训练数据同源无妨, 金标集的独立性由
eval_harness 的 judge 使用不同源模型保证) 生成 100 题金标集:
  principle_recommendation 20 / contradiction_analysis 20 / ariz_guidance 20 /
  case_generation 15 / concept_explanation 15 / innovation_assessment 10
每题: {id, subset, question, reference_answer, keywords(5-8), source_chunk_id, category}

健壮性:
  - RPM=3 限速 (20s 间隔), API 错误指数退避重试;
  - 要求模型输出严格 JSON, 解析失败的题丢弃并补抽下一 chunk;
  - 每题最低校验: reference >= 200 字符; 关键词必须出现在 reference 中
    (不在则剔除该关键词), 过滤后关键词 < 3 个则整题丢弃;
  - 逐条追加输出 jsonl → 断点续跑 (重启时按已有记录恢复配额进度与已用 chunk);
  - 导出人工抽检队列 v4_gold_review.md。

用法:
  venv_v5/bin/python pipeline_v4/src/gold_gen.py --config pipeline_v4/configs/eval_v4.json
  # 试跑 (3 题, 产物到 /tmp):
  venv_v5/bin/python pipeline_v4/src/gold_gen.py --config ... --limit 3 \
      --output /tmp/v4_gold_trial.jsonl --review-file /tmp/v4_gold_review_trial.md
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

DEFAULT_CONFIG = "pipeline_v4/configs/eval_v4.json"

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
    """从模型输出提取 JSON 对象 (容忍 ```json 围栏与前后杂文本)。"""
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
    """最低校验: 字段齐全、reference 够长、关键词过滤后仍达标。返回 (ok, 清洗后obj)。"""
    q = str(obj.get("question", "")).strip()
    ref = str(obj.get("reference_answer", "")).strip()
    kws = obj.get("keywords", [])
    if not q or not ref or not isinstance(kws, list):
        return False, None
    if len(ref) < min_ref_chars:
        return False, None
    kws = [str(k).strip() for k in kws if str(k).strip()]
    kws = [k for k in kws if k in ref]   # 关键词必须确实出现在 reference 中
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
    """按 category 比例分层抽样 (每层至少 2 个), 保证覆盖各文档来源。"""
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
    """断点续跑: 读取已生成题目, 返回 (records, 子集计数, 已用chunk集合)。"""
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


def export_review(records, path: Path):
    lines = ["# v4 金标集人工抽检队列", "",
             f"生成时间: {datetime.now().isoformat(timespec='seconds')} | 共 {len(records)} 题",
             ""]
    for r in records:
        lines += [
            f"## {r['id']} [{r['subset']}] (chunk: {r['source_chunk_id']})",
            "",
            f"**Question**: {r['question']}", "",
            "**Reference**:", "", r["reference_answer"], "",
            "**Keywords**: " + "、".join(r["keywords"]), "",
            "---", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"抽检队列: {path}")


def main():
    ap = argparse.ArgumentParser(description="pipeline_v4 金标评测集生成")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--limit", type=int, default=None,
                    help="最多新接受 N 题后停止 (试跑用)")
    ap.add_argument("--output", default=None, help="覆盖输出 jsonl 路径")
    ap.add_argument("--review-file", default=None, help="覆盖抽检 md 路径")
    args = ap.parse_args()
    with open(resolve(args.config), encoding="utf-8") as f:
        cfg = json.load(f)
    g = cfg["gold_gen"]

    out_path = resolve(args.output or g["output_file"])
    review_path = resolve(args.review_file or g["review_file"])

    chunks = load_corpus(resolve(g["corpus_file"]))
    log(f"语料 chunks: {len(chunks)}")
    n_candidates = int(g["total_questions"] * g["candidate_chunk_multiplier"])
    candidates = stratified_chunks(chunks, n_candidates, g["seed"])
    log(f"分层候选 chunks: {len(candidates)} (覆盖 "
        f"{len({c['category'] for c in candidates})} 个 category)")

    records, per_subset, used_chunks = load_existing(out_path)
    if records:
        log(f"断点续跑: 已有 {len(records)} 题 {dict(per_subset)}")

    quotas = dict(g["quotas"])
    remaining = {s: quotas.get(s, 0) - per_subset.get(s, 0) for s in quotas}
    todo_total = sum(max(0, v) for v in remaining.values())
    log(f"配额: {quotas} | 待生成: {todo_total}")

    if todo_total <= 0:
        log("配额已满, 只导出抽检队列")
        export_review(records, review_path)
        return

    client = get_client(g["base_url"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_new, rejected = 0, 0
    chunk_iter = (c for c in candidates if c["id"] not in used_chunks)

    # 子集轮换: 每次取剩余配额最多的子集
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
            rec = {"id": f"v4_gold_{len(records):03d}",
                   "source_chunk_id": chunk["id"], "category": chunk["category"],
                   **clean}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            records.append(rec)
            remaining[subset] -= 1
            accepted_new += 1
            log(f"[{accepted_new}] {rec['id']} {subset} chunk={chunk['id']} "
                f"kw={len(clean['keywords'])} (该子集剩 {remaining[subset]})")

    log(f"本轮: 接受 {accepted_new} 题, 拒绝 {rejected} 次; 累计 {len(records)} 题")
    still = {s: v for s, v in remaining.items() if v > 0}
    if still:
        log(f"[警告] 候选 chunk 耗尽或到达 limit, 剩余配额: {still}")
    export_review(records, review_path)


if __name__ == "__main__":
    main()
