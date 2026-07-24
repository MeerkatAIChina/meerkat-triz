#!/usr/bin/env python
"""
pipeline_v5 通用回归探针扩题 30 -> 120 (方案 §6.6, 裁决项 #9 KPI 改写为 <5pp)。

沿用 data/processed/general_probe.json 的 6 类 subcategory:
  common_sense / math / logic / writing / code / instruction_following
各由 5 题扩至 20 题 (每类新增 15 题, 共新增 90 题), 同 schema:
  {id, category="general_probe", type="open_ended", subcategory,
   question, expected_keywords}
原 30 题原样保留在前, 新题 id 沿用原前缀续号
  (probe_cs_06..20 / probe_math_06..20 / probe_logic_06..20 /
   probe_write_06..20 / probe_code_06..20 / probe_if_06..20)。

去重: question 级 token 3-gram Jaccard >= 0.5 对既有 30 题及本批新题剔除重生
(tokenize/ngrams/jaccard 逐行抄自 pipeline_v4/src/data_build.py, 与金标同口径)。

纪律: RPM=3 限速, 指数退避; 新增题逐条追加缓存 jsonl 断点续跑;
最终产物 data/processed/v5_data/general_probe_v5.json (120 题, 完整列表)。

用法:
  venv_v5/bin/python pipeline_v5/src/probe_gen_v5.py \
      --config pipeline_v5/configs/eval_v5_probe.json
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = "pipeline_v5/configs/eval_v5_probe.json"

ID_PREFIX = {
    "common_sense": "probe_cs_",
    "math": "probe_math_",
    "logic": "probe_logic_",
    "writing": "probe_write_",
    "code": "probe_code_",
    "instruction_following": "probe_if_",
}

SUBCATEGORY_DESC = {
    "common_sense": "常识题 (有确定简短答案的常识/科学/生活知识)",
    "math": "数学题 (有唯一确定数值答案的算术/代数题, 难度小学到初中)",
    "logic": "逻辑题 (三段论/条件推理等, 有确定结论)",
    "writing": "写作题 (要求写一句话/一小段, 并必须包含指定词语)",
    "code": "代码题 (编程基础知识, 有确定简短答案)",
    "instruction_following": "指令跟随题 (要求模型严格按格式/内容约束作答)",
}

GEN_SYSTEM = (
    "你是大模型评测出题专家, 负责构造'通用能力回归探针'题目, 用于检测领域微调后"
    "模型的通用能力是否退化。题目必须与任何专业领域 (如 TRIZ/创新方法) 无关, "
    "面向通用能力。只输出一个 JSON 数组, 不要输出任何其他文字或 markdown 围栏。"
    "数组元素格式: {\"question\": \"题目(中文)\", "
    "\"expected_keywords\": [\"关键词1\", ...]}。expected_keywords 为 1-3 个"
    "判定回答正确所必须出现的关键词/短语。"
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


def parse_json_array(text: str) -> list:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    start, end = t.find("["), t.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"输出中未找到 JSON 数组: {t[:120]}")
    arr = json.loads(t[start:end + 1])
    if not isinstance(arr, list):
        raise ValueError("解析结果不是 JSON 数组")
    return arr


def call_gen_batch(client, model, system, user, rpm, max_tokens, temperature,
                   max_retries):
    delay = 5
    for attempt in range(max_retries):
        try:
            rate_limit_sleep(rpm)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_tokens=max_tokens, temperature=temperature)
            return parse_json_array(resp.choices[0].message.content)
        except Exception as e:
            log(f"生成调用失败 (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
    return None


def validate_probe(obj):
    """最低校验: question 非空, expected_keywords 为 1-3 个非空串。"""
    q = str(obj.get("question", "")).strip()
    kws = obj.get("expected_keywords", [])
    if not q or not isinstance(kws, list):
        return None
    kws = [str(k).strip() for k in kws if str(k).strip()]
    if not kws or len(kws) > 3:
        return None
    if len(q) < 4:
        return None
    return {"question": q, "expected_keywords": kws}


def max_jaccard(grams, ref_grams_list):
    best_j, best_id = 0.0, None
    if not grams:
        return best_j, best_id
    for rid, rg in ref_grams_list:
        j = jaccard(grams, rg)
        if j > best_j:
            best_j, best_id = j, rid
    return best_j, best_id


def main():
    ap = argparse.ArgumentParser(description="pipeline_v5 通用探针扩题 30->120")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    args = ap.parse_args()
    with open(resolve(args.config), encoding="utf-8") as f:
        cfg = json.load(f)
    g = cfg["probe_gen_v5"]

    probe_v4_path = resolve(g["probe_v4_file"])
    cache_path = resolve(g["new_cache_file"])
    out_path = resolve(g["output_file"])
    target_per_cat = int(g["target_per_category"])      # 20
    batch_size = int(g["batch_size"])                   # 5
    dedup_threshold = float(g.get("dedup_jaccard_threshold", 0.5))

    with open(probe_v4_path, encoding="utf-8") as f:
        original = json.load(f)
    per_cat_orig = Counter(p["subcategory"] for p in original)
    log(f"既有探针 {len(original)} 题: {dict(per_cat_orig)}")

    # 断点续跑: 读取已生成新题
    new_items = []
    if cache_path.is_file():
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    new_items.append(json.loads(line))
    per_cat_new = Counter(p["subcategory"] for p in new_items)
    if new_items:
        log(f"断点续跑: 已有新题 {len(new_items)} 题 {dict(per_cat_new)}")

    # 去重参照: 既有 30 题 + 已接受新题
    ref_qgrams = [(p["id"], question_grams(p["question"])) for p in original]
    ref_qgrams += [(p["id"], question_grams(p["question"])) for p in new_items]

    todo = {sc: target_per_cat - per_cat_orig.get(sc, 0) - per_cat_new.get(sc, 0)
            for sc in ID_PREFIX}
    todo = {sc: n for sc, n in todo.items() if n > 0}
    log(f"各类待生成: {todo}")

    stats = {"accepted": 0, "invalid": 0, "dedup_rejected": 0}

    if todo:
        client = get_client(g["base_url"])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "a", encoding="utf-8") as fcache:
            for sc in list(todo):
                examples = [p for p in original + new_items if p["subcategory"] == sc]
                while todo[sc] > 0:
                    want = min(batch_size, todo[sc] + 2)  # 多要 2 题冗余
                    ex_txt = "\n".join(
                        f"- {e['question']} (关键词: {'、'.join(e['expected_keywords'])})"
                        for e in examples[-8:])
                    user = (
                        f"【任务】出 {want} 道「{SUBCATEGORY_DESC[sc]}」类别的通用探针题。\n"
                        f"【既有题目示例 (不得重复或近似)】\n{ex_txt}\n\n"
                        f"【要求】题目之间难度/题材多样化, 与同类别既有题不重复; "
                        f"每题 expected_keywords 1-3 个; 只输出 JSON 数组。")
                    arr = call_gen_batch(client, g["model"], GEN_SYSTEM, user,
                                         g["rpm"], g["max_tokens"],
                                         g["temperature"], g["max_api_retries"])
                    if arr is None:
                        log(f"[失败] {sc} 本批 API 重试耗尽, 该类暂停 (可重跑续跑)")
                        break
                    for obj in arr:
                        if todo[sc] <= 0:
                            break
                        clean = validate_probe(obj)
                        if clean is None:
                            stats["invalid"] += 1
                            log(f"[拒] {sc} 校验失败: {str(obj)[:80]}")
                            continue
                        qg = question_grams(clean["question"])
                        mj, mj_id = max_jaccard(qg, ref_qgrams)
                        if mj >= dedup_threshold:
                            stats["dedup_rejected"] += 1
                            log(f"[拒-去重] {sc} J={mj:.3f}({mj_id}): "
                                f"{clean['question'][:40]}")
                            continue
                        idx = per_cat_orig.get(sc, 0) + per_cat_new.get(sc, 0) + 1
                        rec = {"id": f"{ID_PREFIX[sc]}{idx:02d}",
                               "category": "general_probe",
                               "type": "open_ended",
                               "subcategory": sc,
                               **clean}
                        fcache.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fcache.flush()
                        new_items.append(rec)
                        per_cat_new[sc] += 1
                        ref_qgrams.append((rec["id"], qg))
                        todo[sc] -= 1
                        stats["accepted"] += 1
                        log(f"[{stats['accepted']}] {rec['id']} {sc} "
                            f"J_max={mj:.3f} (该类剩 {todo[sc]})")

    log(f"生成结束: 新接受 {stats['accepted']}, 校验拒绝 {stats['invalid']}, "
        f"去重拒绝 {stats['dedup_rejected']}")
    unfinished = {sc: n for sc, n in todo.items() if n > 0}
    if unfinished:
        log(f"[警告] 未完成的类别: {unfinished} (重跑本脚本可续跑)")

    # 汇总输出: 原 30 题 + 全部新题 (即使未满额也落盘, 计数如实写入报告)
    final = original + new_items
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    per_cat_final = Counter(p["subcategory"] for p in final)
    log(f"产物: {out_path} 共 {len(final)} 题 {dict(per_cat_final)}")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(probe_v4_path),
        "output": str(out_path),
        "original_count": len(original),
        "new_count": len(new_items),
        "total": len(final),
        "per_subcategory": dict(per_cat_final),
        "target_per_category": target_per_cat,
        "stats": stats,
        "unfinished": unfinished,
        "dedup_jaccard_threshold": dedup_threshold,
        "model": g["model"],
    }
    report_path = out_path.with_suffix(".report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log(f"报告: {report_path}")


if __name__ == "__main__":
    main()
