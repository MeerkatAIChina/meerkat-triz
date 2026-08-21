#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval2 — 新版 LLM 评测流水线（base / v1 / v2 / v3 四方对比）

三阶段：
  --phase generate   GPU 生成（断点续跑；每个 tag 完成后释放显存）
  --phase score      CPU 关键词轨 + Moonshot LLM judge 轨（批量 10 条，RPM=3）
  --phase report     配对 bootstrap / McNemar / Wilson 统计报告
  --phase all        依次执行以上三阶段
  --phase judge_smoke  单批次真实 API 冒烟测试（2 条假数据）

设计说明（与任务书的偏差）：
  扩充评测集 sample_data_expanded.json 实际是 SFT 格式（instruction/input/output），
  不存在 question/type/expected/expected_keywords 字段，也没有 multiple_choice 题。
  因此每题期望关键词从 reference(output) 自动抽取：
    principle_recommendation → reference 中出现的中文原理名（xx原理）
    contradiction_analysis   → reference 中改善/恶化参数名 + 原理名
    case_generation          → CASE_QUALITY_KEYWORDS（与旧评测一致）
    ariz_guidance            → ARIZ_STEP_KEYWORD_MAP 6 步骤（与旧评测一致）
    concept_explanation      → reference 中原理名（可能为空，仅参考轨）
    innovation_assessment    → 不抽取，仅记录响应
  principle 轨相应改为：期望原理全覆盖 0/1（McNemar 用）+ 连续覆盖率。

  judge 输入的 response 截断为前 1000 字符、每批 5 条（moonshot-v1-8k 上下文限制；
  512 tokens 的回答约 500-800 字符，1000 字符基本完整覆盖。ARIZ 后段步骤位于回答
  末尾，过短截断会系统性低估 resource_analysis / solution_evaluation 命中率）。
"""

import argparse
import gc
import json
import logging
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ==================== 路径与常量 ====================

PROJECT_ROOT = "/home/meerkat/mongoose_ai"
sys.path.append(PROJECT_ROOT)

BASE_PATH = f"{PROJECT_ROOT}/models/Qwen3.6-35B-A3B"
ADAPTER_PATHS = {
    "v1": f"{PROJECT_ROOT}/models/meerkat_triz_adapter_v1",
    "v2": f"{PROJECT_ROOT}/models/meerkat_triz_adapter_v2",
    "v3": f"{PROJECT_ROOT}/models/meerkat_triz_adapter_v3",
}
ALL_TAGS = ["base", "v1", "v2", "v3"]

EVAL_DATA = f"{PROJECT_ROOT}/data/processed/sample_data_expanded.json"
PROBE_DATA = f"{PROJECT_ROOT}/data/processed/general_probe.json"
OUT_DIR = Path(f"{PROJECT_ROOT}/results/eval2")

JUDGE_MODEL = "moonshot-v1-8k"
JUDGE_BASE_URL = "https://api.moonshot.cn/v1"
JUDGE_BATCH = 5
JUDGE_INTERVAL = 20.0          # RPM=3
JUDGE_MAX_RETRIES = 3
JUDGE_RESP_CHARS = 1000        # 8k 上下文限制下的截断长度（须覆盖回答末尾的 ARIZ 后段步骤）
JUDGE_Q_CHARS = 150

MAX_NEW_TOKENS = 512
BOOT_N = 10000
BOOT_SEED = 42

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("eval2").info


# ==================== 关键词工具（优先复用项目 benchmark_utils） ====================

def _load_kw_maps():
    """从项目 utils.benchmark_utils 导入关键词映射；失败则用本地副本兜底。"""
    try:
        from utils.benchmark_utils import (
            ARIZ_STEP_KEYWORD_MAP, CONTRADICTION_KEYWORD_MAP,
            PRINCIPLE_NAME_MAP, CASE_QUALITY_KEYWORDS,
        )
        return ARIZ_STEP_KEYWORD_MAP, CONTRADICTION_KEYWORD_MAP, PRINCIPLE_NAME_MAP, CASE_QUALITY_KEYWORDS
    except Exception as e:  # pragma: no cover
        log(f"警告: benchmark_utils 导入失败({e})，使用本地关键词副本")
        ariz = {
            "problem analysis": ["问题分析", "问题识别"],
            "problem model": ["问题模型", "迷你问题"],
            "ideal final result": ["理想最终解", "理想解", "IFR"],
            "contradiction analysis": ["矛盾分析", "技术矛盾", "物理矛盾"],
            "resource analysis": ["资源分析"],
            "solution evaluation": ["方案评估", "方案评价", "解的评估", "方案验证"],
        }
        contra = {
            "strength": ["强度", "坚固", "强"], "weight": ["重量", "轻便", "轻量化", "轻"],
            "composite materials": ["复合材料", "碳纤维"], "porous materials": ["多孔材料", "泡沫金属"],
            "cost": ["成本", "便宜", "廉价"], "speed": ["速度", "快速"],
        }
        case_kw = ["原理", "方案", "创新", "解决", "TRIZ"]
        return ariz, contra, {}, case_kw


ARIZ_MAP, CONTRA_MAP, PRINCIPLE_MAP, CASE_KW = _load_kw_maps()
ARIZ_STEPS = list(ARIZ_MAP.keys())


def _check_keywords_local(response, keywords, keyword_map=None):
    """复刻 utils.benchmark_utils._check_keywords（子串匹配 + 同义词映射）。"""
    response_lower = response.lower()
    matched = 0
    for keyword in keywords:
        candidates = [keyword.lower()]
        if keyword_map and keyword in keyword_map:
            candidates.extend([k.lower() for k in keyword_map[keyword]])
        if any(c in response_lower for c in candidates):
            matched += 1
    return matched, len(keywords)


def kw_hit(response, keyword):
    """中文关键词命中：自身 + 在 PRINCIPLE/CONTRA 映射中查到的全部同义词候选。"""
    rl = response.lower()
    candidates = {keyword.lower()}
    for m in (PRINCIPLE_MAP, CONTRA_MAP):
        for k, aliases in m.items():
            group = {k.lower()} | {a.lower() for a in aliases}
            if keyword.lower() in group:
                candidates |= group
    return any(c in rl for c in candidates)


# ==================== 数据集加载与期望关键词抽取 ====================

PRINCIPLE_RE = re.compile(r"[一-鿿A-Za-z]{2,12}原理")
PARAM_RE = re.compile(r"(?:改善|恶化)的参数[:：]\s*#?\s*\d*\s*([^\n，,；;|]+)")


def _chinese_principle_aliases():
    """PRINCIPLE_NAME_MAP 中全部中文别名（含英文键），按长度降序用于规范抽取。"""
    aliases = set()
    for en, zh_list in PRINCIPLE_MAP.items():
        aliases.add(en)
        aliases.update(zh_list)
    return sorted(aliases, key=len, reverse=True)


def _extract_principles(text):
    """从 reference 抽取原理名。

    清洗规则：
    - 丢弃含"发明"的泛称（如"推荐的TRIZ发明原理"）；
    - 若候选包含 PRINCIPLE_NAME_MAP 的已知别名，归一到该别名
      （避免"运用分割原理"这类带前缀动词的长匹配导致无法命中）。
    """
    aliases = _chinese_principle_aliases()
    seen, out = set(), []
    for raw in PRINCIPLE_RE.findall(text or ""):
        raw = raw.strip()
        if not raw or "发明" in raw:
            continue
        name = next((a for a in aliases if a in raw), raw)
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def load_items(limit=None):
    """加载扩充评测集 + 通用探针集，统一为 eval schema。

    返回 [{id, category, type, question, reference, expected_keywords}, ...]
    """
    items = []
    with open(EVAL_DATA, encoding="utf-8") as f:
        data = json.load(f)
    for cat, samples in data.items():
        for i, s in enumerate(samples):
            q = s.get("instruction", "")
            if s.get("input"):
                q += "\n" + s["input"]
            ref = s.get("output", "")
            if cat == "principle_recommendation":
                typ, kws = "principle", _extract_principles(ref)
            elif cat == "contradiction_analysis":
                typ = "contradiction"
                params = [p.strip() for p in PARAM_RE.findall(ref) if p.strip()]
                kws = params + _extract_principles(ref)
            elif cat == "case_generation":
                typ, kws = "generation", list(CASE_KW)
            elif cat == "ariz_guidance":
                typ, kws = "ariz", []
            elif cat == "concept_explanation":
                typ, kws = "concept", _extract_principles(ref)
            else:  # innovation_assessment 等
                typ, kws = "assessment", []
            items.append({
                "id": f"{cat}#{i:03d}", "category": cat, "type": typ,
                "question": q, "reference": ref, "expected_keywords": kws,
            })
    if os.path.exists(PROBE_DATA):
        with open(PROBE_DATA, encoding="utf-8") as f:
            probes = json.load(f)
        for i, p in enumerate(probes):
            items.append({
                "id": f"general_probe#{i:03d}", "category": "general_probe",
                "type": "probe", "question": p["question"],
                "reference": "", "expected_keywords": p.get("expected_keywords", []),
            })
    if limit:
        items = items[:limit]
    return items


# ==================== 阶段1：GPU 生成 ====================

def _patch_peft():
    """PEFT v0.18 WeightConverter 兼容补丁（必须在 PeftModel 加载前，与旧脚本一致）。"""
    import peft.utils.transformers_weight_conversion as twc

    def _skip_weight_conversion(model, peft_config, adapter_state_dict, adapter_name):
        return adapter_state_dict

    twc.convert_peft_adapter_state_dict_for_transformers = _skip_weight_conversion


def phase_generate(tags, limit=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = load_items(limit)
    log(f"评测题数: {len(items)} (limit={limit})")

    _patch_peft()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from utils.data_utils import format_messages

    log(f"加载 tokenizer: {BASE_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, trust_remote_code=True)

    for tag in tags:
        out_path = OUT_DIR / f"responses_{tag}.json"
        done = {}
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                for r in json.load(f):
                    done[r["id"]] = r
            if len(done) >= len(items):
                log(f"[{tag}] 已存在完整结果 ({len(done)} 条)，跳过")
                continue
            log(f"[{tag}] 断点续跑：已有 {len(done)}/{len(items)} 条")

        log(f"[{tag}] 加载基座模型 (FP16)...")
        base = AutoModelForCausalLM.from_pretrained(
            BASE_PATH, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        if tag == "base":
            model = base
        else:
            log(f"[{tag}] 挂载适配器: {ADAPTER_PATHS[tag]}")
            model = PeftModel.from_pretrained(base, ADAPTER_PATHS[tag])
        model.eval()
        device = next(model.parameters()).device
        log(f"[{tag}] 模型就绪，显存 {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

        results = []
        t0 = time.time()
        n_new = 0
        for i, item in enumerate(items):
            if item["id"] in done:
                results.append(done[item["id"]])
                continue
            prompt = format_messages(tokenizer, user_content=item["question"],
                                     add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS,
                    temperature=0.0, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id)
            resp = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
            rec = {"id": item["id"], "category": item["category"],
                   "question": item["question"], "reference": item["reference"],
                   "expected_keywords": item["expected_keywords"], "response": resp}
            results.append(rec)
            done[item["id"]] = rec
            n_new += 1
            if n_new % 20 == 0:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=1)
                log(f"[{tag}] {len(results)}/{len(items)} "
                    f"({(time.time() - t0) / 60:.1f} min)")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        log(f"[{tag}] 完成 {len(results)} 条 -> {out_path}")

        del model, base
        gc.collect()
        torch.cuda.empty_cache()
        log(f"[{tag}] 显存已释放")


# ==================== 阶段2a：关键词轨打分 ====================

def score_item_kw(rec, item_type):
    """对单条响应做关键词轨打分，返回度量字典。"""
    resp = rec["response"]
    kws = rec.get("expected_keywords", [])
    m = {}
    if item_type == "principle":
        if kws:
            hit = sum(1 for k in kws if kw_hit(resp, k))
            m["principle_coverage"] = hit / len(kws)
            m["principle_correct"] = 1 if hit == len(kws) else 0
    elif item_type == "contradiction":
        if kws:
            hit = sum(1 for k in kws if kw_hit(resp, k))
            m["contradiction_coverage"] = hit / len(kws)
    elif item_type == "generation":
        hit, tot = _check_keywords_local(resp, kws or CASE_KW)
        m["case_coverage"] = hit / tot if tot else 0.0
    elif item_type == "ariz":
        hit, tot = _check_keywords_local(resp, ARIZ_STEPS, ARIZ_MAP)
        m["ariz_step_coverage"] = hit / tot if tot else 0.0
    elif item_type == "concept":
        if kws:
            hit = sum(1 for k in kws if kw_hit(resp, k))
            m["concept_coverage"] = hit / len(kws)
    elif item_type == "probe":
        if kws:
            hit, tot = _check_keywords_local(resp, kws)
            m["probe_coverage"] = hit / tot if tot else 0.0
    return m


def compute_reference_metrics(records):
    """case_generation 的 corpus BLEU / ROUGE（参考轨）。"""
    preds = [r["response"] for r in records if r["category"] == "case_generation" and r.get("reference")]
    refs = [r["reference"] for r in records if r["category"] == "case_generation" and r.get("reference")]
    out = {"n_pairs": len(refs)}
    if not refs:
        return out
    try:
        import sacrebleu
        out["bleu"] = float(sacrebleu.corpus_bleu(preds, [refs], tokenize="zh").score)
    except Exception as e:
        out["bleu_error"] = str(e)
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
        agg = {k: [] for k in ("rouge1", "rouge2", "rougeL")}
        for p, r in zip(preds, refs):
            sc = scorer.score(r, p)
            for k in agg:
                agg[k].append(sc[k].fmeasure)
        out["rouge"] = {k: float(np.mean(v)) for k, v in agg.items()}
    except Exception as e:
        out["rouge_error"] = str(e)
    return out


# ==================== 阶段2b：LLM judge 轨 ====================

ARIZ_STEP_DEFS = {
    "problem_analysis": "问题分析：审视初始问题情境，明确问题边界、目标与已知条件，界定要解决的具体技术问题。",
    "problem_model": "问题模型（迷你问题）：将问题化简为最小化表述，用'最小改动达成目标'的方式重构问题模型。",
    "ideal_final_result": "理想最终解（IFR）：定义理想最终结果——系统自身实现所需功能，而不增加成本、复杂度或有害效应。",
    "contradiction_analysis": "矛盾分析：识别并表述技术矛盾（改善某参数导致另一参数恶化）或物理矛盾（同一参数须同时满足相反要求）。",
    "resource_analysis": "资源分析：盘点系统内外可用的物质、能量、空间、时间、信息等资源，寻找免费或现成的解决资源。",
    "solution_evaluation": "方案评估：对候选解决方案进行评估、比较与验证，选择最优方案并检查其是否接近理想解。",
}
ARIZ_STEP_KEYS = list(ARIZ_STEP_DEFS.keys())

ARIZ_JUDGE_SYSTEM = (
    "你是 TRIZ/ARIZ 方法论评测专家。以下是 ARIZ 算法 6 个关键步骤的标准定义：\n"
    + "\n".join(f"{i+1}. {k}（{v}）" for i, (k, v) in enumerate(ARIZ_STEP_DEFS.items()))
    + "\n\n用户会给你若干条 {id, question, response}。请对每条 response 判断它在语义上是否体现了每个步骤"
      "（概念命中即可，不要求字面出现定义中的词语），每步给 0 或 1，并给一句简短理由。"
      "\n严格输出 JSON 数组，不要输出任何其他文字：\n"
      '[{"id": "...", "steps": {"problem_analysis": 0, "problem_model": 0, '
      '"ideal_final_result": 0, "contradiction_analysis": 0, "resource_analysis": 0, '
      '"solution_evaluation": 0}, "reason": "..."}]'
)

CONTRA_JUDGE_SYSTEM = (
    "你是 TRIZ 技术矛盾分析评测专家。用户会给你若干条 {id, question, expected_keywords, response}，"
    "其中 expected_keywords 是该题参考答案中的关键概念（矛盾参数名或发明原理名）。"
    "请判断每条 response 是否在语义上涵盖了每个关键概念（同义表述、英文、编号引用均算命中，"
    "不要求字面一致），每个概念给 0 或 1，并给一句简短理由。"
    "\n严格输出 JSON 数组，不要输出任何其他文字：\n"
    '[{"id": "...", "keywords": {"概念1": 0, "概念2": 1}, "reason": "..."}]'
)

_LAST_API_CALL = [0.0]


def rate_limit_sleep(interval=JUDGE_INTERVAL):
    wait = interval - (time.time() - _LAST_API_CALL[0])
    if wait > 0:
        time.sleep(wait)
    _LAST_API_CALL[0] = time.time()


def get_judge_client():
    from openai import OpenAI
    key = os.environ.get("MOONSHOT_API_KEY")
    if not key:
        raise RuntimeError("MOONSHOT_API_KEY 未设置（非交互 shell 请 eval bashrc 中的 export 行）")
    return OpenAI(api_key=key, base_url=JUDGE_BASE_URL)


def parse_json_array(text):
    """从模型输出中提取 JSON 数组（容忍 ```json 围栏与前后杂文本）。"""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    start, end = t.find("["), t.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"输出中未找到 JSON 数组: {t[:120]}")
    arr = json.loads(t[start:end + 1])
    if not isinstance(arr, list):
        raise ValueError("解析结果不是数组")
    return arr


def call_judge(client, system, user, max_retries=JUDGE_MAX_RETRIES):
    """带限速与指数退避重试的 judge 调用；成功返回解析后的 list，失败返回 None。"""
    delay = 5
    for attempt in range(max_retries):
        try:
            rate_limit_sleep()
            resp = client.chat.completions.create(
                model=JUDGE_MODEL, temperature=0,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            return parse_json_array(resp.choices[0].message.content)
        except Exception as e:
            log(f"judge 调用失败 (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 3
    return None


def _build_judge_user(kind, batch):
    payload = []
    for r in batch:
        entry = {"id": r["id"], "question": r["question"][:JUDGE_Q_CHARS],
                 "response": r["response"][:JUDGE_RESP_CHARS]}
        if kind == "contradiction":
            entry["expected_keywords"] = r.get("expected_keywords", [])
        payload.append(entry)
    return json.dumps(payload, ensure_ascii=False)


def judge_batch(client, kind, batch):
    """评一批（≤10 条），返回 {id: entry}；批次失败时降级为逐条重试。"""
    system = ARIZ_JUDGE_SYSTEM if kind == "ariz" else CONTRA_JUDGE_SYSTEM
    results = {}

    def valid(arr, items):
        out = {}
        key = "steps" if kind == "ariz" else "keywords"
        for e in arr if isinstance(arr, list) else []:
            if isinstance(e, dict) and "id" in e and isinstance(e.get(key), dict):
                out[e["id"]] = e
        return out

    arr = call_judge(client, system, _build_judge_user(kind, batch))
    results.update(valid(arr, batch))
    missing = [r for r in batch if r["id"] not in results]
    if missing:
        log(f"批次缺失/解析失败 {len(missing)} 条，降级逐条重试")
        for r in missing:
            arr1 = call_judge(client, system, _build_judge_user(kind, [r]))
            got = valid(arr1, [r])
            if got:
                results.update(got)
            else:
                # 失败条目不写缓存：本轮不计分，下次运行自动重试
                log(f"单条 judge 最终失败: {r['id']}（本轮跳过，下轮重试）")
    return results


def run_judge_for_tag(client, tag, records):
    """对单个 tag 的 ariz_guidance + contradiction_analysis 记录跑 judge（带缓存续跑）。"""
    cache_path = OUT_DIR / f"judge_{tag}.json"
    cache = {"ariz": {}, "contradiction": {}}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cache.update(json.load(f))

    jobs = {
        "ariz": [r for r in records if r["category"] == "ariz_guidance" and r["id"] not in cache["ariz"]],
        "contradiction": [r for r in records if r["category"] == "contradiction_analysis"
                          and r["id"] not in cache["contradiction"]],
    }
    for kind, todo in jobs.items():
        if not todo:
            log(f"[{tag}] judge {kind}: 缓存完整，跳过")
            continue
        log(f"[{tag}] judge {kind}: {len(todo)} 条待评 "
            f"(约 {math.ceil(len(todo) / JUDGE_BATCH) * JUDGE_INTERVAL / 60:.1f} 分钟)")
        for i in range(0, len(todo), JUDGE_BATCH):
            batch = todo[i:i + JUDGE_BATCH]
            res = judge_batch(client, kind, batch)
            cache[kind].update(res)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=1)
            log(f"[{tag}] judge {kind}: {min(i + JUDGE_BATCH, len(todo))}/{len(todo)}")
    return cache


def judge_metric(kind, entry, expected_kws=None):
    """judge 条目 -> 覆盖率数值（缺失键按 0 计；JUDGE_FAILED 条目返回 None 剔除）。"""
    if entry.get("reason") == "JUDGE_FAILED":
        return None
    if kind == "ariz":
        steps = entry.get("steps", {})
        return float(np.mean([1.0 if steps.get(k) == 1 else 0.0 for k in ARIZ_STEP_KEYS]))
    kws = entry.get("keywords", {})
    if not expected_kws:
        return None
    return float(np.mean([1.0 if kws.get(k) == 1 else 0.0 for k in expected_kws]))


# ==================== 阶段2：打分主流程 ====================

def phase_score(tags, limit=None, skip_judge=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = load_items(limit)
    type_by_id = {it["id"]: it["type"] for it in items}
    client = None

    for tag in tags:
        resp_path = OUT_DIR / f"responses_{tag}.json"
        if not resp_path.exists():
            log(f"[{tag}] 无响应文件，跳过打分")
            continue
        with open(resp_path, encoding="utf-8") as f:
            records = json.load(f)
        log(f"[{tag}] 打分 {len(records)} 条 (关键词轨)")

        scored = []
        for r in records:
            m = score_item_kw(r, type_by_id.get(r["id"], ""))
            scored.append({"id": r["id"], "category": r["category"], **m})

        judge_cache = {"ariz": {}, "contradiction": {}}
        if not skip_judge:
            if client is None:
                client = get_judge_client()
            judge_cache = run_judge_for_tag(client, tag, records)
            for s in scored:
                if s["category"] == "ariz_guidance" and s["id"] in judge_cache["ariz"]:
                    v = judge_metric("ariz", judge_cache["ariz"][s["id"]])
                    if v is not None:
                        s["judge_ariz"] = v
                elif s["category"] == "contradiction_analysis" and s["id"] in judge_cache["contradiction"]:
                    kws = next((r.get("expected_keywords") for r in records if r["id"] == s["id"]), [])
                    v = judge_metric("contradiction", judge_cache["contradiction"][s["id"]], kws)
                    if v is not None:
                        s["judge_contradiction"] = v

        out = {
            "tag": tag, "n": len(scored), "scored_at": datetime.now().isoformat(),
            "items": scored,
            "reference_metrics": compute_reference_metrics(records),
            "judge_counts": {"ariz": len(judge_cache["ariz"]),
                             "contradiction": len(judge_cache["contradiction"])},
        }
        out_path = OUT_DIR / f"scores_{tag}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        log(f"[{tag}] 打分完成 -> {out_path}")


# ==================== 统计函数（纯函数，可单测） ====================

def wilson_ci(k, n, z=1.959963984540054):
    """Wilson score 区间，返回 (p_hat, lo, hi)。"""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact_p(b, c):
    """McNemar 精确检验（双侧二项），b/c 为不一致对数。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def bootstrap_diff(arr_a, arr_b, n_boot=BOOT_N, seed=BOOT_SEED):
    """配对 bootstrap：B - A 差值的均值与 95% 百分位 CI。arr_a/arr_b 按同一题序对齐。"""
    a = np.asarray(arr_a, dtype=float)
    b = np.asarray(arr_b, dtype=float)
    n = len(a)
    if n == 0:
        return None
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_boot, n))
    diffs = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    return {
        "diff": float(b.mean() - a.mean()),
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "n": n,
    }


def bootstrap_overall_diff(strata_a, strata_b, weights, n_boot=BOOT_N, seed=BOOT_SEED):
    """分层配对 bootstrap：各层独立重抽样后按权重合成 overall 差值。"""
    rng = np.random.RandomState(seed)
    total = np.zeros(n_boot)
    point = 0.0
    for sa, sb, w in zip(strata_a, strata_b, weights):
        a = np.asarray(sa, dtype=float)
        b = np.asarray(sb, dtype=float)
        if len(a) == 0:
            continue
        idx = rng.randint(0, len(a), size=(n_boot, len(a)))
        total += w * (b[idx].mean(axis=1) - a[idx].mean(axis=1))
        point += w * (b.mean() - a.mean())
    return {
        "diff": float(point),
        "ci95": [float(np.percentile(total, 2.5)), float(np.percentile(total, 97.5))],
        "n": sum(len(s) for s in strata_a),
    }


# ==================== 阶段3：统计报告 ====================

# 指标名 -> (category, 字段)
METRIC_DEFS = {
    "principle_accuracy": ("principle_recommendation", "principle_correct"),
    "principle_coverage": ("principle_recommendation", "principle_coverage"),
    "contradiction_coverage": ("contradiction_analysis", "contradiction_coverage"),
    "case_coverage": ("case_generation", "case_coverage"),
    "ariz_step_coverage": ("ariz_guidance", "ariz_step_coverage"),
    "concept_coverage": ("concept_explanation", "concept_coverage"),
    "general_probe_coverage": ("general_probe", "probe_coverage"),
    "judge_contradiction_coverage": ("contradiction_analysis", "judge_contradiction"),
    "judge_ariz_step_coverage": ("ariz_guidance", "judge_ariz"),
}
W_KW = {"principle_accuracy": 0.3, "contradiction_coverage": 0.3,
        "case_coverage": 0.2, "ariz_step_coverage": 0.2}
W_JUDGE = {"principle_accuracy": 0.3, "judge_contradiction_coverage": 0.3,
           "case_coverage": 0.2, "judge_ariz_step_coverage": 0.2}


def _load_scores(tags):
    out = {}
    for tag in tags:
        p = OUT_DIR / f"scores_{tag}.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            out[tag] = {it["id"]: it for it in d["items"]}
    return out


def _metric_arrays(scores, tag_a, tag_b, metric):
    """取两模型在某指标上共同题目的对齐数组。"""
    cat, field = METRIC_DEFS[metric]
    ids = [i for i, it in scores[tag_a].items()
           if it.get("category") == cat and field in it
           and i in scores[tag_b] and field in scores[tag_b][i]]
    ids.sort()
    a = [scores[tag_a][i][field] for i in ids]
    b = [scores[tag_b][i][field] for i in ids]
    return a, b, ids


def _composite(scores, tag_a, tag_b, weights, n_boot, seed):
    strata_a, strata_b, ws = [], [], []
    for metric, w in weights.items():
        a, b, _ = _metric_arrays(scores, tag_a, tag_b, metric)
        if a:
            strata_a.append(a)
            strata_b.append(b)
            ws.append(w)
    if not strata_a:
        return None
    return bootstrap_overall_diff(strata_a, strata_b, ws, n_boot, seed)


def phase_report(tags, n_boot=BOOT_N, seed=BOOT_SEED):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    scores = _load_scores(tags)
    tags = [t for t in tags if t in scores]
    if "base" not in scores:
        raise RuntimeError("缺少 scores_base.json，无法生成对比报告")
    log(f"报告模型: {tags} (bootstrap n={n_boot}, seed={seed})")

    # --- 单模型指标 ---
    models_out = {}
    for tag in tags:
        metrics = {}
        for metric, (cat, field) in METRIC_DEFS.items():
            vals = [it[field] for it in scores[tag].values()
                    if it.get("category") == cat and field in it]
            if vals:
                metrics[metric] = float(np.mean(vals))
                if metric == "principle_accuracy":
                    k = int(sum(vals))
                    p, lo, hi = wilson_ci(k, len(vals))
                    metrics["principle_wilson"] = {"k": k, "n": len(vals),
                                                   "p": p, "ci95": [lo, hi]}
        # overall（点估计）
        for name, weights in (("overall_kw", W_KW), ("overall_judge", W_JUDGE)):
            parts = [metrics.get(m) for m in weights]
            if all(v is not None for v in parts):
                metrics[name] = float(sum(w * v for w, v in zip(weights.values(), parts)))
        ref_path = OUT_DIR / f"scores_{tag}.json"
        with open(ref_path, encoding="utf-8") as f:
            ref_metrics = json.load(f).get("reference_metrics", {})
        models_out[tag] = {"metrics": metrics, "reference_metrics": ref_metrics}

    # --- 配对对比 ---
    pairs = [(t, "base") for t in tags if t != "base"]
    if "v3" in scores and "v2" in scores:
        pairs.append(("v3", "v2"))
    pairs_out = {}
    for tb, ta in pairs:
        entry = {"metrics": {}}
        for metric in METRIC_DEFS:
            a, b, _ = _metric_arrays(scores, ta, tb, metric)
            if not a:
                continue
            r = bootstrap_diff(a, b, n_boot, seed)
            r["significant"] = bool(r["ci95"][0] > 0 or r["ci95"][1] < 0)
            entry["metrics"][metric] = r
        # McNemar（principle 0/1）
        a, b, _ = _metric_arrays(scores, ta, tb, "principle_accuracy")
        if a:
            bb = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)  # base 错 adapter 对
            cc = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
            entry["mcnemar_principle"] = {
                "b_only_{}_correct".format(tb): bb, "only_{}_correct".format(ta): cc,
                "p_exact": mcnemar_exact_p(bb, cc),
            }
        entry["overall_kw"] = _composite(scores, ta, tb, W_KW, n_boot, seed)
        entry["overall_judge"] = _composite(scores, ta, tb, W_JUDGE, n_boot, seed)
        pairs_out[f"{tb}_vs_{ta}"] = entry

    report = {
        "timestamp": ts,
        "config": {"bootstrap_n": n_boot, "seed": seed,
                   "weights_kw": W_KW, "weights_judge": W_JUDGE,
                   "judge_model": JUDGE_MODEL, "judge_batch": JUDGE_BATCH,
                   "judge_resp_chars": JUDGE_RESP_CHARS,
                   "note": "评测集为 SFT 格式，期望关键词由 reference 自动抽取；"
                           "principle 轨为'期望原理全覆盖 0/1'（数据集中无选择题）；"
                           "judge 输入 response 截断 1000 字符、每批 5 条；"
                           "judge 失败条目不缓存不计分（下轮重试）；未使用 BERTScore（venv 无 bert_score）。"},
        "models": models_out,
        "pairs": pairs_out,
    }
    json_path = OUT_DIR / f"report_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    md = render_markdown(report)
    md_path = OUT_DIR / f"report_{ts}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    log(f"报告已保存: {json_path} / {md_path}")
    return report


def _fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"


def _sig(flag):
    return " ✅" if flag else ""


def render_markdown(report):
    L = []
    tags = list(report["models"].keys())
    L.append(f"# eval2 评测报告 ({report['timestamp']})\n")
    L.append(f"> bootstrap={report['config']['bootstrap_n']} 次重采样, seed={report['config']['seed']}; "
             f"judge={report['config']['judge_model']} (批量{report['config']['judge_batch']}, RPM=3)。\n")

    metric_rows = ["overall_kw", "overall_judge", "principle_accuracy", "principle_coverage",
                   "contradiction_coverage", "case_coverage", "ariz_step_coverage",
                   "judge_contradiction_coverage", "judge_ariz_step_coverage",
                   "concept_coverage", "general_probe_coverage"]
    L.append("## 1. 四方指标总览\n")
    L.append("| 指标 | " + " | ".join(tags) + " |")
    L.append("|" + "---|" * (len(tags) + 1))
    for m in metric_rows:
        L.append(f"| {m} | " + " | ".join(_fmt(report["models"][t]["metrics"].get(m)) for t in tags) + " |")
    L.append("")
    for t in tags:
        w = report["models"][t]["metrics"].get("principle_wilson")
        if w:
            L.append(f"- **{t}** principle_accuracy Wilson 95% CI: "
                     f"[{_fmt(w['ci95'][0])}, {_fmt(w['ci95'][1])}] (k={w['k']}/{w['n']})")
    L.append("")

    L.append("## 2. 配对对比（bootstrap 差值, 后者-前者, CI 不含 0 记 ✅）\n")
    for pair, entry in report["pairs"].items():
        L.append(f"### {pair}\n")
        L.append("| 指标 | diff | 95% CI | n | 显著 |")
        L.append("|---|---|---|---|---|")
        for comp in ("overall_kw", "overall_judge"):
            r = entry.get(comp)
            if r:
                sig = r["ci95"][0] > 0 or r["ci95"][1] < 0
                L.append(f"| **{comp}** | **{_fmt(r['diff'])}** | "
                         f"[{_fmt(r['ci95'][0])}, {_fmt(r['ci95'][1])}] | {r['n']} | {_sig(sig)} |")
        for m, r in entry["metrics"].items():
            L.append(f"| {m} | {_fmt(r['diff'])} | "
                     f"[{_fmt(r['ci95'][0])}, {_fmt(r['ci95'][1])}] | {r['n']} | {_sig(r['significant'])} |")
        mc = entry.get("mcnemar_principle")
        if mc:
            ks = list(mc.keys())
            L.append(f"\nMcNemar (principle): {ks[0]}={mc[ks[0]]}, {ks[1]}={mc[ks[1]]}, "
                     f"exact p={_fmt(mc['p_exact'])}"
                     + (" (p<0.05 ✅)" if mc["p_exact"] < 0.05 else ""))
        L.append("")

    L.append("## 3. judge 轨 vs 关键词轨对照\n")
    L.append("| 模型 | contradiction 关键词 | contradiction judge | ariz 关键词 | ariz judge |")
    L.append("|---|---|---|---|---|")
    for t in tags:
        mt = report["models"][t]["metrics"]
        L.append(f"| {t} | {_fmt(mt.get('contradiction_coverage'))} | "
                 f"{_fmt(mt.get('judge_contradiction_coverage'))} | "
                 f"{_fmt(mt.get('ariz_step_coverage'))} | {_fmt(mt.get('judge_ariz_step_coverage'))} |")
    L.append("")

    L.append("## 4. 通用能力回归探针（general_probe）\n")
    L.append("| 模型 | probe 覆盖率 |")
    L.append("|---|---|")
    for t in tags:
        L.append(f"| {t} | {_fmt(report['models'][t]['metrics'].get('general_probe_coverage'))} |")
    L.append("")

    L.append("## 5. 参考轨（case_generation BLEU/ROUGE）\n")
    L.append("| 模型 | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |")
    L.append("|---|---|---|---|---|")
    for t in tags:
        rm = report["models"][t].get("reference_metrics", {})
        rg = rm.get("rouge", {})
        L.append(f"| {t} | {_fmt(rm.get('bleu'), 2)} | {_fmt(rg.get('rouge1'))} | "
                 f"{_fmt(rg.get('rouge2'))} | {_fmt(rg.get('rougeL'))} |")
    L.append("")
    L.append(f"## 备注\n\n{report['config']['note']}\n")
    return "\n".join(L)


# ==================== judge 冒烟测试 ====================

def phase_judge_smoke():
    """真实调用一次 Moonshot API（1 个批次、2 条假数据），验证 JSON 解析与限速。"""
    client = get_judge_client()
    fake = [
        {"id": "smoke#000", "question": "使用ARIZ算法分析：如何在不增加成本的情况下提高打印速度？",
         "reference": "", "expected_keywords": [],
         "response": "先进行问题分析，明确速度与成本的矛盾。定义理想最终解IFR：打印机自身提速。"
                     "技术矛盾为生产率与物质损失。资源分析：现有机械结构与材料。"
                     "方案评估：采用分割原理并行打印头。"},
        {"id": "smoke#001", "question": "使用ARIZ算法分析：如何使折叠屏既轻薄又耐用？",
         "reference": "", "expected_keywords": [],
         "response": "这款手机很不错，建议购买保护壳使用。"},
    ]
    t0 = time.time()
    res = judge_batch(client, "ariz", fake)
    dt = time.time() - t0
    print(json.dumps(res, ensure_ascii=False, indent=2))
    assert set(res.keys()) == {"smoke#000", "smoke#001"}, "id 不完整"
    e0 = res["smoke#000"]
    assert isinstance(e0.get("steps"), dict), "缺少 steps 字段"
    hit = sum(1 for k in ARIZ_STEP_KEYS if e0["steps"].get(k) == 1)
    print(f"SMOKE_OK latency={dt:.1f}s smoke#000 命中步骤数={hit}/6 "
          f"(预期高) smoke#001 steps={res['smoke#001'].get('steps')}")


# ==================== CLI ====================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="eval2: base/v1/v2/v3 四方评测流水线")
    p.add_argument("--phase", required=True,
                   choices=["generate", "score", "report", "all", "judge_smoke"])
    p.add_argument("--models", default="base,v1,v2,v3",
                   help="逗号分隔的模型 tag（base=基座, v1/v2/v3=适配器）")
    p.add_argument("--limit", type=int, default=None, help="只跑前 N 题（调试用）")
    p.add_argument("--skip-judge", action="store_true", help="score 阶段跳过 LLM judge 轨")
    p.add_argument("--bootstrap", type=int, default=BOOT_N, help="bootstrap 重采样次数")
    p.add_argument("--seed", type=int, default=BOOT_SEED)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    tags = [t.strip() for t in args.models.split(",") if t.strip()]
    for t in tags:
        if t != "base" and t not in ADAPTER_PATHS:
            raise SystemExit(f"未知模型 tag: {t}（可选 base,v1,v2,v3）")

    if args.phase == "judge_smoke":
        phase_judge_smoke()
        return
    if args.phase in ("generate", "all"):
        phase_generate(tags, args.limit)
    if args.phase in ("score", "all"):
        phase_score(tags, args.limit, skip_judge=args.skip_judge)
    if args.phase in ("report", "all"):
        phase_report(tags, args.bootstrap, args.seed)


if __name__ == "__main__":
    main()
