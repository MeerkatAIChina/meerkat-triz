#!/usr/bin/env python
"""
pipeline_v4 评测 harness (TRIZ v4) —— 金标集双轨评测 + 配对统计。

流程:
  1. 加载金标集 (默认 data/processed/v4_gold.jsonl, 每题含 question/reference_answer/
     keywords/subset)。
  2. 生成: BF16 加载基座 (必要时先 import compat 打 WeightConverter 补丁; 有
     --adapter-path 则用 PeftModel 挂适配器), 贪心解码 (do_sample=False) 逐题生成。
     生成结果缓存 results/v4_gen_<tag>.jsonl, 缓存完整时直接复用不碰 GPU。
  3. 双轨评分:
     - 关键词轨: 期望关键词命中率 (大小写不敏感包含匹配)。
     - judge 轨: 不用 moonshot-v1-8k (与数据生成器同源); 按 config.judge.candidates
       顺序各发一条 ping 探测, 选第一个可用者; 兜底 moonshot-v1-8k 时结果中大字标注
       "judge 与生成器同源, 结论仅供参考"。judge 按 0-4 分 rubric (准确性/完整性/
       TRIZ正确性/结构) 批量 5 条/请求打分, 解析失败重试 2 次后单条重试, 仍失败记缺失。
       judge 结果缓存 results/v4_judge_<tag>.json 支持续跑。
  4. 统计 (干净重实现, 参考 /tmp/eval_pipeline_v2/eval2.py L582-647):
     与基线结果 (默认 results/ 中最新的 eval_v4_<baseline_tag>_*.json) 逐题配对:
     paired bootstrap 10000 次算差值 95% CI (overall + 各子集), Wilson CI (pass 率),
     McNemar 精确检验 (pass/fail 不一致对)。两轨各自报告, 不混合加权。
  5. 输出 results/eval_v4_<tag>_<timestamp>.json + 同名 .md 摘要; 每题明细全保留。

模式:
  --calibrate     校准模式: 只对 base 输出跑双轨并导出 judge 打分明细
                  (results/eval_v4_calibrate_judge_detail_<ts>.md), 供人工一致性检查。
  --dry-run       冒烟: 注入确定性假生成 + 假 judge 分, 不碰 GPU 不打 API,
                  跑通评分/统计/报告全链路 (需配合 --tag, 不覆盖真实缓存)。
  --probe-judge   只做 judge 模型可用性探测 (每候选一条 ping) 后退出。

本脚本不在本代理任务中启动真实 GPU 评测; 真实评测由 run/chain_v4.sh 串行调度。
"""

import argparse
import glob
import json
import math
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_CONFIG = "pipeline_v4/configs/eval_v4.json"
EMPTY_THINK = "<think>\n\n</think>\n\n"

JUDGE_SYSTEM = (
    "你是 TRIZ 领域资深评审专家, 正在评估一个 AI 助手对 TRIZ 评测题的回答质量。"
    "对每道题, 依据参考答案与期望关键词, 按以下 rubric 打 0-4 整数分 "
    "(0=完全错误/无关, 1=严重缺陷, 2=部分正确, 3=基本正确且较完整, 4=优秀):\n"
    "- accuracy: 事实与概念准确性\n"
    "- completeness: 相对参考答案的完整性\n"
    "- triz_correctness: TRIZ 方法论运用是否正确 (原理编号/矛盾分析/ARIZ步骤等)\n"
    "- structure: 回答结构与条理性\n"
    "- overall: 综合质量 (不是简单平均, 以 TRIZ 正确性为重)\n"
    "只输出一个 JSON 数组, 不要输出任何其他文字或 markdown 围栏。格式:\n"
    '[{"id": "题目id", "accuracy": 0-4, "completeness": 0-4, '
    '"triz_correctness": 0-4, "structure": 0-4, "overall": 0-4}, ...]'
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


# ==================== 金标集加载 ====================

def load_gold(path: Path, limit=None):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if limit:
        items = items[:limit]
    return items


# ==================== 生成 ====================

def render_prompt(tokenizer, system_message, question):
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_message},
         {"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return prompt.replace(EMPTY_THINK, "")


def load_gen_cache(path: Path):
    cache = {}
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    cache[r["id"]] = r["response"]
    return cache


def gpu_generate(items, cfg, adapter_path, cache_path):
    """BF16 加载 + 贪心生成, 逐题追加缓存。compat 必须先于模型加载。"""
    import compat  # noqa: F401  (import 即打 WeightConverter monkey-patch)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    g = cfg["generation"]
    base = str(resolve(cfg["base_model_path"]))
    log(f"加载 BF16 基座: {base}")
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map=g["device"],
        trust_remote_code=True)
    if adapter_path:
        from peft import PeftModel
        ap = str(resolve(adapter_path))
        log(f"挂载适配器: {ap}")
        model = PeftModel.from_pretrained(model, ap)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    log(f"显存占用 {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    done = load_gen_cache(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as fout:
        for i, it in enumerate(items):
            if it["id"] in done:
                continue
            prompt = render_prompt(tok, cfg["chatml"]["system_message"], it["question"])
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            gen_kwargs = {"max_new_tokens": g["max_new_tokens"],
                          "do_sample": False, "pad_token_id": tok.pad_token_id}
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
            resp = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()
            fout.write(json.dumps({"id": it["id"], "response": resp},
                                  ensure_ascii=False) + "\n")
            fout.flush()
            done[it["id"]] = resp
            if (i + 1) % 10 == 0 or i + 1 == len(items):
                log(f"生成进度 {i + 1}/{len(items)}")
    return done


def fake_generate(items):
    """dry-run 用确定性假生成: 按题号奇偶注入部分关键词 + 填充句。"""
    out = {}
    for i, it in enumerate(items):
        kws = it.get("keywords", [])
        take = kws[: max(1, (i * 7 + 3) % (len(kws) + 1))] if kws else []
        out[it["id"]] = ("这是一个用于冒烟测试的模拟回答, 涉及 "
                         + "、".join(take)
                         + " 等概念。首先进行分析, 其次给出方案, 最后总结。")
    return out


# ==================== 关键词轨 ====================

def keyword_score(response, keywords):
    if not keywords:
        return {"kw_hits": 0, "kw_total": 0, "kw_hit_rate": None}
    low = response.lower()
    hits = sum(1 for k in keywords if k.lower() in low)
    return {"kw_hits": hits, "kw_total": len(keywords),
            "kw_hit_rate": hits / len(keywords)}


# ==================== judge 轨 ====================

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


def ping_model(client, model, max_tokens):
    """一条 ping 探测模型可用性; 返回 (ok, detail)。"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping, 回复 ok"}],
            max_tokens=max_tokens)
        return True, (resp.choices[0].message.content or "")[:50]
    except Exception as e:
        return False, str(e)[:200]


def probe_judge_models(cfg):
    """按 candidates 顺序探测, 返回 (选中模型, 探测明细, 是否同源兜底)。"""
    j = cfg["judge"]
    client = get_client(j["base_url"])
    details = []
    chosen = None
    for m in j["candidates"]:
        ok, detail = ping_model(client, m, j["ping_max_tokens"])
        details.append({"model": m, "ok": ok, "detail": detail})
        log(f"judge 探测 {m}: {'OK' if ok else 'FAIL'} ({detail})")
        if ok and chosen is None:
            chosen = m
        time.sleep(1)
    if chosen is None:
        raise RuntimeError(f"所有 judge 候选均不可用: {details}")
    same_origin = (chosen == j["same_origin_fallback_model"])
    return chosen, details, same_origin


def parse_json_array(text):
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


def build_judge_user(batch, responses, max_chars):
    parts = []
    for it in batch:
        resp = responses.get(it["id"], "")[:max_chars]
        parts.append(
            f"【题目 {it['id']}】({it['subset']})\n"
            f"问题: {it['question']}\n"
            f"参考答案: {it['reference_answer'][:max_chars]}\n"
            f"期望关键词: {'、'.join(it.get('keywords', []))}\n"
            f"AI 回答: {resp}\n")
    return "\n".join(parts) + "\n请按 rubric 对上述每题打分, 输出 JSON 数组。"


def call_judge(client, model, user, cfg):
    """批量 judge: API 重试 max_api_retries 次, 解析失败再重试 max_parse_retries 次。
    成功返回 {id: scores}, 失败返回 None。"""
    j = cfg["judge"]
    attempts = j["max_api_retries"] + j["max_parse_retries"]
    delay = 5
    for attempt in range(attempts):
        try:
            rate_limit_sleep(j["rpm"])
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user", "content": user}],
                max_tokens=2000, temperature=0.0)
            arr = parse_json_array(resp.choices[0].message.content)
            out = {}
            for e in arr:
                if isinstance(e, dict) and "id" in e and "overall" in e:
                    out[str(e["id"])] = {
                        k: e.get(k) for k in
                        ("accuracy", "completeness", "triz_correctness",
                         "structure", "overall")}
            if out:
                return out
            raise ValueError("JSON 数组中无有效评分条目")
        except Exception as e:
            log(f"judge 调用失败 (attempt {attempt + 1}/{attempts}): {e}")
            if attempt < attempts - 1:
                time.sleep(delay)
                delay = min(delay * 2, 60)
    return None


def run_judge(items, responses, judge_model, cfg, cache_path, dry_run):
    """返回 {id: scores|None}; 带缓存续跑, 批次解析失败退化为单条。"""
    cache = {}
    if cache_path.is_file():
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    if dry_run:
        for i, it in enumerate(items):
            if it["id"] not in cache:
                v = (i * 13 + 5) % 5
                cache[it["id"]] = {"accuracy": v, "completeness": max(0, v - 1),
                                   "triz_correctness": v, "structure": min(4, v + 1),
                                   "overall": v}
        return cache

    todo = [it for it in items if it["id"] not in cache]
    log(f"judge: 缓存 {len(cache)} 条, 待评 {len(todo)} 条 (model={judge_model})")
    if not todo:
        return cache
    client = get_client(cfg["judge"]["base_url"])
    bs = cfg["judge"]["batch_size"]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for bi in range(0, len(todo), bs):
        batch = todo[bi:bi + bs]
        user = build_judge_user(batch, responses, cfg["judge"]["max_response_chars"])
        res = call_judge(client, judge_model, user, cfg)
        missing = [it for it in batch if res is None or it["id"] not in res]
        if res:
            for it in batch:
                if it["id"] in res:
                    cache[it["id"]] = res[it["id"]]
        # 批次内缺失 → 单条重试, 仍失败记缺失 (None)
        for it in missing:
            single = call_judge(client, judge_model,
                                build_judge_user([it], responses,
                                                 cfg["judge"]["max_response_chars"]), cfg)
            cache[it["id"]] = single.get(it["id"]) if single else None
            if cache[it["id"]] is None:
                log(f"judge 最终缺失: {it['id']}")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        log(f"judge 进度 {min(bi + bs, len(todo))}/{len(todo)}")
    return cache


# ==================== 统计 (参考 /tmp/eval_pipeline_v2/eval2.py L582-647 干净重实现) ====================

def wilson_ci(k, n, z=1.959963984540054):
    """Wilson score 区间, 返回 (p_hat, lo, hi)。"""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact_p(b, c):
    """McNemar 精确检验 (双侧二项), b/c 为不一致对数。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def bootstrap_diff(arr_a, arr_b, n_boot, seed):
    """配对 bootstrap: B - A 差值均值与 95% 百分位 CI (arr 按同一题序对齐)。"""
    n = len(arr_a)
    if n == 0:
        return None
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            s += arr_b[i] - arr_a[i]
        diffs.append(s / n)
    diffs.sort()
    return {"diff": sum(arr_b[i] - arr_a[i] for i in range(n)) / n,
            "ci95": [diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]],
            "n": n}


# ==================== 汇总与对比 ====================

def collect_records(items, responses, judge_scores):
    """每题明细记录。"""
    records = []
    for it in items:
        kw = keyword_score(responses.get(it["id"], ""), it.get("keywords", []))
        js = judge_scores.get(it["id"])
        records.append({"id": it["id"], "subset": it["subset"],
                        "question": it["question"],
                        "source_chunk_id": it.get("source_chunk_id"),
                        "keywords": it.get("keywords", []),
                        "response": responses.get(it["id"], ""),
                        **kw,
                        "judge": js,
                        "judge_overall": (js or {}).get("overall")})
    return records


def track_summary(records, value_fn, pass_fn, name):
    """单轨汇总: overall 均值 + 按子集 + pass 率 Wilson。"""
    vals = [value_fn(r) for r in records]
    vals = [v for v in vals if v is not None]
    subsets = defaultdict(list)
    for r in records:
        v = value_fn(r)
        if v is not None:
            subsets[r["subset"]].append(v)
    passes = [1 if pass_fn(r) else 0 for r in records if value_fn(r) is not None]
    p_hat, lo, hi = wilson_ci(sum(passes), len(passes))
    return {"track": name,
            "n": len(vals),
            "mean": (sum(vals) / len(vals)) if vals else None,
            "per_subset": {s: {"n": len(v), "mean": sum(v) / len(v)}
                           for s, v in sorted(subsets.items())},
            "pass_rate": {"p": p_hat, "wilson_ci95": [lo, hi],
                          "k": sum(passes), "n": len(passes)}}


def paired_comparison(base_records, this_records, cfg):
    """与基线逐题配对: bootstrap 差值 CI (overall+子集) + McNemar, 两轨各自。"""
    st = cfg["stats"]
    base_by_id = {r["id"]: r for r in base_records}
    common = [r for r in this_records if r["id"] in base_by_id]
    out = {"n_common": len(common), "tracks": {}}

    def paired(track_value, track_pass, key):
        pairs = [(track_value(base_by_id[r["id"]]), track_value(r), r["subset"],
                  track_pass(base_by_id[r["id"]]), track_pass(r))
                 for r in common]
        pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
        a = [p[0] for p in pairs]
        b = [p[1] for p in pairs]
        res = {"overall": bootstrap_diff(a, b, st["bootstrap_n"], st["bootstrap_seed"])}
        per = {}
        for s in sorted({p[2] for p in pairs}):
            sa = [p[0] for p in pairs if p[2] == s]
            sb = [p[1] for p in pairs if p[2] == s]
            per[s] = bootstrap_diff(sa, sb, st["bootstrap_n"], st["bootstrap_seed"])
        res["per_subset"] = per
        b10 = sum(1 for p in pairs if not p[3] and p[4])   # base fail → this pass
        b01 = sum(1 for p in pairs if p[3] and not p[4])   # base pass → this fail
        res["mcnemar"] = {"base_fail_this_pass": b10, "base_pass_this_fail": b01,
                          "p": mcnemar_exact_p(b10, b01)}
        return res

    kw_thr, jd_thr = st["kw_pass_threshold"], st["judge_pass_threshold"]
    out["tracks"]["keyword"] = paired(
        lambda r: r["kw_hit_rate"], lambda r: (r["kw_hit_rate"] or 0) >= kw_thr, "keyword")
    out["tracks"]["judge"] = paired(
        lambda r: r["judge_overall"], lambda r: (r["judge_overall"] or 0) >= jd_thr, "judge")
    return out


def find_baseline_results(results_dir: Path, baseline_tag, exclude_tag):
    pat = str(results_dir / f"eval_v4_{baseline_tag}_*.json")
    files = sorted(glob.glob(pat))
    files = [f for f in files if f"eval_v4_{exclude_tag}_" not in f]
    return Path(files[-1]) if files else None


# ==================== 报告 ====================

def write_markdown(path, meta, kw_sum, jd_sum, comparison, same_origin):
    L = []
    L.append(f"# eval_v4 评测报告 — {meta['tag']}")
    L.append("")
    L.append(f"- 时间: {meta['timestamp']}")
    L.append(f"- 适配器: {meta['adapter_path'] or '(纯 base)'}")
    L.append(f"- 评测集: {meta['eval_file']} ({meta['n_items']} 题)")
    L.append(f"- judge 模型: **{meta['judge_model']}** (探测: "
             + "; ".join(f"{d['model']}={'OK' if d['ok'] else 'FAIL'}"
                         for d in meta['judge_probe']) + ")")
    if same_origin:
        L.append("")
        L.append("> ⚠️⚠️ **judge 与生成器同源 (moonshot-v1-8k), 结论仅供参考** ⚠️⚠️")
    L.append("")
    for s in (kw_sum, jd_sum):
        mean = f"{s['mean']:.4f}" if s["mean"] is not None else "N/A"
        L.append(f"## {s['track']} 轨")
        L.append("")
        L.append(f"- overall 均值: **{mean}** (n={s['n']})")
        pr = s["pass_rate"]
        L.append(f"- pass 率: {pr['p']:.3f} [{pr['wilson_ci95'][0]:.3f}, "
                 f"{pr['wilson_ci95'][1]:.3f}] (Wilson, {pr['k']}/{pr['n']})")
        L.append("")
        L.append("| 子集 | n | 均值 |")
        L.append("|---|---|---|")
        for name, d in s["per_subset"].items():
            L.append(f"| {name} | {d['n']} | {d['mean']:.4f} |")
        L.append("")
    if comparison:
        L.append(f"## 与基线配对对比 (共同题 n={comparison['n_common']})")
        L.append("")
        for tname, t in comparison["tracks"].items():
            o = t["overall"]
            if o is None:
                continue
            sig = "显著" if (o["ci95"][0] > 0 or o["ci95"][1] < 0) else "不显著"
            L.append(f"### {tname} 轨")
            L.append(f"- overall 差值 (this-base): {o['diff']:+.4f} "
                     f"[{o['ci95'][0]:+.4f}, {o['ci95'][1]:+.4f}] "
                     f"(paired bootstrap 10000, {sig})")
            mc = t["mcnemar"]
            L.append(f"- McNemar: base→this 翻正 {mc['base_fail_this_pass']}, "
                     f"翻负 {mc['base_pass_this_fail']}, p={mc['p']:.4f}")
            L.append("")
            L.append("| 子集 | 差值 | 95% CI |")
            L.append("|---|---|---|")
            for name, d in t["per_subset"].items():
                if d:
                    L.append(f"| {name} | {d['diff']:+.4f} | "
                             f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] |")
            L.append("")
    path.write_text("\n".join(L), encoding="utf-8")
    log(f"Markdown 摘要: {path}")


def export_calibration_md(path, records, judge_model):
    L = [f"# judge 校准明细 (judge={judge_model})", "",
         "供人工一致性检查: 每题 judge 各维度打分 + AI 回答摘录。", ""]
    for r in records:
        js = r["judge"] or {}
        L.append(f"## {r['id']} [{r['subset']}]")
        L.append(f"- question: {r['question']}")
        L.append(f"- judge: {json.dumps(js, ensure_ascii=False)}")
        L.append(f"- keywords 命中: {r['kw_hits']}/{r['kw_total']}")
        L.append(f"- response 摘录: {r['response'][:400]}")
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")
    log(f"校准明细: {path}")


# ==================== main ====================

def main():
    ap = argparse.ArgumentParser(description="pipeline_v4 金标评测 harness")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--adapter-path", default=None, help="LoRA 适配器目录; 空=纯 base")
    ap.add_argument("--eval-file", default=None, help="默认 config.eval_file")
    ap.add_argument("--tag", default=None, help="结果标签 (默认 base_gold 或适配器名)")
    ap.add_argument("--baseline-results", default=None,
                    help="基线结果 json; 默认 results/ 中最新 eval_v4_<baseline_tag>_*.json")
    ap.add_argument("--calibrate", action="store_true",
                    help="校准模式: 只对 base 跑双轨并导出 judge 打分明细")
    ap.add_argument("--dry-run", action="store_true",
                    help="注入假生成+假judge, 不碰 GPU/API, 验证全链路")
    ap.add_argument("--probe-judge", action="store_true",
                    help="只做 judge 可用性探测后退出")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    with open(resolve(args.config), encoding="utf-8") as f:
        cfg = json.load(f)

    if args.probe_judge:
        chosen, details, same_origin = probe_judge_models(cfg)
        print(json.dumps({"chosen": chosen, "same_origin_fallback": same_origin,
                          "probe": details}, ensure_ascii=False, indent=2))
        return

    eval_file = resolve(args.eval_file or cfg["eval_file"])
    tag = args.tag
    if not tag:
        tag = "base_gold" if not args.adapter_path else \
            Path(args.adapter_path).name.replace("meerkat_triz_adapter_", "") + "_gold"
    if args.calibrate:
        if args.adapter_path:
            log("[致命] --calibrate 只评 base, 不应传 --adapter-path")
            sys.exit(2)
        tag = args.tag or "calibrate_base"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = resolve(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    items = load_gold(eval_file, args.limit)
    log(f"金标集: {len(items)} 题 ({eval_file}) | tag={tag} | dry_run={args.dry_run}")

    # ---- 生成 (缓存优先) ----
    gen_cache = results_dir / cfg["generation"]["cache_file_template"].format(tag=tag)
    if args.dry_run:
        responses = fake_generate(items)
        log(f"dry-run: 注入 {len(responses)} 条假生成")
    else:
        cached = load_gen_cache(gen_cache)
        if all(it["id"] in cached for it in items):
            log(f"生成缓存完整 ({len(cached)} 条), 复用 {gen_cache}, 不碰 GPU")
            responses = cached
        else:
            responses = gpu_generate(items, cfg, args.adapter_path, gen_cache)

    # ---- judge ----
    judge_cache_path = results_dir / cfg["judge"]["cache_file_template"].format(tag=tag)
    if args.dry_run:
        judge_model, probe_details, same_origin = "dry-run-fake-judge", [], False
    else:
        judge_model, probe_details, same_origin = probe_judge_models(cfg)
        if same_origin:
            log("[警告] judge 兜底为 moonshot-v1-8k, 与生成器同源, 结论仅供参考")
    judge_scores = run_judge(items, responses, judge_model, cfg,
                             judge_cache_path, args.dry_run)

    # ---- 汇总 ----
    records = collect_records(items, responses, judge_scores)
    st = cfg["stats"]
    kw_sum = track_summary(records, lambda r: r["kw_hit_rate"],
                           lambda r: (r["kw_hit_rate"] or 0) >= st["kw_pass_threshold"],
                           "keyword")
    jd_sum = track_summary(records, lambda r: r["judge_overall"],
                           lambda r: (r["judge_overall"] or 0) >= st["judge_pass_threshold"],
                           "judge")

    # ---- 与基线配对 ----
    comparison = None
    baseline_path = None
    if args.baseline_results:
        baseline_path = Path(args.baseline_results)
    elif tag != st["baseline_tag"] and not args.calibrate:
        baseline_path = find_baseline_results(results_dir, st["baseline_tag"], tag)
    if baseline_path and baseline_path.is_file():
        with open(baseline_path, encoding="utf-8") as f:
            base_result = json.load(f)
        comparison = paired_comparison(base_result["records"], records, cfg)
        log(f"基线配对: {baseline_path.name} (共同题 {comparison['n_common']})")
    else:
        log("无可用基线结果, 跳过配对统计")

    # ---- 输出 ----
    meta = {"tag": tag, "timestamp": ts,
            "adapter_path": args.adapter_path,
            "eval_file": str(eval_file), "n_items": len(items),
            "judge_model": judge_model, "judge_probe": probe_details,
            "judge_same_origin_fallback": same_origin,
            "dry_run": bool(args.dry_run), "calibrate": bool(args.calibrate),
            "baseline_results": str(baseline_path) if baseline_path else None,
            "stats_method": "paired bootstrap n=10000 + Wilson CI + McNemar "
                            "(干净重实现, 参考 /tmp/eval_pipeline_v2/eval2.py L582-647)",
            "config_snapshot": cfg}
    result = {"meta": meta, "keyword_track": kw_sum, "judge_track": jd_sum,
              "paired_vs_baseline": comparison, "records": records}
    out_json = results_dir / f"eval_v4_{tag}_{ts}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log(f"结果: {out_json}")
    write_markdown(results_dir / f"eval_v4_{tag}_{ts}.md",
                   meta, kw_sum, jd_sum, comparison, same_origin)
    if args.calibrate:
        export_calibration_md(
            results_dir / f"eval_v4_calibrate_judge_detail_{ts}.md",
            records, judge_model)

    print(json.dumps({"status": "OK", "tag": tag, "result_json": str(out_json),
                      "keyword_mean": kw_sum["mean"], "judge_mean": jd_sum["mean"],
                      "judge_model": judge_model,
                      "same_origin_fallback": same_origin},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
