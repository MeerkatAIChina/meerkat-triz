# -*- coding: utf-8 -*-
"""P0 实验包共享工具: API 客户端(限流+429退避)、jsonl 断点续跑、缓存加载。
远端路径: /home/meerkat/mongoose_ai/results/e1/e1_common.py
"""
import json, os, sys, time, glob
from pathlib import Path

PROJECT = Path("/home/meerkat/mongoose_ai")
RESULTS = PROJECT / "results"
RPM = 3
MIN_INTERVAL = 60.0 / RPM  # 20s
_last_call = [0.0]

GOLD_FILES = {
    "base": RESULTS / "eval_v4_base_gold_20260723_105438.json",
    "v2":   RESULTS / "eval_v4_v2_gold_20260723_124807.json",
    "v3":   RESULTS / "eval_v4_v3_gold_20260723_132023.json",
    "v4":   RESULTS / "eval_v4_v4_gold_20260724_004355.json",
    "base_goldfix": RESULTS / "eval_v4_base_goldfix_20260724_055459.json",
}
GOLD_JSONL = PROJECT / "data/processed/v4_gold.jsonl"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_client():
    from openai import OpenAI
    key = os.environ.get("MOONSHOT_API_KEY")
    if not key:
        raise RuntimeError("MOONSHOT_API_KEY 未设置")
    return OpenAI(api_key=key, base_url="https://api.moonshot.cn/v1", timeout=180.0)


def call_chat(client, model, system, user, temperature=0.0, max_tokens=2000):
    """限流 + 429/错误指数退避重试。返回文本或 None(最终失败)。"""
    delay = 30
    for attempt in range(8):
        # 全局限流: 距上次请求 >= MIN_INTERVAL
        wait = MIN_INTERVAL - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_tokens=max_tokens, temperature=temperature)
            return resp.choices[0].message.content
        except Exception as e:
            es = str(e)
            log(f"API 失败 (attempt {attempt+1}/8, model={model}): {es[:160]}")
            if "429" in es or "rate" in es.lower():
                time.sleep(delay)
                delay = min(delay * 2, 600)
            else:
                time.sleep(min(delay, 120))
                delay = min(delay * 2, 300)
    return None


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl(path):
    out = []
    p = Path(path)
    if p.is_file():
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return out


def parse_json_array(text):
    import re
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    start, end = t.find("["), t.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"无 JSON 数组: {t[:120]}")
    return json.loads(t[start:end + 1])


def load_gold():
    return load_jsonl(GOLD_JSONL)


def load_responses(tag):
    """从 v4 金标评测缓存取 {id: response_text}。
    base 优先检查 E0 修复产物 (results/e0_basefix/)。"""
    if tag == "base":
        e0 = RESULTS / "e0_basefix"
        if e0.is_dir() and list(e0.glob("*.done")):
            cands = sorted(e0.glob("*.json")) + sorted(e0.glob("*.jsonl"))
            for c in cands:
                try:
                    if c.suffix == ".jsonl":
                        recs = load_jsonl(c)
                        m = {r["id"]: (r.get("response") if isinstance(r.get("response"), str)
                                       else r.get("response", {}).get("text", "")) for r in recs if "id" in r}
                    else:
                        d = json.load(open(c))
                        recs = d.get("records", [])
                        m = {}
                        for r in recs:
                            resp = r.get("response")
                            m[r["id"]] = resp if isinstance(resp, str) else (resp or {}).get("text", "")
                    m = {k: v for k, v in m.items() if v}
                    if len(m) >= 100:
                        log(f"E0 base 修复缓存生效: {c.name} ({len(m)} 题)")
                        return m, f"e0_basefix:{c.name}"
                except Exception as e:
                    log(f"E0 缓存解析失败 {c.name}: {e}")
    d = json.load(open(GOLD_FILES[tag]))
    m = {}
    for r in d["records"]:
        resp = r.get("response")
        m[r["id"]] = resp if isinstance(resp, str) else (resp or {}).get("text", "")
    return m, f"gold_cache:{GOLD_FILES[tag].name}"


def load_judge_cache(tag):
    """既有 moonshot-v1-32k 分明细 {id: judge_dict}"""
    d = json.load(open(GOLD_FILES[tag]))
    return {r["id"]: r["judge"] for r in d["records"]}


def load_kw_cache(tag):
    d = json.load(open(GOLD_FILES[tag]))
    return {r["id"]: {"kw_hit_rate": r["kw_hit_rate"], "kw_hits": r["kw_hits"],
                      "kw_total": r["kw_total"]} for r in d["records"]}


# ---- v4 harness 同款 judge prompt (E1b 必须与原评分输入一致) ----
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


def build_judge_user(batch, responses, max_chars=1500):
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


# ---- pairwise prompt (E1a / E1c) ----
PAIRWISE_SYSTEM = (
    "你是 TRIZ 领域资深评审专家。给定题目、参考答案与两个 AI 回答(回答A、回答B), "
    "判断哪个回答整体更好。评价维度: 事实与概念准确性、相对参考答案的完整性、"
    "TRIZ 方法论运用正确性(原理编号/矛盾分析/ARIZ步骤等)、结构与条理性。"
    "若两者质量相当则判 tie。只输出 JSON 数组, 不要输出其他文字或 markdown 围栏。格式:\n"
    '[{"id": "题目id", "winner": "A"|"B"|"tie", "reason": "一句话理由"}, ...]'
)

RESP_TRUNC = 2000  # E1a 执行卡: 不截断或截断至 2000 字符


def build_pairwise_user(items):
    """items: [{pid, subset, question, reference_answer, resp_a, resp_b}]"""
    parts = []
    for it in items:
        parts.append(
            f"【题目 {it['pid']}】({it['subset']})\n"
            f"问题: {it['question']}\n"
            f"参考答案: {it['reference_answer'][:1500]}\n"
            f"回答A: {it['resp_a'][:RESP_TRUNC]}\n"
            f"回答B: {it['resp_b'][:RESP_TRUNC]}\n")
    return "\n".join(parts) + "\n请逐题裁决, 输出 JSON 数组。"
