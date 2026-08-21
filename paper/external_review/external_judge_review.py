#!/usr/bin/env python3
"""
异源评委终审 (External Judge Final Review) — Meerkat-TRIZ-v1

协议与 v5 harness 臂 A 完全对齐:
  - 同一 JUDGE_SYSTEM_ARM_A rubric (反冗长条款, 输入不截断)
  - 同一 build_judge_user_arm_a 用户消息格式 (批量 5 题/请求)
  - T=0; 解析失败批次退化单条重试
  - paired bootstrap (stdlib Random(42), 10000 次) + Wilson + McNemar

输入: base / v5a 结果 json + v5_gold.jsonl (取 reference_answer)
输出: 每评委配对差值 + 跨评委 Spearman + 综合判定, md + json 报告

可续跑: 每评委每臂独立缓存; --time-budget 秒数用尽即安全退出, 重跑续评。
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from meerkat_triz.eval.judge_arms import JUDGE_SYSTEM_ARM_A, build_judge_user_arm_a

HERE = Path(__file__).resolve().parent
BASE_JSON = HERE / "eval_v5_base_goldfix_v5_20260726_234434.json"
V5A_JSON = HERE.parent / "v5_execution" / "eval_v5_v5a_gold_20260729_065141.json"
GOLD_JSONL = HERE / "v5_gold.jsonl"

JUDGES = ["claude-sonnet-4-6", "gpt-5.4", "gemini-3.5-flash"]
ARMS = ["base", "v5a"]
BATCH_SIZE = 5
MAX_API_RETRIES = 3
MAX_PARSE_RETRIES = 2
N_BOOT = 10000
SEED = 42

_START = time.time()
_TIME_BUDGET = [10**9]


def log(msg):
    elapsed = time.time() - _START
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)


def budget_exceeded():
    return time.time() - _START > _TIME_BUDGET[0]


# ==================== 数据装配 ====================

def load_items():
    gold = {}
    with open(GOLD_JSONL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                gold[r["id"]] = r
    resps = {}
    moonshot = {}
    for arm, path in (("base", BASE_JSON), ("v5a", V5A_JSON)):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        resps[arm] = {}
        for r in data["records"]:
            resps[arm][r["id"]] = r["response"]
            if arm == "base":
                moonshot.setdefault(r["id"], {})["base"] = r.get("judge_overall")
            else:
                moonshot.setdefault(r["id"], {})["v5a"] = r.get("judge_overall")
    items = []
    for qid, g in gold.items():
        items.append({"id": qid, "subset": g["subset"],
                      "question": g["question"],
                      "reference_answer": g["reference_answer"],
                      "keywords": g.get("keywords", [])})
    items.sort(key=lambda x: x["id"])
    return items, resps, moonshot


# ==================== judge 调用 ====================

def parse_json_array(text):
    t = re.sub(r"^```(?:json)?\s*", "", text.strip())
    t = re.sub(r"\s*```$", "", t)
    s, e = t.find("["), t.rfind("]")
    if s == -1 or e <= s:
        raise ValueError(f"未找到 JSON 数组: {t[:120]}")
    arr = json.loads(t[s:e + 1])
    if not isinstance(arr, list):
        raise ValueError("非数组")
    return arr


def call_judge(client, model, user):
    from openai import PermissionDeniedError
    attempts = MAX_API_RETRIES + MAX_PARSE_RETRIES
    delay = 5
    for attempt in range(attempts):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": JUDGE_SYSTEM_ARM_A},
                          {"role": "user", "content": user}],
                max_tokens=2000, temperature=0.0)
            out = {}
            for e in parse_json_array(resp.choices[0].message.content):
                if isinstance(e, dict) and "id" in e and "overall" in e:
                    out[str(e["id"])] = {
                        k: e.get(k) for k in
                        ("accuracy", "completeness", "triz_correctness",
                         "structure", "overall")}
            if out:
                return out
            raise ValueError("无有效评分条目")
        except PermissionDeniedError as ex:
            if "blocked" in str(ex).lower():
                log(f"  judge {model} 被安全过滤 block (403)")
                return "__BLOCKED__"
            log(f"  judge {model} 失败 ({attempt + 1}/{attempts}): {str(ex)[:120]}")
            if attempt < attempts - 1:
                time.sleep(delay)
                delay = min(delay * 2, 60)
        except Exception as ex:
            log(f"  judge {model} 失败 ({attempt + 1}/{attempts}): {str(ex)[:120]}")
            if attempt < attempts - 1:
                time.sleep(delay)
                delay = min(delay * 2, 60)
    return None


def run_judge_arm(client, model, arm, items, responses):
    """返回 {id: scores|None|"__BLOCKED__"}; 缓存续跑。"""
    cache_path = HERE / f"cache_{model.replace('/', '_')}_{arm}.json"
    cache = {}
    if cache_path.is_file():
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    todo = [it for it in items if it["id"] not in cache]
    if not todo:
        log(f"{model}/{arm}: 缓存完整 ({len(cache)})")
        return cache
    log(f"{model}/{arm}: 缓存 {len(cache)}, 待评 {len(todo)}")
    for bi in range(0, len(todo), BATCH_SIZE):
        if budget_exceeded():
            log(f"  时间预算用尽, 退出 (本轮完成 {bi}/{len(todo)})")
            break
        batch = todo[bi:bi + BATCH_SIZE]
        user = build_judge_user_arm_a(batch, responses[arm])
        res = call_judge(client, model, user)
        if res == "__BLOCKED__":
            missing = batch[:]
        else:
            missing = [it for it in batch if res is None or it["id"] not in res]
            if res and isinstance(res, dict):
                for it in batch:
                    if it["id"] in res:
                        cache[it["id"]] = res[it["id"]]
        for it in missing:
            single = call_judge(client, model,
                                build_judge_user_arm_a([it], responses[arm]))
            if single == "__BLOCKED__":
                cache[it["id"]] = "__BLOCKED__"
            else:
                cache[it["id"]] = single.get(it["id"]) if single else None
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=1)
        log(f"  进度 {min(bi + BATCH_SIZE, len(todo))}/{len(todo)}")
    return cache


# ==================== 统计 (与 harness 同协议) ====================

def bootstrap_diff(a, b):
    import random
    n = len(a)
    if n == 0:
        return {"diff": 0.0, "ci95": [0.0, 0.0], "n": 0}
    rng = random.Random(SEED)
    diffs = []
    for _ in range(N_BOOT):
        s = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            s += b[i] - a[i]
        diffs.append(s / n)
    diffs.sort()
    return {"diff": sum(b[i] - a[i] for i in range(n)) / n,
            "ci95": [diffs[int(0.025 * N_BOOT)], diffs[int(0.975 * N_BOOT) - 1]],
            "n": n}


def spearman(x, y):
    """逐题 Spearman 秩相关 (无 scipy, 手写)。"""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((rx[i] - mx) ** 2 for i in range(n))
    vy = sum((ry[i] - my) ** 2 for i in range(n))
    return cov / (vx * vy) ** 0.5 if vx and vy else 0.0


# ==================== 主流程 ====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-budget", type=int, default=270,
                    help="秒; 用尽即安全退出可续跑")
    ap.add_argument("--judges", nargs="*", default=JUDGES)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    _TIME_BUDGET[0] = args.time_budget

    items, responses, moonshot = load_items()
    log(f"装配: {len(items)} 题, 双臂 base/v5a")

    all_scores = {}
    if not args.analyze_only:
        from openai import OpenAI
        key = os.environ.get("TENSORIS_API_KEY")
        if not key:
            sys.exit("TENSORIS_API_KEY 未设置")
        client = OpenAI(api_key=key, base_url="https://api.tensoris.ai/v1")
        for model in args.judges:
            all_scores[model] = {}
            for arm in ARMS:
                all_scores[model][arm] = run_judge_arm(
                    client, model, arm, items, responses)
    else:
        for model in args.judges:
            all_scores[model] = {}
            for arm in ARMS:
                cp = HERE / f"cache_{model.replace('/', '_')}_{arm}.json"
                with open(cp, encoding="utf-8") as f:
                    all_scores[model][arm] = json.load(f)

    # ---- 汇总 ----
    report = {"meta": {"protocol": "同 v5 臂 A: 反冗长 rubric, 不截断, T=0, "
                                   "batch=5, paired bootstrap n=10000 seed=42",
                        "judges": args.judges, "n_items": len(items)},
              "moonshot_reference": None, "judges": {}}

    # moonshot 参照差值 (来自原结果)
    ms_pairs = [(moonshot[q["id"]]["base"], moonshot[q["id"]]["v5a"])
                for q in items
                if moonshot[q["id"]]["base"] is not None
                and moonshot[q["id"]]["v5a"] is not None]
    ms = bootstrap_diff([p[0] for p in ms_pairs], [p[1] for p in ms_pairs])
    report["moonshot_reference"] = ms

    for model in args.judges:
        sc = all_scores[model]
        pairs, subset_pairs = [], {}
        ext_vs_ms = {"x": [], "y": []}
        n_missing = 0
        for q in items:
            sb_raw = sc["base"].get(q["id"])
            sv_raw = sc["v5a"].get(q["id"])
            sb = sb_raw.get("overall") if isinstance(sb_raw, dict) else None
            sv = sv_raw.get("overall") if isinstance(sv_raw, dict) else None
            if sb is None or sv is None:
                n_missing += 1
                continue
            pairs.append((sb, sv))
            subset_pairs.setdefault(q["subset"], []).append((sb, sv))
            mb = moonshot[q["id"]]["base"]
            if mb is not None:
                ext_vs_ms["x"].append(mb)
                ext_vs_ms["y"].append(sb)
        overall = bootstrap_diff([p[0] for p in pairs], [p[1] for p in pairs])
        per = {s: bootstrap_diff([p[0] for p in ps], [p[1] for p in ps])
               for s, ps in sorted(subset_pairs.items())}
        report["judges"][model] = {
            "n_scored": len(pairs), "n_missing": n_missing,
            "overall": overall, "per_subset": per,
            "mean_base": sum(p[0] for p in pairs) / len(pairs) if pairs else None,
            "mean_v5a": sum(p[1] for p in pairs) / len(pairs) if pairs else None,
            "spearman_vs_moonshot_base": round(
                spearman(ext_vs_ms["x"], ext_vs_ms["y"]), 4) if ext_vs_ms["x"] else None}

    with open(HERE / "external_review_result.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ---- md 报告 ----
    L = ["# 异源评委终审报告 — Meerkat-TRIZ-v1 (v5a vs base, v5 金标 300 题)", ""]
    L.append(f"- 协议: {report['meta']['protocol']}")
    L.append(f"- 评委: {', '.join(args.judges)} (真异源: anthropic/openai/google)")
    L.append("")
    L.append("| 评委 | n | base 均分 | v5a 均分 | 配对差值 [95% CI] | 显著 | 与 moonshot 逐题 ρ |")
    L.append("|---|---|---|---|---|---|---|")
    ms_sig = "显著" if (ms["ci95"][0] > 0 or ms["ci95"][1] < 0) else "不显著"
    L.append(f"| moonshot-v1-32k (参照, 同族) | {ms['n']} | — | — | "
             f"{ms['diff']:+.4f} [{ms['ci95'][0]:+.4f}, {ms['ci95'][1]:+.4f}] | {ms_sig} | — |")
    for model, d in report["judges"].items():
        o = d["overall"]
        sig = "显著" if (o["ci95"][0] > 0 or o["ci95"][1] < 0) else "不显著"
        sp = d['spearman_vs_moonshot_base']
        sp_str = f"{sp}" if sp is not None else "—"
        mb_str = f"{d['mean_base']:.3f}" if d['mean_base'] is not None else "—"
        mv_str = f"{d['mean_v5a']:.3f}" if d['mean_v5a'] is not None else "—"
        L.append(f"| {model} | {d['n_scored']} | {mb_str} | "
                 f"{mv_str} | {o['diff']:+.4f} "
                 f"[{o['ci95'][0]:+.4f}, {o['ci95'][1]:+.4f}] | {sig} | "
                 f"{sp_str} |")
    L.append("")
    for model, d in report["judges"].items():
        L.append(f"## {model} 子集差值")
        L.append("")
        L.append("| 子集 | 差值 | 95% CI |")
        L.append("|---|---|---|")
        for s, o in d["per_subset"].items():
            L.append(f"| {s} | {o['diff']:+.4f} | [{o['ci95'][0]:+.4f}, {o['ci95'][1]:+.4f}] |")
        L.append("")
    with open(HERE / "external_review_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    log("报告: external_review_report.md / external_review_result.json")


if __name__ == "__main__":
    main()
