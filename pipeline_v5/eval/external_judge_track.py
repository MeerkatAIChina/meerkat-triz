#!/usr/bin/env python
"""
pipeline_v5 异源评委复核轨 (决策门 G8 数据源)。

对任意 candidate vs anchor 两个评测结果 json, 用外部评委逐字复跑
臂 A 协议 (同 JUDGE_SYSTEM_ARM_A rubric / build_judge_user_arm_a 批量 5 题 /
T=0 / 403 降级单条 / paired bootstrap Random(42) n=10000),
产出决策门 external_review 碎片并可并入 scores.json。

用法:
  venv/bin/python pipeline_v5/eval/external_judge_track.py \
      --candidate-json results/eval_v5_v5b_gold_<ts>.json \
      --anchor-json    results/eval_v5_base_goldfix_v5_<ts>.json \
      --gold-jsonl     data/processed/v5_data/v5_gold.jsonl \
      --cmp-name v5_vs_base \
      --workdir results/ext_review_v5b \
      [--judges claude-sonnet-4-6 gpt-5.4 gemini-3.5-flash] \
      [--time-budget 270] [--analyze-only] [--merge-scores results/v5/v5_scores.json]

环境: TENSORIS_API_KEY 必须设置 (https://api.tensoris.ai/v1)。
可续跑: 每评委每臂独立缓存 <workdir>/cache_<judge>_<arm>.json。
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from judge_arms import JUDGE_SYSTEM_ARM_A, build_judge_user_arm_a  # noqa: E402

DEFAULT_JUDGES = ["claude-sonnet-4-6", "gpt-5.4", "gemini-3.5-flash"]
TENSORIS_BASE_URL = "https://api.tensoris.ai/v1"
BATCH_SIZE = 5
MAX_API_RETRIES = 3
MAX_PARSE_RETRIES = 2
N_BOOT = 10000
SEED = 42

_START = time.time()
_TIME_BUDGET = [10**9]


def log(msg):
    print(f"[{time.time() - _START:7.1f}s] {msg}", flush=True)


def budget_exceeded():
    return time.time() - _START > _TIME_BUDGET[0]


# ==================== 数据装配 ====================

def load_items(candidate_json, anchor_json, gold_jsonl):
    gold = {}
    with open(gold_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                gold[r["id"]] = r
    resps, moonshot = {}, {}
    for arm, path in (("anchor", anchor_json), ("candidate", candidate_json)):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        resps[arm] = {}
        for r in data["records"]:
            resps[arm][r["id"]] = r["response"]
            moonshot.setdefault(r["id"], {})[arm] = r.get("judge_overall")
    items = [{"id": qid, "subset": g.get("subset", ""),
              "question": g["question"],
              "reference_answer": g.get("reference_answer", ""),
              "keywords": g.get("keywords", [])}
             for qid, g in gold.items()]
    items.sort(key=lambda x: x["id"])
    # 只保留双臂都有回答的题
    items = [it for it in items
             if it["id"] in resps["anchor"] and it["id"] in resps["candidate"]]
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


def run_judge_arm(client, model, arm, items, responses, workdir):
    cache_path = workdir / f"cache_{model.replace('/', '_')}_{arm}.json"
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


# ==================== 统计 ====================

def paired_bootstrap_diff(a, b):
    """逐题配对 bootstrap: 同一下标有放回重抽样 (与 harness 口径一致)。"""
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
    ap = argparse.ArgumentParser(description="异源评委复核轨 (G8 数据源)")
    ap.add_argument("--candidate-json", required=True)
    ap.add_argument("--anchor-json", required=True)
    ap.add_argument("--gold-jsonl", required=True)
    ap.add_argument("--cmp-name", required=True,
                    choices=["v5_vs_base", "v5_vs_v2", "v6_vs_base"],
                    help="写入 external_review 的哪个对比块")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--judges", nargs="*", default=DEFAULT_JUDGES)
    ap.add_argument("--time-budget", type=int, default=270)
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--merge-scores", default=None,
                    help="可选: 将碎片并入既有 scores.json (原地更新)")
    args = ap.parse_args()
    _TIME_BUDGET[0] = args.time_budget

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    items, responses, moonshot = load_items(
        args.candidate_json, args.anchor_json, args.gold_jsonl)
    log(f"装配: {len(items)} 题 (双臂均有回答), cmp={args.cmp_name}")

    all_scores = {}
    if not args.analyze_only:
        from openai import OpenAI
        key = os.environ.get("TENSORIS_API_KEY")
        if not key:
            sys.exit("TENSORIS_API_KEY 未设置")
        client = OpenAI(api_key=key, base_url=TENSORIS_BASE_URL)
        for model in args.judges:
            all_scores[model] = {}
            for arm in ("anchor", "candidate"):
                all_scores[model][arm] = run_judge_arm(
                    client, model, arm, items, responses, workdir)
    else:
        for model in args.judges:
            all_scores[model] = {}
            for arm in ("anchor", "candidate"):
                cp = workdir / f"cache_{model.replace('/', '_')}_{arm}.json"
                with open(cp, encoding="utf-8") as f:
                    all_scores[model][arm] = json.load(f)

    # ---- 汇总 ----
    report = {"meta": {
        "protocol": "同 v5 臂 A: 反冗长 rubric, 不截断, T=0, batch=5, "
                    "paired bootstrap n=10000 seed=42",
        "judges": args.judges, "n_items": len(items),
        "cmp_name": args.cmp_name,
        "candidate": args.candidate_json, "anchor": args.anchor_json},
        "moonshot_reference": None, "judges": {}}

    ms_pairs = [(moonshot[q["id"]]["anchor"], moonshot[q["id"]]["candidate"])
                for q in items
                if moonshot[q["id"]]["anchor"] is not None
                and moonshot[q["id"]]["candidate"] is not None]
    if ms_pairs:
        report["moonshot_reference"] = paired_bootstrap_diff(
            [p[0] for p in ms_pairs], [p[1] for p in ms_pairs])

    for model in args.judges:
        sc = all_scores[model]
        pairs, subset_pairs = [], {}
        ext_vs_ms = {"x": [], "y": []}
        n_missing = 0
        for q in items:
            sa_raw = sc["anchor"].get(q["id"])
            sc_raw = sc["candidate"].get(q["id"])
            sa = sa_raw.get("overall") if isinstance(sa_raw, dict) else None
            sv = sc_raw.get("overall") if isinstance(sc_raw, dict) else None
            if sa is None or sv is None:
                n_missing += 1
                continue
            pairs.append((sa, sv))
            subset_pairs.setdefault(q["subset"], []).append((sa, sv))
            mb = moonshot[q["id"]]["anchor"]
            if mb is not None:
                ext_vs_ms["x"].append(mb)
                ext_vs_ms["y"].append(sa)
        overall = paired_bootstrap_diff([p[0] for p in pairs],
                                        [p[1] for p in pairs])
        per = {s: paired_bootstrap_diff([p[0] for p in ps], [p[1] for p in ps])
               for s, ps in sorted(subset_pairs.items())}
        report["judges"][model] = {
            "n_scored": len(pairs), "n_missing": n_missing,
            "overall": overall, "per_subset": per,
            "mean_anchor": sum(p[0] for p in pairs) / len(pairs) if pairs else None,
            "mean_candidate": sum(p[1] for p in pairs) / len(pairs) if pairs else None,
            "spearman_vs_moonshot_anchor": round(
                spearman(ext_vs_ms["x"], ext_vs_ms["y"]), 4)
            if ext_vs_ms["x"] else None}

    with open(workdir / "external_review_detail.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ---- 决策门碎片 ----
    fragment = {"judges": args.judges, args.cmp_name: {}}
    for model, d in report["judges"].items():
        o = d["overall"]
        fragment[args.cmp_name][model] = {
            "diff": o["diff"], "ci95": o["ci95"], "n": o["n"]}
    frag_path = workdir / "external_review_fragment.json"
    with open(frag_path, "w", encoding="utf-8") as f:
        json.dump(fragment, f, ensure_ascii=False, indent=2)
    log(f"碎片: {frag_path}")

    if args.merge_scores:
        with open(args.merge_scores, encoding="utf-8") as f:
            scores = json.load(f)
        er = scores.setdefault("external_review", {"judges": []})
        for j in args.judges:
            if j not in er["judges"]:
                er["judges"].append(j)
        er[args.cmp_name] = fragment[args.cmp_name]
        with open(args.merge_scores, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
        log(f"已并入 scores: {args.merge_scores} (external_review.{args.cmp_name})")

    # ---- md 简报 ----
    L = [f"# 异源评委复核简报 ({args.cmp_name})", ""]
    L.append(f"- 协议: {report['meta']['protocol']}")
    L.append(f"- 评委: {', '.join(args.judges)}")
    L.append("")
    L.append("| 评委 | n | anchor 均分 | candidate 均分 | 配对差值 [95% CI] | 显著 |")
    L.append("|---|---|---|---|---|---|")
    ms = report.get("moonshot_reference")
    if ms:
        sig = "显著" if (ms["ci95"][0] > 0 or ms["ci95"][1] < 0) else "不显著"
        L.append(f"| moonshot (同族参照) | {ms['n']} | — | — | {ms['diff']:+.4f} "
                 f"[{ms['ci95'][0]:+.4f}, {ms['ci95'][1]:+.4f}] | {sig} |")
    for model, d in report["judges"].items():
        o = d["overall"]
        sig = "显著" if (o["ci95"][0] > 0 or o["ci95"][1] < 0) else "不显著"
        ma = f"{d['mean_anchor']:.3f}" if d['mean_anchor'] is not None else "—"
        mc = f"{d['mean_candidate']:.3f}" if d['mean_candidate'] is not None else "—"
        L.append(f"| {model} | {d['n_scored']} | {ma} | {mc} | {o['diff']:+.4f} "
                 f"[{o['ci95'][0]:+.4f}, {o['ci95'][1]:+.4f}] | {sig} |")
    with open(workdir / "external_review_brief.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    log("简报: external_review_brief.md")


if __name__ == "__main__":
    main()
