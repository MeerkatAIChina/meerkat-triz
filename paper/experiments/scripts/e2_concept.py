# -*- coding: utf-8 -*-
"""E2 v4 concept_explanation 退化归因。
Phase1(CPU): v4_train vs v2 语料 concept 子集条数/长度分布, cap-1500 门进出分析, Mann-Whitney U。
Phase2(API): 15 题 x {base,v2,v4} judge 重判 0-4 + 期望关键词 explicit/synonym/absent 分类。
Phase3(统计): McNemar + 配对 bootstrap 判定噪声/真实。
输出 results/e2/"""
import sys, json, math, random, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "e1"))
from e1_common import (get_client, call_chat, append_jsonl, load_jsonl,
                       parse_json_array, load_gold, load_responses,
                       load_judge_cache, load_kw_cache, log, RESULTS, PROJECT)

E2 = RESULTS / "e2"
OUT_JUDGE = E2 / "e2_rejudge.jsonl"
JUDGE = "moonshot-v1-32k"
BATCH = 5

KW_SYSTEM = (
    "你是 TRIZ 领域资深评审专家, 正在评估一个 AI 助手对 TRIZ 概念解释题的回答质量。"
    "对每道题: (1) 按 rubric 打 0-4 整数分 "
    "(0=完全错误/无关, 1=严重缺陷, 2=部分正确, 3=基本正确且较完整, 4=优秀), "
    "维度: accuracy 准确性 / completeness 完整性 / triz_correctness TRIZ正确性 / structure 结构 / overall 综合; "
    "(2) 对每个期望关键词, 判断其在回答中的覆盖状态: "
    '"explicit"=原词或其直接变形出现; "synonym"=未出现原词但用同义表述覆盖了同一概念'
    '(如 "理想解" 之于 "最终理想解"); "absent"=该概念真正缺失。'
    "只输出 JSON 数组, 格式:\n"
    '[{"id": "题目id", "accuracy": 0-4, "completeness": 0-4, "triz_correctness": 0-4, '
    '"structure": 0-4, "overall": 0-4, "keywords": {"关键词": "explicit|synonym|absent", ...}}, ...]'
)


def mw_u(xs, ys):
    """Mann-Whitney U + 正态近似 z/p (无 scipy)。"""
    data = [(x, 0) for x in xs] + [(y, 1) for y in ys]
    data.sort(key=lambda t: t[0])
    ranks = [0.0] * len(data)
    i = 0
    while i < len(data):
        j = i
        while j < len(data) and data[j][0] == data[i][0]:
            j += 1
        r = (i + j - 1) / 2 + 1
        for k in range(i, j):
            ranks[k] = r
        i = j
    r1 = sum(ranks[k] for k in range(len(data)) if data[k][1] == 0)
    n1, n2 = len(xs), len(ys)
    u1 = r1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u1 - mu) / sigma if sigma else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    return {"u": u1, "z": z, "p_approx": p, "n1": n1, "n2": n2}


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (0, 0, 0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, max(0, (c - w) / d), min(1, (c + w) / d))


def paired_bootstrap(a, b, n_boot=10000, seed=42):
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    boots = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        boots.append(s)
    boots.sort()
    return (sum(diffs) / n, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)])


def mcnemar(a_pass, b_pass):
    b_ = sum(1 for x, y in zip(a_pass, b_pass) if x and not y)
    c_ = sum(1 for x, y in zip(a_pass, b_pass) if not x and y)
    n = b_ + c_
    if n == 0:
        return {"b": 0, "c": 0, "p": 1.0}
    # 精确二项 p
    k = min(b_, c_)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n)
    return {"b": b_, "c": c_, "p": p}


def text_len_v4(rec):
    return len(rec.get("completion", ""))


def text_len_v2(rec):
    t = rec.get("text", "")
    # v2 text 为 chatml 全文, 取 assistant 段近似 output
    idx = t.rfind("<|im_start|>assistant")
    return len(t[idx:]) if idx >= 0 else len(t)


def phase1():
    log("E2 Phase1 数据侧分析")
    v4c = [json.loads(l) for l in open(PROJECT / "data/processed/v4_train.jsonl")
           if json.loads(l).get("subset") == "concept_explanation"]
    v2c = []
    for f in ("v2_train.jsonl", "v2_validation.jsonl", "v2_test.jsonl"):
        for l in open(PROJECT / f"data/processed/{f}"):
            d = json.loads(l)
            if d.get("subset") == "concept_explanation":
                v2c.append(d)
    rep = json.load(open(RESULTS / "v4_data_report.json"))
    gate = [g for g in rep["gates"] if g["gate"] == "rebalance_caps"][0]

    l4 = sorted(text_len_v4(r) for r in v4c)
    l2 = sorted(text_len_v2(r) for r in v2c)

    def dist(xs):
        n = len(xs)
        return {"n": n, "mean": sum(xs) / n, "p50": xs[n // 2],
                "p95": xs[int(n * 0.95)], "min": xs[0], "max": xs[-1]}

    out = {
        "v4_train_concept": dist(l4),
        "v2_corpus_concept": dist(l2),
        "cap_gate": gate,
        "mwu_v4train_vs_v2corpus_len": mw_u(l4, l2),
        "note": "v4 cap 门: concept 3527->1500, 门内优先保留 output 更长者; "
                "v2 语料长度取 assistant 段近似; cap 前 pre-rebalance 明细未单独存档, "
                "以 v2 全量 concept 作为 cap 前分布近似 (decontamination 丢0, near_dedup 全门仅丢72)",
    }
    E2.mkdir(exist_ok=True)
    (E2 / "e2_data_attr.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    log(f"Phase1 完成: v4 n={len(l4)} mean={out['v4_train_concept']['mean']:.0f}, "
        f"v2 n={len(l2)} mean={out['v2_corpus_concept']['mean']:.0f}, "
        f"MWU p={out['mwu_v4train_vs_v2corpus_len']['p_approx']:.4f}")
    return out


def phase2(gold_c):
    log("E2 Phase2 judge 重判 + 关键词覆盖分类")
    resps = {}
    for mtag in ("base", "v2", "v4"):
        resps[mtag], src = load_responses(mtag)
    client = get_client()
    fails_in_a_row = 0
    while True:
        done = {(r["qid"], r["model"]) for r in load_jsonl(OUT_JUDGE)}
        todo = [(m, it) for m in ("base", "v2", "v4") for it in gold_c
                if (it["id"], m) not in done]
        log(f"Phase2 进度 {len(done)}/45, 剩余 {len(todo)}")
        if not todo:
            break
        chunk = todo[:BATCH]
        parts = []
        for mtag, it in chunk:
            resp = resps[mtag][it["id"]][:1500]
            parts.append(
                f"【题目 {it['id']}】\n问题: {it['question']}\n"
                f"参考答案: {it['reference_answer'][:1500]}\n"
                f"期望关键词: {'、'.join(it['keywords'])}\nAI 回答: {resp}\n")
        user = "\n".join(parts) + "\n请逐题输出评分与关键词覆盖, JSON 数组。"
        text = call_chat(client, JUDGE, KW_SYSTEM, user, temperature=0.0, max_tokens=3000)
        ok = 0
        if text:
            try:
                arr = parse_json_array(text)
                by_id = {str(e.get("id")): e for e in arr if isinstance(e, dict)}
                for mtag, it in chunk:
                    e = by_id.get(it["id"])
                    if e and "overall" in e:
                        append_jsonl(OUT_JUDGE, {
                            "qid": it["id"], "model": mtag,
                            "accuracy": e.get("accuracy"),
                            "completeness": e.get("completeness"),
                            "triz_correctness": e.get("triz_correctness"),
                            "structure": e.get("structure"),
                            "overall": e.get("overall"),
                            "kw_status": {k: v for k, v in (e.get("keywords") or {}).items()},
                            "judge": JUDGE})
                        ok += 1
            except Exception as ex:
                log(f"解析失败: {ex}")
        if ok < len(chunk):
            done_now = {(r["qid"], r["model"]) for r in load_jsonl(OUT_JUDGE)}
            for mtag, it in chunk:
                if (it["id"], mtag) in done_now:
                    continue
                u2 = (f"【题目 {it['id']}】\n问题: {it['question']}\n"
                      f"参考答案: {it['reference_answer'][:1500]}\n"
                      f"期望关键词: {'、'.join(it['keywords'])}\n"
                      f"AI 回答: {resps[mtag][it['id']][:1500]}\n"
                      "请输出评分与关键词覆盖, JSON 数组。")
                t2 = call_chat(client, JUDGE, KW_SYSTEM, u2, temperature=0.0, max_tokens=1500)
                try:
                    e = parse_json_array(t2)[0]
                    if "overall" not in e:
                        raise ValueError("no overall")
                    append_jsonl(OUT_JUDGE, {
                        "qid": it["id"], "model": mtag,
                        "accuracy": e.get("accuracy"),
                        "completeness": e.get("completeness"),
                        "triz_correctness": e.get("triz_correctness"),
                        "structure": e.get("structure"),
                        "overall": e.get("overall"),
                        "kw_status": {k: v for k, v in (e.get("keywords") or {}).items()},
                        "judge": JUDGE})
                    ok += 1
                except Exception as ex:
                    log(f"单条失败 {mtag}/{it['id']}: {ex}")
        fails_in_a_row = 0 if ok > 0 else fails_in_a_row + 1
        if fails_in_a_row >= 5:
            log("连续 5 批 0 成功, 退出待续跑")
            sys.exit(1)
        if ok == 0:
            time.sleep(60)


def phase3(gold_c):
    log("E2 Phase3 统计分析")
    kw = {m: load_kw_cache(m) for m in ("base", "v2", "v4")}
    judge32 = {m: load_judge_cache(m) for m in ("base", "v2", "v4")}
    re = {(r["qid"], r["model"]): r for r in load_jsonl(OUT_JUDGE)}
    ids = [it["id"] for it in gold_c]
    kws = {it["id"]: it["keywords"] for it in gold_c}

    report = {"n": len(ids), "per_item": [], "stats": {}, "kw_reclass": {}}
    # 逐题表
    for qid in ids:
        row = {"qid": qid, "keywords": kws[qid]}
        for m in ("base", "v2", "v4"):
            r = re.get((qid, m), {})
            row[m] = {"kw_hit_rate": kw[m][qid]["kw_hit_rate"],
                      "judge32_overall": judge32[m][qid]["overall"],
                      "rejudge_overall": r.get("overall"),
                      "kw_status": r.get("kw_status", {})}
        report["per_item"].append(row)

    # 漏判词分类汇总: v4 漏(absent/synonym) 而 base/v2 explicit 的词
    miss_syn, miss_abs = {}, {}
    for row in report["per_item"]:
        st4, stb, st2 = row["v4"]["kw_status"], row["base"]["kw_status"], row["v2"]["kw_status"]
        for k in row["keywords"]:
            v4s = st4.get(k, "absent")
            others_exp = (stb.get(k) == "explicit") or (st2.get(k) == "explicit")
            if others_exp and v4s == "synonym":
                miss_syn.setdefault(k, []).append(row["qid"])
            elif others_exp and v4s == "absent":
                miss_abs.setdefault(k, []).append(row["qid"])
    report["kw_reclass"] = {
        "synonym_artifact": {k: v for k, v in sorted(miss_syn.items(), key=lambda x: -len(x[1]))},
        "true_missing": {k: v for k, v in sorted(miss_abs.items(), key=lambda x: -len(x[1]))}}

    # 统计: v4 vs base, v4 vs v2 — kw 轨 + judge 轨(32k 原分 + 重判分)
    for pair in (("v4", "base"), ("v4", "v2")):
        a, b = pair
        sa_kw = [kw[a][i]["kw_hit_rate"] for i in ids]
        sb_kw = [kw[b][i]["kw_hit_rate"] for i in ids]
        sa_j = [judge32[a][i]["overall"] for i in ids]
        sb_j = [judge32[b][i]["overall"] for i in ids]
        key = f"{a}_vs_{b}"
        report["stats"][key] = {
            "kw_paired_bootstrap": paired_bootstrap(sa_kw, sb_kw),
            "kw_mcnemar(pass>=0.5)": mcnemar([x >= 0.5 for x in sa_kw], [x >= 0.5 for x in sb_kw]),
            "judge32_paired_bootstrap": paired_bootstrap(sa_j, sb_j),
            "judge32_mcnemar(pass>=3)": mcnemar([x >= 3 for x in sa_j], [x >= 3 for x in sb_j]),
        }
        if re:
            ra = [re.get((i, a), {}).get("overall") for i in ids]
            rb = [re.get((i, b), {}).get("overall") for i in ids]
            if all(x is not None for x in ra + rb):
                report["stats"][key]["rejudge_paired_bootstrap"] = paired_bootstrap(ra, rb)
    (E2 / "e2_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    log("Phase3 完成, e2_report.json 已写")
    (E2 / "e2.done").write_text("done\n")
    return report


def main():
    E2.mkdir(exist_ok=True)
    if not (E2 / "e2_data_attr.json").exists():
        phase1()
    else:
        log("Phase1 已有产物, 跳过")
    gold_c = [it for it in load_gold() if it["subset"] == "concept_explanation"]
    log(f"concept 题数 {len(gold_c)}")
    phase2(gold_c)
    phase3(gold_c)
    log("E2 全部完成")


if __name__ == "__main__":
    main()
