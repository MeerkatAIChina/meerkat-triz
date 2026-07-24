#!/usr/bin/env python
"""
pipeline_v5 评测 harness —— v5 协议全量改造版 (§6.1–6.8, §11.3)。

相对 pipeline_v4/src/eval_harness.py 的 diff:
  1. render_prompt 保留空 think 块 (不剥离, E0 铁律), 生成后才剥离闭合块;
     冒烟断言 assert_empty_think_retained。
  2. 生成后四道质量门 (+英文草稿检测, 微调模型追加长度规则), 任一不过
     触发一次 bad_words_ids 兜底重生成, 仍不过计入失败率并冻结评测;
     质量门汇总上报告首页。
  3. 关键词轨: 子串 + keyword_map_v5.json 别名表; 报告给更新前/后双分数;
     漏判审计队列落盘 (results/v5_miss_audit_<tag>.jsonl)。
  4. judge 双臂: 臂 A (反冗长 rubric + 输入不截断) 常跑; 臂 B 同长度桶
     配对在与基线配对时计算并报告。T=0 硬断言; judge 钉死
     moonshot-v1-32k, 偏离红标; 评委家族谱系入 meta。
  5. overrefusal 检查 (微调模型): 金标回答拒答模板命中率 >2% 不过门。
  6. base 锚点纪律: baseline_tag 必须 base_goldfix 谱系; --anchor-check
     对旧缓存逐题 kw 一致性检查 (|Δ|>0.2 的题数 >10% 冻结报警)。
  7. 统计: stdlib Random(42) 指纹自检 (stats_utils import 即检);
     meta 记录 seed/配置/judge 版本/温度/截断/锚点 tag。

模式:
  --dry-run      假生成 + 假 judge, 不碰 GPU/API, 跑通全链路 (配 --tag)
  --probe-judge  只探测 judge 可用性
  --calibrate    只对 base 跑双轨并导出 judge 明细
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# 文件位于 <project_root>/pipeline_v5/eval/eval_harness_v5.py → parents[2] = 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(_HERE))

import stats_utils  # noqa: F401  (import 即指纹自检)
from stats_utils import bootstrap_diff, mcnemar_exact_p, wilson_ci
from render import (EMPTY_THINK, render_prompt, assert_empty_think_retained,
                    strip_closed_think)
import quality_gates as qg
import keyword_scorer as ks
import judge_arms as ja
import overrefusal as orf

DEFAULT_CONFIG = "pipeline_v5/eval/configs/eval_v5.json"
ANCHOR_LINEAGE = "base_goldfix"  # 锚点谱系一票否决 (§6.1)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_gold(path, limit=None):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items[:limit] if limit else items


# ==================== 生成 ====================

def load_gen_cache(path):
    cache = {}
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    cache[r["id"]] = r["response"]
    return cache


def gpu_generate(items, cfg, adapter_path, cache_path, base_responses=None):
    """BF16 + 贪心生成; 质量门 + bad_words_ids 兜底一次; 仍 invalid 冻结。"""
    import compat  # noqa: F401
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
        log(f"挂载适配器: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(resolve(adapter_path)))
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bad_words = g.get("fallback_bad_words", ["<think>", "</think>"])
    bad_ids = [tok.encode(w, add_special_tokens=False) for w in bad_words]

    def gen_one(prompt, use_fallback):
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        kw = {"max_new_tokens": g["max_new_tokens"], "do_sample": False,
              "pad_token_id": tok.pad_token_id}
        if use_fallback:
            kw["bad_words_ids"] = bad_ids
        with torch.no_grad():
            out = model.generate(**inputs, **kw)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=True)

    finetuned = bool(adapter_path)
    done = load_gen_cache(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_final = []
    with open(cache_path, "a", encoding="utf-8") as fout:
        for i, it in enumerate(items):
            if it["id"] in done:
                continue
            prompt = render_prompt(tok, cfg["chatml"]["system_message"],
                                   it["question"])
            assert_empty_think_retained(prompt)  # §6.1 冒烟断言
            resp = strip_closed_think(gen_one(prompt, use_fallback=False))
            base_len = len(base_responses[it["id"]].strip()) \
                if finetuned and base_responses and it["id"] in base_responses \
                else None
            r = qg.run_gates(resp, finetuned=finetuned, base_len=base_len)
            mode = "direct"
            if not r["pass"]:
                resp2 = strip_closed_think(gen_one(prompt, use_fallback=True))
                r2 = qg.run_gates(resp2, finetuned=finetuned, base_len=base_len)
                mode = "fallback"
                if r2["pass"]:
                    resp, r = resp2, r2
                else:
                    invalid_final.append(it["id"])
            fout.write(json.dumps({"id": it["id"], "response": resp,
                                   "mode": mode,
                                   "gate": r}, ensure_ascii=False) + "\n")
            fout.flush()
            done[it["id"]] = resp
            if (i + 1) % 10 == 0 or i + 1 == len(items):
                log(f"生成进度 {i + 1}/{len(items)} (invalid={len(invalid_final)})")
    if invalid_final:
        log(f"[冻结] 质量门终审 invalid {len(invalid_final)} 题: "
            f"{invalid_final[:20]} —— 计入失败率并冻结评测 (§6.1)")
        sys.exit(3)
    return done


def fake_generate(items, base_responses=None):
    """dry-run 确定性假生成: 中文作答, 含部分期望关键词, 长度 ~300+。"""
    out = {}
    for i, it in enumerate(items):
        kws = it.get("keywords", [])
        take = kws[: max(1, (i * 7 + 3) % (len(kws) + 1))] if kws else []
        body = ("这是一个用于 v5 冒烟测试的模拟回答。首先进行问题分析, "
                "识别系统中的关键矛盾; 接着构建矛盾, 明确技术矛盾与物理矛盾; "
                "然后定义理想解IFR, 即系统在不增加复杂度的前提下自行消除有害作用; "
                "随后进行资源分析, 盘点物质、能量、空间与信息资源; "
                "最后完成方案评估, 给出推荐路径。本题涉及 "
                + "、".join(take) +
                " 等概念。综上, 建议按 ARIZ 流程逐步求解, 并对方案做多轮验证。")
        out[it["id"]] = body
    return out


# ==================== judge 轨 (臂 A; T=0; 钉死 32k) ====================

_LAST_CALL = [0.0]


def rate_limit_sleep(rpm):
    wait = 60.0 / rpm - (time.time() - _LAST_CALL[0])
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL[0] = time.time()


def get_client(base_url):
    from openai import OpenAI
    key = os.environ.get("MOONSHOT_API_KEY")
    if not key:
        raise RuntimeError("MOONSHOT_API_KEY 未设置")
    return OpenAI(api_key=key, base_url=base_url)


def probe_judge(cfg):
    """钉死 moonshot-v1-32k; 不可用按回退链, 偏离红标。"""
    j = cfg["judge"]
    client = get_client(j["base_url"])
    details, chosen = [], None
    for m in [ja.JUDGE_PINNED] + [c for c in ja.JUDGE_CHAIN_FULL
                                  if c != ja.JUDGE_PINNED]:
        try:
            resp = client.chat.completions.create(
                model=m, messages=[{"role": "user", "content": "ping, 回复 ok"}],
                max_tokens=j["ping_max_tokens"], temperature=0.0)
            ok, detail = True, (resp.choices[0].message.content or "")[:50]
        except Exception as e:
            ok, detail = False, str(e)[:200]
        details.append({"model": m, "ok": ok, "detail": detail})
        log(f"judge 探测 {m}: {'OK' if ok else 'FAIL'}")
        if ok:
            chosen = m
            break
        time.sleep(1)
    if chosen is None:
        raise RuntimeError(f"judge 全部不可用: {details}")
    pinned_ok, deviation = ja.assert_pinned_judge(chosen)
    if not pinned_ok:
        log(f"[红标] {deviation}")
    return chosen, details, deviation


def parse_json_array(text):
    import re
    t = re.sub(r"^```(?:json)?\s*", "", text.strip())
    t = re.sub(r"\s*```$", "", t)
    s, e = t.find("["), t.rfind("]")
    if s == -1 or e <= s:
        raise ValueError(f"未找到 JSON 数组: {t[:120]}")
    arr = json.loads(t[s:e + 1])
    if not isinstance(arr, list):
        raise ValueError("非数组")
    return arr


def call_judge(client, model, user, cfg):
    j = cfg["judge"]
    ja.assert_temperature_zero(0.0)  # T=0 锁定硬断言 (§6.4)
    attempts = j["max_api_retries"] + j["max_parse_retries"]
    delay = 5
    for attempt in range(attempts):
        try:
            rate_limit_sleep(j["rpm"])
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": ja.JUDGE_SYSTEM_ARM_A},
                          {"role": "user", "content": user}],
                max_tokens=2000, temperature=0.0)  # T=0 硬编码
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
        except Exception as e:
            log(f"judge 失败 ({attempt + 1}/{attempts}): {e}")
            if attempt < attempts - 1:
                time.sleep(delay)
                delay = min(delay * 2, 60)
    return None


def run_judge(items, responses, judge_model, cfg, cache_path, dry_run):
    cache = {}
    if cache_path.is_file():
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    if dry_run:
        for i, it in enumerate(items):
            if it["id"] not in cache:
                v = (i * 13 + 5) % 5
                cache[it["id"]] = {"accuracy": v, "completeness": max(0, v - 1),
                                   "triz_correctness": v,
                                   "structure": min(4, v + 1), "overall": v}
        return cache
    todo = [it for it in items if it["id"] not in cache]
    log(f"judge: 缓存 {len(cache)}, 待评 {len(todo)} (model={judge_model}, 臂A)")
    if not todo:
        return cache
    client = get_client(cfg["judge"]["base_url"])
    bs = cfg["judge"]["batch_size"]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for bi in range(0, len(todo), bs):
        batch = todo[bi:bi + bs]
        user = ja.build_judge_user_arm_a(batch, responses)  # 不截断
        res = call_judge(client, judge_model, user, cfg)
        missing = [it for it in batch if res is None or it["id"] not in res]
        if res:
            for it in batch:
                if it["id"] in res:
                    cache[it["id"]] = res[it["id"]]
        for it in missing:
            single = call_judge(client, judge_model,
                                ja.build_judge_user_arm_a([it], responses), cfg)
            cache[it["id"]] = single.get(it["id"]) if single else None
            if cache[it["id"]] is None:
                log(f"judge 最终缺失: {it['id']}")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        log(f"judge 进度 {min(bi + bs, len(todo))}/{len(todo)}")
    return cache


# ==================== 汇总/配对 ====================

def collect_records(items, responses, judge_scores, alias_map, gate_per_item):
    records = []
    for it in items:
        qid = it["id"]
        post = ks.keyword_score(responses.get(qid, ""), it.get("keywords", []),
                                alias_map)
        pre = ks.keyword_score(responses.get(qid, ""), it.get("keywords", []),
                               None)
        js = judge_scores.get(qid)
        records.append({"id": qid, "subset": it["subset"],
                        "question": it["question"],
                        "keywords": it.get("keywords", []),
                        "response": responses.get(qid, ""),
                        "kw_pre": pre, "kw_post": post,
                        "kw_hit_rate": post["kw_hit_rate"],
                        "judge": js,
                        "judge_overall": (js or {}).get("overall"),
                        "gate": (gate_per_item or {}).get(qid)})
    return records


def track_summary(records, value_fn, pass_fn, name):
    vals = [value_fn(r) for r in records if value_fn(r) is not None]
    subsets = defaultdict(list)
    for r in records:
        v = value_fn(r)
        if v is not None:
            subsets[r["subset"]].append(v)
    passes = [1 if pass_fn(r) else 0 for r in records if value_fn(r) is not None]
    p, lo, hi = wilson_ci(sum(passes), len(passes))
    return {"track": name, "n": len(vals),
            "mean": (sum(vals) / len(vals)) if vals else None,
            "per_subset": {s: {"n": len(v), "mean": sum(v) / len(v),
                               **({"label": "描述性"} if len(v) < 30 else {})}
                           for s, v in sorted(subsets.items())},
            "pass_rate": {"p": p, "wilson_ci95": [lo, hi],
                          "k": sum(passes), "n": len(passes)}}


def paired_comparison(base_records, this_records, cfg):
    st = cfg["stats"]
    base_by_id = {r["id"]: r for r in base_records}
    common = [r for r in this_records if r["id"] in base_by_id]
    out = {"n_common": len(common), "tracks": {}}

    def paired(tv, tp):
        pairs = [(tv(base_by_id[r["id"]]), tv(r), r["subset"],
                  tp(base_by_id[r["id"]]), tp(r)) for r in common]
        pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
        a, b = [p[0] for p in pairs], [p[1] for p in pairs]
        res = {"overall": bootstrap_diff(a, b)}
        per = {}
        for s in sorted({p[2] for p in pairs}):
            sa = [p[0] for p in pairs if p[2] == s]
            sb = [p[1] for p in pairs if p[2] == s]
            d = bootstrap_diff(sa, sb)
            if d and d["n"] < 30:
                d["label"] = f"描述性(n={d['n']})"
            per[s] = d
        res["per_subset"] = per
        b10 = sum(1 for p in pairs if not p[3] and p[4])
        b01 = sum(1 for p in pairs if p[3] and not p[4])
        res["mcnemar"] = {"base_fail_this_pass": b10,
                          "base_pass_this_fail": b01,
                          "p": mcnemar_exact_p(b10, b01)}
        return res

    thr = st["kw_pass_threshold"], st["judge_pass_threshold"]
    out["tracks"]["keyword"] = paired(
        lambda r: r["kw_hit_rate"], lambda r: (r["kw_hit_rate"] or 0) >= thr[0])
    out["tracks"]["judge_armA"] = paired(
        lambda r: r["judge_overall"], lambda r: (r["judge_overall"] or 0) >= thr[1])
    return out


def anchor_consistency_check(old_records, new_records):
    """§6.1 双锚点一致性: kw 分差 |Δ|>0.2 的题数 >10% → 冻结报警。"""
    old = {r["id"]: r.get("kw_hit_rate") for r in old_records}
    deltas = []
    for r in new_records:
        if r["id"] in old and old[r["id"]] is not None \
                and r.get("kw_hit_rate") is not None:
            deltas.append((r["id"], abs(r["kw_hit_rate"] - old[r["id"]])))
    if not deltas:
        return {"checked": 0, "alarm": False, "note": "无可比题"}
    big = [qid for qid, d in deltas if d > 0.2]
    frac = len(big) / len(deltas)
    return {"checked": len(deltas), "n_big_delta": len(big),
            "frac_big_delta": round(frac, 4), "alarm": frac > 0.10,
            "big_delta_ids": big[:50],
            "rule": "|Δkw|>0.2 的题数 >10% 冻结报警; 结论不一致时以同 harness 新锚点为准"}


def find_baseline(results_dir, baseline_tag, exclude_tag):
    files = sorted(glob.glob(str(results_dir / f"eval_v5_{baseline_tag}_*.json")))
    files = [f for f in files if f"eval_v5_{exclude_tag}_" not in f]
    return Path(files[-1]) if files else None


# ==================== 报告 ====================

def write_markdown(path, meta, gate_md, kw_sum, kw_pre_mean, jd_sum,
                   comparison, arm_b, audit, overrefusal_res, deviation):
    L = [f"# eval_v5 评测报告 — {meta['tag']}", ""]
    L.append(gate_md)  # §6.1: 质量门汇总首页
    if deviation:
        L.append(f"> 🔴 **judge 版本偏离红标**: {deviation}")
        L.append("")
    L.append(f"- 时间: {meta['timestamp']} | 适配器: {meta['adapter_path'] or '(纯 base)'}")
    L.append(f"- 评测集: {meta['eval_file']} ({meta['n_items']} 题)")
    lin = meta["judge_lineage"]
    L.append(f"- judge: **{meta['judge_model']}** | 家族: {lin['family']} | "
             f"谱系声明: {lin['declaration']} | T=0 锁定 | 臂A 输入不截断")
    L.append(f"- 锚点 tag: {meta.get('anchor_tag')} (谱系必须 {ANCHOR_LINEAGE})")
    L.append(f"- seed={meta['stats_meta']['seed']} "
             f"bootstrap={meta['stats_meta']['n_boot']} (stdlib Random, 指纹自检通过)")
    L.append("")
    for s, extra in ((kw_sum, " (别名表后)"), (jd_sum, " (臂A)")):
        mean = f"{s['mean']:.4f}" if s["mean"] is not None else "N/A"
        L.append(f"## {s['track']}{extra}")
        L.append(f"- overall 均值: **{mean}** (n={s['n']})")
        if s is kw_sum and kw_pre_mean is not None:
            L.append(f"- 别名表更新前: {kw_pre_mean:.4f} → 更新后: {mean} (双分数, §6.2)")
        pr = s["pass_rate"]
        L.append(f"- pass 率: {pr['p']:.3f} [{pr['wilson_ci95'][0]:.3f}, "
                 f"{pr['wilson_ci95'][1]:.3f}] (Wilson)")
        L.append("")
        L.append("| 子集 | n | 均值 | 标注 |")
        L.append("|---|---|---|---|")
        for name, d in s["per_subset"].items():
            L.append(f"| {name} | {d['n']} | {d['mean']:.4f} | "
                     f"{d.get('label', '')} |")
        L.append("")
    if audit:
        L.append(f"## 漏判审计 (§6.2)")
        L.append(f"- 漏判率 (rubric≥0.5 且 kw<0.5): "
                 f"{audit['n_queue']}/{audit['n_scored']} = "
                 f"{(audit['miss_rate'] or 0):.1%} | "
                 f"{'建议冻结别名表' if audit['freeze_recommended'] else '继续回流'}")
        L.append(f"- 审计队列: {meta.get('audit_file')}")
        L.append("")
    if overrefusal_res:
        L.append(f"## overrefusal 检查")
        L.append(f"- 拒答模板命中: {overrefusal_res['n_hits']}/{overrefusal_res['n']} "
                 f"= {overrefusal_res['hit_rate']:.2%} "
                 f"(阈值 ≤2%) → **{'过门' if overrefusal_res['pass'] else '不过门'}**")
        L.append("")
    if comparison:
        L.append(f"## 与基线配对 (共同题 n={comparison['n_common']})")
        for tname, t in comparison["tracks"].items():
            o = t["overall"]
            if not o:
                continue
            sig = "显著" if (o["ci95"][0] > 0 or o["ci95"][1] < 0) else "不显著"
            L.append(f"### {tname}")
            L.append(f"- overall 差值: {o['diff']:+.4f} "
                     f"[{o['ci95'][0]:+.4f}, {o['ci95'][1]:+.4f}] ({sig})")
            mc = t["mcnemar"]
            L.append(f"- McNemar: 翻正 {mc['base_fail_this_pass']}, "
                     f"翻负 {mc['base_pass_this_fail']}, p={mc['p']:.4g}")
            L.append("")
            L.append("| 子集 | 差值 | 95% CI | 标注 |")
            L.append("|---|---|---|---|")
            for name, d in t["per_subset"].items():
                if d:
                    L.append(f"| {name} | {d['diff']:+.4f} | "
                             f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] | "
                             f"{d.get('label', '')} |")
            L.append("")
    if arm_b:
        L.append("## 臂 B 同长度桶对比 (§6.4)")
        L.append(f"- {arm_b['note']}")
        L.append(f"- 分桶计数: {json.dumps(arm_b['bucket_counts'], ensure_ascii=False)}")
        if arm_b.get("no_pair"):
            L.append(f"- 无同桶配对 {len(arm_b['no_pair'])} 题 (以臂 A 为准)")
        L.append("")
    L.append("## 功效不足结论清单 (§6.8-⑦)")
    L.append("- 全部 n<30 子集结论为描述性, 不作独立否决依据; 探针/子集 "
             "MDE 随正式评测数据生成后补充。")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")
    log(f"Markdown: {path}")


# ==================== main ====================

def main():
    ap = argparse.ArgumentParser(description="pipeline_v5 金标评测 harness")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--eval-file", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--baseline-results", default=None)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probe-judge", action="store_true")
    ap.add_argument("--anchor-check", default=None,
                    help="旧锚点结果 json, 与新锚点做一致性检查 (§6.1)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    with open(resolve(args.config), encoding="utf-8") as f:
        cfg = json.load(f)

    if args.probe_judge:
        chosen, details, deviation = probe_judge(cfg)
        print(json.dumps({"chosen": chosen, "deviation": deviation,
                          "lineage": ja.judge_lineage(chosen),
                          "probe": details}, ensure_ascii=False, indent=2))
        return

    eval_file = resolve(args.eval_file or cfg["eval_file"])
    finetuned = bool(args.adapter_path)
    tag = args.tag or ("base_goldfix_v5" if not finetuned else
                       Path(args.adapter_path).name + "_v5")
    if args.calibrate:
        if finetuned:
            log("[致命] --calibrate 只评 base")
            sys.exit(2)
        tag = args.tag or "calibrate_base_v5"

    # §6.1 锚点谱系一票否决: 非 dry-run 的正式 base 评测 tag 须为 base_goldfix 谱系
    if not finetuned and not args.dry_run and ANCHOR_LINEAGE not in tag \
            and not args.calibrate:
        log(f"[致命] base 锚点 tag {tag!r} 非 {ANCHOR_LINEAGE} 谱系, 一票否决")
        sys.exit(2)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = resolve(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    items = load_gold(eval_file, args.limit)
    log(f"金标 {len(items)} 题 | tag={tag} | dry_run={args.dry_run}")

    alias_path = _HERE / "keyword_map_v5.json"
    alias_map = ks.load_alias_map(alias_path)
    log(f"别名表: {alias_path.name} ({len(alias_map)} 条 confirmed 关键词)")

    # ---- 基线 (微调模型长度规则 + 配对都需要 base 响应) ----
    baseline_path = None
    base_records, base_responses = None, None
    if args.baseline_results:
        baseline_path = Path(args.baseline_results)
    elif tag != cfg["stats"]["baseline_tag"] and not args.calibrate:
        baseline_path = find_baseline(results_dir, cfg["stats"]["baseline_tag"], tag)
    if baseline_path and baseline_path.is_file():
        with open(baseline_path, encoding="utf-8") as f:
            base_result = json.load(f)
        base_records = base_result["records"]
        base_responses = {r["id"]: r["response"] for r in base_records}
        log(f"基线: {baseline_path.name}")
        # §6.1 缓存 tag 校验
        btag = base_result.get("meta", {}).get("tag", "")
        if ANCHOR_LINEAGE not in btag and "dryrun" not in btag:
            log(f"[致命] 基线锚点 tag {btag!r} 非 {ANCHOR_LINEAGE} 谱系, 一票否决")
            sys.exit(2)

    # ---- 生成 ----
    gen_cache = results_dir / cfg["generation"]["cache_file_template"].format(tag=tag)
    if args.dry_run:
        responses = fake_generate(items)
        log(f"dry-run: 注入 {len(responses)} 条假生成")
    else:
        cached = load_gen_cache(gen_cache)
        if all(it["id"] in cached for it in items):
            log(f"生成缓存完整 ({len(cached)}), 复用, 不碰 GPU")
            responses = cached
        else:
            responses = gpu_generate(items, cfg, args.adapter_path, gen_cache,
                                     base_responses)

    # ---- 质量门 (全部回答, 含缓存复用路径) ----
    gate_per_item, gate_summary = qg.run_gates_batch(
        responses, finetuned=finetuned, base_responses=base_responses)
    gate_md = qg.gate_summary_markdown(gate_summary)
    log(f"质量门: {gate_summary['n_pass']}/{gate_summary['n_generated']} 过门, "
        f"invalid {gate_summary['n_invalid']}")
    if gate_summary["n_invalid"] and not args.dry_run:
        log("[冻结] 存在 invalid 回答 (缓存路径无兜底机会), 冻结评测 (§6.1)")
        sys.exit(3)

    # ---- judge (臂 A) ----
    judge_cache = results_dir / cfg["judge"]["cache_file_template"].format(tag=tag)
    if args.dry_run:
        judge_model, probe_details, deviation = "dry-run-fake-judge", [], ""
    else:
        judge_model, probe_details, deviation = probe_judge(cfg)
    judge_scores = run_judge(items, responses, judge_model, cfg,
                             judge_cache, args.dry_run)

    # ---- 汇总 ----
    records = collect_records(items, responses, judge_scores, alias_map,
                              gate_per_item)
    st = cfg["stats"]
    kw_sum = track_summary(records, lambda r: r["kw_hit_rate"],
                           lambda r: (r["kw_hit_rate"] or 0) >= st["kw_pass_threshold"],
                           "keyword")
    pre_mean, _, _ = ks.dual_score(
        [{"id": r["id"], "response": r["response"], "keywords": r["keywords"]}
         for r in records], alias_map)
    jd_sum = track_summary(records, lambda r: r["judge_overall"],
                           lambda r: (r["judge_overall"] or 0) >= st["judge_pass_threshold"],
                           "judge_armA")

    # ---- 漏判审计队列 ----
    audit_file = results_dir / f"v5_miss_audit_{tag}.jsonl"
    if audit_file.exists() and not args.dry_run:
        audit_file.unlink()  # 本轮重算 (历史 jsonl 由 git/归档保留)
    audit = ks.miss_audit(
        records, {r["id"]: {"post": r["kw_post"]} for r in records},
        lambda r: (r["judge_overall"] / 4.0 if r["judge_overall"] is not None
                   else None),
        out_jsonl=str(audit_file))
    log(f"漏判审计: {audit['n_queue']}/{audit['n_scored']} "
        f"({(audit['miss_rate'] or 0):.1%}) → {audit_file.name}")

    # ---- overrefusal (微调模型) ----
    overrefusal_res = None
    if finetuned:
        patterns = orf.load_templates(_HERE / "refusal_templates_v5.json")
        overrefusal_res = orf.check_overrefusal(responses, patterns)
        log(f"overrefusal: {overrefusal_res['n_hits']}/{overrefusal_res['n']} "
            f"({'过门' if overrefusal_res['pass'] else '不过门'})")

    # ---- 配对 + 臂 B ----
    comparison, arm_b = None, None
    if base_records:
        comparison = paired_comparison(base_records, records, cfg)
        arm_b = ja.arm_b_pairs(base_responses, responses)
        log(f"臂 B: {arm_b['note']}")

    # ---- 双锚点一致性 (§6.1) ----
    anchor_check = None
    if args.anchor_check:
        with open(resolve(args.anchor_check), encoding="utf-8") as f:
            old = json.load(f)
        anchor_check = anchor_consistency_check(old["records"], records)
        log(f"双锚点一致性: {json.dumps(anchor_check, ensure_ascii=False)}")
        if anchor_check["alarm"]:
            log("[冻结报警] |Δkw|>0.2 题数 >10%, 锚点一致性失败")

    # ---- 输出 ----
    meta = {"tag": tag, "timestamp": ts, "adapter_path": args.adapter_path,
            "eval_file": str(eval_file), "n_items": len(items),
            "judge_model": judge_model, "judge_probe": probe_details,
            "judge_lineage": ja.judge_lineage(judge_model),
            "judge_deviation": deviation,
            "judge_temperature": 0.0, "judge_truncation": "none (臂A不截断)",
            "anchor_tag": cfg["stats"]["baseline_tag"],
            "dry_run": bool(args.dry_run), "calibrate": bool(args.calibrate),
            "baseline_results": str(baseline_path) if baseline_path else None,
            "alias_map": str(alias_path.name),
            "audit_file": audit_file.name,
            "stats_meta": {"seed": 42, "n_boot": 10000, "rng": "stdlib",
                            "fingerprint": "self-check passed"},
            "config_snapshot": cfg}
    result = {"meta": meta, "quality_gates": gate_summary,
              "keyword_track_post_alias": kw_sum,
              "keyword_pre_alias_mean": pre_mean,
              "judge_track_armA": jd_sum,
              "miss_audit": {k: v for k, v in audit.items() if k != "queue"},
              "overrefusal": overrefusal_res,
              "paired_vs_baseline": comparison,
              "armB_bucket_pairs": arm_b,
              "anchor_consistency": anchor_check,
              "records": records}
    out_json = results_dir / f"eval_v5_{tag}_{ts}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log(f"结果: {out_json}")
    write_markdown(results_dir / f"eval_v5_{tag}_{ts}.md",
                   meta, gate_md, kw_sum, pre_mean, jd_sum,
                   comparison, arm_b, audit, overrefusal_res, deviation)

    print(json.dumps({"status": "OK", "tag": tag,
                      "result_json": str(out_json),
                      "gate_pass": f"{gate_summary['n_pass']}/{gate_summary['n_generated']}",
                      "kw_post_alias_mean": kw_sum["mean"],
                      "kw_pre_alias_mean": pre_mean,
                      "judge_armA_mean": jd_sum["mean"],
                      "miss_audit_n": audit["n_queue"],
                      "overrefusal": overrefusal_res,
                      "judge_model": judge_model},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
