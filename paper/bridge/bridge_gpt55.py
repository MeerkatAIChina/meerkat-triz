#!/usr/bin/env python3
"""D6 桥接实验: 新/旧评委 (步骤 2 翻转率 + 步骤 3 臂重评分)。

协议与 contest_judge.py 逐字一致: 臂 A rubric, batch=5, T=0,
403 降级单条, 缓存断点续跑 (逐批落盘)。评委由 BRIDGE_JUDGE 指定。

用法:
  BRIDGE_JUDGE=gpt-5.5 TENSORIS_API_KEY=... python3 bridge_gpt55.py flip
  BRIDGE_JUDGE=gpt-5.4 TENSORIS_API_KEY=... python3 bridge_gpt55.py score [arm ...]
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT / "pipeline_v5" / "eval"))

from external_judge_track import call_judge, log  # noqa: E402
from judge_arms import build_judge_user_arm_a  # noqa: E402

GOLD = PROJECT / "paper" / "external_review" / "v5_gold.jsonl"
ARMS = {
    "base": PROJECT / "paper" / "contest" / "gen_base.jsonl",
    "v5a": PROJECT / "paper" / "contest" / "gen_v5a.jsonl",
    "base_eqlen": PROJECT / "paper" / "external_review_v5b" / "v5_gen_base_eqlen.jsonl",
}
JUDGE = os.environ.get("BRIDGE_JUDGE", "gpt-5.5")
BATCH_SIZE = 5
WORKERS = 8


def load_gold():
    gold = {}
    with open(GOLD, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                gold[r["id"]] = r
    return [{"id": q, "subset": g.get("subset", ""), "question": g["question"],
             "reference_answer": g.get("reference_answer", ""),
             "keywords": g.get("keywords", [])}
            for q, g in sorted(gold.items())]


def load_responses(path):
    resps = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("response"):
                    resps[r["id"]] = r["response"]
    return resps


def judge_items(client, items, resps, tag, cache=None, cache_path=None):
    """对 items 评分 (batch=5, 单条降级), 逐批落盘, 返回 {id: scores}。"""
    out = cache if cache is not None else {}
    todo = [it for it in items if it["id"] not in out]
    batches = [todo[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    wlock = threading.Lock()

    def flush():
        if cache_path:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=1)

    def do_batch(batch):
        user = build_judge_user_arm_a(batch, resps)
        res = call_judge(client, JUDGE, user)
        missing = batch[:] if (res == "__BLOCKED__" or not isinstance(res, dict)) \
            else [it for it in batch if it["id"] not in res]
        if isinstance(res, dict):
            with wlock:
                for it in batch:
                    if it["id"] in res:
                        out[it["id"]] = res[it["id"]]
        for it in missing:
            single = call_judge(client, JUDGE,
                                build_judge_user_arm_a([it], resps))
            with wlock:
                if single == "__BLOCKED__":
                    out[it["id"]] = "__BLOCKED__"
                else:
                    out[it["id"]] = single.get(it["id"]) if isinstance(single, dict) else None
        with wlock:
            flush()
            log(f"  {tag} 进度 {len(out)}/{len(items)}")

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(do_batch, b) for b in batches]
        for f in as_completed(futs):
            f.result()
    flush()
    return out


def mode_flip(client, items, resps):
    """50 题 x 3 次重复, 计算逐题翻转率 (任一维度不一致即记翻转)。"""
    sample = items[:50]
    runs = []
    for k in range(3):
        log(f"=== 翻转率 run {k + 1}/3 ===")
        runs.append(judge_items(client, sample, resps, f"flip{k + 1}"))
    dims = ("accuracy", "completeness", "triz_correctness", "structure", "overall")
    flips = 0
    detail = []
    for it in sample:
        q = it["id"]
        cells = []
        for r in runs:
            s = r.get(q)
            cells.append(None if not isinstance(s, dict) else
                         tuple(s.get(d) for d in dims))
        if len({c for c in cells if c is not None}) > 1 or any(c is None for c in cells):
            flips += 1
            detail.append((q, cells))
    rate = flips / len(sample)
    log(f"翻转率: {flips}/{len(sample)} = {rate:.3f} (D6 门限 ≤0.02)")
    for q, cells in detail[:8]:
        log(f"  翻转 {q}: {cells}")
    rep = {"judge": JUDGE, "n": len(sample), "runs": 3,
           "flips": flips, "flip_rate": rate,
           "pass_D6": rate <= 0.02,
           "detail": [{"id": q, "cells": [list(c) if c else None for c in cells]}
                      for q, cells in detail]}
    rpt = HERE / f"flip_report_{JUDGE}.json"
    with open(rpt, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    log(f"报告: {rpt}")


def mode_score(client, items, arm_names):
    for arm in arm_names:
        cache_path = HERE / f"contest_cache_{JUDGE}_{arm}.json"
        cache = json.load(open(cache_path, encoding="utf-8")) \
            if cache_path.is_file() else {}
        resps = load_responses(ARMS[arm])
        todo = [it for it in items if it["id"] in resps and it["id"] not in cache]
        log(f"=== {JUDGE} / {arm}: 缓存 {len(cache)}, 待评 {len(todo)} ===")
        if not todo:
            continue
        judge_items(client, [it for it in items if it["id"] in resps],
                    resps, arm, cache=cache, cache_path=cache_path)
        log(f"  {arm} 完成, 缓存 {len(cache)}")
    log("臂重评分完成")


def main():
    from openai import OpenAI
    key = os.environ.get("TENSORIS_API_KEY")
    if not key:
        raise RuntimeError("TENSORIS_API_KEY 未设置")
    client = OpenAI(api_key=key, base_url="https://api.tensoris.ai/v1")
    items = load_gold()
    mode = sys.argv[1] if len(sys.argv) > 1 else "flip"
    if mode == "flip":
        resps = load_responses(ARMS["base"])
        mode_flip(client, items, resps)
    else:
        arm_names = sys.argv[2:] or ["base_eqlen"]
        mode_score(client, items, arm_names)


if __name__ == "__main__":
    main()
