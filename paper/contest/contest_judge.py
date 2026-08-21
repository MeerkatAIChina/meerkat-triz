#!/usr/bin/env python3
"""TRIZ-Bench 擂台评委计分: 选手臂 × 家族回避评委矩阵。

协议: 与终审完全同款 (臂 A rubric, 批量 5 题, T=0, 403 降级单条,
     缓存 contest_cache_<judge>_<arm>.json 断点续跑)。
纪律: 评委与选手同家族回避 (claude 系列互评禁止; gpt/gemini 同理)。
     moonshot 主轨不在本脚本 (在远端用 MOONSHOT_API_KEY 跑, 见 README)。

输入: gen_<model>.jsonl (contestant_gen_v5.py 产物) + v5_gold.jsonl
输出: paper/contest/contest_cache_<judge>_<arm>.json

用法:
  PYTHONPATH=meerkat-triz/src python3 paper/contest/contest_judge.py \
      [--arms gpt-5.4 ...] [--time-budget 270]
"""

import argparse
import json
import threading
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT / "pipeline_v5" / "eval"))

from external_judge_track import call_judge, log  # noqa: E402
from judge_arms import JUDGE_SYSTEM_ARM_A, build_judge_user_arm_a  # noqa: E402

GOLD = PROJECT / "paper" / "external_review" / "v5_gold.jsonl"
BATCH_SIZE = 5
JUDGE_WORKERS = 8

FAMILY = {"claude": "anthropic", "gpt": "openai", "gemini": "google"}
JUDGES = ["claude-sonnet-4-6", "gpt-5.4", "gemini-3.5-flash"]


def family_of(model):
    return FAMILY.get(model.split("-")[0], model.split("-")[0])


def allowed_judges(arm):
    fam = family_of(arm)
    return [j for j in JUDGES if family_of(j) != fam]


def load_gold():
    gold = {}
    with open(GOLD, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                gold[r["id"]] = r
    items = [{"id": qid, "subset": g.get("subset", ""),
              "question": g["question"],
              "reference_answer": g.get("reference_answer", ""),
              "keywords": g.get("keywords", [])}
             for qid, g in sorted(gold.items())]
    return items


def load_responses(arm):
    path = HERE / f"gen_{arm}.jsonl"
    resps = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("response"):
                    resps[r["id"]] = r["response"]
    return resps


def judge_arm(client, judge, arm, items, resps, budget):
    cache_path = HERE / f"contest_cache_{judge}_{arm}.json"
    cache = {}
    if cache_path.is_file():
        cache = json.load(open(cache_path, encoding="utf-8"))
    todo = [it for it in items if it["id"] in resps and it["id"] not in cache]
    if not todo:
        log(f"{judge}/{arm}: 缓存完整 ({len(cache)})")
        return
    log(f"{judge}/{arm}: 缓存 {len(cache)}, 待评 {len(todo)}")
    t0 = time.time()
    wlock = threading.Lock()
    batches = [todo[bi:bi + BATCH_SIZE] for bi in range(0, len(todo), BATCH_SIZE)]
    state = {"done_batches": 0, "stop": False}

    def flush():
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=1)

    def do_batch(batch):
        if state["stop"]:
            return
        user = build_judge_user_arm_a(batch, resps)
        res = call_judge(client, judge, user)
        with wlock:
            if res == "__BLOCKED__":
                missing = batch[:]
            else:
                missing = [it for it in batch if res is None or it["id"] not in res]
                if isinstance(res, dict):
                    for it in batch:
                        if it["id"] in res:
                            cache[it["id"]] = res[it["id"]]
        for it in missing:
            single = call_judge(client, judge,
                                build_judge_user_arm_a([it], resps))
            with wlock:
                if single == "__BLOCKED__":
                    cache[it["id"]] = "__BLOCKED__"
                else:
                    cache[it["id"]] = single.get(it["id"]) if single else None
        with wlock:
            state["done_batches"] += 1
            flush()
            if state["done_batches"] % 5 == 0:
                done_n = len([1 for it in items if it["id"] in cache])
                log(f"  {judge}/{arm} 进度 {done_n}/{len(resps)}")
            if time.time() - t0 > budget:
                state["stop"] = True

    with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as ex:
        futs = [ex.submit(do_batch, b) for b in batches]
        for f in as_completed(futs):
            f.result()
    if state["stop"]:
        log(f"  时间预算用尽, 安全退出")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*",
                    default=["gpt-5.4", "claude-sonnet-4-6",
                             "claude-opus-4-8", "gemini-3.5-flash"])
    ap.add_argument("--time-budget", type=int, default=10**9,
                    help="每 (评委,臂) 组合的秒数上限")
    args = ap.parse_args()

    from openai import OpenAI
    key = os.environ.get("TENSORIS_API_KEY")
    if not key:
        raise RuntimeError("TENSORIS_API_KEY 未设置")
    client = OpenAI(api_key=key, base_url="https://api.tensoris.ai/v1")

    items = load_gold()
    for arm in args.arms:
        resps = load_responses(arm)
        judges = allowed_judges(arm)
        log(f"=== 选手 {arm}: {len(resps)} 条回答, 评委 {judges} ===")
        for j in judges:
            judge_arm(client, j, arm, items, resps, args.time_budget)
    log("全部评委任务完成")


if __name__ == "__main__":
    main()
