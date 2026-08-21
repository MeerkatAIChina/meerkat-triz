#!/usr/bin/env python3
"""v6 异族评委翻转率探针 —— 与 bridge_gpt55.py flip 模式协议逐字一致:
臂 A rubric / batch=5 / T=0 / 403 降级单条 / 3 次重复 / 任一维度不一致即记翻转。
数据源改为 v6 gold + v6 锚点(base) 前 50 题。
用法: BRIDGE_JUDGE=<model> venv_v5/bin/python flip_probe_v6.py
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE  # 脚本置于项目根目录
sys.path.insert(0, str(PROJECT / "pipeline_v5" / "eval"))

from external_judge_track import call_judge, log  # noqa: E402
from judge_arms import build_judge_user_arm_a  # noqa: E402

GOLD = PROJECT / "data" / "processed" / "v5_data" / "v5_gold.jsonl"
ANCHOR = PROJECT / "results" / "v5" / "eval_v5_base_goldfix_v5_qwen38_20260815_023643.json"
JUDGE = os.environ.get("BRIDGE_JUDGE", "claude-opus-4-8")
BATCH_SIZE = 5


def load_items_and_responses():
    gold = {}
    with open(GOLD, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                gold[r["id"]] = r
    anchor = json.load(open(ANCHOR, encoding="utf-8"))
    resps = {r["id"]: r["response"] for r in anchor["records"]
             if r.get("response")}
    items = [{"id": q, "subset": g.get("subset", ""), "question": g["question"],
              "reference_answer": g.get("reference_answer", ""),
              "keywords": g.get("keywords", [])}
             for q, g in sorted(gold.items())]
    # 只保留 base 有回答的题
    items = [it for it in items if it["id"] in resps]
    return items, resps


def judge_items(client, items, resps, tag):
    out = {}
    todo = list(items)
    for bi in range(0, len(todo), BATCH_SIZE):
        batch = todo[bi:bi + BATCH_SIZE]
        user = build_judge_user_arm_a(batch, resps)
        res = call_judge(client, JUDGE, user)
        if isinstance(res, dict):
            for it in batch:
                if it["id"] in res:
                    out[it["id"]] = res[it["id"]]
        missing = [it for it in batch
                   if it["id"] not in out and res != "__BLOCKED__"]
        if res == "__BLOCKED__":
            missing = batch[:]
        for it in missing:
            single = call_judge(client, JUDGE,
                                build_judge_user_arm_a([it], resps))
            if single == "__BLOCKED__":
                out[it["id"]] = "__BLOCKED__"
            elif isinstance(single, dict):
                out[it["id"]] = single.get(it["id"])
        log(f"  {tag} 进度 {len(out)}/{len(items)}")
    return out


def mode_flip(client, items, resps):
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
    rpt = HERE / f"flip_report_{JUDGE.replace('/', '_')}.json"
    with open(rpt, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    log(f"报告: {rpt}")


def main():
    from openai import OpenAI
    key = os.environ.get("TENSORIS_API_KEY")
    if not key:
        raise RuntimeError("TENSORIS_API_KEY 未设置")
    client = OpenAI(api_key=key, base_url="https://api.tensoris.ai/v1")
    items, resps = load_items_and_responses()
    log(f"装配: {len(items)} 题有 base 回答, judge={JUDGE}")
    mode_flip(client, items, resps)


if __name__ == "__main__":
    main()
