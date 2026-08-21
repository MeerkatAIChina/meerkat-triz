#!/usr/bin/env python
"""远端 moonshot 主轨擂台计分 (与臂 A 协议一致, 批量 5 题, T=0)。

输入: results/v5/contest_gen_<model>.jsonl + v5_gold.jsonl
输出: results/v5/contest_cache_moonshot_<model>.json (与本地擂台缓存同格式)
断点续跑; 纯 API 任务, 与 GPU 训练无冲突。

用法:
  cd /home/meerkat/mongoose_ai
  venv_v5/bin/python pipeline_v5/eval/moonshot_contest_judge.py [--arms a b]
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from judge_arms import JUDGE_SYSTEM_ARM_A, build_judge_user_arm_a  # noqa: E402

GOLD = PROJECT / "data/processed/v5_data/v5_gold.jsonl"
RESULTS = PROJECT / "results/v5"
MODEL = "moonshot-v1-32k"
BATCH_SIZE = 5
WORKERS = 4
RPM = 6
MAX_API_RETRIES = 5

_RATE_LOCK = threading.Lock()
_LAST = [0.0]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rate_gate():
    with _RATE_LOCK:
        wait = 60.0 / RPM - (time.time() - _LAST[0])
        if wait > 0:
            time.sleep(wait)
        _LAST[0] = time.time()


def get_client():
    from openai import OpenAI
    key = os.environ.get("MOONSHOT_API_KEY")
    if not key:
        raise RuntimeError("MOONSHOT_API_KEY 未设置")
    return OpenAI(api_key=key, base_url="https://api.moonshot.cn/v1")


def parse_json_array(text):
    import re, json as _j
    t = re.sub(r"^```(?:json)?\s*", "", text.strip())
    t = re.sub(r"\s*```$", "", t)
    s, e = t.find("["), t.rfind("]")
    if s == -1 or e <= s:
        raise ValueError("未找到 JSON 数组: " + t[:100])
    return _j.loads(t[s:e + 1])


def call_judge(client, user):
    delay = 5
    for attempt in range(MAX_API_RETRIES):
        try:
            rate_gate()
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": JUDGE_SYSTEM_ARM_A},
                          {"role": "user", "content": user}],
                max_tokens=2000, temperature=0.0)
            out = {}
            for e in parse_json_array(resp.choices[0].message.content):
                if isinstance(e, dict) and "id" in e and "overall" in e:
                    out[str(e["id"])] = {k: e.get(k) for k in
                                         ("accuracy", "completeness",
                                          "triz_correctness", "structure",
                                          "overall")}
            if out:
                return out
            raise ValueError("无有效评分条目")
        except Exception as ex:
            log(f"  moonshot 失败 ({attempt + 1}/{MAX_API_RETRIES}): {str(ex)[:120]}")
            if attempt < MAX_API_RETRIES - 1:
                time.sleep(delay)
                delay = min(delay * 2, 60)
    return None


def judge_arm(client, arm, items, resps):
    cache_path = RESULTS / f"contest_cache_moonshot_{arm}.json"
    cache = {}
    if cache_path.is_file():
        cache = json.load(open(cache_path, encoding="utf-8"))
    todo = [it for it in items if it["id"] in resps and it["id"] not in cache]
    if not todo:
        log(f"moonshot/{arm}: 缓存完整 ({len(cache)})")
        return
    log(f"moonshot/{arm}: 缓存 {len(cache)}, 待评 {len(todo)}")
    wlock = threading.Lock()
    batches = [todo[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    state = {"n": 0}

    def flush():
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=1)

    def do_batch(batch):
        res = call_judge(client, build_judge_user_arm_a(batch, resps))
        with wlock:
            missing = batch[:] if res is None else \
                [it for it in batch if it["id"] not in res]
            if isinstance(res, dict):
                for it in batch:
                    if it["id"] in res:
                        cache[it["id"]] = res[it["id"]]
        for it in missing:
            single = call_judge(client, build_judge_user_arm_a([it], resps))
            with wlock:
                cache[it["id"]] = single.get(it["id"]) if single else None
        with wlock:
            state["n"] += 1
            flush()
            if state["n"] % 5 == 0:
                log(f"  moonshot/{arm} 进度 {len(cache)}/{len(resps)}")

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(do_batch, b) for b in batches]
        for f in as_completed(futs):
            f.result()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*",
                    default=["gpt-5.4", "claude-sonnet-4-6",
                             "claude-opus-4-8", "gemini-3.5-flash"])
    args = ap.parse_args()
    gold = {}
    for line in open(GOLD, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            gold[r["id"]] = r
    items = [{"id": q, "subset": g.get("subset", ""), "question": g["question"],
              "reference_answer": g.get("reference_answer", ""),
              "keywords": g.get("keywords", [])}
             for q, g in sorted(gold.items())]
    client = get_client()
    for arm in args.arms:
        resps = {}
        for line in open(RESULTS / f"contest_gen_{arm}.jsonl", encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("response"):
                    resps[r["id"]] = r["response"]
        log(f"=== {arm}: {len(resps)} 条 ===")
        judge_arm(client, arm, items, resps)
    log("moonshot 主轨全部完成")


if __name__ == "__main__":
    main()
