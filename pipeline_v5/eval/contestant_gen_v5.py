#!/usr/bin/env python
"""TRIZ-Bench 擂台: 前沿模型选手生成 (300 题金标, v5 评测协议)。

协议对齐 (与 base/v5a 金标生成完全一致):
  system: 你是 TRIZ 创新方法论专家助手, 用中文专业回答用户关于 TRIZ 理论、
          发明原理、矛盾分析、ARIZ 算法等方面的问题。
  max_tokens=2048, 逐题单条调用。

选手: gpt-5.4 / claude-sonnet-4-6 / claude-opus-4-8 / gemini-3.5-flash
通道: tensoris (TENSORIS_API_KEY)。断点续跑 (按 id 跳过)。

产物: paper/contest/gen_<model>.jsonl  每行 {id, response, model, mode}
用法: python3 pipeline_v5/eval/contestant_gen_v5.py [--models a b] [--limit N]
"""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
GOLD = PROJECT / "paper" / "external_review" / "v5_gold.jsonl"
OUT_DIR = PROJECT / "paper" / "contest"

BASE_URL = "https://api.tensoris.ai/v1"
MODELS = ["gpt-5.4", "claude-sonnet-4-6", "claude-opus-4-8", "gemini-3.5-flash"]
SYSTEM = ("你是 TRIZ 创新方法论专家助手, 用中文专业回答用户关于 TRIZ 理论、"
          "发明原理、矛盾分析、ARIZ 算法等方面的问题。")
MAX_TOKENS = 2048
WORKERS = 12
MIN_INTERVAL = 1.0        # 全局限速 ~30 RPM
MAX_API_RETRIES = 5

_RATE_LOCK = threading.Lock()
_LAST = [0.0]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def rate_gate():
    with _RATE_LOCK:
        wait = MIN_INTERVAL - (time.time() - _LAST[0])
        if wait > 0:
            time.sleep(wait)
        _LAST[0] = time.time()


def get_client():
    from openai import OpenAI
    key = os.environ.get("TENSORIS_API_KEY")
    if not key:
        raise RuntimeError("TENSORIS_API_KEY 未设置")
    return OpenAI(api_key=key, base_url=BASE_URL)


def gen_model(client, model, items, out_path):
    done = set()
    if out_path.is_file():
        for line in open(out_path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("response"):
                    done.add(r["id"])
    todo = [it for it in items if it["id"] not in done]
    log(f"{model}: 已完成 {len(done)}, 待生成 {len(todo)}")
    if not todo:
        return
    fout = open(out_path, "a", encoding="utf-8")
    wlock = threading.Lock()
    state = {"n": 0, "fail": 0}

    def one(it):
        delay = 5
        for attempt in range(MAX_API_RETRIES):
            try:
                rate_gate()
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": it["question"]}],
                    max_tokens=MAX_TOKENS)
                text = resp.choices[0].message.content or ""
                if not text.strip():
                    raise ValueError("empty content")
                with wlock:
                    fout.write(json.dumps(
                        {"id": it["id"], "response": text, "model": model,
                         "mode": "contest_v1"}, ensure_ascii=False) + "\n")
                    fout.flush()
                    state["n"] += 1
                    if state["n"] % 25 == 0:
                        log(f"{model}: {state['n']}/{len(todo)}")
                return
            except Exception as e:
                log(f"{model}/{it['id']} attempt {attempt+1} 失败: {str(e)[:120]}")
                time.sleep(delay)
                delay = min(delay * 2, 90)
        with wlock:
            state["fail"] += 1
            fout.write(json.dumps({"id": it["id"], "response": "", "model": model,
                                   "mode": "contest_v1", "error": "api_fail"},
                                  ensure_ascii=False) + "\n")
            fout.flush()

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(one, it) for it in todo]
        for f in as_completed(futs):
            f.result()
    fout.close()
    log(f"{model}: 完成 {state['n']}, 失败 {state['fail']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    items = [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]
    if args.limit:
        items = items[: args.limit]
    log(f"金标 {len(items)} 题")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = get_client()
    for m in args.models:
        gen_model(client, m, items, OUT_DIR / f"gen_{m}.jsonl")
    log("全部选手完成")


if __name__ == "__main__":
    main()
