#!/usr/bin/env python
"""E0: 增量 judge 驱动 — 与 GPU 生成并行, 边出边评。

复用 eval_harness 的 run_judge/probe_judge_models (同款缓存
results/v4_judge_base_goldfix.json, RPM=3, T=0, 指数退避, 断点续跑)。
轮询 gen 缓存 results/v4_gen_base_goldfix.jsonl, 把已生成但未评分的题送 judge;
gen 满 100 且 judge 满 100 后退出。随后由续跑手册第 2/3 步接管。
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "pipeline_v4" / "src"))
import eval_harness as eh  # noqa: E402

TAG = "base_goldfix"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    with open(PROJECT_ROOT / "pipeline_v4/configs/eval_v4.json", encoding="utf-8") as f:
        cfg = json.load(f)
    items = eh.load_gold(PROJECT_ROOT / cfg["eval_file"])
    results_dir = PROJECT_ROOT / cfg["results_dir"]
    gen_cache = results_dir / cfg["generation"]["cache_file_template"].format(tag=TAG)
    judge_cache = results_dir / cfg["judge"]["cache_file_template"].format(tag=TAG)

    judge_model, details, same_origin = eh.probe_judge_models(cfg)
    log(f"judge 选定: {judge_model} (same_origin_fallback={same_origin})")
    if judge_model != "moonshot-v1-32k":
        log(f"[注意] judge 不是 moonshot-v1-32k, 实际为 {judge_model}")

    while True:
        responses = eh.load_gen_cache(gen_cache)
        scored = set()
        if judge_cache.is_file():
            with open(judge_cache, encoding="utf-8") as f:
                scored = set(json.load(f))
        ready = [it for it in items
                 if it["id"] in responses and it["id"] not in scored]
        log(f"gen {len(responses)}/100, judge {len(scored)}/100, 本轮送评 {len(ready)}")
        if ready:
            eh.run_judge(ready, responses, judge_model, cfg, judge_cache,
                         dry_run=False)
        if len(responses) >= len(items) and len(scored) + len(ready) >= len(items):
            # 最终确认
            with open(judge_cache, encoding="utf-8") as f:
                final = json.load(f)
            missing = [it["id"] for it in items
                       if it["id"] not in final or final[it["id"]] is None]
            log(f"judge 完成度 {len(final)}/100, 缺失 {missing}")
            if len(final) >= len(items):
                log("JUDGE_ALL_DONE")
                return
        time.sleep(120)


if __name__ == "__main__":
    main()
