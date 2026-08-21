#!/usr/bin/env python
"""
v5b PR 存量审计: 分类 v5a 训练集 196 条 principle_recommendation 样本。

背景: 归因发现 gated_corpus 162 条 v2 遗产 PR 中存在错标
(如"ARIZ 中有哪些资源"被标成 PR); 54 条 styleC 长答同样需审
(如"专利规避设计有哪些核心原则"非原理推荐题)。

分类 (kimi-k2.5, 批量 10 条/请求, RPM=3, 断点续跑):
  true_pr  : 明确要求"针对某问题/矛盾推荐 TRIZ 发明原理"
  related  : 围绕具体发明原理的应用/解释 (与原理推荐强相关, 保留)
  mislabel : 与发明原理无关 (剔除)

产物 (data/processed/v5b_data/):
  pr_legacy_audit.jsonl   每条 {idx, verdict, reason}
  pr_legacy_audit.md      人工复核摘要

用法: venv_v5/bin/python pipeline_v5/src/pr_legacy_audit_v5b.py
"""

import json
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN = PROJECT_ROOT / "data/processed/v5_data/final/v5_train_v5a.jsonl"
OUT_DIR = PROJECT_ROOT / "data/processed/v5b_data"
OUT = OUT_DIR / "pr_legacy_audit.jsonl"
SUMMARY = OUT_DIR / "pr_legacy_audit.md"

MODEL = "kimi-k2.5"
BASE_URL = "https://api.moonshot.cn/v1"
RPM = 3
BATCH = 10
WORKERS = 3
MAX_API_RETRIES = 5

AUDIT_SYSTEM = (
    "你是数据集质量审计员。给定若干条标注为 'principle_recommendation'"
    "(TRIZ 发明原理推荐) 的训练样本 (问题+答案开头), 逐条判定类别:\n"
    "  true_pr  = 问题明确要求针对某个问题/矛盾推荐或选择 TRIZ 发明原理;\n"
    "  related  = 问题围绕具体发明原理的应用、解释、举例 "
    "(虽不是推荐题但与发明原理直接相关);\n"
    "  mislabel = 问题与 TRIZ 发明原理无关 (如纯 ARIZ/纯概念/纯管理题)。\n"
    "只输出 JSON 数组: [{\"idx\": 序号, \"verdict\": \"true_pr|related|mislabel\", "
    "\"reason\": \"<=20字\"}, ...]"
)

ANS_RE = re.compile(r"\[.*\]", re.S)

_RATE_LOCK = threading.Lock()
_LAST_START = [0.0]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def rate_gate():
    with _RATE_LOCK:
        wait = 60.0 / RPM - (time.time() - _LAST_START[0])
        if wait > 0:
            time.sleep(wait)
        _LAST_START[0] = time.time()


def get_client():
    from openai import OpenAI
    key = os.environ.get("MOONSHOT_API_KEY")
    if not key:
        raise RuntimeError("MOONSHOT_API_KEY 未设置")
    return OpenAI(api_key=key, base_url=BASE_URL)


def extract_user(prompt):
    m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt, re.S)
    return m.group(1).strip() if m else prompt[:300]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pr = []
    with open(TRAIN, encoding="utf-8") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            if d.get("subset") == "principle_recommendation":
                pr.append({"line_no": i,
                           "question": extract_user(d["prompt"]),
                           "completion": d["completion"]})
    log(f"PR 存量: {len(pr)} 条")

    done = set()
    if OUT.is_file():
        for line in open(OUT, encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line)["line_no"])
    todo = [p for p in pr if p["line_no"] not in done]
    log(f"已审 {len(done)}, 待审 {len(todo)}")

    client = get_client()
    fout = open(OUT, "a", encoding="utf-8")
    write_lock = threading.Lock()

    def audit_batch(bi, batch):
        parts = []
        for j, it in enumerate(batch, 1):
            parts.append(f"[{j}] 问题: {it['question'][:200]}\n"
                         f"    答案开头: {it['completion'][:150]}")
        user = "\n".join(parts) + "\n请逐条判定, 输出 JSON 数组。"
        delay = 5
        for attempt in range(MAX_API_RETRIES):
            try:
                rate_gate()
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": AUDIT_SYSTEM},
                              {"role": "user", "content": user}],
                    max_tokens=2000)
                text = resp.choices[0].message.content
                m = ANS_RE.search(text)
                arr = json.loads(m.group(0))
                by_idx = {e["idx"]: e for e in arr if isinstance(e, dict)}
                with write_lock:
                    for j, it in enumerate(batch, 1):
                        e = by_idx.get(j)
                        if e and e.get("verdict") in ("true_pr", "related", "mislabel"):
                            fout.write(json.dumps(
                                {"line_no": it["line_no"], "verdict": e["verdict"],
                                 "reason": e.get("reason", ""),
                                 "question_head": it["question"][:60]},
                                ensure_ascii=False) + "\n")
                        else:
                            fout.write(json.dumps(
                                {"line_no": it["line_no"], "verdict": "parse_fail",
                                 "reason": "", "question_head": it["question"][:60]},
                                ensure_ascii=False) + "\n")
                    fout.flush()
                log(f"批次 {bi} 完成")
                return
            except Exception as e:
                log(f"批次 {bi} 失败 (attempt {attempt+1}): {str(e)[:120]}")
                time.sleep(delay)
                delay = min(delay * 2, 90)
        with write_lock:
            for it in batch:
                fout.write(json.dumps(
                    {"line_no": it["line_no"], "verdict": "api_fail",
                     "reason": "", "question_head": it["question"][:60]},
                    ensure_ascii=False) + "\n")
            fout.flush()

    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(audit_batch, bi + 1, b) for bi, b in enumerate(batches)]
        for f in as_completed(futs):
            f.result()
    fout.close()

    # 汇总
    verdicts = []
    for line in open(OUT, encoding="utf-8"):
        if line.strip():
            verdicts.append(json.loads(line))
    cnt = Counter(v["verdict"] for v in verdicts)
    lines = ["# v5b PR 存量审计汇总", "",
             f"时间: {datetime.now().isoformat(timespec='seconds')} | "
             f"总数 {len(verdicts)}", "",
             f"- true_pr (保留): {cnt.get('true_pr', 0)}",
             f"- related (保留): {cnt.get('related', 0)}",
             f"- mislabel (剔除): {cnt.get('mislabel', 0)}",
             f"- 判定失败 (人工): {cnt.get('parse_fail', 0) + cnt.get('api_fail', 0)}",
             "", "## 剔除清单 (mislabel)", ""]
    for v in verdicts:
        if v["verdict"] == "mislabel":
            lines.append(f"- line {v['line_no']}: {v['question_head']} — {v['reason']}")
    lines += ["", "## 判定失败清单 (需人工)", ""]
    for v in verdicts:
        if v["verdict"] in ("parse_fail", "api_fail"):
            lines.append(f"- line {v['line_no']}: {v['question_head']}")
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    log(f"汇总: {dict(cnt)} -> {SUMMARY}")


if __name__ == "__main__":
    main()
