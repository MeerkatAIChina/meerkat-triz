#!/usr/bin/env python
"""等长对照生成 (论文 Limitation ii, v5b 后方法论前置, 2026-07-31)。

目的: 解开 judge 轨的长度混淆——让 base 在逐题等长约束下重答 300 题,
篇幅目标 = 同题 v5a 答案长度。若 v5a−base 的 judge 差在等长后消失,
则该差主要为长度伪影; 若仍存在, 则为真实质量差。

协议: 除用户内容追加【篇幅要求】外, 与 eval_v5.json 完全一致
(TRIZ 专家 system / 贪心 / max_new_tokens=2048 / ChatML / strip_closed_think)。
断点续跑: 已完成的 id 跳过。

用法: venv_v5/bin/python pipeline_v5/eval/eqlen_gen_v5.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path("/home/meerkat/mongoose_ai")
sys.path.insert(0, str(PROJECT_ROOT / "pipeline_v5" / "eval"))
sys.path.insert(0, str(PROJECT_ROOT / "pipeline_v5" / "src"))

import compat  # noqa: F401,E402  peft×transformers 兼容补丁, 必须在模型加载前

from render import render_prompt, assert_empty_think_retained, strip_closed_think  # noqa: E402

BASE_MODEL = PROJECT_ROOT / "models/Qwen3.6-35B-A3B"
GOLD = PROJECT_ROOT / "data/processed/v5_data/v5_gold.jsonl"
V5A_GEN = PROJECT_ROOT / "results/v5/v5_gen_v5a_gold.jsonl"
OUT = PROJECT_ROOT / "results/v5/v5_gen_base_eqlen.jsonl"
SYSTEM_MESSAGE = ("你是 TRIZ 创新方法论专家助手, 用中文专业回答用户关于 "
                  "TRIZ 理论、发明原理、矛盾分析、ARIZ 算法等方面的问题。")
MAX_NEW_TOKENS = 2048


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def main():
    gold = [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]
    v5a = {json.loads(l)["id"]: json.loads(l)["response"]
           for l in open(V5A_GEN, encoding="utf-8") if l.strip()}
    done = {}
    if OUT.is_file():
        for l in open(OUT, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                done[r["id"]] = r
    todo = [it for it in gold if it["id"] not in done]
    log(f"等长生成: 已完成 {len(done)}/{len(gold)}, 待生成 {len(todo)}")
    if not todo:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"加载 BF16 基座: {BASE_MODEL}")
    tok = AutoTokenizer.from_pretrained(str(BASE_MODEL), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL), torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True)
    model.eval()

    t_start = time.time()
    with open(OUT, "a", encoding="utf-8") as f:
        for i, it in enumerate(todo, 1):
            n = len(v5a[it["id"]].strip())
            user = (it["question"].strip() +
                    f"\n\n【篇幅要求】请将回答控制在约 {n} 字 (允许 ±10% 浮动)。")
            prompt = render_prompt(tok, SYSTEM_MESSAGE, user)
            assert_empty_think_retained(prompt)
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                    pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True)
            resp = strip_closed_think(text)
            f.write(json.dumps({"id": it["id"], "mode": "eqlen",
                                "target_len": n, "actual_len": len(resp.strip()),
                                "response": resp}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t_start
                eta = el / i * (len(todo) - i) / 60
                log(f"{i}/{len(todo)} 本步 {time.time()-t0:.1f}s ETA {eta:.0f}min")

    log("等长生成完成")


if __name__ == "__main__":
    main()
