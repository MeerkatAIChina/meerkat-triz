#!/usr/bin/env python
"""
E6 通用能力探针评测 (决策门 G6 数据): base / v2 / v5a × 120 题。

协议 (与 v5 harness 同源):
  - BF16 加载 Qwen3.6-35B-A3B, 可选挂载 LoRA 适配器;
  - render_prompt 保留空 think 块 (E0 铁律, 冒烟断言);
  - 贪心生成 (do_sample=False), max_new_tokens=1024;
  - 中性系统 prompt (通用能力, 不用 TRIZ 专家人格);
  - 生成后 strip_closed_think 仅剥离闭合 think 块。

产物 (断点续跑): results/e6_probe/gen_<arm>.jsonl  (每行 {id, subcategory, response})
评分在本地做 (关键词命中率 + 配对 bootstrap), 本脚本只负责生成。

用法:
  venv_v5/bin/python pipeline_v5/eval/probe_gen_eval_v5.py --arms base,v2,v5a
"""

import argparse
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

ARMS = {
    "base": None,
    "v2": "models/meerkat_triz_adapter_v2",
    "v5a": "models/meerkat_triz_adapter_v5",
}
PROBE_FILE = PROJECT_ROOT / "data/processed/v5_data/general_probe_v5.json"
OUT_DIR = PROJECT_ROOT / "results" / "e6_probe"
BASE_MODEL = PROJECT_ROOT / "models" / "Qwen3.6-35B-A3B"
SYSTEM_MESSAGE = "你是乐于助人的 AI 助手。请用中文简洁、直接地回答用户的问题。"
MAX_NEW_TOKENS = 1024


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def gen_arm(arm: str, adapter_rel) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUT_DIR / f"gen_{arm}.jsonl"
    done = {}
    if cache.is_file():
        with open(cache, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done[r["id"]] = r
    with open(PROBE_FILE, encoding="utf-8") as f:
        items = json.load(f)
    todo = [it for it in items if it["id"] not in done]
    log(f"[{arm}] 已完成 {len(done)}/{len(items)}, 待生成 {len(todo)}")
    if not todo:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"[{arm}] 加载 BF16 基座: {BASE_MODEL}")
    tok = AutoTokenizer.from_pretrained(str(BASE_MODEL), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL), torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True)
    if adapter_rel:
        from peft import PeftModel
        adapter_path = PROJECT_ROOT / adapter_rel
        log(f"[{arm}] 挂载适配器: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    t_start = time.time()
    with open(cache, "a", encoding="utf-8") as f:
        for i, it in enumerate(todo, 1):
            prompt = render_prompt(tok, SYSTEM_MESSAGE, it["question"])
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
            f.write(json.dumps({
                "id": it["id"], "subcategory": it["subcategory"],
                "response": resp}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t_start
                eta = el / i * (len(todo) - i) / 60
                log(f"[{arm}] {i}/{len(todo)} 本步 {time.time()-t0:.1f}s "
                    f"ETA {eta:.1f}min")
    del model
    torch.cuda.empty_cache()
    log(f"[{arm}] 完成, 用时 {(time.time()-t_start)/60:.1f}min")


def main():
    ap = argparse.ArgumentParser(description="E6 通用探针生成 (base/v2/v5a)")
    ap.add_argument("--arms", default="base,v2,v5a")
    args = ap.parse_args()
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if arm not in ARMS:
            raise SystemExit(f"未知臂: {arm} (可选: {list(ARMS)})")
        gen_arm(arm, ARMS[arm])
    log("全部臂完成")


if __name__ == "__main__":
    main()
