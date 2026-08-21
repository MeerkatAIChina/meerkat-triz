#!/usr/bin/env python
"""E0 诊断: 一次模型加载, 测试多种 anti-think 生成策略 (各 96 token)。"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pipeline_v4" / "src"))

EMPTY_THINK = "<think>\n\n</think>\n\n"
PREFILL = "好的,下面直接给出回答:\n"

with open(PROJECT_ROOT / "pipeline_v4/configs/eval_v4.json", encoding="utf-8") as f:
    cfg = json.load(f)
items = [json.loads(l) for l in open(PROJECT_ROOT / cfg["eval_file"], encoding="utf-8") if l.strip()]
q = items[0]["question"]

import compat  # noqa: F401
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(str(PROJECT_ROOT / cfg["base_model_path"]),
                                    trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    str(PROJECT_ROOT / cfg["base_model_path"]), torch_dtype=torch.bfloat16,
    device_map="cuda:0", trust_remote_code=True)
model.eval()
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def render(prefill=None):
    p = tok.apply_chat_template(
        [{"role": "system", "content": cfg["chatml"]["system_message"]},
         {"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p = p.replace(EMPTY_THINK, "")
    return p + (prefill or "")

think_ids = tok("<think>", add_special_tokens=False)["input_ids"]
think_end_ids = tok("</think>", add_special_tokens=False)["input_ids"]
print("think ids:", think_ids, "think_end ids:", think_end_ids, flush=True)
bad = [think_ids, think_end_ids]

def gen(prompt, max_new=96, bad_words=None):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    kw = {"max_new_tokens": max_new, "do_sample": False,
          "pad_token_id": tok.pad_token_id}
    if bad_words:
        kw["bad_words_ids"] = bad_words
    with torch.no_grad():
        out = model.generate(**inputs, **kw)
    seq = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(seq, skip_special_tokens=False)

print("\n===== S1: bad_words 禁 think, 无 prefill =====", flush=True)
print(repr(gen(render(), bad_words=bad)[:600]), flush=True)
print("\n===== S2: prefill 无 bad_words (raw, 含 special) =====", flush=True)
print(repr(gen(render(PREFILL))[:600]), flush=True)
print("\n===== S3: prefill + bad_words =====", flush=True)
print(repr(gen(render(PREFILL), bad_words=bad)[:600]), flush=True)
print("\nDIAG_DONE", flush=True)
