#!/usr/bin/env python
"""E0 诊断2: 保留 enable_thinking=False 渲染的空 think 块, 不剥离。"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pipeline_v4" / "src"))

with open(PROJECT_ROOT / "pipeline_v4/configs/eval_v4.json", encoding="utf-8") as f:
    cfg = json.load(f)
items = [json.loads(l) for l in open(PROJECT_ROOT / cfg["eval_file"], encoding="utf-8") if l.strip()]

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

def render(q):
    # enable_thinking=False 渲染, 保留空 think 块
    return tok.apply_chat_template(
        [{"role": "system", "content": cfg["chatml"]["system_message"]},
         {"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)

def gen(prompt, max_new=256):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    seq = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(seq, skip_special_tokens=False)

for it in items[:3]:
    p = render(it["question"])
    print(f"\n===== {it['id']} prompt 尾 80 字符 =====", flush=True)
    print(repr(p[-80:]), flush=True)
    print("----- 生成 (256 tok, raw) -----", flush=True)
    print(repr(gen(p)[:500]), flush=True)
print("\nDIAG2_DONE", flush=True)
