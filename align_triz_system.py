#!/usr/bin/env python3
"""把 v5b 训练数据的 system 提示对齐评测 system (TRIZ 专家助手)。"""
import json

SRC = "/home/chinux/jupyterlab/meerkatai/data/processed/v5b_data/final/v5_train_v5b.jsonl"
DST = "/home/chinux/jupyterlab/meerkatai/data/processed/v5b_data/final/v5_train_v5b_trizsys.jsonl"
EVAL_SYS = "你是 TRIZ 创新方法论专家助手, 用中文专业回答用户关于 TRIZ 理论、发明原理、矛盾分析、ARIZ 算法等方面的问题。"

n = 0
with open(SRC) as f, open(DST, "w") as out:
    for line in f:
        d = json.loads(line)
        prompt = d.get("prompt", "")
        if "<|im_start|>system" in prompt:
            marker = "<|im_start|>system"
            sys_pos = prompt.find(marker)
            end_pos = prompt.find("<|im_end|>", sys_pos)
            if end_pos > sys_pos:
                new_system = marker + "\n" + EVAL_SYS
                prompt = prompt[:sys_pos] + new_system + prompt[end_pos:]
                d["prompt"] = prompt
        out.write(json.dumps(d, ensure_ascii=False) + "\n")
        n += 1

print("已生成对齐评测 system 的数据:", DST, "|", n, "条")

# 验证第一条
with open(DST) as f:
    first = json.loads(f.readline())
    sys_part = first["prompt"].split("<|im_end|>")[0]
    print("第一条 system:", sys_part[:100])
