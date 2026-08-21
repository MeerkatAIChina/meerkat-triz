#!/usr/bin/env python
"""一次性修复: v4_1 发货适配器 LoRA 权重 F32→BF16 转置 + 重验证 + adapter_info.json 更新。

背景 (2026-07-29): v4_1 训练从 checkpoint-100 resume 后, transformers 5.10.1 的
checkpoint 加载把 LoRA 权重上转为 FP32 (resume 后新保存的 checkpoint lora_B 全 F32;
run1 保存的为 BF16, 已被 save_total_limit 轮转删除)。早停 (step 1200, best@900) 后
发货验证 lora_B:dtype全BF16 FAILED → adapter_info status=FAILED, 链条中止。

本脚本把已发货适配器与 best/ 的 adapter_model.safetensors 中非 BF16 浮点张量转置回
BF16 (数值影响可忽略: run1 即 BF16 训练, 评测加载亦为 BF16), 然后用
checkpointing.validate_adapter_dir 重验证并更新 adapter_info.json。

用法: venv_v5/bin/python pipeline_v4/src/repair_adapter_dtype.py
"""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "pipeline_v4" / "src"))
import checkpointing  # noqa: E402


def cast_to_bf16(st_path: Path) -> int:
    tensors = load_file(str(st_path))
    n_cast = 0
    for k, v in tensors.items():
        if v.is_floating_point() and v.dtype != torch.bfloat16:
            tensors[k] = v.to(torch.bfloat16)
            n_cast += 1
    if n_cast:
        save_file(tensors, str(st_path), metadata={"format": "pt"})
    return n_cast


def main():
    adapter_dir = PROJECT_ROOT / "models/meerkat_triz_adapter_v4_1"
    best_dir = PROJECT_ROOT / "checkpoints/qlora_triz_v4_1/best"
    for d in (adapter_dir, best_dir):
        st = d / "adapter_model.safetensors"
        if st.is_file():
            print(f"{d}: 转置 {cast_to_bf16(st)} 个张量 → BF16")

    result = checkpointing.validate_adapter_dir(str(adapter_dir))
    print(json.dumps({"status": result["status"],
                      "checks": [[c["name"], c["status"], c["detail"]]
                                 for c in result["checks"]]},
                     ensure_ascii=False, indent=1))

    info_path = adapter_dir / "adapter_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["status"] = result["status"]
    info["validation"] = {"status": result["status"], "checks": result["checks"]}
    info["sha256"] = result["sha256"]
    info["dtype_repair"] = ("2026-07-29: resume 上转 FP32 的 LoRA 权重已转置回 BF16 "
                            "(repair_adapter_dtype.py); 数值影响可忽略, 评测加载本为 BF16")
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"adapter_info.json 已更新: status={result['status']}")
    sys.exit(0 if result["status"] == "PASSED" else 1)


if __name__ == "__main__":
    main()
