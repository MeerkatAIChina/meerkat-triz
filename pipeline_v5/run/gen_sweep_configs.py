#!/usr/bin/env python
"""生成 v5 Day2 P0 六组小扫配置 (lr × rsLoRA)。

小扫矩阵 (v5_优化微调方案.md §5.3 P0 / sec2_training.md §8.1):
  lr ∈ {1e-4, 2e-4, 5e-4} × use_rslora ∈ {False, True}, 各 0.5 epoch
  (= ceil(10698/8)×0.5 = 1338×0.5 = 669 步), 其余与 train_v5_base.json 全同,
  eval_steps=save_steps=100。
输出: pipeline_v5/configs/sweep/sweep_lr{...}_rs{...}.json × 6
"""
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "pipeline_v5/configs/train_v5_base.json"
OUT = PROJECT_ROOT / "pipeline_v5/configs/sweep"

LRS = [1e-4, 2e-4, 5e-4]
RS = [False, True]
D = 10698  # v5_train.jsonl 条数


def lr_tag(lr: float) -> str:
    return f"{lr:.0e}".replace("-0", "-")  # 1e-4 / 2e-4 / 5e-4


def main() -> None:
    with open(BASE) as f:
        base = json.load(f)
    steps_per_epoch = math.ceil(D / (base["per_device_train_batch_size"]
                                     * base["gradient_accumulation_steps"]))
    sweep_steps = int(steps_per_epoch * 0.5)  # 1338 * 0.5 = 669
    OUT.mkdir(parents=True, exist_ok=True)
    for lr in LRS:
        for rs in RS:
            name = f"sweep_lr{lr_tag(lr)}_rs{rs}"
            cfg = dict(base)
            cfg["run_name"] = name
            cfg["learning_rate"] = lr
            cfg["max_steps"] = sweep_steps
            cfg["output_dir"] = f"checkpoints/{name}"
            cfg["adapter_output_dir"] = f"models/sweep_adapters/{name}"
            cfg["lora"] = dict(base["lora"], use_rslora=rs)
            cfg["_notes"] = dict(base.get("_notes", {}),
                                 sweep=f"P0 小扫臂: lr={lr}, use_rslora={rs}, "
                                       f"0.5 epoch = {sweep_steps} 步 "
                                       f"(ceil({D}/8)={steps_per_epoch} × 0.5)")
            path = OUT / f"{name}.json"
            with open(path, "w") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            print(f"written: {path}")
    print(f"共 {len(LRS) * len(RS)} 组, 每组 {sweep_steps} 步 (0.5 epoch)")


if __name__ == "__main__":
    sys.exit(main())
