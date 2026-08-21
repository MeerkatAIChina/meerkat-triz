"""pipeline_v4 冒烟测试: dry-run 数据加载 + checkpointing 验证逻辑 (不碰 GPU)。"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = "/home/chinux/jupyterlab/meerkatai"
PY = os.path.join(ROOT, "venv_v5/bin/python")

tmp = tempfile.mkdtemp(prefix="v4_smoke_")

# ---- 1. 10 样本假 jsonl + dry-run ----
train_path = os.path.join(tmp, "fake_train.jsonl")
val_path = os.path.join(tmp, "fake_val.jsonl")
for path, n in ((train_path, 10), (val_path, 4)):
    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps({
                "prompt": f"<|im_start|>user\n问题{i}: 什么是TRIZ分割原理?<|im_end|>\n<|im_start|>assistant\n",
                "completion": f"分割原理是将物体分成独立部分。示例回答{i}。",
                "subset": "smoke" if i % 2 == 0 else "smoke_b",
            }, ensure_ascii=False) + "\n")

r = subprocess.run(
    [PY, "pipeline_v4/src/train.py", "--config", "pipeline_v4/configs/train_v4.json",
     "--train-file", train_path, "--val-file", val_path, "--dry-run"],
    cwd=ROOT, capture_output=True, text=True, timeout=110)
print("=== dry-run exit:", r.returncode)
print(r.stdout[-2500:])
assert r.returncode == 0, r.stderr[-2000:]
assert '"train_samples": 10' in r.stdout and '"val_samples": 4' in r.stdout
assert "subset 分布" in r.stdout
print("SMOKE 1 PASSED: dry-run 数据加载 + subset 统计 + 配置摘要")

# ---- 1b. 数据文件缺失时的优雅报错 ----
r = subprocess.run(
    [PY, "pipeline_v4/src/train.py", "--config", "pipeline_v4/configs/train_v4.json",
     "--train-file", os.path.join(tmp, "nonexistent.jsonl"),
     "--val-file", val_path, "--dry-run"],
    cwd=ROOT, capture_output=True, text=True, timeout=110)
print("=== missing-file exit:", r.returncode)
assert r.returncode == 2 and "请先运行数据构建" in r.stdout
print("SMOKE 1b PASSED: 数据缺失优雅报错 exit=2")

# ---- 2. checkpointing.validate_adapter_dir: 全零 lora_B → FAILED ----
sys.path.insert(0, os.path.join(ROOT, "pipeline_v4/src"))
import torch
from safetensors.torch import save_file
import checkpointing

def make_adapter(dirpath, zero_b: bool, dtype=torch.bfloat16):
    os.makedirs(dirpath, exist_ok=True)
    tensors = {}
    for i in range(3):
        tensors[f"base_model.model.layers.{i}.q_proj.lora_A.weight"] = torch.randn(64, 4096).to(dtype)
        b = torch.zeros(4096, 64) if zero_b else torch.randn(4096, 64)
        tensors[f"base_model.model.layers.{i}.q_proj.lora_B.weight"] = b.to(dtype)
    save_file(tensors, os.path.join(dirpath, "adapter_model.safetensors"))
    with open(os.path.join(dirpath, "adapter_config.json"), "w") as f:
        json.dump({"r": 64, "lora_alpha": 128, "peft_type": "LORA",
                   "target_modules": ["q_proj"]}, f)

bad_dir = os.path.join(tmp, "adapter_zero")
make_adapter(bad_dir, zero_b=True)
res = checkpointing.validate_adapter_dir(bad_dir)
print("=== zero-lora_B validation:", res["status"])
assert res["status"] == "FAILED"
failed_names = [c["name"] for c in res["checks"] if c["status"] == "FAILED"]
assert "lora_B:全部非零" in failed_names, failed_names
print("SMOKE 2 PASSED: 全零 lora_B 被正确判 FAILED:", failed_names)

# ---- 3. 非零 BF16 → PASSED; F32 lora_B → FAILED ----
good_dir = os.path.join(tmp, "adapter_good")
make_adapter(good_dir, zero_b=False)
res = checkpointing.validate_adapter_dir(good_dir)
print("=== good adapter validation:", res["status"])
assert res["status"] == "PASSED", res["checks"]
assert len(res["sha256"]) == 2
print("SMOKE 3 PASSED: 非零 BF16 适配器 PASSED, sha256 已记录")

f32_dir = os.path.join(tmp, "adapter_f32")
make_adapter(f32_dir, zero_b=False, dtype=torch.float32)
res = checkpointing.validate_adapter_dir(f32_dir)
assert res["status"] == "FAILED"
assert "lora_B:dtype全BF16" in [c["name"] for c in res["checks"] if c["status"] == "FAILED"]
print("SMOKE 3b PASSED: F32 lora_B 被正确判 FAILED (事故形态)")

# ---- 4. find_best_checkpoint 精确归因 ----
ckpt_root = os.path.join(tmp, "ckpts")
# checkpoint-100: 自己 step=100 的 eval_loss=0.5; log_history 里也含 step=200 的 0.3
# 旧实现会把 0.3 记到 checkpoint-100 头上, 新实现必须只认 0.5
for step, entries in (
    (100, [{"step": 100, "eval_loss": 0.5}, {"step": 200, "eval_loss": 0.3}]),
    (200, [{"step": 100, "eval_loss": 0.5}, {"step": 200, "eval_loss": 0.3}]),
):
    d = os.path.join(ckpt_root, f"checkpoint-{step}")
    os.makedirs(d)
    with open(os.path.join(d, "trainer_state.json"), "w") as f:
        json.dump({"log_history": entries}, f)
best = checkpointing.find_best_checkpoint(ckpt_root)
print("=== precise attribution best:", best)
assert best["step"] == 200 and abs(best["eval_loss"] - 0.3) < 1e-9
print("SMOKE 4 PASSED: best 归因到 checkpoint-200 (eval_loss=0.3), 无张冠李戴")

# ---- 5. BestCheckpointCallback: eval 后另存 best/ 防轮转 ----
from transformers import TrainerCallback, TrainerState, TrainingArguments
cls = checkpointing.build_best_checkpoint_callback(TrainerCallback, ckpt_root, log_fn=lambda m: None)
cb = cls()
args = TrainingArguments(output_dir=ckpt_root)
state = TrainerState()
state.global_step = 100
cb.on_evaluate(args, state, None, metrics={"eval_loss": 0.5})
assert os.path.isdir(os.path.join(ckpt_root, "best")), "eval 后应立即另存 best/"
state.global_step = 200
cb.on_evaluate(args, state, None, metrics={"eval_loss": 0.3})
with open(os.path.join(ckpt_root, "best", "trainer_state.json")) as f:
    promoted = json.load(f)
assert promoted["log_history"][0]["step"] == 100  # 内容来自 checkpoint-200 的 trainer_state
assert cb.best_eval_step == 200 and len(cb.promotion_history) == 2
# eval step 无落盘 checkpoint 时 → pending, on_save 补存
state.global_step = 300  # 无 checkpoint-300 目录
cb.on_evaluate(args, state, None, metrics={"eval_loss": 0.2})
assert cb.pending_step == 300
os.makedirs(os.path.join(ckpt_root, "checkpoint-400"))
with open(os.path.join(ckpt_root, "checkpoint-400", "trainer_state.json"), "w") as f:
    json.dump({"log_history": [{"step": 300, "eval_loss": 0.2}]}, f)
state.global_step = 400
cb.on_save(args, state, None)
assert cb.pending_step is None and cb.promotion_history[-1]["eval_step"] == 300
print("SMOKE 5 PASSED: BestCheckpointCallback 即时另存 + eval/save 不对齐补存")

print("\nALL SMOKE TESTS PASSED")
