import re
p = "/home/chinux/jupyterlab/meerkatai/checkpoints/qlora_triz_v6_qwen38/train.log"
text = open(p).read()
lines = text.splitlines()

print("=== 完整 eval 历史 ===")
eval_pattern = r"'eval_loss':\s*'([\d.]+)'.*?'eval_entropy':\s*'([\d.]+)'.*?'eval_mean_token_accuracy':\s*'([\d.]+)'.*?'epoch':\s*'([\d.]+)'"
evals = list(re.finditer(eval_pattern, text))
for i, m in enumerate(evals):
    step = (i + 1) * 100
    print(f"  step {step:3d}: eval_loss={m.group(1)}, entropy={m.group(2)}, acc={m.group(3)}, epoch={m.group(4)}")

print("\n=== NEW BEST 历史 ===")
for line in lines:
    if "NEW BEST" in line:
        print(f"  {line.strip()[:140]}")

print("\n=== 最近训练 loss ===")
train_loss_pattern = r"'loss':\s*'([\d.]+)'.*?'epoch':\s*'([\d.]+)'"
train_losses = list(re.finditer(train_loss_pattern, text))
for m in train_losses[-5:]:
    print(f"  loss={m.group(1)} @ epoch={m.group(2)}")

# 最新训练步数
pattern = r"(\d+)/(\d+)\s+\[(\d+):(\d+)<.*?([\d.]+)s/it\]"
matches = list(re.finditer(pattern, text))
if matches:
    latest = matches[-1]
    step = int(latest.group(1))
    total = int(latest.group(2))
    elapsed = int(latest.group(3)) * 60 + int(latest.group(4))
    print(f"\n=== 最新进度 ===")
    print(f"  步 {step}/{total} ({step/total*100:.1f}%)")
    print(f"  已运行: {elapsed//60}分{elapsed%60}秒")
    
    # 计算真实平均速度
    first_match = None
    for m in matches:
        if int(m.group(1)) >= 10:
            first_match = m
            break
    if first_match:
        first_step = int(first_match.group(1))
        first_elapsed = int(first_match.group(3)) * 60 + int(first_match.group(4))
        real_speed = (elapsed - first_elapsed) / (step - first_step)
        remaining = total - step
        eta_seconds = remaining * real_speed
        eta_hours = eta_seconds / 3600
        print(f"  真实平均速度: {real_speed:.2f}s/步")
        print(f"  预计剩余: {remaining} 步 ≈ {eta_hours:.1f} 小时")

print("\n=== 异常扫描 ===")
abnormal = []
for line in lines:
    l = line.lower()
    if any(k in l for k in ["oom", "out of memory", "nan", "error", "assertion", "fail", "killed", "segmentation", "cuda"]):
        abnormal.append(line.strip()[:200])
if abnormal:
    for a in abnormal[-5:]:
        print(f"  {a}")
else:
    print("  无异常")
