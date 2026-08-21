import re
p = "/home/chinux/jupyterlab/meerkatai/checkpoints/qlora_triz_v6_qwen38/train.log"
text = open(p).read()

# 只匹配训练进度 (total=5548)
pattern = r"(\d+)/5548\s+\[(\d+):(\d+)<.*?([\d.]+)s/it\]"
matches = list(re.finditer(pattern, text))

if matches:
    latest = matches[-1]
    step = int(latest.group(1))
    elapsed = int(latest.group(2)) * 60 + int(latest.group(3))
    speed = float(latest.group(4))
    print(f"=== 训练进度 ===")
    print(f"  步 {step}/5548 ({step/5548*100:.1f}%)")
    print(f"  已运行: {elapsed//60}分{elapsed%60}秒")
    print(f"  当前速度: {speed:.2f}s/it")
    
    # 计算真实平均速度 (排除eval时间)
    first_match = None
    for m in matches:
        if int(m.group(1)) >= 10:
            first_match = m
            break
    if first_match:
        first_step = int(first_match.group(1))
        first_elapsed = int(first_match.group(2)) * 60 + int(first_match.group(3))
        real_speed = (elapsed - first_elapsed) / (step - first_step)
        remaining = 5548 - step
        eta_seconds = remaining * real_speed
        eta_hours = eta_seconds / 3600
        print(f"  真实平均速度: {real_speed:.2f}s/步")
        print(f"  预计剩余: {remaining} 步 ≈ {eta_hours:.1f} 小时")

# 当前 epoch
epoch_pattern = r"epoch.*?'([\d.]+)'"
epochs = list(re.finditer(epoch_pattern, text))
if epochs:
    print(f"\n  当前 epoch: {epochs[-1].group(1)}")
