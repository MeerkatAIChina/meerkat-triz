import re
p = "/home/chinux/jupyterlab/meerkatai/checkpoints/qlora_triz_v6_qwen38/train.log"
text = open(p).read()

# 提取所有 eval 记录
eval_pattern = r"'eval_loss':\s*'([\d.]+)'.*?'epoch':\s*'([\d.]+)'"
evals = list(re.finditer(eval_pattern, text))
print(f"=== 共 {len(evals)} 次 eval 记录 ===")
for i, m in enumerate(evals):
    print(f"  eval {i+1}: loss={m.group(1)} @ epoch={m.group(2)}")

# 提取 NEW BEST
print("\n=== NEW BEST 记录 ===")
for line in text.splitlines():
    if "NEW BEST" in line:
        print(f"  {line.strip()}")

# 最新步数
pattern = r"(\d+)/(\d+)\s+\[(\d+):(\d+)<.*?([\d.]+)s/it\]"
matches = list(re.finditer(pattern, text))
if matches:
    latest = matches[-1]
    step = int(latest.group(1))
    total = int(latest.group(2))
    elapsed = int(latest.group(3)) * 60 + int(latest.group(4))
    speed = float(latest.group(5))
    print(f"\n=== 最新进度 ===")
    print(f"  步 {step}/{total} ({step/total*100:.1f}%)")
    print(f"  已运行: {elapsed//60}分{elapsed%60}秒")
    print(f"  当前速度: {speed:.2f}s/it")
    
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
        print(f"  真实平均速度(从步{first_step}起): {real_speed:.2f}s/步")

# 异常扫描
print("\n=== 异常扫描 ===")
abnormal = []
for line in text.splitlines():
    l = line.lower()
    if any(k in l for k in ["oom", "out of memory", "nan", "error", "assertion", "fail", "killed", "segmentation", "cuda"]):
        abnormal.append(line.strip()[:200])
if abnormal:
    for a in abnormal[-5:]:
        print(f"  {a}")
else:
    print("  无异常")
