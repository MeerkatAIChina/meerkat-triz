import re
p = "/home/chinux/jupyterlab/meerkatai/checkpoints/qlora_triz_v6_qwen38/train.log"
text = open(p).read()

# 查找所有包含 eval_loss 的行附近的内容
lines = text.splitlines()
for i, line in enumerate(lines):
    if "'eval_loss'" in line:
        print(f"=== eval at line {i} ===")
        for j in range(max(0, i-3), min(len(lines), i+4)):
            print(f"  {lines[j]}")
        print()

# 查找 best-ckpt 相关
print("=== best-ckpt 记录 ===")
for i, line in enumerate(lines):
    if "best-ckpt" in line.lower() or "NEW BEST" in line:
        print(f"  line {i}: {line.strip()[:180]}")

# 查看最后几行确认当前状态
print("\n=== 日志最后 10 行 ===")
for line in lines[-10:]:
    print(f"  {line.strip()[:120]}")
