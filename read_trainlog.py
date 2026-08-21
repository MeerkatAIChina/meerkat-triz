import re
p = "/home/chinux/jupyterlab/meerkatai/checkpoints/qlora_triz_v6_qwen38/train.log"
text = open(p).read()

print("=== eval 记录 ===")
for m in re.finditer(r"'eval_loss':\s*'([\d.]+)'.*?'epoch':\s*'([\d.]+)'", text):
    print(f"  eval_loss={m.group(1)} @ epoch={m.group(2)}")

print("\n=== 训练步速估算 ===")
steps = re.findall(r"(\d+)/(\d+)\s+\[([\d:]+)<", text)
if steps:
    last = steps[-1]
    print(f"  最新步: {last[0]}/{last[1]}, 已用时间: {last[2]}")
    
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
    print("  无异常关键词命中")
