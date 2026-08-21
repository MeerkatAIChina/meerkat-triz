import json
import sys
p = "/home/chinux/jupyterlab/meerkatai/results/v5/eval_v5_base_goldfix_v5_qwen38_20260815_023643.json"
d = json.load(open(p))
summary = d.get("summary", {})
print("=== summary keys ===")
for k in sorted(summary.keys()):
    v = summary[k]
    if isinstance(v, (int, float, str, bool)) or v is None:
        print(f"{k}: {v}")
    elif isinstance(v, dict) and len(v) < 5:
        print(f"{k}: {v}")
    else:
        print(f"{k}: <{type(v).__name__} len={len(v) if hasattr(v,'__len__') else 'N/A'}>")
print("\n=== gate_fail_counts ===")
for item in d.get("gate_fail_counts", []):
    print(item)
print("\n=== overrefusal ===")
print(d.get("overrefusal", "NOT FOUND"))
