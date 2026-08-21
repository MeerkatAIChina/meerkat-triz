import json
p = "/home/chinux/jupyterlab/meerkatai/results/v5/eval_v5_base_goldfix_v5_qwen38_20260815_023643.json"
d = json.load(open(p))
print("=== top-level keys ===")
for k in sorted(d.keys()):
    v = d[k]
    t = type(v).__name__
    if isinstance(v, dict):
        print(f"  {k}: dict keys={list(v.keys())}")
    elif isinstance(v, list):
        print(f"  {k}: list len={len(v)}")
    else:
        print(f"  {k}: {v}")
print("\n=== entries sample ===")
entries = d.get("entries", [])
if entries:
    e = entries[0]
    print(f"  entry keys: {list(e.keys())}")
    for k in sorted(e.keys()):
        v = e[k]
        if isinstance(v, str) and len(v) > 80:
            print(f"    {k}: <str len={len(v)}>")
        elif isinstance(v, dict):
            print(f"    {k}: dict keys={list(v.keys())}")
        else:
            print(f"    {k}: {v}")
