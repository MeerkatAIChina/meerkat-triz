import json
p = "/home/chinux/jupyterlab/meerkatai/results/v5/eval_v5_base_goldfix_v5_qwen38_20260815_023643.json"
d = json.load(open(p))

print("=== judge_track_armA ===")
jta = d["judge_track_armA"]
print(f"  n={jta['n']} mean={jta['mean']} pass_rate={jta.get('pass_rate')}")
ps = jta.get("per_subset", [])
if isinstance(ps, dict):
    for k, v in ps.items():
        print(f"    {k}: {v}")
elif isinstance(ps, list):
    for s in ps:
        if isinstance(s, dict):
            print(f"    {s.get('subset')}: n={s.get('n')} mean={s.get('mean')}")
        else:
            print(f"    {s}")

print("\n=== quality_gates ===")
qg = d["quality_gates"]
for k in sorted(qg.keys()):
    v = qg[k]
    if isinstance(v, list):
        print(f"  {k}: list len={len(v)}")
        if v and len(v) <= 10:
            for item in v:
                print(f"    {item}")
    elif isinstance(v, dict):
        print(f"  {k}: dict")
        for kk, vv in v.items():
            if isinstance(vv, list) and len(vv) > 5:
                print(f"    {kk}: list len={len(vv)}")
            else:
                print(f"    {kk}: {vv}")
    else:
        print(f"  {k}: {v}")

print("\n=== miss_audit ===")
ma = d["miss_audit"]
for k, v in ma.items():
    print(f"  {k}: {v}")

print("\n=== overrefusal ===")
print(f"  {d.get('overrefusal')}")

print("\n=== records refusal scan ===")
refusal_keywords = ["无法回答", "我不能", "I cannot", "I can't", "sorry", "apologize", "拒绝", "not able"]
count = 0
for r in d.get("records", []):
    gen = r.get("generation", "")
    low = str(gen).lower()
    for kw in refusal_keywords:
        if kw in low:
            count += 1
            if count <= 3:
                print(f"  hit {count}: {gen[:120]}")
            break
print(f"  total refusal-like records: {count}")
