import requests

# 不传 max_tokens，模拟 WebUI 的失控场景
r = requests.post("http://127.0.0.1:8000/v1/chat/completions", json={
    "model": "Meerkat-TRIZ-v1",
    "messages": [{"role": "user", "content": "你如何证明你的创新能力比基础模型强"}],
}, timeout=1800)
d = r.json()
u = d["usage"]
print(f"finish={d['choices'][0]['finish_reason']} completion_tokens={u['completion_tokens']}")
assert u["completion_tokens"] <= 16384, "cap not working"
print("cap OK: 服务器默认上限 16384 生效")
