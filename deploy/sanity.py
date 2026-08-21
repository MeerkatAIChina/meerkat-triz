import requests


def run(tag, model, prompt, **kw):
    payload = {"model": model, "max_tokens": 4096,
               "messages": [{"role": "user", "content": prompt}], **kw}
    r = requests.post("http://127.0.0.1:8000/v1/chat/completions", json=payload, timeout=900)
    d = r.json()
    msg = d["choices"][0]["message"]
    content = msg.get("content") or ""
    print(f"--- {tag}: finish={d['choices'][0]['finish_reason']} "
          f"content_chars={len(content)} tokens={d['usage']['completion_tokens']}")
    print("    head:", content[:120].replace(chr(10), " "))


P = "设计一款更轻但不能降低强度的自行车车架，请用 TRIZ 技术矛盾分析。"
run("adapter-default", "Meerkat-TRIZ-v1", P)
run("adapter-temp0", "Meerkat-TRIZ-v1", P, temperature=0)
