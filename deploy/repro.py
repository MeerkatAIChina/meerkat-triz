import requests

P = "你经过了微调吗？"


def run(model, **kw):
    payload = {"model": model, "max_tokens": 3000,
               "messages": [{"role": "user", "content": P}], **kw}
    r = requests.post("http://127.0.0.1:8000/v1/chat/completions", json=payload, timeout=900)
    d = r.json()
    msg = d["choices"][0]["message"]
    txt = (msg.get("reasoning") or "") + " ||CONTENT|| " + (msg.get("content") or "")
    # loop detection: find the most repeated 60-char block
    best = 0
    for i in range(0, min(len(txt) - 60, 4000), 200):
        block = txt[i:i + 60]
        best = max(best, txt.count(block))
    print(f"--- {model} {kw}")
    print(f"    finish={d['choices'][0]['finish_reason']} total_chars={len(txt)} max_block_repeat={best}")
    if best >= 3:
        idx = txt.find(block)
        print("    repeated block:", block[:80].replace(chr(10), " "))


run("Qwen3.6-35B-A3B-NVFP4")  # 不传 temperature = 服务器默认(0)
run("Qwen3.6-35B-A3B-NVFP4", temperature=0.6, top_p=0.95)
run("Meerkat-TRIZ-v1")
run("Meerkat-TRIZ-v1", temperature=0.6, top_p=0.95)
