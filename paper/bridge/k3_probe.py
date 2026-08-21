#!/usr/bin/env python3
"""kimi-k3 评委适配性探针：确定性翻转率 + 长上下文 + 延迟。

用法: MOONSHOT_API_KEY=... python3 k3_probe.py
输出: 3 次同请求的分值翻转情况 + 一次长文本冒烟。
"""
import json
import os
import time
import urllib.request

KEY = os.environ["MOONSHOT_API_KEY"]
URL = "https://api.moonshot.cn/v1/chat/completions"

JUDGE_PROMPT = (
    "你是 TRIZ 回答质量评委。按准确性/完整性/TRIZ正确性/结构四个维度给 0-4 分，"
    "只输出 JSON: {\"accuracy\": N, \"completeness\": N, "
    "\"triz_correctness\": N, \"structure\": N, \"overall\": N}\n\n"
    "问题: 什么是 TRIZ 的分割原理？请举例说明。\n"
    "回答: 分割原理是 TRIZ 40 个发明原理中的第 1 条，指将物体分成相互独立的"
    "部分，或使物体易于拆卸。例如组合式家具、模块化手机设计。分割可以提高"
    "灵活性和可维护性。"
)


def chat(model, messages, max_tokens=1024, temperature=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if temperature is not None:
        body["temperature"] = temperature
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            d = json.loads(e.read())
        except Exception:
            d = {"error": {"message": f"HTTP {e.code}"}}
    dt = time.time() - t0
    if "error" in d:
        return {"error": d["error"], "dt": dt}
    m = d["choices"][0]["message"]
    return {"content": m.get("content", ""),
            "reasoning_chars": len(m.get("reasoning_content") or ""),
            "usage": d.get("usage"), "dt": dt}


def main():
    print("== 1. 温度参数兼容性 ==")
    for t in (0, 0.0, None, 1):
        r = chat("kimi-k3", [{"role": "user", "content": "回复 OK"}],
                 max_tokens=32, temperature=t)
        print(f"  temperature={t}: {'ERROR ' + str(r['error'])[:80] if 'error' in r else 'accepted'}")

    print("\n== 2. 确定性翻转率 (评委 prompt x5, 默认温度) ==")
    outs = []
    for i in range(5):
        r = chat("kimi-k3", [{"role": "user", "content": JUDGE_PROMPT}])
        if "error" in r:
            print(f"  run{i}: ERROR {str(r['error'])[:80]}")
            continue
        c = r["content"].strip()
        outs.append(c)
        print(f"  run{i}: {c[:90]}  ({r['dt']:.1f}s, reasoning {r['reasoning_chars']}ch)")
    if outs:
        uniq = len(set(outs))
        print(f"  → 唯一输出 {uniq}/{len(outs)}", "完全确定 ✓" if uniq == 1 else "存在翻转 ✗")

    print("\n== 3. 长上下文冒烟 (32k 字符输入) ==")
    long_doc = ("TRIZ 理论背景资料。" * 4000)[:32000]
    r = chat("kimi-k3", [{"role": "user", "content":
                          f"以下是一份资料，请用一句话总结它的主题。\n\n{long_doc}"}],
             max_tokens=256)
    if "error" in r:
        print(f"  ERROR {str(r['error'])[:120]}")
    else:
        print(f"  ok ({r['dt']:.1f}s) usage={r['usage']}")
        print(f"  content: {r['content'][:100]}")

    print("\n== 4. 对照: moonshot-v1-32k 同 prompt ==")
    r = chat("moonshot-v1-32k", [{"role": "user", "content": JUDGE_PROMPT}],
             temperature=0)
    if "error" in r:
        print(f"  ERROR {str(r['error'])[:80]}")
    else:
        print(f"  {r['content'].strip()[:90]}  ({r['dt']:.1f}s)")


if __name__ == "__main__":
    main()
