#!/usr/bin/env python
"""usage_feedback 分诊器：把采集到的真实使用数据分成三路。

输入:  usage_feedback/raw/<date>/ (owui_dump.json + pi_sessions/)
输出:  usage_feedback/triage/<date>/
  - candidate_sft.jsonl    评委高分且无失败信号 → v5c SFT 候选
  - candidate_prefs.jsonl  👍/👎 反馈 + 评委分差 → DPO 偏好对候选
  - failures.md            失败案例清单（身份否认/重复循环/失控/空输出）
  - summary.md             总量、分布、主题概览

评委: moonshot (MOONSHOT_API_KEY, 默认 moonshot-v1-32k, JUDGE_MODEL 可覆盖)。
本地启发式失败检测不依赖 API，始终运行。
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

MOONSHOT_URL = "https://api.moonshot.cn/v1/chat/completions"
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "moonshot-v1-32k")

# ---------------- 归一化 ----------------

def norm_owui(dump_path):
    convs = []
    d = json.load(open(dump_path))
    for c in d.get("chats", []):
        chat = c.get("chat") or {}
        if isinstance(chat, str):
            try:
                chat = json.loads(chat)
            except Exception:
                continue
        msgs = []
        hist = chat.get("history", {})
        if isinstance(hist, dict) and hist.get("messages"):
            for m in hist["messages"].values():
                msgs.append({"role": m.get("role"), "content": m.get("content", ""),
                             "model": (m.get("models") or [None])[0] if isinstance(m.get("models"), list) else m.get("model")})
        elif isinstance(chat.get("messages"), list):
            msgs = [{"role": m.get("role"), "content": m.get("content", ""), "model": m.get("model")}
                    for m in chat["messages"]]
        msgs = [m for m in msgs if m.get("role") in ("user", "assistant") and m.get("content")]
        if msgs:
            convs.append({"source": "open-webui", "id": c["id"], "title": c.get("title", ""),
                          "ts": c.get("updated_at") or c.get("created_at"), "messages": msgs})
    # feedback 评分: data.rating (+1/-1), snapshot 里带 message
    prefs = []
    for f in d.get("feedback", []):
        try:
            data = f.get("data") or {}
            rating = data.get("rating")
            if rating is not None:
                prefs.append({"source": "open-webui", "rating": rating,
                              "model_id": data.get("model_id"), "ts": f.get("created_at"),
                              "meta": f.get("meta")})
        except Exception:
            continue
    return convs, prefs


def norm_pi(sess_dir):
    convs = []
    for p in Path(sess_dir).rglob("*.jsonl"):
        msgs, tools, model = [], [], None
        for line in open(p, encoding="utf-8", errors="ignore"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("type") == "model_change":
                model = d.get("modelId")
            if d.get("type") != "message":
                continue
            m = d.get("message", {})
            role, parts = m.get("role"), m.get("content")
            if isinstance(parts, list):
                text = "".join(x.get("text", "") for x in parts if isinstance(x, dict) and x.get("type") == "text")
                for x in parts:
                    if isinstance(x, dict) and x.get("type") == "toolCall":
                        tools.append(x.get("name", "?"))
            else:
                text = parts or ""
            if role in ("user", "assistant") and text.strip():
                msgs.append({"role": role, "content": text, "model": model})
        if msgs:
            convs.append({"source": "pi", "id": p.stem, "title": msgs[0]["content"][:40],
                          "ts": d.get("timestamp"), "messages": msgs, "tool_calls": tools})
    return convs

# ---------------- 失败启发式（本地，无 API） ----------------

IDENTITY_Q = re.compile(r"(你是什么模型|你是谁|Meerkat-TRIZ.*是你|what model are you|who are you)", re.I)
IDENTITY_DENY = re.compile(r"(不是\s*Meerkat|我不是\s*Meerkat|我是\s*Qwen|通义千问.{0,10}自主研发)")
RUNAWAY_CHARS = 30000


def detect_failures(conv):
    flags = []
    full_a = "\n".join(m["content"] for m in conv["messages"] if m["role"] == "assistant")
    joined_q = "\n".join(m["content"] for m in conv["messages"] if m["role"] == "user")
    if IDENTITY_Q.search(joined_q) and IDENTITY_DENY.search(full_a):
        flags.append("identity_denial")
    for m in conv["messages"]:
        c = m["content"]
        if m["role"] == "assistant" and len(c) > 500:
            for i in range(0, min(len(c) - 80, 6000), 200):
                if c.count(c[i:i + 80]) >= 3:
                    flags.append("repetition_loop")
                    break
        if m["role"] == "assistant" and not c.strip():
            flags.append("empty_output")
    if len(full_a) > RUNAWAY_CHARS:
        flags.append("runaway_generation")
    return sorted(set(flags))

# ---------------- moonshot 评委 ----------------

RUBRIC = """你是 TRIZ 领域评测专家。给定一段用户与 TRIZ 微调模型的真实对话，按 1-5 打分：
1. triz_correctness: TRIZ 概念使用是否准确（矛盾矩阵/发明原理/物理矛盾等不编造）
2. structure: 回答是否结构化、可执行
3. data_value: 作为训练数据的价值（5=教科书级范例, 1=无价值）
4. theme: 用 2-6 个字概括对话主题
只输出 JSON: {"triz_correctness": int, "structure": int, "data_value": int, "theme": str, "reason": str}"""


def judge_score(conv, client_kwargs):
    import urllib.request
    text = "\n".join(f"[{m['role']}] {m['content'][:3000]}" for m in conv["messages"][:12])[:24000]
    body = json.dumps({
        "model": JUDGE_MODEL, "temperature": 0, "max_tokens": 300,
        "messages": [{"role": "system", "content": RUBRIC},
                     {"role": "user", "content": text}],
    }).encode()
    req = urllib.request.Request(MOONSHOT_URL, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {client_kwargs['api_key']}"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.load(r)["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", out, re.S)
            return json.loads(m.group(0)) if m else None
        except Exception as e:
            print(f"  judge retry{attempt + 1}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    return None

# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()
    raw = Path(args.raw_dir)
    date = raw.name
    out = Path(args.out) if args.out else raw.parent.parent / "triage" / date
    out.mkdir(parents=True, exist_ok=True)

    convs, prefs = [], []
    if (raw / "owui_dump.json").exists():
        c, p = norm_owui(raw / "owui_dump.json")
        convs += c
        prefs += p
    if (raw / "pi_sessions").exists():
        convs += norm_pi(raw / "pi_sessions")

    api_key = os.environ.get("MOONSHOT_API_KEY", "")
    use_judge = bool(api_key) and not args.no_judge
    print(f"[triage] {len(convs)} conversations, {len(prefs)} feedback, judge={'on' if use_judge else 'off'}")

    sft, fail_rows, score_dist = [], [], {}
    themes = {}
    for i, conv in enumerate(convs):
        flags = detect_failures(conv)
        score = judge_score(conv, {"api_key": api_key}) if use_judge else None
        row = {"id": conv["id"], "source": conv["source"], "title": conv["title"],
               "n_msgs": len(conv["messages"]), "tool_calls": conv.get("tool_calls", []),
               "failure_flags": flags, "judge": score, "messages": conv["messages"]}
        if flags:
            fail_rows.append(row)
        elif score and score.get("data_value", 0) >= 4 and score.get("triz_correctness", 0) >= 4:
            sft.append(row)
        dv = score.get("data_value") if score else None
        score_dist[dv] = score_dist.get(dv, 0) + 1
        if score and score.get("theme"):
            themes[score["theme"]] = themes.get(score["theme"], 0) + 1
        print(f"  [{i + 1}/{len(convs)}] {conv['source']}:{conv['title'][:30]} flags={flags} score={score}", flush=True)

    with open(out / "candidate_sft.jsonl", "w", encoding="utf-8") as f:
        for r in sft:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "candidate_prefs.jsonl", "w", encoding="utf-8") as f:
        for p in prefs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(out / "failures.md", "w", encoding="utf-8") as f:
        f.write(f"# 失败案例清单 ({date})\n\n")
        for r in fail_rows:
            f.write(f"## {r['source']}:{r['id'][:12]} — {r['title']}\n")
            f.write(f"- flags: {', '.join(r['failure_flags'])}\n")
            if r["judge"]:
                f.write(f"- judge: {r['judge'].get('reason', '')}\n")
            f.write("\n")
    with open(out / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# usage_feedback 分诊摘要 ({date})\n\n")
        f.write(f"- 会话总数: {len(convs)} (open-webui {sum(1 for c in convs if c['source'] == 'open-webui')}, pi {sum(1 for c in convs if c['source'] == 'pi')})\n")
        f.write(f"- 👍/👎 反馈: {len(prefs)}\n")
        f.write(f"- SFT 候选: {len(sft)} | 失败案例: {len(fail_rows)}\n")
        f.write(f"- data_value 分布: {score_dist}\n")
        f.write(f"- 主题分布: {dict(sorted(themes.items(), key=lambda x: -x[1]))}\n")
    print(f"[triage] done -> {out}")
    print(f"  sft={len(sft)} prefs={len(prefs)} failures={len(fail_rows)}")


if __name__ == "__main__":
    main()
