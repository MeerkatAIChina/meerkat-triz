# -*- coding: utf-8 -*-
"""E1b 多评委交叉: 用第二评委对 {base,v2,v4} 金标 responses 按 v4 harness 同一 rubric 重打分。
评委探测顺序: kimi-k2-0711-preview -> moonshot-v1-8k (kimi-k2 当前 404, 8k 为同族弱异源兜底)。
输入构造与 v4 harness 完全一致 (JUDGE_SYSTEM + 1500 字符截断), 保证与既有 32k 分可比。
输出 results/e1/e1b_rejudge_{judge_tag}.jsonl"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from e1_common import (get_client, call_chat, append_jsonl, load_jsonl,
                       parse_json_array, load_gold, load_responses,
                       JUDGE_SYSTEM, build_judge_user, log, RESULTS)

CANDIDATES = ["kimi-k2-0711-preview", "moonshot-v1-8k"]
BATCH = 5


def probe(client):
    for m in CANDIDATES:
        try:
            r = client.chat.completions.create(
                model=m, messages=[{"role": "user", "content": "ping, 回复 ok"}],
                max_tokens=5, temperature=0)
            log(f"judge 探测 {m}: OK")
            return m
        except Exception as e:
            log(f"judge 探测 {m}: FAIL {str(e)[:120]}")
    raise RuntimeError("无可用第二评委")


def main():
    (RESULTS / "e1").mkdir(exist_ok=True)
    client = get_client()
    judge = probe(client)
    tag = judge.replace("-", "_")
    out = RESULTS / "e1" / f"e1b_rejudge_{tag}.jsonl"
    gold = load_gold()
    resps = {}
    for mtag in ("base", "v2", "v4"):
        resps[mtag], src = load_responses(mtag)
        log(f"{mtag} responses <- {src}")

    def build_todo():
        done = {(r["qid"], r["model"]) for r in load_jsonl(out)}
        todo = []
        for mtag in ("base", "v2", "v4"):
            for it in gold:
                if (it["id"], mtag) not in done:
                    todo.append((mtag, it))
        return todo, len(done)

    fails_in_a_row = 0
    while True:
        todo, ndone = build_todo()
        log(f"E1b 进度 {ndone}/300, 剩余 {len(todo)}, judge={judge}")
        if not todo:
            break
        chunk = todo[:BATCH]
        items = [it for _, it in chunk]
        resp_map = {it["id"]: resps[mtag][it["id"]] for mtag, it in chunk}
        user = build_judge_user(items, resp_map, max_chars=1500)
        text = call_chat(client, judge, JUDGE_SYSTEM, user,
                         temperature=0.0, max_tokens=2000)
        ok = 0
        if text:
            try:
                arr = parse_json_array(text)
                by_id = {str(e.get("id")): e for e in arr if isinstance(e, dict)}
                for mtag, it in chunk:
                    e = by_id.get(it["id"])
                    if e and "overall" in e:
                        append_jsonl(out, {
                            "qid": it["id"], "model": mtag, "subset": it["subset"],
                            "accuracy": e.get("accuracy"),
                            "completeness": e.get("completeness"),
                            "triz_correctness": e.get("triz_correctness"),
                            "structure": e.get("structure"),
                            "overall": e.get("overall"), "judge": judge})
                        ok += 1
            except Exception as ex:
                log(f"解析失败: {ex}")
        if ok < len(chunk):
            done_now = {(r["qid"], r["model"]) for r in load_jsonl(out)}
            for mtag, it in chunk:
                if (it["id"], mtag) in done_now:
                    continue
                t2 = call_chat(client, judge, JUDGE_SYSTEM,
                               build_judge_user([it], {it["id"]: resps[mtag][it["id"]]}, 1500),
                               temperature=0.0, max_tokens=600)
                try:
                    e = parse_json_array(t2)[0]
                    if "overall" not in e:
                        raise ValueError("no overall")
                    append_jsonl(out, {
                        "qid": it["id"], "model": mtag, "subset": it["subset"],
                        "accuracy": e.get("accuracy"),
                        "completeness": e.get("completeness"),
                        "triz_correctness": e.get("triz_correctness"),
                        "structure": e.get("structure"),
                        "overall": e.get("overall"), "judge": judge})
                    ok += 1
                except Exception as ex:
                    log(f"单条失败 {mtag}/{it['id']}: {ex}")
        fails_in_a_row = 0 if ok > 0 else fails_in_a_row + 1
        if fails_in_a_row >= 5:
            log("连续 5 批 0 成功, 退出待续跑")
            sys.exit(1)
        if ok == 0:
            time.sleep(60)
    meta = {"judge": judge,
            "note": "kimi-k2-0711-preview 探测 404 时兜底 moonshot-v1-8k(同族弱异源, 非完全异源)",
            "input_format": "v4 harness JUDGE_SYSTEM + 1500 chars 截断, 与既有 32k 分输入一致"}
    (RESULTS / "e1" / "e1b_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log("E1b 完成")
    (RESULTS / "e1" / "e1b.done").write_text("done\n")


if __name__ == "__main__":
    main()
