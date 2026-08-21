# -*- coding: utf-8 -*-
"""E1a' 干净锚点位置交换: pairwise v4 vs base_goldfix, AB/BA 双序 100 题, 32k, T=0。
输出 results/e1/e1a_position_swap_goldfix.jsonl"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from e1_common import (get_client, call_chat, append_jsonl, load_jsonl,
                       parse_json_array, load_gold, load_responses,
                       PAIRWISE_SYSTEM, build_pairwise_user, log, RESULTS)

OUT = RESULTS / "e1" / "e1a_position_swap_goldfix.jsonl"
JUDGE = "moonshot-v1-32k"
BATCH = 5


def build_todo(gold, resp_v4, resp_base, src_base):
    done = {(r["qid"], r["order"]) for r in load_jsonl(OUT)}
    todo = []
    for it in gold:
        for order in ("AB", "BA"):
            if (it["id"], order) not in done:
                if order == "AB":
                    a, b = resp_v4[it["id"]], resp_base[it["id"]]
                else:
                    a, b = resp_base[it["id"]], resp_v4[it["id"]]
                todo.append({"pid": f"{it['id']}#{order}", "qid": it["id"],
                             "order": order, "subset": it["subset"],
                             "question": it["question"],
                             "reference_answer": it["reference_answer"],
                             "resp_a": a, "resp_b": b})
    return todo, len(done)


def main():
    gold = load_gold()
    resp_v4, src_v4 = load_responses("v4")
    resp_base, src_base = load_responses("base_goldfix")
    log(f"E1a' responses: v4={src_v4} base={src_base}")
    client = get_client()
    fails_in_a_row = 0
    while True:
        todo, ndone = build_todo(gold, resp_v4, resp_base, src_base)
        log(f"E1a' 进度 {ndone}/200, 剩余 {len(todo)}")
        if not todo:
            break
        batch = todo[:BATCH]
        text = call_chat(client, JUDGE, PAIRWISE_SYSTEM,
                         build_pairwise_user(batch), temperature=0.0, max_tokens=2000)
        ok = 0
        if text:
            try:
                arr = parse_json_array(text)
                by_id = {str(e.get("id")): e for e in arr if isinstance(e, dict)}
                for it in batch:
                    e = by_id.get(it["pid"])
                    if e and str(e.get("winner", "")).upper() in ("A", "B", "TIE"):
                        w = str(e["winner"]).upper()
                        append_jsonl(OUT, {
                            "qid": it["qid"], "order": it["order"],
                            "subset": it["subset"],
                            "winner_pos": "tie" if w == "TIE" else w,
                            "reason": str(e.get("reason", ""))[:300],
                            "judge": JUDGE, "temp": 0.0,
                            "resp_trunc": 2000, "base_src": src_base})
                        ok += 1
            except Exception as ex:
                log(f"解析失败: {ex}")
        if ok < len(batch):
            done_now = {(r["qid"], r["order"]) for r in load_jsonl(OUT)}
            for it in batch:
                if (it["qid"], it["order"]) in done_now:
                    continue
                t2 = call_chat(client, JUDGE, PAIRWISE_SYSTEM,
                               build_pairwise_user([it]), temperature=0.0, max_tokens=600)
                try:
                    e = parse_json_array(t2)[0]
                    w = str(e.get("winner", "")).upper()
                    if w not in ("A", "B", "TIE"):
                        raise ValueError(f"bad winner {w}")
                    append_jsonl(OUT, {
                        "qid": it["qid"], "order": it["order"],
                        "subset": it["subset"],
                        "winner_pos": "tie" if w == "TIE" else w,
                        "reason": str(e.get("reason", ""))[:300],
                        "judge": JUDGE, "temp": 0.0, "resp_trunc": 2000,
                        "base_src": src_base})
                    ok += 1
                except Exception as ex:
                    log(f"单条失败 {it['pid']}: {ex}")
        fails_in_a_row = 0 if ok > 0 else fails_in_a_row + 1
        if fails_in_a_row >= 5:
            log("连续 5 批 0 成功, 退出待续跑")
            sys.exit(1)
        if ok == 0:
            time.sleep(60)
    log("E1a' 完成")
    (RESULTS / "e1" / "e1a_goldfix.done").write_text("done\n")


if __name__ == "__main__":
    main()
