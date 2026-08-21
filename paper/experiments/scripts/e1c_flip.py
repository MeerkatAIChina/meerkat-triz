# -*- coding: utf-8 -*-
"""E1c 翻转率 (Coin Flip 复现): 分层 40 题 pairwise v4 vs base,
AB/BA 双序 x 5 次重复 x T in {0, 0.7} = 800 裁决。
输出 results/e1/e1c_flip.jsonl (断点续跑)。"""
import sys, json, random, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from e1_common import (get_client, call_chat, append_jsonl, load_jsonl,
                       parse_json_array, load_gold, load_responses,
                       PAIRWISE_SYSTEM, build_pairwise_user, log, RESULTS)

OUT = RESULTS / "e1" / "e1c_flip.jsonl"
JUDGE = "moonshot-v1-32k"
BATCH = 5
REPS = 5
TEMPS = [0.0, 0.7]
N_Q = 40
TOTAL = N_Q * 2 * REPS * len(TEMPS)


def stratified_sample(gold, n=N_Q, seed=42):
    by_sub = {}
    for it in gold:
        by_sub.setdefault(it["subset"], []).append(it)
    rng = random.Random(seed)
    picked = []
    total = len(gold)
    for sub, items in sorted(by_sub.items()):
        k = round(len(items) / total * n)
        items = sorted(items, key=lambda x: x["id"])
        rng.shuffle(items)
        picked.extend(items[:k])
    return sorted(picked, key=lambda x: x["id"])


def build_todo(gold, resp_v4, resp_base):
    done = {(r["qid"], r["order"], r["rep"], r["temp"]) for r in load_jsonl(OUT)}
    todo = []
    for it in gold:
        for order in ("AB", "BA"):
            if order == "AB":
                a, b = resp_v4[it["id"]], resp_base[it["id"]]
            else:
                a, b = resp_base[it["id"]], resp_v4[it["id"]]
            for temp in TEMPS:
                for rep in range(REPS):
                    if (it["id"], order, rep, temp) not in done:
                        todo.append({"pid": f"{it['id']}#{order}#T{temp}#R{rep}",
                                     "qid": it["id"], "order": order,
                                     "subset": it["subset"], "temp": temp, "rep": rep,
                                     "question": it["question"],
                                     "reference_answer": it["reference_answer"],
                                     "resp_a": a, "resp_b": b})
    return todo, len(done)


def main():
    (RESULTS / "e1").mkdir(exist_ok=True)
    gold = stratified_sample(load_gold())
    log(f"E1c 分层抽样 {len(gold)} 题: " +
        json.dumps({s: sum(1 for it in gold if it["subset"] == s)
                    for s in sorted({it['subset'] for it in gold})}, ensure_ascii=False))
    resp_v4, src_v4 = load_responses("v4")
    resp_base, src_base = load_responses("base")
    log(f"responses: v4={src_v4} base={src_base}")

    client = get_client()
    fails_in_a_row = 0
    while True:
        todo, ndone = build_todo(gold, resp_v4, resp_base)
        log(f"进度 {ndone}/{TOTAL}, 剩余 {len(todo)}")
        if not todo:
            break
        # 同 batch 内 temp 一致
        t0 = todo[0]["temp"]
        batch = [t for t in todo if t["temp"] == t0][:BATCH]
        user = build_pairwise_user(batch)
        text = call_chat(client, JUDGE, PAIRWISE_SYSTEM, user,
                         temperature=t0, max_tokens=2000)
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
                            "subset": it["subset"], "temp": it["temp"],
                            "rep": it["rep"],
                            "winner_pos": "tie" if w == "TIE" else w,
                            "reason": str(e.get("reason", ""))[:300],
                            "judge": JUDGE, "base_src": src_base})
                        ok += 1
            except Exception as ex:
                log(f"解析失败: {ex}")
        if ok < len(batch):
            for it in batch:
                key = (it["qid"], it["order"], it["rep"], it["temp"])
                if key in {(r["qid"], r["order"], r["rep"], r["temp"]) for r in load_jsonl(OUT)}:
                    continue
                t2 = call_chat(client, JUDGE, PAIRWISE_SYSTEM,
                               build_pairwise_user([it]),
                               temperature=it["temp"], max_tokens=600)
                try:
                    e = parse_json_array(t2)[0]
                    w = str(e.get("winner", "")).upper()
                    if w not in ("A", "B", "TIE"):
                        raise ValueError(f"bad winner {w}")
                    append_jsonl(OUT, {
                        "qid": it["qid"], "order": it["order"],
                        "subset": it["subset"], "temp": it["temp"],
                        "rep": it["rep"],
                        "winner_pos": "tie" if w == "TIE" else w,
                        "reason": str(e.get("reason", ""))[:300],
                        "judge": JUDGE, "base_src": src_base})
                    ok += 1
                except Exception as ex:
                    log(f"单条失败 {it['pid']}: {ex}")
        fails_in_a_row = 0 if ok > 0 else fails_in_a_row + 1
        if fails_in_a_row >= 5:
            log("连续 5 批 0 成功, 退出待续跑")
            sys.exit(1)
        if ok == 0:
            time.sleep(60)
    log("E1c 完成")
    (RESULTS / "e1" / "e1c.done").write_text("done\n")


if __name__ == "__main__":
    main()
