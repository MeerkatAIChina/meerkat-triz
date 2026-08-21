# -*- coding: utf-8 -*-
"""E1b 补臂: moonshot-v1-8k 给 base_goldfix 100 题按 v4 harness 同一 rubric 打分。
32k 臂直接复用 eval_v4_base_goldfix 缓存内的 judge 分, 无需重跑。
输出追加 results/e1/e1b_rejudge_moonshot_v1_8k.jsonl (model=base_goldfix)。"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from e1_common import (get_client, call_chat, append_jsonl, load_jsonl,
                       parse_json_array, load_gold, load_responses,
                       JUDGE_SYSTEM, build_judge_user, log, RESULTS)

JUDGE = "moonshot-v1-8k"
OUT = RESULTS / "e1" / "e1b_rejudge_moonshot_v1_8k.jsonl"
BATCH = 5


def main():
    gold = load_gold()
    resps, src = load_responses("base_goldfix")
    log(f"E1b补臂 base_goldfix responses <- {src}")
    client = get_client()
    fails_in_a_row = 0
    while True:
        done = {(r["qid"], r["model"]) for r in load_jsonl(OUT)}
        todo = [it for it in gold if (it["id"], "base_goldfix") not in done]
        log(f"E1b补臂 进度 {100 - len(todo)}/100, 剩余 {len(todo)}")
        if not todo:
            break
        chunk = todo[:BATCH]
        resp_map = {it["id"]: resps[it["id"]] for it in chunk}
        text = call_chat(client, JUDGE, JUDGE_SYSTEM,
                         build_judge_user(chunk, resp_map, 1500),
                         temperature=0.0, max_tokens=2000)
        ok = 0
        if text:
            try:
                arr = parse_json_array(text)
                by_id = {str(e.get("id")): e for e in arr if isinstance(e, dict)}
                for it in chunk:
                    e = by_id.get(it["id"])
                    if e and "overall" in e:
                        append_jsonl(OUT, {
                            "qid": it["id"], "model": "base_goldfix",
                            "subset": it["subset"],
                            "accuracy": e.get("accuracy"),
                            "completeness": e.get("completeness"),
                            "triz_correctness": e.get("triz_correctness"),
                            "structure": e.get("structure"),
                            "overall": e.get("overall"), "judge": JUDGE})
                        ok += 1
            except Exception as ex:
                log(f"解析失败: {ex}")
        if ok < len(chunk):
            done_now = {(r["qid"], r["model"]) for r in load_jsonl(OUT)}
            for it in chunk:
                if (it["id"], "base_goldfix") in done_now:
                    continue
                t2 = call_chat(client, JUDGE, JUDGE_SYSTEM,
                               build_judge_user([it], {it["id"]: resps[it["id"]]}, 1500),
                               temperature=0.0, max_tokens=600)
                try:
                    e = parse_json_array(t2)[0]
                    if "overall" not in e:
                        raise ValueError("no overall")
                    append_jsonl(OUT, {
                        "qid": it["id"], "model": "base_goldfix",
                        "subset": it["subset"],
                        "accuracy": e.get("accuracy"),
                        "completeness": e.get("completeness"),
                        "triz_correctness": e.get("triz_correctness"),
                        "structure": e.get("structure"),
                        "overall": e.get("overall"), "judge": JUDGE})
                    ok += 1
                except Exception as ex:
                    log(f"单条失败 {it['id']}: {ex}")
        fails_in_a_row = 0 if ok > 0 else fails_in_a_row + 1
        if fails_in_a_row >= 5:
            log("连续 5 批 0 成功, 退出待续跑")
            sys.exit(1)
        if ok == 0:
            time.sleep(60)
    log("E1b补臂 完成")
    (RESULTS / "e1" / "e1b_goldfix.done").write_text("done\n")


if __name__ == "__main__":
    main()
