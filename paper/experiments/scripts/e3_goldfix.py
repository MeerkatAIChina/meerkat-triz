# -*- coding: utf-8 -*-
"""E3' 干净锚点 ARIZ rubric 重判: base_goldfix 的 20 道 ariz 题, 6 步骤 rubric。
输出追加 results/e3/e3_ariz_rubric.jsonl (model=base_goldfix)。"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "e1"))
from e1_common import (get_client, call_chat, append_jsonl, load_jsonl,
                       parse_json_array, load_gold, load_responses,
                       log, RESULTS)
from e3_ariz_rubric import ARIZ_SYSTEM, build_user, JUDGE

OUT = RESULTS / "e3" / "e3_ariz_rubric.jsonl"
BATCH = 5


def main():
    gold = [it for it in load_gold() if it["subset"] == "ariz_guidance"]
    resps = {"base_goldfix": load_responses("base_goldfix")[0]}
    log(f"E3' ariz {len(gold)} 题 x base_goldfix")
    client = get_client()
    fails_in_a_row = 0
    while True:
        done = {(r["qid"], r["model"]) for r in load_jsonl(OUT)}
        todo = [("base_goldfix", it) for it in gold
                if (it["id"], "base_goldfix") not in done]
        log(f"E3' 进度 {20 - len(todo)}/20, 剩余 {len(todo)}")
        if not todo:
            break
        chunk = todo[:BATCH]
        text = call_chat(client, JUDGE, ARIZ_SYSTEM, build_user(chunk, resps),
                         temperature=0.0, max_tokens=3000)
        ok = 0
        if text:
            try:
                arr = parse_json_array(text)
                by_id = {str(e.get("id")): e for e in arr if isinstance(e, dict)}
                for mtag, it in chunk:
                    e = by_id.get(it["id"])
                    if e and "step1" in e:
                        append_jsonl(OUT, {
                            "qid": it["id"], "model": mtag,
                            **{f"step{i}": int(e.get(f"step{i}", 0)) for i in range(1, 7)},
                            "evidence": {f"step{i}": str((e.get("evidence") or {}).get(f"step{i}", ""))[:200]
                                         for i in range(1, 7)},
                            "judge": JUDGE, "resp_trunc": 3000})
                        ok += 1
            except Exception as ex:
                log(f"解析失败: {ex}")
        if ok < len(chunk):
            done_now = {(r["qid"], r["model"]) for r in load_jsonl(OUT)}
            for mtag, it in chunk:
                if (it["id"], mtag) in done_now:
                    continue
                t2 = call_chat(client, JUDGE, ARIZ_SYSTEM,
                               build_user([(mtag, it)], resps),
                               temperature=0.0, max_tokens=1500)
                try:
                    e = parse_json_array(t2)[0]
                    if "step1" not in e:
                        raise ValueError("no step1")
                    append_jsonl(OUT, {
                        "qid": it["id"], "model": mtag,
                        **{f"step{i}": int(e.get(f"step{i}", 0)) for i in range(1, 7)},
                        "evidence": {f"step{i}": str((e.get("evidence") or {}).get(f"step{i}", ""))[:200]
                                     for i in range(1, 7)},
                        "judge": JUDGE, "resp_trunc": 3000})
                    ok += 1
                except Exception as ex:
                    log(f"单条失败 {it['id']}: {ex}")
        fails_in_a_row = 0 if ok > 0 else fails_in_a_row + 1
        if fails_in_a_row >= 5:
            log("连续 5 批 0 成功, 退出待续跑")
            sys.exit(1)
        if ok == 0:
            time.sleep(60)
    log("E3' 完成")
    (RESULTS / "e3" / "e3_goldfix.done").write_text("done\n")


if __name__ == "__main__":
    main()
