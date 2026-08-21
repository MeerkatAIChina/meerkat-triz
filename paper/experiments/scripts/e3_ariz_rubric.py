# -*- coding: utf-8 -*-
"""E3 ARIZ rubric 逐项重判: 金标 ariz 20 题 x {base,v2,v4},
judge 按 ARIZ 6 步骤逐项 0/1 + 证据短语。T=0, moonshot-v1-32k。
输出 results/e3/e3_ariz_rubric.jsonl"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "e1"))
from e1_common import (get_client, call_chat, append_jsonl, load_jsonl,
                       parse_json_array, load_gold, load_responses,
                       log, RESULTS)

OUT = RESULTS / "e3" / "e3_ariz_rubric.jsonl"
JUDGE = "moonshot-v1-32k"
BATCH = 5
RESP_TRUNC = 3000  # ARIZ 回答长, 500 字符截断不可用; 3000 覆盖 v2/v4 全长

ARIZ_SYSTEM = (
    "你是 TRIZ/ARIZ 方法论资深评审。给定题目、参考答案与一个 AI 回答, "
    "逐项判断该回答是否体现了 ARIZ 算法流程的以下 6 个步骤(语义覆盖, 不限措辞, "
    "同义表述视为覆盖, 例如 'IFR'、'理想解'、'最终理想结果' 都算理想解表述):\n"
    "- step1 问题与系统分析: 分析问题情境, 明确技术系统及其功能/问题所在。"
    "  正例: '该技术系统的核心问题是……' '分析当前研磨系统……'\n"
    "- step2 矛盾确定: 明确表述技术矛盾(改善X导致Y恶化)或物理矛盾。"
    "  正例: '这里的技术矛盾是: 提高温度可改善沉积速率, 但会导致薄膜应力增大'\n"
    "- step3 理想解表述: 提出最终理想解(IFR)或理想化方向。"
    "  正例: '理想状态下, 系统应在不增加能耗的前提下自动……' 'IFR: ……'\n"
    "- step4 资源分析: 分析可利用的系统内部/外部资源(物质、场、空间、时间、信息等)。"
    "  正例: '可利用现有的磁场资源……' '系统中闲置的资源包括……'\n"
    "- step5 方案生成: 运用发明原理/标准解/科学效应等给出具体解决方案。"
    "  正例: '应用发明原理35(物理化学状态变化)……' '建议采用……方案'\n"
    "- step6 方案评估与实施: 对方案进行比较/评估并给出实施建议或步骤。"
    "  正例: '方案一成本低但效率有限, 方案二……推荐方案二, 实施分三步……'\n"
    "对每题每步输出 0(未体现) 或 1(体现), 并给出证据短语(从回答中摘抄体现该步的关键表述, "
    "未体现则给空字符串)。只输出 JSON 数组, 格式:\n"
    '[{"id": "题目id", "step1": 0, "step2": 1, "step3": 0, "step4": 1, "step5": 1, '
    '"step6": 0, "evidence": {"step1": "", "step2": "……", ...}}, ...]'
)


def build_user(chunk, resps):
    parts = []
    for mtag, it in chunk:
        resp = resps[mtag][it["id"]][:RESP_TRUNC]
        parts.append(
            f"【题目 {it['id']}】\n问题: {it['question']}\n"
            f"参考答案: {it['reference_answer'][:1500]}\nAI 回答: {resp}\n")
    return "\n".join(parts) + "\n请逐题逐项判定, 输出 JSON 数组。"


def main():
    (RESULTS / "e3").mkdir(exist_ok=True)
    gold = [it for it in load_gold() if it["subset"] == "ariz_guidance"]
    log(f"E3 ariz 题数 {len(gold)}")
    resps = {}
    for mtag in ("base", "v2", "v4"):
        resps[mtag], src = load_responses(mtag)
        log(f"{mtag} responses <- {src}")

    client = get_client()
    fails_in_a_row = 0
    while True:
        done = {(r["qid"], r["model"]) for r in load_jsonl(OUT)}
        todo = [(m, it) for m in ("base", "v2", "v4") for it in gold
                if (it["id"], m) not in done]
        log(f"E3 进度 {len(done)}/60, 剩余 {len(todo)}")
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
                            "judge": JUDGE, "resp_trunc": RESP_TRUNC})
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
                        "judge": JUDGE, "resp_trunc": RESP_TRUNC})
                    ok += 1
                except Exception as ex:
                    log(f"单条失败 {mtag}/{it['id']}: {ex}")
        fails_in_a_row = 0 if ok > 0 else fails_in_a_row + 1
        if fails_in_a_row >= 5:
            log("连续 5 批 0 成功, 退出待续跑")
            sys.exit(1)
        if ok == 0:
            time.sleep(60)
    log("E3 完成")
    (RESULTS / "e3" / "e3.done").write_text("done\n")


if __name__ == "__main__":
    main()
