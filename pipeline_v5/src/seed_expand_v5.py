#!/usr/bin/env python3
"""
Worker H 种子扩写 - 步骤2: Moonshot API 扩写被弃短种子 (Owner 裁决方案③)。

每条保留原 instruction 与子集标签, 基于原 output (截尾后文本) 扩写为
150-600 字符的专业完整回答: 保持原答案结论与术语, 补全推理与结构,
禁止模板化收尾 (R2 黑名单 + tail80 频次>=3 硬门)。

纪律: RPM=3 全局限速, 批量 5 条/请求, 429/异常指数退避, jsonl 追加断点续跑。
质量门 (与 Worker A 同标准):
  - 长度 150-600 硬校验; 区间外单条重生成 1 次, 仍不合格丢弃并计数
  - 模板收尾: tail80 全库频次>=3 拒绝; 命中 R2 黑名单 (b1/b2) 拒绝
产物:
  data/processed/v5_data/seed_expanded.jsonl      扩写存活 (追加, 断点续跑)
  data/processed/v5_data/seed_expand_dropped.jsonl 丢弃及原因
  data/processed/v5_data/seed_expand_gen.log      运行日志 (tee)
"""
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data/processed/v5_data"

P_CAND = OUT_DIR / "seed_expand_candidates.jsonl"
P_BLACKLIST = OUT_DIR / "seed_expand_blacklist.json"
P_OUT = OUT_DIR / "seed_expanded.jsonl"
P_DROP = OUT_DIR / "seed_expand_dropped.jsonl"

MODEL = "kimi-k2.5"
BASE_URL = "https://api.moonshot.cn/v1"
RPM = 3
TEMPERATURE = None            # kimi-k2.x 仅允许 temperature=1, 不传参
MAX_TOKENS_BATCH = 6000       # 5 条 x ≤600 字 + 格式开销, 余量充足
MAX_TOKENS_SINGLE = 2000
MAX_API_RETRIES = 5
BATCH = 5
WORKERS = 3
LEN_MIN, LEN_MAX = 150, 600
TEMPLATE_FREQ_LIMIT = 3       # tail80 频次>=3 即黑名单 (同 Worker A R2 口径)

GEN_SYSTEM = (
    "你是 TRIZ/创新方法领域专家助手。用户会给出若干条问题及其已有的草稿答案, "
    "请逐条把草稿扩写为完整的专业回答。\n"
    "【扩写纪律】\n"
    "1) 严格保持原草稿答案的技术结论、术语与观点不变, 只允许补充推理过程、"
    "结构化展开、必要的解释与例子, 不得推翻或替换原结论;\n"
    "2) 每条扩写后 250-550 个汉字字符 (硬范围 150-600, 越界会被系统拒绝);\n"
    "3) 禁止模板化、口号式收尾 (如\"建议进一步分析/建议进行FTO分析/综合评估"
    "该方案可行\"一类套话); 各条的收尾句式必须多样化, 自然结束于具体内容;\n"
    "4) 直接输出正文, 不要重复问题, 不要加\"扩写后:\"等前缀。\n"
    "【输出格式】不要使用 JSON。对第 i 条, 严格输出:\n"
    "【回答i】\n(扩写正文)\n【/回答i】"
)

ANS_RE = re.compile(r"【回答\s*(\d+)\s*】\s*(.+?)\s*(?:【/回答\s*\d*\s*】|$)", re.S)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


_RATE_LOCK = threading.Lock()
_LAST_START = [0.0]

def rate_gate(rpm=RPM):
    with _RATE_LOCK:
        wait = 60.0 / rpm - (time.time() - _LAST_START[0])
        if wait > 0:
            time.sleep(wait)
        _LAST_START[0] = time.time()


def get_client():
    from openai import OpenAI
    key = os.environ.get("MOONSHOT_API_KEY")
    if not key:
        raise RuntimeError("MOONSHOT_API_KEY 未设置")
    return OpenAI(api_key=key, base_url=BASE_URL)


def call_api(client, messages, max_tokens):
    kwargs = {"model": MODEL, "messages": messages, "max_tokens": max_tokens}
    if TEMPERATURE is not None:
        kwargs["temperature"] = TEMPERATURE
    delay = 5
    for attempt in range(MAX_API_RETRIES):
        try:
            rate_gate()
            resp = client.chat.completions.create(**kwargs)
            ch = resp.choices[0]
            return ch.message.content, ch.finish_reason
        except Exception as e:
            log(f"调用失败 (attempt {attempt + 1}/{MAX_API_RETRIES}): {str(e)[:150]}")
            if attempt < MAX_API_RETRIES - 1:
                time.sleep(delay)
                delay = min(delay * 2, 90)
    return None, "fail"


def parse_answers(text: str):
    out = {}
    for m in ANS_RE.finditer(text):
        try:
            idx = int(m.group(1))
        except ValueError:
            continue
        ans = m.group(2).strip()
        if ans:
            out[idx] = ans
    return out


def batch_user_prompt(batch):
    lines = ["请把以下各条的草稿答案扩写为完整专业回答, "
             "严格使用【回答i】...【/回答i】格式, 每条 250-550 字:\n"]
    for i, it in enumerate(batch, 1):
        q = it["instruction"]
        if it.get("input"):
            q += f"\n(背景: {it['input']})"
        lines.append(f"【问题{i}】({it['subset']}) {q}\n【草稿答案{i}】\n{it['output_base']}")
    return "\n\n".join(lines)


def single_user_prompt(item, extra=""):
    q = item["instruction"]
    if item.get("input"):
        q += f"\n(背景: {item['input']})"
    msg = (f"请把以下草稿答案扩写为完整专业回答, 以【回答1】开头、【/回答1】结尾, "
           f"250-550 字:\n\n【问题】({item['subset']}) {q}\n"
           f"【草稿答案】\n{item['output_base']}")
    if extra:
        msg += f"\n\n{extra}"
    return msg


def gen_batch(client, batch):
    msgs = [{"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": batch_user_prompt(batch)}]
    content, finish = call_api(client, msgs, MAX_TOKENS_BATCH)
    if content is None:
        return None
    out = parse_answers(content)
    if not out:
        log(f"批量解析为空 (finish={finish}, len={len(content)})")
        return None
    return out


def gen_single(client, item, extra=""):
    msgs = [{"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": single_user_prompt(item, extra)}]
    content, finish = call_api(client, msgs, MAX_TOKENS_SINGLE)
    if content is None:
        return None
    out = parse_answers(content)
    if out:
        return out.get(1) or next(iter(out.values()))
    t = re.sub(r"【/?回答\s*\d*\s*】", "", content).strip()
    return t or None


def last_sentence(text: str) -> str:
    t = (text or "").rstrip()
    if not t:
        return ""
    i = max(t.rfind(c, 0, len(t) - 1) for c in "。!！?？\n")
    return t[i + 1:].strip() if i >= 0 else t


def make_validator(b1, b2, tail80_cnt, lock):
    def validate(ans):
        if not ans:
            return False, "empty"
        n = len(ans)
        if n < LEN_MIN:
            return False, f"too_short({n})"
        if n > LEN_MAX:
            return False, f"too_long({n})"
        if ans[-80:] in b1 or last_sentence(ans) in b2:
            return False, "r2_blacklist_tail"
        with lock:
            c = tail80_cnt.get(ans[-80:], 0)
        if c >= TEMPLATE_FREQ_LIMIT - 1:
            return False, "tail80_template"
        return True, "ok"
    return validate


def main():
    cands = [json.loads(l) for l in open(P_CAND, encoding="utf-8") if l.strip()]
    bl = json.loads(P_BLACKLIST.read_text(encoding="utf-8"))
    b1, b2 = set(bl["b1_tail80"]), set(bl["b2_last_sentence"])

    done, dropped = set(), set()
    tail80_cnt = Counter()
    if P_OUT.is_file():
        for l in open(P_OUT, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                done.add(r["group_id"])
                tail80_cnt[r["output"][-80:]] += 1
    if P_DROP.is_file():
        for l in open(P_DROP, encoding="utf-8"):
            if l.strip():
                dropped.add(json.loads(l)["group_id"])
    todo = [c for c in cands if c["group_id"] not in done and c["group_id"] not in dropped]
    log(f"候选 {len(cands)} | 已完成 {len(done)} | 已丢弃 {len(dropped)} | 待扩写 {len(todo)}")
    if not todo:
        log("无待办, 退出")
        return

    client = get_client()
    validate = make_validator(b1, b2, tail80_cnt, threading.Lock())
    out_f = open(P_OUT, "a", encoding="utf-8")
    drop_f = open(P_DROP, "a", encoding="utf-8")
    wlock = threading.Lock()
    counters = Counter()

    def accept(item, ans, attempt, batch_no):
        rec = {"instruction": item["instruction"], "input": item.get("input", ""),
               "output": ans, "subset": item["subset"], "group_id": item["group_id"],
               "output_base": item["output_base"], "drop_reason": item["drop_reason"],
               "gen_params": {"model": MODEL, "temperature": 1, "attempt": attempt,
                              "batch_no": batch_no, "kind": "seed_expansion"}}
        with wlock:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            tail80_cnt[ans[-80:]] += 1
            counters["ok"] += 1

    def drop(item, reason, batch_no):
        rec = {"group_id": item["group_id"], "subset": item["subset"],
               "drop_reason": item["drop_reason"], "fail_reason": reason,
               "batch_no": batch_no}
        with wlock:
            drop_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            drop_f.flush()
            counters[f"drop_{reason.split('(')[0]}"] += 1

    def process_batch(bi, batch):
        answers = gen_batch(client, batch)
        if answers is None:
            for item in batch:  # 整批失败: 逐条单发兜底 (仍计 attempt 1)
                ans = gen_single(client, item)
                ok, reason = validate(ans)
                if ok:
                    accept(item, ans, 1, bi)
                else:
                    ans2 = gen_single(client, item,
                                      f"上次输出未通过校验 ({reason}), 请修正后重新输出。")
                    ok2, reason2 = validate(ans2)
                    if ok2:
                        accept(item, ans2, 2, bi)
                    else:
                        drop(item, reason2, bi)
            return
        for i, item in enumerate(batch, 1):
            ans = answers.get(i)
            ok, reason = validate(ans)
            if ok:
                accept(item, ans, 1, bi)
                continue
            counters[f"retry_{reason.split('(')[0]}"] += 1
            ans2 = gen_single(client, item,
                              f"上次输出未通过校验 ({reason}), 请修正后重新输出。")
            ok2, reason2 = validate(ans2)
            if ok2:
                accept(item, ans2, 2, bi)
            else:
                drop(item, reason2, bi)

    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    log(f"共 {len(batches)} 个批次, {WORKERS} 线程 (RPM={RPM} 全局限速)")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process_batch, bi + 1, b): b
                for bi, b in enumerate(batches)}
        n_done_batches = 0
        for fu in as_completed(futs):
            try:
                fu.result()
            except Exception as e:  # 兜底: 整批记丢弃, 不中断全局
                log(f"批次异常: {str(e)[:150]}")
                for item in futs[fu]:
                    drop(item, "batch_exception", -1)
            n_done_batches += 1
            if n_done_batches % 5 == 0 or n_done_batches == len(batches):
                done_now = counters["ok"]
                log(f"批次进度 {n_done_batches}/{len(batches)} | 扩写存活 {done_now} | "
                    f"丢弃 {sum(v for k, v in counters.items() if k.startswith('drop_'))} | "
                    f"耗时 {time.time() - t0:.0f}s")
    out_f.close()
    drop_f.close()
    log(f"完成: {dict(counters)}")


if __name__ == "__main__":
    main()
