#!/usr/bin/env python
"""
pipeline_v5 风格 C 长答生成 (方案 §4.1 方案 C)。

输入: data/processed/v5_data/styleC_longanswer_sampling.jsonl
  (3,445 条, Worker A 按子集分层抽样, 含 group_id)
输出: data/processed/v5_data/styleC_long_answers.jsonl
  字段: group_id / subset / instruction / system / completion / gen_params
  system 固定为: 详细模式:给出完整的结构化分析,包含术语定义、推理过程与实施建议

质量门 (与任务书一致):
  - 长度硬校验 1200-2500 字符: 区间外重生成 1 次 (单条), 仍不合格丢弃并计数
    (丢弃清单落 styleC_long_dropped.jsonl);
  - 模板收尾黑名单: 与 Worker A 种子清洗同一机制 (末 80 字符频次 >=3 的串),
    预防式拒绝命中样本 (重生成 1 次, 仍命中丢弃);
  - 10% 抽检队列落 styleC_long_review.md (seed=42 随机);
  - 批量 5 条/请求, 429/异常指数退避, 追加 jsonl 断点续跑 (按 group_id 跳过)。

v3 修订 (2026-07-24):
  - 模型切换 moonshot-v1-32k -> kimi-k2.5 (同属 Moonshot API; 实测 v1 批量输出
    收敛 ~600 字/条、batch=2 仅 ~800 字/条, 无法满足 1200 字硬下限;
    k2.5 批量 5 条实测 1442-1772 字全部达标; k2.x 仅允许 temperature=1, 调用不传温度);
  - 输出协议: 分隔符【回答i】...【/回答i】(长文本 JSON 转义易碎);
  - 吞吐: 6 线程流水线, 全局令牌桶保证请求启动间隔 >=20s (RPM=3 纪律不变,
    k2.5 单请求延迟 ~140s, 串行执行 26h, 流水线后恢复任务书 ~4h 预期);
  - 系统提示内嵌 1341 字完整示例锚定长度与结构。

用法:
  venv_v5/bin/python pipeline_v5/src/styleC_gen_v5.py [--limit N] [--review-only]
"""

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

IN_DEFAULT = "data/processed/v5_data/styleC_longanswer_sampling.jsonl"
OUT_DEFAULT = "data/processed/v5_data/styleC_long_answers.jsonl"
DROP_DEFAULT = "data/processed/v5_data/styleC_long_dropped.jsonl"
REVIEW_DEFAULT = "data/processed/v5_data/styleC_long_review.md"
REPORT_DEFAULT = "data/processed/v5_data/styleC_long_report.json"

MODEL = "kimi-k2.5"
BASE_URL = "https://api.moonshot.cn/v1"
RPM = 3                      # 请求启动速率 (全局令牌桶)
TEMPERATURE = None           # kimi-k2.x 仅允许 temperature=1, 不传参
MAX_TOKENS_BATCH = 16000
MAX_TOKENS_SINGLE = 4000
MAX_API_RETRIES = 5
BATCH = 5
WORKERS = 6
LEN_MIN, LEN_MAX = 1200, 2500
TEMPLATE_FREQ_LIMIT = 3      # 末80字符频次 >=3 即黑名单 (同种子清洗 R2 口径)

SYSTEM_FOR_RECORD = "详细模式:给出完整的结构化分析,包含术语定义、推理过程与实施建议"

# ~1341 字完整示例, 锚定输出长度与结构
FEWSHOT_EXAMPLE = """【输出示例】
问题: 什么是技术矛盾？它在TRIZ理论中起什么作用？
【回答1】
技术矛盾是TRIZ理论中最基础、最核心的概念之一，理解它是掌握整套创新方法的前提。

一、术语定义

技术矛盾（Technical Contradiction）指的是：当我们试图改善技术系统的某一个参数时，会导致另一个参数同时恶化的两难处境。阿奇舒勒在对大量发明专利进行分析后发现，绝大多数发明问题的本质都是技术矛盾。例如，我们希望提高飞机机翼的强度（改善参数），通常需要增加结构材料，但这会使重量上升（恶化参数）；强度与重量之间就构成了一对典型的技术矛盾。TRIZ将工程中千变万化的具体参数归纳为39个通用工程参数，如重量、强度、速度、温度、可靠性、可制造性等，任何具体的技术矛盾都可以转译为通用参数之间的矛盾对。与技术矛盾相关的另一个重要概念是物理矛盾，即对同一个参数提出两个相反的要求，它是技术矛盾进一步提炼后的更深层表达。此外，解决技术矛盾的主要工具是矛盾矩阵和40个发明原理：矛盾矩阵的行与列分别对应改善参数与恶化参数，矩阵单元格给出统计上高频使用的发明原理编号，为解题者提供方向性指引。

二、推理过程

从技术矛盾的定义出发，可以推导它在TRIZ体系中的作用机制。第一步，技术矛盾是问题的"标准化接口"：工程师面对的具体困境千差万别，但只要能将其转译为"改善哪个参数、恶化哪个参数"，就能把陌生问题映射到TRIZ的标准问题框架上，从而调用通用解法。第二步，技术矛盾界定了创新的层级：折中、妥协式设计并不消除矛盾，只是在两端之间取平衡点；而发明的本质是消除矛盾，即在不引起恶化的前提下实现改善，这正是高水平专利与平庸设计的分水岭。第三步，技术矛盾决定了工具的选择路径：识别矛盾对之后，查矛盾矩阵得到若干候选发明原理，再结合具体情境筛选，例如强度与重量的矛盾常对应"复合材料""预先作用""分割"等原理。第四步，技术矛盾还是通向物理矛盾和理想解的阶梯：当技术矛盾难以直接消除时，可以把它进一步深化为物理矛盾，利用分离原理（时间分离、空间分离、条件分离、系统级分离）求解，使系统逼近理想最终结果（IFR）。因此，技术矛盾不仅是一个描述性概念，更是整个TRIZ解题流程的枢纽，把"问题定义"与"解法生成"两个阶段连接起来。

三、实施建议

在实际工程工作中运用技术矛盾概念时，建议遵循以下做法。第一，训练"矛盾化表述"的习惯：遇到改进需求时，强制自己写出"如果改善X，则Y会恶化"的句式，避免直接跳到方案层面。第二，认真完成参数转译：把行业特定参数对照39个通用工程参数表进行归类，转译准确与否直接决定矛盾矩阵推荐的相关性。第三，不要满足于矩阵给出的第一个原理：通常矩阵会给出三到四个候选原理，应逐一展开设想，组合使用往往比单一原理更有效。第四，建立团队级的矛盾案例库：把本行业已解决的典型矛盾对及其有效原理记录下来，形成组织知识资产，可显著缩短后续项目的分析时间。第五，警惕"伪矛盾"：有些看似矛盾的局面其实源于约束条件设定错误，重新审视设计边界后矛盾可能自然消失。第六，将技术矛盾分析与功能分析、因果链分析结合使用，先定位问题根源，再构建矛盾，可以避免在错误的对象上浪费分析精力。通过持续练习，工程师可以把矛盾思维内化为直觉反应，这是TRIZ从工具转化为能力的关键一步。
【/回答1】
【示例结束】"""

GEN_SYSTEM = (
    "你是 TRIZ/创新方法领域专家助手, 当前处于详细模式: 给出完整的结构化分析, "
    "包含术语定义、推理过程与实施建议。"
    "用户会给出若干条问题, 请逐条撰写详细长答。\n"
    "【输出格式】不要使用 JSON。对第 i 条问题, 严格按以下格式输出:\n"
    "【回答i】\n(长答正文)\n【/回答i】\n"
    "【长度要求】每条长答 1350-1700 个汉字字符, 低于 1200 字将被系统直接拒绝"
    "(字数不足是最常见失败原因, 请务必充分展开); 结构依次包含:\n"
    "一、术语定义(约 350 字); 二、推理过程(约 650 字, 分步骤); 三、实施建议"
    "(约 500 字, 分条列出)。\n"
    "【多样性要求】各条长答的表述、举例与收尾句式必须多样化, 禁止固定模板收尾。\n"
    "以下是一条例答, 请严格参照其篇幅量级与结构深度:\n\n"
    + FEWSHOT_EXAMPLE
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- 全局限速器: 请求启动间隔 >= 60/RPM 秒 (跨线程) ----
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


ANS_RE = re.compile(r"【回答\s*(\d+)\s*】\s*(.+?)\s*(?:【/回答\s*\d*\s*】|$)", re.S)


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


def call_api(client, messages, max_tokens):
    """限速 + 指数退避; 返回 (content, finish_reason), 重试耗尽返回 (None, 'fail')。"""
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


def batch_user_prompt(batch):
    lines = ["请为以下问题逐条生成长答, 严格使用【回答i】...【/回答i】格式, "
             "每条 1350-1700 字:\n"]
    for i, it in enumerate(batch, 1):
        q = it["instruction"]
        if it.get("input"):
            q += f"\n(背景: {it['input']})"
        lines.append(f"【问题{i}】({it['subset']}) {q}")
    return "\n\n".join(lines)


def single_user_prompt(item, extra=""):
    q = item["instruction"]
    if item.get("input"):
        q += f"\n(背景: {item['input']})"
    msg = (f"请为以下问题生成长答, 以【回答1】开头、【/回答1】结尾:\n\n"
           f"【问题】({item['subset']}) {q}")
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


def validate(ans, tail80_cnt):
    if not ans:
        return False, "empty"
    n = len(ans)
    if n < LEN_MIN:
        return False, f"too_short({n})"
    if n > LEN_MAX:
        return False, f"too_long({n})"
    if tail80_cnt.get(ans[-80:], 0) >= TEMPLATE_FREQ_LIMIT - 1:
        return False, "tail80_template"
    return True, "ok"


def load_done_ids(path: Path):
    done = set()
    tail80_cnt = Counter()
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done.add(r["group_id"])
                    tail80_cnt[r["completion"][-80:]] += 1
    return done, tail80_cnt


def write_report(records_path, report_path, counters, dropped_path):
    records = []
    if records_path.is_file():
        with open(records_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    lens = [len(r["completion"]) for r in records]
    lens_sorted = sorted(lens)
    tail80 = Counter(r["completion"][-80:] for r in records)
    attempts = Counter(r["gen_params"].get("attempt", 1) for r in records)
    rep = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "completed": len(records),
        "per_subset": dict(Counter(r["subset"] for r in records)),
        "gen_params": {"model": MODEL, "temperature": TEMPERATURE,
                       "rpm": RPM, "batch": BATCH, "workers": WORKERS,
                       "system": SYSTEM_FOR_RECORD},
        "quality_gates": {
            "len_range": [LEN_MIN, LEN_MAX],
            "len_min": min(lens, default=0),
            "len_max": max(lens, default=0),
            "len_mean": round(sum(lens) / len(lens), 1) if lens else 0,
            "len_p50": lens_sorted[len(lens) // 2] if lens else 0,
            "attempt_dist": {str(k): v for k, v in attempts.items()},
            "tail80_freq_ge3": [t for t, c in tail80.items() if c >= 3],
            "tail80_max_freq": max(tail80.values(), default=0),
            "rejected": counters,
            "dropped_file": str(dropped_path),
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)


def export_review(records_path, review_path, ratio=0.1, seed=42):
    records = []
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    rng = random.Random(seed)
    sample = rng.sample(records, max(1, int(len(records) * ratio)))
    sample.sort(key=lambda r: r["group_id"])
    lines = ["# 风格 C 长答 10% 人工抽检队列", "",
             f"导出时间: {datetime.now().isoformat(timespec='seconds')} | "
             f"总完成 {len(records)} 条, 抽检 {len(sample)} 条 (seed={seed})", ""]
    for r in sample:
        comp = r["completion"]
        lines += [
            f"## {r['group_id']} [{r['subset']}] ({len(comp)} 字符)",
            "",
            f"自动初检: 长度 {'PASS' if LEN_MIN <= len(comp) <= LEN_MAX else 'FAIL'}"
            f" | 人工状态: 待人工", "",
            f"**Instruction**: {r['instruction']}", "",
            "**Completion**:", "", comp, "", "---", ""]
    with open(review_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"抽检队列: {review_path} ({len(sample)} 条)")


def main():
    ap = argparse.ArgumentParser(description="v5 风格 C 长答生成")
    ap.add_argument("--input", default=IN_DEFAULT)
    ap.add_argument("--output", default=OUT_DEFAULT)
    ap.add_argument("--dropped", default=DROP_DEFAULT)
    ap.add_argument("--review", default=REVIEW_DEFAULT)
    ap.add_argument("--report", default=REPORT_DEFAULT)
    ap.add_argument("--limit", type=int, default=None, help="最多新完成 N 条 (试跑)")
    ap.add_argument("--review-only", action="store_true")
    args = ap.parse_args()

    def res(p):
        p = Path(p)
        return p if p.is_absolute() else PROJECT_ROOT / p

    in_path, out_path = res(args.input), res(args.output)
    drop_path, review_path, report_path = res(args.dropped), res(args.review), res(args.report)

    if args.review_only:
        export_review(out_path, review_path)
        return

    with open(in_path, encoding="utf-8") as f:
        todo_all = [json.loads(line) for line in f if line.strip()]
    log(f"抽样清单: {len(todo_all)} 条")

    done_ids, tail80_cnt = load_done_ids(out_path)
    if done_ids:
        log(f"断点续跑: 已完成 {len(done_ids)} 条")
    todo = [t for t in todo_all if t["group_id"] not in done_ids]
    if args.limit is not None:
        todo = todo[: args.limit * 2]  # 预留拒/丢损耗
    log(f"待生成: {len(todo)} 条")

    counters = Counter()
    state = {"completed": 0}
    write_lock = threading.Lock()
    cnt_lock = threading.Lock()
    client = get_client()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = open(out_path, "a", encoding="utf-8")
    fdrop = open(drop_path, "a", encoding="utf-8")

    def accept(item, ans, attempt, batch_no):
        rec = {
            "group_id": item["group_id"], "subset": item["subset"],
            "instruction": item["instruction"], "input": item.get("input", ""),
            "system": SYSTEM_FOR_RECORD, "completion": ans,
            "gen_params": {"model": MODEL, "temperature": 1,
                           "attempt": attempt, "batch_no": batch_no,
                           "protocol": "delimiter_v3"},
        }
        with write_lock:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
        with cnt_lock:
            tail80_cnt[ans[-80:]] += 1
            state["completed"] += 1
            n = state["completed"]
        if n % 25 == 0:
            log(f"本轮完成 {n} | 计数 {dict(counters)}")
            write_report(out_path, report_path, dict(counters), drop_path)

    def process_batch(bi, batch):
        """批量生成 + 校验; 失败项返回待单条重生成清单。"""
        if args.limit is not None and state["completed"] >= args.limit:
            return []
        answers = gen_batch(client, batch)
        retry_items = []
        for i, item in enumerate(batch, 1):
            if args.limit is not None and state["completed"] >= args.limit:
                break
            ans = answers.get(i) if answers else None
            with cnt_lock:
                ok, reason = validate(ans, tail80_cnt)
            if ok:
                accept(item, ans, 1, bi)
            else:
                with cnt_lock:
                    counters[f"batch_{reason.split('(')[0]}"] += 1
                retry_items.append((item, reason, bi))
        return retry_items

    def process_single(item, reason, bi):
        """重生成 1 次; 仍不合格丢弃并计数。"""
        if args.limit is not None and state["completed"] >= args.limit:
            return
        m = re.search(r"\((\d+)\)", reason)
        prev = m.group(1) if m else "不足"
        if "too_short" in reason:
            extra = (f"注意: 上次回答仅 {prev} 字被系统拒绝, 本次请务必写到 "
                     "1400 字以上, 术语定义、推理过程、实施建议三部分都充分展开。")
        elif "too_long" in reason:
            extra = "注意: 上次回答过长被系统拒绝, 本次请控制在 1600 字以内。"
        elif reason == "tail80_template":
            extra = "注意: 请使用与常见模板不同的个性化收尾句式。"
        else:
            extra = ""
        ans = gen_single(client, item, extra)
        with cnt_lock:
            ok, reason2 = validate(ans, tail80_cnt)
        if ok:
            accept(item, ans, 2, bi)
        else:
            with cnt_lock:
                counters[f"dropped_{reason2.split('(')[0]}"] += 1
            with write_lock:
                fdrop.write(json.dumps(
                    {"group_id": item["group_id"], "subset": item["subset"],
                     "instruction": item["instruction"], "reason": reason2,
                     "last_len": len(ans) if ans else 0,
                     "dropped_at": datetime.now().isoformat(timespec="seconds")},
                    ensure_ascii=False) + "\n")
                fdrop.flush()

    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    log(f"共 {len(batches)} 个批次, {WORKERS} 线程流水线 (RPM={RPM} 全局限速)")
    t0 = time.time()
    retry_queue = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process_batch, bi + 1, b): b
                for bi, b in enumerate(batches)}
        for fut in as_completed(futs):
            retry_queue.extend(fut.result())
    log(f"批量阶段结束: 完成 {state['completed']}, 待单条重生成 {len(retry_queue)}, "
        f"耗时 {time.time() - t0:.0f}s")

    if retry_queue and (args.limit is None or state["completed"] < args.limit):
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = [ex.submit(process_single, it, rsn, bi)
                    for it, rsn, bi in retry_queue]
            for fut in as_completed(futs):
                fut.result()

    fout.close()
    fdrop.close()
    log(f"本轮新完成 {state['completed']} 条; 计数 {dict(counters)}")
    write_report(out_path, report_path, dict(counters), drop_path)

    done_ids2, _ = load_done_ids(out_path)
    total_remaining = sum(1 for t in todo_all if t["group_id"] not in done_ids2)
    if total_remaining == 0:
        export_review(out_path, review_path)
    else:
        log(f"尚有 {total_remaining} 条未完成, 暂不导出终版抽检队列 "
            f"(可用 --review-only 随时导出当前进度的抽检)")


if __name__ == "__main__":
    main()
