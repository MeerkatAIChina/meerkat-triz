#!/usr/bin/env python
"""
pipeline_v5b PR 定向注入生成器 (2026-07-29 归因后立项)。

归因结论 (真缺失, 量+质双重):
  - v5a 训练集 principle_recommendation 仅 196 条 (六大子集最少),
    72% 为"简洁模式<=300字"短答, 详细模式长答仅 54 条;
  - 失败模式: 模型把"推荐原理"答成方法论讲座, hedging 不 commit,
    原理选择偏离金标; 外部评委 gemini -0.308 显著为负。

本脚本从 triz_corpus 未被金标占用的 chunk 中单阶段生成 (问题, 长答) 对:
  - 题型对齐金标: "针对 X 情境/矛盾, 推荐 2-3 个 TRIZ 发明原理并论证";
  - 答案范式: 直给原理编号+名称 -> TRIZ 定义 -> 针对该问题的应用逻辑;
  - 40 原理轮转候选强制均衡覆盖 (每 chunk 指派 3 个优先候选, 适用则用);
  - 质量门: 长度 800-1600 / >=2 个不同原理编号 / 题型校验 / tail80 黑名单 /
    vs 金标 3-gram Jaccard 去污染 (题>=0.4 答>=0.5 丢弃);
  - 批量 5 条/请求, RPM=3 全局限速, 6 线程流水线, jsonl 追加断点续跑,
    10% 抽检队列 (seed=42)。

产物 (data/processed/v5b_data/):
  pr_inject_sampling.jsonl   抽样+候选原理指派清单
  pr_inject_answers.jsonl    合格 (问题, 长答) 对
  pr_inject_dropped.jsonl    丢弃及原因
  pr_inject_report.json      运行报告
  pr_inject_review.md        10% 抽检队列

用法:
  venv_v5/bin/python pipeline_v5/src/pr_inject_gen_v5b.py [--limit N] [--review-only]
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

CORPUS = "data/processed/corpus/triz_corpus.jsonl"
GOLD = "data/processed/v5_data/v5_gold.jsonl"
OUT_DIR = "data/processed/v5b_data"

MODEL = "kimi-k2.5"
BASE_URL = "https://api.moonshot.cn/v1"
RPM = 3
TEMPERATURE = None            # kimi-k2.x 仅允许 temperature=1
MAX_TOKENS_BATCH = 16000
MAX_TOKENS_SINGLE = 4000
MAX_API_RETRIES = 5
BATCH = 5
WORKERS = 6
LEN_MIN, LEN_MAX = 800, 1600
TEMPLATE_FREQ_LIMIT = 3
N_SAMPLE = 420                # 目标 ~350 合格 (损耗冗余)
JACCARD_Q = 0.4               # 题面 vs 金标题面
JACCARD_A = 0.5               # 答案 vs 金标参考答案

SYSTEM_FOR_RECORD = "详细模式:给出完整的结构化分析,包含术语定义、推理过程与实施建议"

# 40 发明原理 (标准中文名)
PRINCIPLES = [
    (1, "分割"), (2, "抽取"), (3, "局部质量"), (4, "不对称"), (5, "合并"),
    (6, "多用性"), (7, "嵌套"), (8, "配重"), (9, "预先反作用"), (10, "预先作用"),
    (11, "预先防范"), (12, "等势"), (13, "反向作用"), (14, "曲面化"),
    (15, "动态化"), (16, "不足或过度作用"), (17, "维度转换"), (18, "机械振动"),
    (19, "周期性作用"), (20, "有效作用的连续性"), (21, "快速作用"),
    (22, "变害为利"), (23, "反馈"), (24, "中介物"), (25, "自服务"),
    (26, "复制"), (27, "廉价替代品"), (28, "机械系统替代"),
    (29, "气动与液压结构"), (30, "柔性壳体或薄膜"), (31, "多孔材料"),
    (32, "颜色改变"), (33, "同质性"), (34, "抛弃与再生"), (35, "参数变化"),
    (36, "相变"), (37, "热膨胀"), (38, "强氧化"), (39, "惰性环境"),
    (40, "复合材料"),
]

FEWSHOT = """【输出示例】
【问题1】
某锂电池制造企业在极片烘干工序中面临技术矛盾：提高烘干温度可以加快溶剂挥发速度、提升产能，但温度过高会导致极片涂层开裂、粘结剂分解，严重影响电池循环寿命。请推荐两个 TRIZ 发明原理并解释为什么适用于解决这个矛盾。
【回答1】
该矛盾的本质是"温度"这一参数被提出了双向要求：工艺端需要高温以保证挥发速率，材料端需要低温以维持涂层完整性，属于典型的技术矛盾（速度/生产率 vs 可靠性/质量）。

推荐原理一：参数变化原理（原理 35）。该原理主张改变物体的物理或化学状态参数，如温度、压力、浓度、粘度等，以避开矛盾区间。在极片烘干中，与其在"固定高温"与"固定低温"之间折中，不如改变烘干曲线的参数组合：采用阶梯升温——初期低温大风量快速带走表面溶剂（避免表层结膜），中期升温至溶剂沸点附近加速内部挥发，后期降温缓释内应力。参数从"恒定值"变为"时变函数"，同一台设备在不同阶段满足相互冲突的要求。

推荐原理二：机械振动原理（原理 18）。该原理建议使物体发生振动，或提高振动频率以强化过程。引入红外辐射配合热风脉动或超声辅助，可以在不提高整体温度的前提下增强溶剂分子的逸出动能：振动能量直接作用于溶剂分子与涂层界面的传质过程，相当于把"加热"这一粗暴的能量输入替换为"定向传质强化"，烘干速率提升而涂层本体温度保持低位。

实施要点：先在实验线上标定阶梯温度曲线与振动参数的耦合窗口，以极片剥离强度和残余溶剂含量为双指标验收；注意振动幅度须低于涂层流平临界值，防止引入新的厚度不均缺陷。两条原理可独立使用，叠加时先验证参数变化原理建立基础工艺，再叠加振动强化。
【/回答1】
【示例结束】"""

GEN_SYSTEM = (
    "你是 TRIZ/创新方法领域资深专家。用户会给出若干段工程/创新领域语料, "
    "请为每段语料完成两步创作:\n"
    "第一步【出题】: 基于语料中的技术情境, 构造一道'发明原理推荐题', "
    "要求: 明确指出一个具体的技术矛盾或工程困境 (改善什么会恶化什么), "
    "然后要求推荐 2-3 个 TRIZ 发明原理并说明理由。题目必须自洽、脱离语料也能读懂。\n"
    "第二步【作答】: 为该题撰写模范答案, 结构:\n"
    "  (a) 矛盾界定 (~200字): 一句话点明矛盾双方参数;\n"
    "  (b) 原理推荐与论证 (~700字): 每个原理按'原理编号+名称 -> TRIZ 标准定义 -> "
    "针对本题情境的具体应用逻辑'展开, 直给结论, 禁止'可能涉及/可以考虑'一类闪烁其词;\n"
    "  (c) 实施要点 (~250字): 落地验证指标与风险边界。\n"
    "【硬要求】答案 1000-1500 汉字 (低于 800 或高于 1600 被系统直接拒绝); "
    "必须显式写出至少 2 个原理的编号与名称 (如'原理 35 参数变化'); "
    "各条答案的句式与收尾必须多样化, 禁止模板套话。\n"
    "【输出格式】不要使用 JSON。对第 i 段语料严格输出:\n"
    "【问题i】\n(题目)\n【回答i】\n(答案)\n【/回答i】\n\n"
    + FEWSHOT
)

ANS_RE = re.compile(r"【回答\s*(\d+)\s*】\s*(.+?)\s*(?:【/回答\s*\d*\s*】|$)", re.S)
Q_RE = re.compile(r"【问题\s*(\d+)\s*】\s*(.+?)\s*(?=【回答\s*\d*\s*】)", re.S)
PR_NUM_RE = re.compile(r"原理\s*[#第]?\s*(\d{1,2})")
HEDGE_RE = re.compile(r"可能涉及|可以考虑|或许可以|一般来说可以")


def log(msg):
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


# ---- 3-gram Jaccard 去污染 (字符级, 同 assemble_v5 口径) ----

def ngrams(text, n=3):
    t = re.sub(r"\s+", "", text)
    return {t[i:i + n] for i in range(len(t) - n + 1)} if len(t) >= n else {t}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_gold_index(path):
    q_idx, a_idx = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                g = json.loads(line)
                q_idx.append(ngrams(g["question"]))
                a_idx.append(ngrams(g["reference_answer"]))
    return q_idx, a_idx


def contaminated(question, answer, q_idx, a_idx):
    qg = ngrams(question)
    if any(jaccard(qg, g) >= JACCARD_Q for g in q_idx):
        return True, "decon_question"
    ag = ngrams(answer)
    if any(jaccard(ag, g) >= JACCARD_A for g in a_idx):
        return True, "decon_answer"
    return False, ""


# ---- 校验 ----

def validate(question, answer, tail80_cnt):
    if not question or not answer:
        return False, "empty"
    if "原理" not in question or not re.search(r"推荐|选择|适用|应用", question):
        return False, "question_not_pr"
    n = len(answer)
    if n < LEN_MIN:
        return False, f"too_short({n})"
    if n > LEN_MAX:
        return False, f"too_long({n})"
    nums = set(PR_NUM_RE.findall(answer))
    nums = {x for x in nums if 1 <= int(x) <= 40}
    if len(nums) < 2:
        return False, f"principle_count({len(nums)})"
    if len(HEDGE_RE.findall(answer)) > 2:
        return False, "hedging"
    if tail80_cnt.get(answer[-80:], 0) >= TEMPLATE_FREQ_LIMIT - 1:
        return False, "tail80_template"
    return True, "ok"


# ---- 抽样 ----

def build_sampling(corpus_path, gold_path, out_path, n=N_SAMPLE, seed=42):
    gold_chunks = set()
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                gold_chunks.add(json.loads(line)["source_chunk_id"])
    chunks = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                if c["id"] not in gold_chunks and len(c.get("text", "")) >= 300:
                    chunks.append(c)
    rng = random.Random(seed)
    rng.shuffle(chunks)
    picked = chunks[:n]
    # 40 原理轮转: 每条指派 3 个优先候选 ( stride 错位保证均匀 )
    recs = []
    for i, c in enumerate(picked):
        base = (i * 3) % 40
        cand = [PRINCIPLES[(base + k) % 40] for k in range(3)]
        recs.append({
            "group_id": f"prinj_{i:04d}",
            "chunk_id": c["id"],
            "chunk_text": c["text"][:2000],
            "candidates": [{"num": p, "name": nm} for p, nm in cand],
        })
    with open(out_path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"抽样完成: {len(recs)} 条 (语料 {len(chunks)} 可用, 金标占用 {len(gold_chunks)} 已排除)")
    return recs


def batch_user_prompt(batch):
    lines = []
    for i, it in enumerate(batch, 1):
        cand = "、".join(f"原理 {c['num']} {c['name']}" for c in it["candidates"])
        lines.append(
            f"【语料{i}】(优先候选: {cand}; 若与情境确实不契合可另选更贴切原理)\n"
            f"{it['chunk_text']}")
    return ("请为以下语料逐段完成出题+作答, 严格使用【问题i】【回答i】...【/回答i】格式, "
            "每条答案 1000-1500 字:\n\n" + "\n\n".join(lines))


def single_user_prompt(item, extra=""):
    cand = "、".join(f"原理 {c['num']} {c['name']}" for c in item["candidates"])
    msg = (f"请为以下语料完成出题+作答, 以【问题1】【回答1】开头、【/回答1】结尾:\n\n"
           f"【语料】(优先候选: {cand}; 若与情境确实不契合可另选)\n{item['chunk_text']}")
    if extra:
        msg += f"\n\n{extra}"
    return msg


def parse_qa(text):
    qs = {int(m.group(1)): m.group(2).strip() for m in Q_RE.finditer(text)}
    ans = {int(m.group(1)): m.group(2).strip() for m in ANS_RE.finditer(text)}
    return qs, ans


def load_done_ids(path):
    done, tail80_cnt = set(), Counter()
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
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
    lens = sorted(len(r["completion"]) for r in records)
    tail80 = Counter(r["completion"][-80:] for r in records)
    pr_hits = Counter()
    for r in records:
        for x in set(PR_NUM_RE.findall(r["completion"])):
            if 1 <= int(x) <= 40:
                pr_hits[int(x)] += 1
    rep = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "completed": len(records),
        "gen_params": {"model": MODEL, "rpm": RPM, "batch": BATCH,
                       "workers": WORKERS, "system": SYSTEM_FOR_RECORD},
        "quality_gates": {
            "len_range": [LEN_MIN, LEN_MAX],
            "len_min": min(lens, default=0),
            "len_max": max(lens, default=0),
            "len_mean": round(sum(lens) / len(lens), 1) if lens else 0,
            "len_p50": lens[len(lens) // 2] if lens else 0,
            "principle_coverage": {str(k): pr_hits[k] for k in sorted(pr_hits)},
            "principles_missing": [p for p, _ in PRINCIPLES if pr_hits[p] == 0],
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
    lines = ["# v5b PR 注入 10% 人工抽检队列", "",
             f"导出时间: {datetime.now().isoformat(timespec='seconds')} | "
             f"总完成 {len(records)} 条, 抽检 {len(sample)} 条 (seed={seed})", ""]
    for r in sample:
        lines += [
            f"## {r['group_id']} [{r['chunk_id']}] ({len(r['completion'])} 字符)",
            "",
            f"**Question**: {r['instruction']}", "",
            "**Completion**:", "", r["completion"], "", "---", ""]
    with open(review_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"抽检队列: {review_path} ({len(sample)} 条)")


def main():
    ap = argparse.ArgumentParser(description="v5b PR 定向注入生成")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--review-only", action="store_true")
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    sampling_path = out_dir / "pr_inject_sampling.jsonl"
    out_path = out_dir / "pr_inject_answers.jsonl"
    drop_path = out_dir / "pr_inject_dropped.jsonl"
    review_path = out_dir / "pr_inject_review.md"
    report_path = out_dir / "pr_inject_report.json"

    if args.review_only:
        export_review(out_path, review_path)
        return

    if not sampling_path.is_file():
        build_sampling(PROJECT_ROOT / CORPUS, PROJECT_ROOT / GOLD, sampling_path)
    todo_all = [json.loads(l) for l in open(sampling_path, encoding="utf-8") if l.strip()]
    log(f"抽样清单: {len(todo_all)} 条")

    q_idx, a_idx = load_gold_index(PROJECT_ROOT / GOLD)
    log(f"金标去污染索引: {len(q_idx)} 题")

    done_ids, tail80_cnt = load_done_ids(out_path)
    if done_ids:
        log(f"断点续跑: 已完成 {len(done_ids)} 条")
    todo = [t for t in todo_all if t["group_id"] not in done_ids]
    if args.limit is not None:
        todo = todo[: args.limit * 2]
    log(f"待生成: {len(todo)} 条")

    counters = Counter()
    state = {"completed": 0}
    write_lock = threading.Lock()
    cnt_lock = threading.Lock()
    client = get_client()
    fout = open(out_path, "a", encoding="utf-8")
    fdrop = open(drop_path, "a", encoding="utf-8")

    def accept(item, question, ans, attempt, batch_no):
        rec = {
            "group_id": item["group_id"], "chunk_id": item["chunk_id"],
            "subset": "principle_recommendation",
            "instruction": question,
            "system": SYSTEM_FOR_RECORD, "completion": ans,
            "gen_params": {"model": MODEL, "attempt": attempt,
                           "batch_no": batch_no, "protocol": "pr_inject_v1",
                           "candidates": item["candidates"]},
        }
        with write_lock:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
        with cnt_lock:
            tail80_cnt[ans[-80:]] += 1
            state["completed"] += 1
            n = state["completed"]
        if n % 20 == 0:
            log(f"本轮完成 {n} | 计数 {dict(counters)}")
            write_report(out_path, report_path, dict(counters), drop_path)

    def drop(item, reason, ans=""):
        with cnt_lock:
            counters[f"dropped_{reason.split('(')[0]}"] += 1
        with write_lock:
            fdrop.write(json.dumps(
                {"group_id": item["group_id"], "chunk_id": item["chunk_id"],
                 "reason": reason, "last_len": len(ans),
                 "dropped_at": datetime.now().isoformat(timespec="seconds")},
                ensure_ascii=False) + "\n")
            fdrop.flush()

    def process_batch(bi, batch):
        if args.limit is not None and state["completed"] >= args.limit:
            return []
        content, finish = call_api(
            client,
            [{"role": "system", "content": GEN_SYSTEM},
             {"role": "user", "content": batch_user_prompt(batch)}],
            MAX_TOKENS_BATCH)
        retry_items = []
        if content is None:
            return [(it, "api_fail", bi) for it in batch]
        qs, ans_map = parse_qa(content)
        for i, item in enumerate(batch, 1):
            if args.limit is not None and state["completed"] >= args.limit:
                break
            q, ans = qs.get(i, ""), ans_map.get(i, "")
            with cnt_lock:
                ok, reason = validate(q, ans, tail80_cnt)
            if ok:
                bad, dreason = contaminated(q, ans, q_idx, a_idx)
                if bad:
                    ok, reason = False, dreason
            if ok:
                accept(item, q, ans, 1, bi)
            else:
                with cnt_lock:
                    counters[f"batch_{reason.split('(')[0]}"] += 1
                retry_items.append((item, reason, bi))
        return retry_items

    def process_single(item, reason, bi):
        if args.limit is not None and state["completed"] >= args.limit:
            return
        if reason.startswith("decon"):
            drop(item, reason)  # 污染不重试, 直接弃
            return
        m = re.search(r"\((\d+)\)", reason)
        prev = m.group(1) if m else "不足"
        if "too_short" in reason:
            extra = (f"注意: 上次答案仅 {prev} 字被系统拒绝, 本次务必写到 1100 字以上, "
                     "矛盾界定、原理论证、实施要点三部分都充分展开。")
        elif "too_long" in reason:
            extra = "注意: 上次答案过长被拒绝, 本次控制在 1400 字以内。"
        elif "principle_count" in reason:
            extra = "注意: 上次答案未显式写出至少 2 个原理的编号与名称, 本次必须写出 (如'原理 35 参数变化')。"
        elif "hedging" in reason:
            extra = "注意: 禁止'可能涉及/可以考虑'式闪烁其词, 直接给出确定推荐。"
        elif "tail80_template" in reason:
            extra = "注意: 请使用与常见模板不同的个性化收尾句式。"
        else:
            extra = ""
        content, finish = call_api(
            client,
            [{"role": "system", "content": GEN_SYSTEM},
             {"role": "user", "content": single_user_prompt(item, extra)}],
            MAX_TOKENS_SINGLE)
        q, ans = "", ""
        if content:
            qs, ans_map = parse_qa(content)
            q = qs.get(1, "")
            ans = ans_map.get(1) or (next(iter(ans_map.values())) if ans_map else "")
            if not ans:
                ans = re.sub(r"【/?(回答|问题)\s*\d*\s*】", "", content).strip()
        with cnt_lock:
            ok, reason2 = validate(q, ans, tail80_cnt)
        if ok:
            bad, dreason = contaminated(q, ans, q_idx, a_idx)
            if bad:
                ok, reason2 = False, dreason
        if ok:
            accept(item, q, ans, 2, bi)
        else:
            drop(item, reason2, ans)

    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    log(f"共 {len(batches)} 个批次, {WORKERS} 线程流水线 (RPM={RPM})")
    t0 = time.time()
    retry_queue = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process_batch, bi + 1, b): b for bi, b in enumerate(batches)}
        for fut in as_completed(futs):
            retry_queue.extend(fut.result())
    log(f"批量阶段结束: 完成 {state['completed']}, 待单条重生成 {len(retry_queue)}, "
        f"耗时 {time.time() - t0:.0f}s")

    if retry_queue and (args.limit is None or state["completed"] < args.limit):
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = [ex.submit(process_single, it, rsn, bi) for it, rsn, bi in retry_queue]
            for fut in as_completed(futs):
                fut.result()

    fout.close()
    fdrop.close()
    log(f"本轮新完成 {state['completed']} 条; 计数 {dict(counters)}")
    write_report(out_path, report_path, dict(counters), drop_path)

    done_ids2, _ = load_done_ids(out_path)
    remaining = sum(1 for t in todo_all if t["group_id"] not in done_ids2)
    if remaining == 0:
        export_review(out_path, review_path)
    else:
        log(f"尚有 {remaining} 条未完成, 可用 --review-only 导出当前抽检")


if __name__ == "__main__":
    main()
