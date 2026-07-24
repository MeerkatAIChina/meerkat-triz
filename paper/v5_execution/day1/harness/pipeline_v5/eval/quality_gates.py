#!/usr/bin/env python
"""
pipeline_v5 生成质量门 (§6.1 四道门 + 微调模型追加规则 + 英文草稿检测)。

评分前执行; 任一不过标记 invalid 并触发一次 bad_words_ids 兜底重生成,
仍不过计入失败率并**冻结评测**。质量门汇总须出现在每份评测报告首页。

门 (§6.1 原文):
  ① think 残留检测
  ② 非空
  ③ 中文占比 ≥ 0.3
  ④ 长度 ≥ 100 字符
  ④b 英文草稿检测 (E0 事故形态: 未闭合英文 think 草稿被剥离标记后的
      残文 —— 以拉丁字母为主且含典型草稿语标记)
  微调模型追加: ≥50 字符 且 不低于同题 base 长度 3%

E0 参照: 修复后 100/100 mode=direct、0 兜底、0 invalid、
长度 1246–4080 均值 3250 字符 (E0_report.md §3)。
"""

import re

from render import has_think_residue

ZH_MIN_RATIO = 0.3
MIN_LEN_BASE = 100
MIN_LEN_FINETUNED = 50
FINETUNED_MIN_BASE_RATIO = 0.03

_ZH_RE = re.compile(r"[一-鿿]")
# 英文草稿语标记 (E0 观察到的 thinking 草稿高频开场; 仅在前 300 字符内匹配)
_DRAFT_MARKERS_RE = re.compile(
    r"(?:^|\s)(Let me|Okay|Ok,|The user|We need|I need|First,? I|"
    r"Let's|So, the|To answer|I should|Step by step)",
    re.IGNORECASE)


def zh_ratio(text):
    """中文字符 / 非空白字符 (无字符返回 0.0)。"""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    zh = sum(1 for c in chars if _ZH_RE.match(c))
    return zh / len(chars)


def latin_ratio(text):
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    lat = sum(1 for c in chars if c.isascii() and c.isalpha())
    return lat / len(chars)


def gate_think_residual(resp):
    ok = not has_think_residue(resp)
    return ok, "think 残留" if not ok else ""


def gate_non_empty(resp):
    ok = bool(resp and resp.strip())
    return ok, "空回答" if not ok else ""


def gate_zh_ratio(resp):
    r = zh_ratio(resp)
    ok = r >= ZH_MIN_RATIO
    return ok, "" if ok else f"中文占比 {r:.3f} < {ZH_MIN_RATIO}"


def gate_min_length(resp, finetuned=False, base_len=None):
    n = len(resp.strip())
    floor = MIN_LEN_FINETUNED if finetuned else MIN_LEN_BASE
    if n < floor:
        return False, f"长度 {n} < {floor}"
    if finetuned and base_len is not None:
        if n < FINETUNED_MIN_BASE_RATIO * base_len:
            return False, (f"长度 {n} < 同题 base 长度 {base_len} 的 "
                           f"{FINETUNED_MIN_BASE_RATIO:.0%}")
    return True, ""


def gate_english_draft(resp):
    """英文草稿检测: 拉丁字母为主 (>70%) 且开头含草稿语标记 → 判为
    未闭合英文 think 草稿残文 (E0 污染形态)。"""
    if not resp.strip():
        return True, ""  # 空由门②处理
    head = resp[:300]
    if latin_ratio(resp) > 0.7 and _DRAFT_MARKERS_RE.search(head):
        return False, "英文 think 草稿残文 (拉丁为主 + 草稿语标记)"
    return True, ""


GATE_ORDER = ["think_residual", "non_empty", "zh_ratio",
              "min_length", "english_draft"]


def run_gates(resp, finetuned=False, base_len=None):
    """对单条回答跑全部质量门。返回 {"pass": bool, "failures": [...], "detail": {}}。"""
    failures = []
    detail = {"len": len(resp.strip()), "zh_ratio": round(zh_ratio(resp), 4)}
    checks = {
        "think_residual": gate_think_residual(resp),
        "non_empty": gate_non_empty(resp),
        "zh_ratio": gate_zh_ratio(resp),
        "min_length": gate_min_length(resp, finetuned, base_len),
        "english_draft": gate_english_draft(resp),
    }
    for name in GATE_ORDER:
        ok, why = checks[name]
        if not ok:
            failures.append({"gate": name, "reason": why})
    return {"pass": not failures, "failures": failures, "detail": detail}


def run_gates_batch(responses, finetuned=False, base_responses=None):
    """批量质量门。

    responses: {id: response}; base_responses: {id: base_response} (微调模型
    追加长度规则用)。返回 (per_item, summary):
      per_item[id] = run_gates(...) 结果 + "invalid": bool
      summary = 报告首页用汇总 (过门条数/生成条数/各门失败计数/invalid 清单)
    """
    per_item = {}
    gate_fail_counts = {g: 0 for g in GATE_ORDER}
    invalid_ids = []
    for qid, resp in responses.items():
        base_len = None
        if finetuned and base_responses and qid in base_responses:
            base_len = len(base_responses[qid].strip())
        r = run_gates(resp, finetuned=finetuned, base_len=base_len)
        r["invalid"] = not r["pass"]
        per_item[qid] = r
        for f in r["failures"]:
            gate_fail_counts[f["gate"]] += 1
        if r["invalid"]:
            invalid_ids.append(qid)
    n = len(responses)
    summary = {
        "n_generated": n,
        "n_pass": n - len(invalid_ids),
        "n_invalid": len(invalid_ids),
        "invalid_ids": invalid_ids,
        "gate_fail_counts": gate_fail_counts,
        "finetuned_extra_rules": bool(finetuned),
        "policy": ("任一不过标记 invalid 并触发一次 bad_words_ids 兜底重生成; "
                   "仍不过计入失败率并冻结评测 (§6.1)"),
    }
    return per_item, summary


def gate_summary_markdown(summary, title="生成质量门汇总"):
    """报告首页块 (§6.1: 质量门汇总须出现在每份评测报告首页)。"""
    L = [f"## {title}", ""]
    L.append(f"- 生成条数: **{summary['n_generated']}** | "
             f"过门: **{summary['n_pass']}** | "
             f"invalid: **{summary['n_invalid']}**")
    if summary["finetuned_extra_rules"]:
        L.append("- 微调模型追加规则已启用: ≥50 字符 且 ≥ 同题 base 长度 3%")
    L.append("")
    L.append("| 门 | 失败条数 |")
    L.append("|---|---|")
    names = {"think_residual": "① think 残留", "non_empty": "② 非空",
             "zh_ratio": "③ 中文占比 ≥0.3", "min_length": "④ 长度下限",
             "english_draft": "④b 英文草稿"}
    for g in GATE_ORDER:
        L.append(f"| {names[g]} | {summary['gate_fail_counts'][g]} |")
    if summary["invalid_ids"]:
        L.append("")
        L.append(f"- invalid 题: {', '.join(summary['invalid_ids'][:50])}")
    L.append(f"- 处置策略: {summary['policy']}")
    L.append("")
    return "\n".join(L)
