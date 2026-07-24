#!/usr/bin/env python
"""
pipeline_v5 eval 单元测试 (无 pytest 依赖, 直接 python 运行)。

覆盖:
  - E0 回归: 空 think 块保留 (3 题冒烟断言中文正式作答形态)
  - 生成后 think 残留检测 (微调模型同规则)
  - 质量门四道 + 英文草稿 + 微调追加长度规则
  - 关键词别名表 + 双分数 + 漏判审计
  - judge 双臂 (不截断/反冗长/分桶) + T=0 断言 + 版本钉死
  - pairwise 双序合并 + >10% 单序作废断言
  - overrefusal >2% 不过门
  - 决策门 2.0 回溯验证: v4 代入必须复现 keep_v2 (§6.7 门设计验收测试)
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import stats_utils  # noqa: F401  (import 即指纹自检)
from stats_utils import wilson_ci, mcnemar_exact_p
from render import (EMPTY_THINK, render_prompt, assert_empty_think_retained,
                    strip_closed_think, has_think_residue)
import quality_gates as qg
import keyword_scorer as ks
import judge_arms as ja
import pairwise as pw
import overrefusal as orf
import decision_gate as dg

PASS_COUNT = [0]


def ok(name):
    PASS_COUNT[0] += 1
    print(f"  PASS {name}")


# ---------- 假 tokenizer: 模拟 Qwen3.6 ChatML + enable_thinking=False ----------
class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=True):
        s = ""
        for m in messages:
            s += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            s += "<|im_start|>assistant\n"
            if not enable_thinking:
                s += EMPTY_THINK
        return s


GOOD_ZH = ("首先进行问题分析, 识别技术矛盾; 接着构建矛盾并定义理想解IFR; "
           "然后做资源分析, 盘点可用物质与能量资源; 最后完成方案评估, "
           "给出基于发明原理的推荐路径, 并说明预期效果与验证方式。")


def test_e0_regression_think_retained():
    tok = FakeTokenizer()
    # 3 题冒烟 (§6.1: 写入单元测试, 3 题冒烟断言中文正式作答)
    for q in ["什么是技术矛盾?", "解释理想解IFR。", "ARIZ 第五步做什么?"]:
        prompt = render_prompt(tok, "系统", q)
        assert EMPTY_THINK in prompt, "空 think 块被剥离 —— E0 污染路径回归!"
        assert_empty_think_retained(prompt)
    # v4 旧行为 (剥 think) 必须被断言拦截
    stripped = render_prompt(tok, "系统", "题").replace(EMPTY_THINK, "")
    try:
        assert_empty_think_retained(stripped)
        raise SystemExit("v4 剥离路径未被拦截, 测试失败")
    except AssertionError:
        pass
    ok("E0 回归: 空 think 保留 + 剥离路径被断言拦截")


def test_strip_and_residue():
    dirty = "<think>\nLet me think about TRIZ...\n</think>\n\n" + GOOD_ZH
    cleaned = strip_closed_think(dirty)
    assert cleaned == GOOD_ZH.strip() or GOOD_ZH in cleaned
    assert not has_think_residue(cleaned)
    # 未闭合草稿 (E0 事故形态) 必须被检测
    draft = "<think>\nLet me analyze this TRIZ problem. The user asks..."
    assert has_think_residue(draft)
    ok("生成后闭合 think 剥离 + 未闭合草稿残留检测 (微调模型同规则)")


def test_quality_gates():
    r = qg.run_gates(GOOD_ZH * 3)  # ~300+ 字符中文
    assert r["pass"], r["failures"]
    assert not qg.run_gates("<think>abc" + GOOD_ZH)["pass"]          # ① think 残留
    assert not qg.run_gates("   ")["pass"]                            # ② 空
    assert not qg.run_gates("This is a pure English answer about TRIZ "
                            "methodology and inventive principles. " * 3)["pass"]  # ③ 非中文
    assert not qg.run_gates("太短了")["pass"]                          # ④ 长度下限
    # ④b 英文草稿检测: E0 形态的草稿残文
    draft = ("Okay, let me analyze this problem. The user is asking about "
             "TRIZ contradiction matrix. " * 4)
    r2 = qg.run_gates(draft)
    assert not r2["pass"] and any(f["gate"] in ("english_draft", "zh_ratio")
                                  for f in r2["failures"])
    # 微调追加规则: ≥50 字符且 ≥同题 base 3% (3000×3%=90)
    short_ft = GOOD_ZH  # ~110 字符
    assert qg.run_gates(short_ft, finetuned=True, base_len=3000)["pass"]
    tiny_ft = "简短回答。"
    assert not qg.run_gates(tiny_ft, finetuned=True, base_len=3000)["pass"]
    # 中文合法但拉丁为主的混合文不被误伤 (含 TRIZ 术语的正式中文作答)
    mixed = GOOD_ZH + " (参考 40 Principles 与 Contradiction Matrix)"
    assert qg.run_gates(mixed * 2)["pass"]
    # 汇总结构
    per, summ = qg.run_gates_batch({"q1": GOOD_ZH * 3, "q2": "<think>x"})
    assert summ["n_generated"] == 2 and summ["n_pass"] == 1
    assert summ["gate_fail_counts"]["think_residual"] == 1
    md = qg.gate_summary_markdown(summ)
    assert "质量门" in md and "invalid" in md
    ok("质量门: 四道 + 英文草稿 + 微调追加长度规则 + 汇总首页块")


def test_keyword_alias():
    amap = ks.load_alias_map(HERE.parent / "keyword_map_v5.json")
    assert "资源分析" in amap and "resource analysis" in \
        [a.lower() for a in amap["资源分析"]]
    resp_en = "Step 4: Resource Analysis of available substances and fields."
    pre = ks.keyword_score(resp_en, ["资源分析"], None)
    post = ks.keyword_score(resp_en, ["资源分析"], amap)
    assert pre["kw_hit_rate"] == 0.0 and post["kw_hit_rate"] == 1.0
    assert post["alias_hits"].get("资源分析") == "resource analysis"
    # 双分数
    records = [{"id": "t1", "response": resp_en, "keywords": ["资源分析"]},
               {"id": "t2", "response": GOOD_ZH, "keywords": ["问题分析"]}]
    pre_m, post_m, per = ks.dual_score(records, amap)
    assert post_m > pre_m
    # 漏判审计: rubric 过 kw 不过 → 入队
    audit = ks.miss_audit(
        [{"id": "t1", "subset": "ariz_guidance", "response": "完全无关的回答",
          "keywords": ["资源分析"]}],
        {"t1": {"post": ks.keyword_score("完全无关的回答", ["资源分析"], amap)}},
        lambda r: 0.75)
    assert audit["n_queue"] == 1 and audit["miss_rate"] == 1.0
    assert not audit["freeze_recommended"]
    ok("关键词轨: 别名表命中 + 更新前/后双分数 + 漏判审计队列")


def test_judge_arms():
    # 臂 A: 反冗长条款 + 不截断
    assert "回答长度本身不构成加分项" in ja.JUDGE_SYSTEM_ARM_A
    assert "冗余重复内容" in ja.JUDGE_SYSTEM_ARM_A
    long_resp = "长回答。" * 3000  # >1500 字符, v4 会截断
    user = ja.build_judge_user_arm_a(
        [{"id": "g1", "subset": "concept_explanation", "question": "Q",
          "reference_answer": "R", "keywords": ["k"]}], {"g1": long_resp})
    assert long_resp in user, "臂 A 输入被截断!"
    # 臂 B: 分桶
    assert ja.length_bucket(100) == "<500"
    assert ja.length_bucket(499) == "<500"
    assert ja.length_bucket(500) == "500-1500"
    assert ja.length_bucket(1499) == "500-1500"
    assert ja.length_bucket(1500) == "1500-3000"
    assert ja.length_bucket(2999) == "1500-3000"
    assert ja.length_bucket(3000) == ">3000"
    pairs = ja.arm_b_pairs({"a": "x" * 100, "b": "x" * 100, "c": "x" * 4000},
                           {"a": "y" * 200, "b": "y" * 3000, "c": "y" * 5000})
    assert ("a", "<500") in pairs["pairs"] and len(pairs["pairs"]) == 2
    assert "b" in pairs["no_pair"]  # 不同桶 → 无同桶配对
    # T=0 断言
    ja.assert_temperature_zero(0.0)
    try:
        ja.assert_temperature_zero(0.7)
        raise SystemExit("T≠0 未被拦截")
    except AssertionError:
        pass
    # 版本钉死
    ok32k, dev = ja.assert_pinned_judge("moonshot-v1-32k")
    assert ok32k and dev == ""
    okk2, dev2 = ja.assert_pinned_judge("kimi-k2-0711-preview")
    assert not okk2 and "偏离" in dev2
    # 谱系
    lin = ja.judge_lineage("moonshot-v1-32k")
    assert lin["same_family_as_data"] is True and "同族" in lin["declaration"]
    assert ja.judge_lineage("gpt-4o")["same_family_as_data"] is False
    # 冲突检测
    conf, _ = ja.arms_conflict({"diff": 0.3, "ci95": [0.1, 0.5]},
                               {"diff": -0.3, "ci95": [-0.5, -0.1]})
    assert conf
    ok("judge 双臂: 反冗长 + 不截断 + 分桶 + T=0 断言 + 钉死 + 谱系")


def test_pairwise():
    ab = {"q1": "A", "q2": "A", "q3": "B", "q4": "tie"}
    ba = {"q1": "B", "q2": "B", "q3": "A", "q4": "tie"}  # 完全一致 (候选视角)
    rep = pw.merge_double_order(ab, ba)
    assert rep["position_inconsistency_rate"] == 0.0
    assert rep["single_order_void"] is False
    # 候选 q1,q2 胜, q3 负, q4 平 → 5/8
    assert abs(rep["candidate_winrate_merged"] - 0.625) < 1e-9
    assert rep["b_position_winrate"] is not None
    assert len(rep["candidate_winrate_wilson_ci95"]) == 2
    # 不一致 >10% → 单序作废断言
    # AB 序候选全胜; BA 序前 2 题候选(B 位)胜、后 8 题候选负 → 8/10 不一致
    ab2 = {f"q{i}": "A" for i in range(10)}
    ba2 = {f"q{i}": ("B" if i < 2 else "A") for i in range(10)}
    try:
        pw.merge_double_order(ab2, ba2, strict=True)
        raise SystemExit(">10% 不一致未触发作废断言")
    except AssertionError:
        pass
    rep2 = pw.merge_double_order(ab2, ba2, strict=False)
    assert rep2["single_order_void"] is True
    assert rep2["position_inconsistency_rate"] == 0.8
    ok("pairwise: AB/BA 合并胜率 + Wilson + 位置不一致率 + >10% 作废断言")


def test_overrefusal():
    pats = orf.load_templates(HERE.parent / "refusal_templates_v5.json")
    assert pats, "拒答模板为空"
    resps = {f"q{i:03d}": GOOD_ZH for i in range(100)}
    resps["q007"] = "很抱歉, 我无法提供该信息, 建议您咨询持证专业人士。"
    resps["q042"] = "这个问题我不能提供答案。"
    r = orf.check_overrefusal(resps, pats)
    assert r["n_hits"] == 2 and r["hit_rate"] == 0.02 and r["pass"]  # 恰好 2% 过门
    resps["q099"] = "作为AI助手，我不能回答。"
    r2 = orf.check_overrefusal(resps, pats)
    assert r2["hit_rate"] == 0.03 and not r2["pass"]  # >2% 不过门
    ok("overrefusal: 2% 阈值边界 (2% 过 / 3% 不过)")


def test_decision_gate_backtest():
    with open(HERE.parent / "backtest" / "gate_scores_v4_backtest.json",
              encoding="utf-8") as f:
        scores = json.load(f)
    dec = dg.run_decision_gate(scores)
    by_gate = {g["gate"]: g["status"] for g in dec["gates"]}
    # §6.7 回溯验证预期: G1/G2/G3 不过 → 拒绝 v4 替代 v2
    assert by_gate["G1"] == "FAIL", by_gate
    assert by_gate["G2"] == "FAIL", by_gate
    assert by_gate["G3"] == "FAIL", by_gate
    assert dec["verdict"] == "keep_v2", dec["verdict"]
    # 假想全过案例 → ship
    good = json.loads(json.dumps(scores))
    good["v5_vs_base"]["judge_armA"]["overall"] = {
        "diff": 0.05, "ci95": [-0.10, 0.20], "n": 200}
    good["v5_vs_v2"]["judge_armA"]["overall"] = {
        "diff": 0.10, "ci95": [0.02, 0.18], "n": 200}
    good["v5_vs_v2"]["keyword"]["per_subset"]["concept_explanation"] = {
        "diff": 0.01, "ci95": [-0.03, 0.05], "n": 30}
    good["probe"] = {"v5_minus_v2_overall_pp": -1.0}
    good["overrefusal"] = {"hit_rate": 0.0, "pass": True}
    dec2 = dg.run_decision_gate(good)
    assert dec2["verdict"] == "ship_v5", \
        [g for g in dec2["gates"] if g["status"] != "PASS"]
    # G6 探针超界 → FAIL
    bad6 = json.loads(json.dumps(good))
    bad6["probe"] = {"v5_minus_v2_overall_pp": -6.0}
    assert dg.run_decision_gate(bad6)["verdict"] == "keep_v2"
    # G7 双轨反向 → FAIL
    bad7 = json.loads(json.dumps(good))
    bad7["v5_vs_v2"]["keyword"]["overall"] = {
        "diff": -0.10, "ci95": [-0.15, -0.05], "n": 200}
    assert dg.run_decision_gate(bad7)["verdict"] == "keep_v2"
    ok("决策门 2.0: 回溯复现 keep_v2 + ship/G6/G7 边界案例")


def test_stats():
    p, lo, hi = wilson_ci(67, 100)
    assert abs(p - 0.67) < 1e-9 and lo < p < hi
    assert mcnemar_exact_p(55, 4) == 1.6979681549678105e-12
    ok("统计: Wilson + McNemar 指纹 (bootstrap 指纹 import 时已自检)")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"运行 {len(tests)} 组测试:")
    for t in tests:
        t()
    print(f"\n全部通过: {PASS_COUNT[0]} 组 ✓")


if __name__ == "__main__":
    main()
