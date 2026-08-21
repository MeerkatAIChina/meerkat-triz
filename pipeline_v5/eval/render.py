#!/usr/bin/env python
"""
pipeline_v5 干净锚点: prompt 渲染与 think 块处理 (§6.1, §11.3)。

铁律 (E0 已证实, E0_report.md §2):
  Qwen3.6-35B-A3B 是 thinking-native 基座。apply_chat_template(
  enable_thinking=False) 渲染出的空 think 块 `<think>\\n\\n</think>\\n\\n`
  必须保留 —— 它是"思考已结束"锚点; 剥离后 100/100 自吐未闭合英文
  think 草稿, 正式答案为空。E0 诊断矩阵中"保留空 think 块"是唯一成功
  路径 (3/3), assistant prefill 与 bad_words_ids 均失败。

  因此:
  - render_prompt 禁止任何后处理剥离 (v4 事故即 `.replace(EMPTY_THINK, "")`);
  - 空 think 保留有单元测试硬化 (tests/test_v5.py::test_render_keeps_empty_think);
  - 生成完成后才允许 strip_closed_think 剥离**闭合** think 块;
  - 微调模型生成后同样过 think 残留检测 (quality_gates.gate_think_residual)。
"""

import re

EMPTY_THINK = "<think>\n\n</think>\n\n"

_CLOSED_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def render_prompt(tokenizer, system_message, question):
    """渲染 prompt, 保留空 think 块 (与 v4 的唯一差异: 不剥离)。"""
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_message},
         {"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    # v5: 禁止 .replace(EMPTY_THINK, "") —— E0 回归测试守护
    return prompt


def assert_empty_think_retained(prompt):
    """冒烟断言: 渲染结果必须含空 think 块 (§6.1 '写入单元测试')。"""
    assert EMPTY_THINK in prompt, \
        "渲染结果缺失空 think 块 —— E0 污染路径回归, 立即冻结"


def strip_closed_think(text):
    """生成后处理: 仅剥离**闭合** think 块; 未闭合草稿原样保留
    (交由质量门 think 残留检测判 invalid)。"""
    return _CLOSED_THINK_RE.sub("", text).strip()


def has_think_residue(text):
    """think 残留检测: 任何 `<think>` / `</think>` 标记残留即为残留
    (未闭合草稿、嵌套草稿、剥离失败均会被捕获)。"""
    return ("<think>" in text) or ("</think>" in text)
