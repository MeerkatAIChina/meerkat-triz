#!/usr/bin/env python
"""Qwen3.8-27B 换基座复训前三步预检 (v6 前置步骤②, 2026-08-14)。

烧训练算力之前, 依次验证:
  P1 模块名对齐: 新基座 named_modules 是否覆盖 v5a 配方的 12 个 target_modules
     (qwen3_5 混合线性注意力架构, 预期有 in_proj_qkv/z/b/a, 但以实测为准);
     同时确认语言模型子模块路径 (多模态 ForConditionalGeneration 封装)。
  P2 数据契约: v5a 训练/验证集在新 tokenizer 下的编码 —
     2048 max_length 截断率、特殊 token 变化、prompt/response 边界完整性。
  P3 生成冒烟: 裸基座跑 5 题金标, 过质量门 ① (think 残留) 与 ③ (中文占比),
     验证 keep_empty_think_block 假设与生成配置。

用法 (DGX 上, venv_v5 或兼容环境):
  python pipeline_v5/eval/precheck_qwen38.py \
      --model models/Qwen3.8-27B \
      --train-file data/processed/v5_data/final/v5_train_v5a.jsonl \
      --val-file   data/processed/v5_data/final/v5_validation.jsonl \
      --gold-file  data/processed/v5_data/v5_gold.jsonl \
      --report results/v5/precheck_qwen38.md

退出码: 0 = 三步全 PASS; 3 = 任一步 FAIL (沿用 v5 train.py 断言语义)。
P3 需要 GPU 与完整模型权重; --skip-generate 可只做 P1+P2 (纯 CPU/ tokenizer)。
"""
import argparse
import json
import sys

TARGET_MODULES_V5A = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    "gate_proj", "up_proj", "down_proj",
]

MAX_LENGTH = 2048


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def p1_module_check(model_path, report):
    """P1: 配置级检查 (不加载权重) — 架构/层数/线性注意力参数 + 模块名静态推断。"""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text = getattr(cfg, "text_config", cfg)
    info = {
        "model_type": getattr(cfg, "model_type", None),
        "is_multimodal": hasattr(cfg, "vision_config"),
        "num_hidden_layers": getattr(text, "num_hidden_layers", None),
        "hidden_size": getattr(text, "hidden_size", None),
        "linear_attn_layers": (
            text.layer_types.count("linear_attention")
            if hasattr(text, "layer_types") else None),
        "full_attn_layers": (
            text.layer_types.count("full_attention")
            if hasattr(text, "layer_types") else None),
        "vocab_size": getattr(text, "vocab_size", None),
    }
    report.append(f"## P1 模块与架构检查\n")
    report.append("```json\n" + json.dumps(info, ensure_ascii=False, indent=2) + "\n```\n")
    if not info["is_multimodal"]:
        report.append("- ⚠️ 未检测到 vision_config, 与 Qwen3.8-27B 已知 config 不符, 检查模型路径\n")
    report.append(
        "- 架构为 qwen3_5 混合线性注意力 (3:1 linear:full), 预期存在 "
        "`in_proj_qkv/in_proj_z/in_proj_b/in_proj_a`; "
        "**权重级 named_modules 核对在 P3 加载后进行** (见 P3 报告段)\n")
    return True  # 配置级不否决, 模块名实测在 P3


def p2_data_contract(model_path, train_file, val_file, report):
    """P2: 新 tokenizer 下的数据契约。"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    report.append(f"\n## P2 数据契约 (tokenizer: {model_path})\n")
    fails = []

    for name, path in [("train", train_file), ("val", val_file)]:
        rows = load_jsonl(path)
        lens, truncated, boundary_bad = [], 0, 0
        for r in rows:
            # v5 数据为预模板化 ChatML 文本 (prompt/completion), 直接拼接编码
            if r.get("prompt") is not None and r.get("completion") is not None:
                prompt, completion = str(r["prompt"]), str(r["completion"])
                # 新基座 eos 从 <|endoftext|> 变为 <|im_end|>: 模拟 train.py
                # 的实际拼接 (prompt + completion + eos)
                full = prompt + completion + (tok.eos_token or "")
                ids = tok(full, add_special_tokens=False).input_ids
                if len(ids) > MAX_LENGTH:
                    # 截断边界: 右对齐截断时 completion 尾部必须完整
                    comp_ids = tok(completion, add_special_tokens=False).input_ids
                    if ids[-len(comp_ids):] != comp_ids:
                        boundary_bad += 1
            else:
                msgs = r.get("messages") or r.get("conversation")
                if not msgs:
                    text = r.get("text", "")
                    ids = tok(text, add_special_tokens=True).input_ids
                else:
                    text = tok.apply_chat_template(msgs, tokenize=False)
                    ids = tok(text, add_special_tokens=False).input_ids
                    if len(ids) > MAX_LENGTH:
                        resp = msgs[-1].get("content", "")
                        resp_ids = tok(resp, add_special_tokens=False).input_ids
                        if ids[-len(resp_ids):] != resp_ids:
                            boundary_bad += 1
            lens.append(len(ids))
            if len(ids) > MAX_LENGTH:
                truncated += 1
        n = len(rows)
        trunc_rate = truncated / max(n, 1)
        report.append(
            f"- `{path}`: n={n} | 中位长度 {sorted(lens)[n // 2]} tok | "
            f"最大 {max(lens)} | **截断率 {trunc_rate:.2%}** ({truncated} 条) | "
            f"response 边界受损 {boundary_bad} 条\n")
        # 空编码硬门: 中位长度 0 说明字段 schema 未被识别, 统计不可信
        if sorted(lens)[n // 2] == 0:
            fails.append(f"{name} 中位长度 0 — 数据 schema 未识别, 统计不可信")
        # v5 §11.2-3 的隐含契约: 截断率不可显著高于 v5a 旧基座水平
        if trunc_rate > 0.02:
            fails.append(f"{name} 截断率 {trunc_rate:.2%} > 2%")
        if boundary_bad > 0:
            fails.append(f"{name} 有 {boundary_bad} 条截断伤到 response 区段")

    # 特殊 token 漂移
    specials = {k: v for k, v in tok.special_tokens_map.items()}
    report.append(f"- 特殊 token: ```json {json.dumps(specials, ensure_ascii=False)} ```\n")
    # tokenizer 完整性硬门: 词表远小于标注值说明文件未下载全/加载回退,
    # 此时上面所有长度统计均不可信 (2026-08-14 实测 vocab_size=1 的静默假 PASS)
    if tok.vocab_size < 200000:
        fails.append(f"tokenizer 加载异常 vocab_size={tok.vocab_size} (<200000), "
                     f"长度统计不可信; 检查 tokenizer.json/tokenizer_config.json 是否完整")
    report.append(f"- vocab_size={tok.vocab_size} (config 标注 248320)\n")

    ok = not fails
    report.append(f"\n**P2 判定: {'PASS' if ok else 'FAIL — ' + '; '.join(fails)}**\n")
    return ok


def p3_smoke_generate(model_path, gold_file, report, limit=5):
    """P3: 裸基座生成冒烟 + 权重级模块名核对。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    report.append(f"\n## P3 生成冒烟 ({limit} 题金标, 裸基座)\n")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
    except Exception as e:  # 多模态封装加载失败时的诊断
        report.append(f"- **模型加载失败**: `{type(e).__name__}: {e}`\n"
                      f"- 处置: 降级 transformers 至 5.8.0.dev0 或改用 "
                      f"Qwen3_5ForConditionalGeneration 的 language_model 子路径后重试\n")
        report.append("\n**P3 判定: FAIL — 模型加载失败**\n")
        return False
    model.eval()

    # 权重级 target_modules 核对 (P1 的实测部分)
    names = [n for n, _ in model.named_modules()]
    missing = [m for m in TARGET_MODULES_V5A
               if not any(n.endswith("." + m) or n == m for n in names)]
    if missing:
        report.append(f"- ⚠️ 实际权重中未找到 target_modules: {missing}; "
                      f"复训配置需按实际模块名修订\n")
    else:
        report.append("- v5a 配方 12 个 target_modules 在实际权重中全部命中 ✅\n")
    vision_hit = [m for m in TARGET_MODULES_V5A
                  if any(("visual" in n or "vision" in n) and n.endswith(m)
                         for n in names)]
    if vision_hit:
        report.append(f"- ⚠️ 以下模块名同时出现在 vision tower: {vision_hit}; "
                      f"LoRA target 需限定 language_model 路径, 避免误打视觉侧\n")

    # 生成冒烟
    gold = load_jsonl(gold_file)[:limit]
    sys_msg = ("你是 TRIZ 创新方法论专家助手, 用中文专业回答用户关于 TRIZ 理论、"
               "发明原理、矛盾分析、ARIZ 算法等方面的问题。")
    gate_fail = 0
    for r in gold:
        msgs = [{"role": "system", "content": sys_msg},
                {"role": "user", "content": r["question"]}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to(model.device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=attn,
                                 max_new_tokens=512, do_sample=False)
        text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        think_leak = "<think>" in text or "</think>" in text
        zh = sum(1 for c in text if "一" <= c <= "鿿") / max(len(text), 1)
        status = []
        if think_leak:
            status.append("think残留")
        if len(text.strip()) < 100:
            status.append("过短")
        if zh < 0.3:
            status.append("中文占比低")
        if status:
            gate_fail += 1
        report.append(f"- {r.get('id', '?')}: len={len(text)} zh={zh:.2f} "
                      f"{'❌ ' + ','.join(status) if status else '✅'}\n")
    ok = gate_fail == 0
    if ok:
        verdict = "PASS"
    else:
        verdict = (f"FAIL — {gate_fail}/{limit} 题过质量门失败; "
                   f"若 think 残留为主因, 将 anchor 配置 "
                   f"keep_empty_think_block 改为 false 重试")
    report.append(f"\n**P3 判定: {verdict}**\n")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--val-file", required=True)
    ap.add_argument("--gold-file", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--skip-generate", action="store_true",
                    help="只做 P1+P2 (不需要 GPU/权重)")
    ap.add_argument("--smoke-limit", type=int, default=5)
    args = ap.parse_args()

    report = [f"# Qwen3.8-27B 复训前预检报告\n",
              f"- model: `{args.model}`\n"]
    ok = True
    ok &= p1_module_check(args.model, report)
    ok &= p2_data_contract(args.model, args.train_file, args.val_file, report)
    if not args.skip_generate:
        ok &= p3_smoke_generate(args.model, args.gold_file, report,
                                limit=args.smoke_limit)
    else:
        report.append("\n## P3 跳过 (--skip-generate); 上 DGX 后必须补跑\n")

    text = "".join(report)
    print(text)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
