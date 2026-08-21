#!/usr/bin/env python
"""E0: 干净 base 金标回答再生成 (base_goldfix)。

问题: v4 金标评测中 base 的 100 条回答有 91 条是未闭合英文 think 草稿
(1024 token 预算被思考烧光, 正式答案未产生), 两轨评分测的是草稿。

修复策略 (经 e0_diag/e0_diag2 冒烟验证):
  1. enable_thinking=False 渲染后 **保留空 think 块** (<think>\n\n</think>\n\n)。
     诊断发现: harness 原实现剥掉空 think 块后, base 模型失去"思考已结束"锚点,
     100/100 自吐 <think> 英文草稿烧光预算; prefill/bad_words 均导致立即 EOS。
     保留空 think 块后 base 直接产出正常中文结构化作答 (3/3 冒烟通过)。
  2. 兜底: 若仍出现未闭合 think / 非中文作答 → 同 prompt + bad_words_ids
     禁用 think token 重生成一次; 仍无效则标记 *_invalid。
  生成后剥离任何闭合 <think>...</think> 块再入缓存。

输出:
  --out     : harness 兼容缓存 jsonl ({"id","response"} 逐行追加, response 已剥 think)
  --raw-out : 原始生成 + 处理元数据 (e0_basefix/raw_gen.jsonl), 供审计
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 放 pipeline_v4/src/ 旁运行时为 mongoose_ai
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pipeline_v4" / "src"))

EMPTY_THINK = "<think>\n\n</think>\n\n"
PREFILL = "好的,下面直接给出回答:\n"
THINK_CLOSED = re.compile(r"<think>.*?</think>", re.DOTALL)
CJK = re.compile(r"[一-鿿]")


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def strip_think(text):
    """剥离闭合 think 块; 返回 (clean, meta)。"""
    had_open = "<think>" in text
    had_close = "</think>" in text
    clean = THINK_CLOSED.sub("", text)
    if "<think>" in clean:  # 未闭合 → 其后全是草稿
        clean = clean.split("<think>")[0]
    clean = clean.strip()
    return clean, {"had_think_open": had_open, "had_think_close": had_close}


def cjk_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if CJK.match(c)) / len(letters)


def is_valid_answer(clean):
    return len(clean) >= 20 and cjk_ratio(clean) >= 0.3


def render_prompt(tok, system_message, question, prefill=None):
    # 关键修复: 保留 enable_thinking=False 渲染的空 think 块 (不 .replace)
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": system_message},
         {"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if prefill:
        prompt = prompt + prefill
    return prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_v4/configs/eval_v4.json")
    ap.add_argument("--eval-file", default=None)
    ap.add_argument("--out", required=True, help="harness 兼容缓存 jsonl (剥 think 后)")
    ap.add_argument("--raw-out", required=True, help="原始生成 + 元数据 jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    with open(PROJECT_ROOT / args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    eval_file = Path(args.eval_file) if args.eval_file else PROJECT_ROOT / cfg["eval_file"]
    items = [json.loads(l) for l in open(eval_file, encoding="utf-8") if l.strip()]
    if args.limit:
        items = items[: args.limit]
    log(f"金标 {len(items)} 题 | out={args.out} | max_new_tokens={args.max_new_tokens}")

    out_path, raw_path = Path(args.out), Path(args.raw_out)
    done = set()
    if out_path.is_file():
        for l in open(out_path, encoding="utf-8"):
            if l.strip():
                done.add(json.loads(l)["id"])
    log(f"缓存已有 {len(done)} 条, 断点续跑")

    import compat  # noqa: F401  WeightConverter monkey-patch
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = str(PROJECT_ROOT / cfg["base_model_path"])
    log(f"加载 BF16 基座: {base}")
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map=cfg["generation"]["device"],
        trust_remote_code=True)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bad_words = [tok("<think>", add_special_tokens=False)["input_ids"],
                 tok("</think>", add_special_tokens=False)["input_ids"]]
    log(f"think token 禁用表 (兜底用): {bad_words}")

    def gen(prompt, use_bad_words=False):
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        kw = {"max_new_tokens": args.max_new_tokens, "do_sample": False,
              "pad_token_id": tok.pad_token_id}
        if use_bad_words:
            kw["bad_words_ids"] = bad_words
        with torch.no_grad():
            out = model.generate(**inputs, **kw)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()

    stats = {"direct_ok": 0, "badwords_fallback": 0, "still_invalid": 0,
             "think_closed": 0, "think_unclosed": 0, "no_think": 0}
    sysmsg = cfg["chatml"]["system_message"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as fout, \
         open(raw_path, "a", encoding="utf-8") as fraw:
        for i, it in enumerate(items):
            if it["id"] in done:
                continue
            t0 = time.time()
            raw1 = gen(render_prompt(tok, sysmsg, it["question"]))
            clean, meta = strip_think(raw1)
            mode = "direct"
            if meta["had_think_open"] and meta["had_think_close"]:
                stats["think_closed"] += 1
            elif meta["had_think_open"]:
                stats["think_unclosed"] += 1
            else:
                stats["no_think"] += 1
            raw2 = ""
            if not is_valid_answer(clean):
                # 兜底: 同 prompt + 禁 think token 重生成
                raw2 = gen(render_prompt(tok, sysmsg, it["question"]),
                           use_bad_words=True)
                clean2, meta2 = strip_think(raw2)
                mode = "badwords"
                stats["badwords_fallback"] += 1
                if is_valid_answer(clean2):
                    clean = clean2
                else:
                    stats["still_invalid"] += 1
                    # 保留两者中较长者, 标记 invalid 供后续排查
                    if len(clean2) > len(clean):
                        clean = clean2
                    mode = "badwords_invalid"
            else:
                stats["direct_ok"] += 1
            fout.write(json.dumps({"id": it["id"], "response": clean},
                                  ensure_ascii=False) + "\n")
            fout.flush()
            fraw.write(json.dumps(
                {"id": it["id"], "mode": mode, "meta_direct": meta,
                 "raw_direct": raw1, "raw_fallback": raw2,
                 "clean_len": len(clean), "cjk_ratio": round(cjk_ratio(clean), 3),
                 "gen_seconds": round(time.time() - t0, 1)},
                ensure_ascii=False) + "\n")
            fraw.flush()
            log(f"[{i+1}/{len(items)}] {it['id']} mode={mode} len={len(clean)} "
                f"({time.time()-t0:.0f}s)")
    log(f"完成: {json.dumps(stats, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
