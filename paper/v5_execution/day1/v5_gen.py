#!/usr/bin/env python
"""v5 Day1: base/v2/v4 在新金标 100 题 (v5_gold_new100) 上的对照生成。

协议 = E0 干净锚点协议 (E0_report.md §3 + v5_优化微调方案.md §6.1):
  1. enable_thinking=False 渲染后 **保留空 think 块** (<think>\n\n</think>\n\n),
     禁止后处理剥离; 生成完成后才剥离闭合 think 块。
  2. BF16 加载, 贪心 (do_sample=False), max_new_tokens=2048。
  3. 四道质量门: ① think 残留 ② 非空+中文占比>=0.3 ③ 长度下限
     (base>=100 字符; 适配器>=50 且不低于同题 base 长度 3%) ④ 英文草稿检测
     (前 300 字符 CJK 占比 <0.1 视为英文草稿)。
     任一不过 → 同 prompt + bad_words_ids 禁 think token 兜底重生成一次;
     仍不过计入 invalid。
适配器 (v2/v4) output 若无 think 结构直接作答属正常 (§6.1)。
断点续跑: out 文件已有 id 直接跳过。
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path("/home/meerkat/mongoose_ai")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pipeline_v4" / "src"))

THINK_CLOSED = re.compile(r"<think>.*?</think>", re.DOTALL)
CJK = re.compile(r"[一-鿿]")


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def strip_think(text):
    had_open = "<think>" in text
    had_close = "</think>" in text
    clean = THINK_CLOSED.sub("", text)
    if "<think>" in clean:
        clean = clean.split("<think>")[0]
    return clean.strip(), {"had_think_open": had_open, "had_think_close": had_close}


def cjk_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if CJK.match(c)) / len(letters)


def gate_report(clean, min_len):
    """四道质量门, 返回 (ok, gates dict)。"""
    g1_think_residual = ("<think>" in clean) or ("</think>" in clean)
    g2_nonempty_zh = len(clean) > 0 and cjk_ratio(clean) >= 0.3
    g3_length = len(clean) >= min_len
    head = clean[:300]
    g4_english_draft = len(clean) > 0 and cjk_ratio(head) < 0.1
    ok = (not g1_think_residual) and g2_nonempty_zh and g3_length \
        and (not g4_english_draft)
    return ok, {
        "think_residual": g1_think_residual,
        "nonempty_zh": g2_nonempty_zh,
        "length_ok": g3_length,
        "english_draft": g4_english_draft,
    }


def render_prompt(tok, system_message, question):
    # 关键: 保留 enable_thinking=False 渲染的空 think 块 (不剥离)
    return tok.apply_chat_template(
        [{"role": "system", "content": system_message},
         {"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default=None,
                    help="LoRA 适配器目录 (相对 PROJECT_ROOT); 缺省=纯 base")
    ap.add_argument("--eval-file",
                    default="data/processed/v5_data/v5_gold_new100.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw-out", required=True)
    ap.add_argument("--base-cache", default=None,
                    help="base responses jsonl, 用于适配器长度下限 3% 规则")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    with open(PROJECT_ROOT / "pipeline_v4/configs/eval_v4.json",
              encoding="utf-8") as f:
        cfg = json.load(f)
    sysmsg = cfg["chatml"]["system_message"]

    items = [json.loads(l) for l in
             open(PROJECT_ROOT / args.eval_file, encoding="utf-8") if l.strip()]
    if args.limit:
        items = items[: args.limit]
    log(f"tag={args.tag} | 题目 {len(items)} | adapter={args.adapter or '(base)'}")

    base_lens = {}
    if args.base_cache and Path(args.base_cache).is_file():
        for l in open(args.base_cache, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                base_lens[r["id"]] = len(r["response"])
        log(f"base 缓存长度表 {len(base_lens)} 条 (适配器 3% 规则)")

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
        base, torch_dtype=torch.bfloat16,
        device_map=cfg["generation"]["device"], trust_remote_code=True)
    if args.adapter:
        from peft import PeftModel
        adir = str(PROJECT_ROOT / args.adapter)
        log(f"挂载 LoRA 适配器: {adir}")
        model = PeftModel.from_pretrained(model, adir)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bad_words = [tok("<think>", add_special_tokens=False)["input_ids"],
                 tok("</think>", add_special_tokens=False)["input_ids"]]

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

    def min_len_for(qid):
        if not args.adapter:
            return 100
        floor = 50
        if qid in base_lens:
            floor = max(floor, int(base_lens[qid] * 0.03))
        return floor

    stats = {"direct_ok": 0, "badwords_fallback": 0, "still_invalid": 0,
             "think_closed": 0, "think_unclosed": 0, "no_think": 0}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as fout, \
         open(raw_path, "a", encoding="utf-8") as fraw:
        for i, it in enumerate(items):
            if it["id"] in done:
                continue
            t0 = time.time()
            ml = min_len_for(it["id"])
            raw1 = gen(render_prompt(tok, sysmsg, it["question"]))
            clean, meta = strip_think(raw1)
            mode = "direct"
            if meta["had_think_open"] and meta["had_think_close"]:
                stats["think_closed"] += 1
            elif meta["had_think_open"]:
                stats["think_unclosed"] += 1
            else:
                stats["no_think"] += 1
            ok, gates = gate_report(clean, ml)
            raw2 = ""
            if not ok:
                raw2 = gen(render_prompt(tok, sysmsg, it["question"]),
                           use_bad_words=True)
                clean2, meta2 = strip_think(raw2)
                mode = "badwords"
                stats["badwords_fallback"] += 1
                ok2, gates2 = gate_report(clean2, ml)
                if ok2:
                    clean, gates = clean2, gates2
                else:
                    stats["still_invalid"] += 1
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
                 "gates": gates, "min_len": ml,
                 "raw_direct": raw1, "raw_fallback": raw2,
                 "clean_len": len(clean),
                 "cjk_ratio": round(cjk_ratio(clean), 3),
                 "gen_seconds": round(time.time() - t0, 1)},
                ensure_ascii=False) + "\n")
            fraw.flush()
            log(f"[{i+1}/{len(items)}] {it['id']} mode={mode} len={len(clean)} "
                f"({time.time()-t0:.0f}s)")
    log(f"完成: {json.dumps(stats, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
