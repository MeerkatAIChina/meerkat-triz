"""
诊断评测：基座模型 vs LoRA适配器 (Layer 2 TRIZ + Layer 3 性能)

在同一进程、同一 FP16 加载方式、同一测试集下先后评测基座与适配器，
保证 apples-to-apples 对比。结果保存为 results/adapter_vs_base_<ts>.json。

用法 (DGX Spark):
    venv_v5/bin/python scripts/eval_adapter_vs_base.py
    # 低成本 before/after 交叉验证: held-out test 困惑度对比
    venv_v5/bin/python scripts/eval_adapter_vs_base.py --ppl
    venv_v5/bin/python scripts/eval_adapter_vs_base.py --ppl data/processed/test.jsonl
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.append("/home/meerkat/mongoose_ai")

# 修复 PEFT v0.18 WeightConverter 兼容性 (必须在 PeftModel 加载前)
import peft.utils.transformers_weight_conversion as twc


def _skip_weight_conversion(model, peft_config, adapter_state_dict, adapter_name):
    return adapter_state_dict


twc.convert_peft_adapter_state_dict_for_transformers = _skip_weight_conversion

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402

from utils.benchmark_utils import run_triz_evaluation, run_performance_benchmark  # noqa: E402

BASE_PATH = "/home/meerkat/mongoose_ai/models/Qwen3.6-35B-A3B"
ADAPTER_PATH = "/home/meerkat/mongoose_ai/models/meerkat_triz_adapter_v1"
RESULTS_DIR = "/home/meerkat/mongoose_ai/results"
TEST_DATA = "/home/meerkat/mongoose_ai/data/sample_data.json"
DEFAULT_PPL_DATA = "/home/meerkat/mongoose_ai/data/processed/test.jsonl"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="基座 vs LoRA 适配器诊断评测")
    parser.add_argument(
        "--ppl",
        nargs="?",
        const=DEFAULT_PPL_DATA,
        default=None,
        metavar="JSONL",
        help="启用困惑度对比模式: 对给定 jsonl (默认 data/processed/test.jsonl, "
             "训练留出的 157 条 held-out test) 分别计算基座与适配器的平均 loss/困惑度",
    )
    return parser.parse_args()


def load_base_and_tokenizer():
    """加载 tokenizer 与 FP16 基座模型 (TRIZ/性能/PPL 模式共用)。"""
    log("加载 tokenizer 与基座模型 (FP16)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    log(f"基座加载完成, 显存 {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    return tokenizer, base


def summarize(tag, triz, perf):
    return {
        "model": tag,
        "overall_score": triz.get("overall_score"),
        "principle_accuracy": triz["principle_accuracy"].get("accuracy"),
        "contradiction_resolution": triz["contradiction_resolution"].get("average_score"),
        "case_coverage": triz["case_quality"].get("average_coverage"),
        "ariz_completeness": triz["ariz_completeness"].get("completeness"),
        "throughput_tokens_per_sec": perf.get("throughput_tokens_per_sec"),
        "latency_p50_ms": perf.get("latency_p50_ms"),
        # 与 run_performance_benchmark 返回的真实键名一致
        "peak_memory_gb": perf.get("memory_peak_gb"),
    }


def compute_ppl(model, tokenizer, jsonl_path, max_length=2048):
    """对 jsonl 数据集计算平均 loss 与困惑度 (token 加权 corpus 级)。"""
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if not samples:
        raise ValueError(f"PPL 数据集为空: {jsonl_path}")
    log(f"PPL 数据集: {jsonl_path} ({len(samples)} 条)")

    model.eval()
    device = next(model.parameters()).device
    losses = []
    total_nll = 0.0
    total_tokens = 0

    for i, s in enumerate(samples):
        text = s.get("text")
        if not text:
            # 回退: 由 instruction/input/output 重建对话文本 (无 system)
            user_content = s.get("instruction", "")
            if s.get("input"):
                user_content += "\n" + s["input"]
            text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": s.get("output", "")},
                ],
                tokenize=False,
            )
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(device)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        n_tokens = enc["input_ids"].shape[1] - 1  # 移位后的有效预测 token 数
        losses.append(out.loss.item())
        total_nll += out.loss.item() * n_tokens
        total_tokens += n_tokens
        if (i + 1) % 20 == 0:
            log(f"  PPL 进度 {i + 1}/{len(samples)}")

    avg_loss = sum(losses) / len(losses)
    token_avg_loss = total_nll / total_tokens
    return {
        "num_samples": len(samples),
        "avg_loss": avg_loss,
        "perplexity": math.exp(token_avg_loss),
        "total_tokens": total_tokens,
    }


def run_ppl_comparison(tokenizer, base, jsonl_path):
    """PPL 模式: 基座 vs 适配器在 held-out test 上的困惑度对比。"""
    log("=== BASE: held-out test 困惑度 ===")
    ppl_base = compute_ppl(base, tokenizer, jsonl_path)
    log(f"BASE   avg_loss={ppl_base['avg_loss']:.4f} ppl={ppl_base['perplexity']:.4f}")

    log("挂载 LoRA 适配器...")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    log("适配器挂载完成")

    log("=== ADAPTER: held-out test 困惑度 ===")
    ppl_adapter = compute_ppl(model, tokenizer, jsonl_path)
    log(f"ADAPTER avg_loss={ppl_adapter['avg_loss']:.4f} ppl={ppl_adapter['perplexity']:.4f}")

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "mode": "ppl",
        "ppl_data": jsonl_path,
        "base": ppl_base,
        "adapter": ppl_adapter,
        "delta": {
            "avg_loss": ppl_adapter["avg_loss"] - ppl_base["avg_loss"],
            "perplexity": ppl_adapter["perplexity"] - ppl_base["perplexity"],
        },
    }

    out = Path(RESULTS_DIR) / f"ppl_adapter_vs_base_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    log(f"PPL 对比结果已保存: {out}")
    print("\n===== PPL 汇总 =====")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    t0 = time.time()
    tokenizer, base = load_base_and_tokenizer()

    if args.ppl is not None:
        # 困惑度对比模式: 低成本 before/after 交叉验证
        run_ppl_comparison(tokenizer, base, args.ppl)
        log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
        return

    log("=== BASE: Layer 2 TRIZ 评测 ===")
    triz_base = run_triz_evaluation(
        model=base, tokenizer=tokenizer, output_dir=RESULTS_DIR,
        test_data_path=TEST_DATA, max_new_tokens=512, temperature=0.0,
    )
    log("=== BASE: Layer 3 性能评测 ===")
    perf_base = run_performance_benchmark(model=base, tokenizer=tokenizer, output_dir=RESULTS_DIR)

    log("挂载 LoRA 适配器...")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    log("适配器挂载完成")

    log("=== ADAPTER: Layer 2 TRIZ 评测 ===")
    triz_adapter = run_triz_evaluation(
        model=model, tokenizer=tokenizer, output_dir=RESULTS_DIR,
        test_data_path=TEST_DATA, max_new_tokens=512, temperature=0.0,
    )
    log("=== ADAPTER: Layer 3 性能评测 ===")
    perf_adapter = run_performance_benchmark(model=model, tokenizer=tokenizer, output_dir=RESULTS_DIR)

    base_summary = summarize("base_fp16", triz_base, perf_base)
    adapter_summary = summarize("meerkat_triz_adapter_v1", triz_adapter, perf_adapter)

    # 显式 None 传播: 任一缺失则 delta 记 None 并列入 delta_missing, 不再静默当 0
    delta = {}
    delta_missing = []
    for k in ["overall_score", "principle_accuracy", "contradiction_resolution",
              "case_coverage", "ariz_completeness"]:
        b_val = base_summary.get(k)
        a_val = adapter_summary.get(k)
        if b_val is None or a_val is None:
            delta[k] = None
            delta_missing.append(k)
        else:
            delta[k] = a_val - b_val
    if delta_missing:
        log(f"警告: 以下指标缺失, delta 记为 None: {delta_missing}")

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "test_data": TEST_DATA,
        "base": base_summary,
        "adapter": adapter_summary,
        "delta": delta,
        "delta_missing": delta_missing,
        "full_triz_base": triz_base,
        "full_triz_adapter": triz_adapter,
    }

    out = Path(RESULTS_DIR) / f"adapter_vs_base_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    log(f"对比结果已保存: {out}")
    log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    print("\n===== 汇总 =====")
    print(json.dumps({k: comparison[k] for k in ["base", "adapter", "delta", "delta_missing"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
