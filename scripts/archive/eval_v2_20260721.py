"""
诊断评测：基座模型 vs LoRA适配器 (Layer 2 TRIZ + Layer 3 性能)

在同一进程、同一 FP16 加载方式、同一测试集下先后评测基座与适配器，
保证 apples-to-apples 对比。结果保存为 results/eval_v2_<ts>.json。

用法 (DGX Spark):
    venv_v5/bin/python scripts/eval_adapter_vs_base.py
"""

import json
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
ADAPTER_PATH = "/home/meerkat/mongoose_ai/models/meerkat_triz_adapter_v2"
RESULTS_DIR = "/home/meerkat/mongoose_ai/results"
TEST_DATA = "/home/meerkat/mongoose_ai/data/sample_data.json"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


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
        "peak_memory_gb": perf.get("peak_memory_gb"),
    }


def main():
    t0 = time.time()
    log("加载 tokenizer 与基座模型 (FP16)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    log(f"基座加载完成, 显存 {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    log("=== BASE: Layer 2 TRIZ 评测 ===")
    triz_base = run_triz_evaluation(
        model=base, tokenizer=tokenizer, output_dir=RESULTS_DIR,
        test_data_path=TEST_DATA, max_new_tokens=512, temperature=0.7,
    )
    log("=== BASE: Layer 3 性能评测 ===")
    perf_base = run_performance_benchmark(model=base, tokenizer=tokenizer, output_dir=RESULTS_DIR)

    log("挂载 LoRA 适配器...")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    log("适配器挂载完成")

    log("=== ADAPTER: Layer 2 TRIZ 评测 ===")
    triz_adapter = run_triz_evaluation(
        model=model, tokenizer=tokenizer, output_dir=RESULTS_DIR,
        test_data_path=TEST_DATA, max_new_tokens=512, temperature=0.7,
    )
    log("=== ADAPTER: Layer 3 性能评测 ===")
    perf_adapter = run_performance_benchmark(model=model, tokenizer=tokenizer, output_dir=RESULTS_DIR)

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "test_data": TEST_DATA,
        "base": summarize("base_fp16", triz_base, perf_base),
        "adapter": summarize("meerkat_triz_adapter_v2", triz_adapter, perf_adapter),
        "delta": {
            k: (summarize("a", triz_adapter, perf_adapter)[k] or 0)
               - (summarize("b", triz_base, perf_base)[k] or 0)
            for k in ["overall_score", "principle_accuracy", "contradiction_resolution",
                      "case_coverage", "ariz_completeness"]
        },
        "full_triz_base": triz_base,
        "full_triz_adapter": triz_adapter,
    }

    out = Path(RESULTS_DIR) / f"eval_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    log(f"对比结果已保存: {out}")
    log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    print("\n===== 汇总 =====")
    print(json.dumps({k: comparison[k] for k in ["base", "adapter", "delta"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
