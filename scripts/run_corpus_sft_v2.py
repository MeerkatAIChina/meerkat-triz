"""
Corpus SFT V2 生成运行器 (DGX Spark)

从 TRIZ-raw corpus 全量生成多角度 SFT 数据集：
- 每个 chunk 生成 3 个不同角度样本 (concept/application/case/comparison)
- 按来源类别引导子集分类 (修正 V1 的失衡)
- 质量门: 清洗空 think 块 / 最短回答 150 字符 / instruction 去重
- 断点续跑 (checkpoint_corpus_sft_v2)
- 生成后构建 ChatML train/val/test jsonl (与 V1 同 schema, 剥离空 think 前缀)

用法 (DGX Spark):
    # 冒烟测试 (30 chunks ≈ 90 样本, 验证质量与成本)
    venv_v5/bin/python scripts/run_corpus_sft_v2.py --max-samples 30

    # 全量运行 (3,914 chunks × 3 ≈ 11.7K 样本, ~7小时 @ 3 RPM)
    venv_v5/bin/python scripts/run_corpus_sft_v2.py
"""

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.append("/home/meerkat/mongoose_ai")

from config import DATA_CONFIG  # noqa: E402
from utils.corpus_to_sft import CorpusSFTGenerator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_corpus_sft_v2")

BASE = "/home/meerkat/mongoose_ai"
CORPUS_PATH = f"{BASE}/data/processed/corpus/triz_corpus.jsonl"
MODEL_PATH = f"{BASE}/models/Qwen3.6-35B-A3B"
OUTPUT_DIR = f"{BASE}/data/raw/corpus_sft_v2"
CHECKPOINT_DIR = f"{BASE}/data/processed/checkpoint_corpus_sft_v2"
PROCESSED_DIR = f"{BASE}/data/processed"

SPLIT_RATIO = DATA_CONFIG["split_ratio"]  # train 0.85 / val 0.10 / test 0.05
EMPTY_THINK = "<think>\n\n</think>\n\n"


def build_chatml_jsonl(grouped, output_prefix, seed=42):
    """将分组样本转为 ChatML jsonl 并划分 train/val/test (复用 tokenizer 模板)。"""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    system_message = DATA_CONFIG["chatml"]["system_message"]

    all_samples = []
    for subset_name, samples in grouped.items():
        for s in samples:
            all_samples.append({**s, "subset": subset_name})

    random.Random(seed).shuffle(all_samples)
    n = len(all_samples)
    n_train = int(n * SPLIT_RATIO["train"])
    n_val = int(n * SPLIT_RATIO["validation"])
    splits = {
        "train": all_samples[:n_train],
        "validation": all_samples[n_train:n_train + n_val],
        "test": all_samples[n_train + n_val:],
    }

    think_stripped = 0
    for split_name, samples in splits.items():
        path = Path(PROCESSED_DIR) / f"{output_prefix}{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                user_content = s["instruction"]
                if s.get("input"):
                    user_content += "\n" + s["input"]
                text = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": s["output"]},
                    ],
                    tokenize=False,
                )
                if EMPTY_THINK in text:
                    text = text.replace(EMPTY_THINK, "")
                    think_stripped += 1
                # instruction/input/output 是训练 formatting_func 的必需字段; text 用于检查
                f.write(json.dumps({
                    "instruction": s["instruction"],
                    "input": s.get("input", ""),
                    "output": s["output"],
                    "subset": s["subset"],
                    "text": text,
                }, ensure_ascii=False) + "\n")
        logger.info(f"{split_name}: {len(samples)} 条 → {path}")

    return {k: len(v) for k, v in splits.items()}, think_stripped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="moonshot-v1-8k")
    parser.add_argument("--rpm", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--samples-per-chunk", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="限制处理的 chunk 数 (冒烟测试用)")
    parser.add_argument("--output-prefix", default="v2_",
                        help="jsonl 文件名前缀 (默认 v2_; 冒烟测试建议 test_)")
    parser.add_argument("--skip-build", action="store_true",
                        help="只生成子集 JSON, 不构建 ChatML jsonl")
    parser.add_argument("--rebuild-only", action="store_true",
                        help="跳过 API 生成, 从 checkpoint 的样本重建子集 JSON 与 jsonl "
                             "(用于运行中的任务结束后按修正后的 schema 重建)")
    args = parser.parse_args()

    t0 = datetime.now()

    if args.rebuild_only:
        from utils.corpus_to_sft import VALID_SUBSETS, apply_v2_quality_gates

        checkpoint_file = Path(CHECKPOINT_DIR) / "corpus_sft_checkpoint.json"
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        all_samples = checkpoint.get("samples", [])
        logger.info(f"从 checkpoint 加载 {len(all_samples)} 条样本")

        all_samples, gate_stats = apply_v2_quality_gates(all_samples, min_output_chars=150)
        logger.info(f"质量门: {gate_stats}")

        grouped = {s: [] for s in VALID_SUBSETS}
        for s in all_samples:
            grouped[s["subset"]].append({
                "instruction": s["instruction"],
                "input": s["input"],
                "output": s["output"],
            })

        generator = CorpusSFTGenerator(
            model=args.model, rpm=args.rpm,
            output_dir=OUTPUT_DIR, checkpoint_dir=CHECKPOINT_DIR,
        )
        saved = generator.save_subsets(grouped)
        logger.info(f"子集文件已保存: {len(saved)} 个 → {OUTPUT_DIR}")

        split_counts, think_stripped = build_chatml_jsonl(grouped, args.output_prefix)
        stats = {
            "rebuild_only": True,
            "quality_gates": gate_stats,
            "subset_distribution": {k: len(v) for k, v in grouped.items()},
            "chatml_splits": split_counts,
            "chatml_think_stripped": think_stripped,
        }
        logger.info("=== 重建完成 ===")
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    logger.info(f"=== Corpus SFT V2 生成开始 {t0.isoformat()} ===")
    logger.info(f"模型: {args.model} | RPM: {args.rpm} | 每批 {args.batch_size} chunks "
                f"× {args.samples_per_chunk} 样本 | max_samples: {args.max_samples}")

    generator = CorpusSFTGenerator(
        model=args.model,
        rpm=args.rpm,
        output_dir=OUTPUT_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    grouped, stats = generator.generate_from_corpus(
        CORPUS_PATH,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        samples_per_chunk=args.samples_per_chunk,
        min_output_chars=150,
        dedup=True,
        category_hints=True,
    )

    saved = generator.save_subsets(grouped)
    logger.info(f"子集文件已保存: {len(saved)} 个 → {OUTPUT_DIR}")

    if not args.skip_build:
        split_counts, think_stripped = build_chatml_jsonl(grouped, args.output_prefix)
        stats["chatml_splits"] = split_counts
        stats["chatml_think_stripped"] = think_stripped

    stats["elapsed_minutes"] = round((datetime.now() - t0).total_seconds() / 60, 1)
    stats["args"] = vars(args)

    stats_path = Path(OUTPUT_DIR) / f"v2_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"=== 完成, 耗时 {stats['elapsed_minutes']} 分钟 ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
