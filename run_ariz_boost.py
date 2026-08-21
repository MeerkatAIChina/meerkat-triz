#!/usr/bin/env python3
"""ARIZ 定向补充生成 (v2 boost): 只为 ariz_guidance 子集生成新样本。

- 语料筛选: 仅 CATEGORY_SUBSET_HINTS 中含 ariz_guidance 的 3 个来源类别
  (2019年创新方法培训班课件 / TRIZ网课 / 教材和测试题, 共 352 chunks)
- ARIZ 专用 system prompt, 解析后强制 subset=ariz_guidance
- 断点续跑: data/processed/checkpoint_corpus_sft_v2_ariz_boost/
- 与既有 v2 全量样本按归一化 instruction 交叉去重
- 产物: data/processed/corpus_sft_v2_ariz_boost/ariz_guidance.json

用法 (DGX Spark):
    venv_v5/bin/python /tmp/run_ariz_boost.py                  # 全量 (~40 分钟 @ 3 RPM)
    venv_v5/bin/python /tmp/run_ariz_boost.py --max-chunks 3   # 冒烟测试
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.append("/home/meerkat/mongoose_ai")

from openai import RateLimitError, BadRequestError  # noqa: E402
from utils.corpus_to_sft import (  # noqa: E402
    CorpusSFTGenerator,
    _format_chunks_for_prompt,
    _parse_json_response,
    _normalize_instruction,
    apply_v2_quality_gates,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ariz_boost")

BASE = "/home/meerkat/mongoose_ai"
CORPUS_PATH = f"{BASE}/data/processed/corpus/triz_corpus.jsonl"
OUTPUT_DIR = f"{BASE}/data/processed/corpus_sft_v2_ariz_boost"
CHECKPOINT_DIR = f"{BASE}/data/processed/checkpoint_corpus_sft_v2_ariz_boost"
EXISTING_CKPT = f"{BASE}/data/processed/checkpoint_corpus_sft_v2/corpus_sft_checkpoint.json"

ARIZ_CATEGORIES = {"2019年创新方法培训班课件", "TRIZ网课", "教材和测试题"}

ARIZ_SYSTEM_PROMPT = """你是一个TRIZ领域数据生成专家，专注于 ARIZ（发明问题解决算法）。你的任务是根据给定的TRIZ原始材料片段，生成高质量的 ARIZ 指导类中文指令微调样本。

对每个输入片段，生成指定数量的样本，每个样本从不同 ARIZ 角度切入（轮换使用）：
- 步骤指导：讲解 ARIZ 的阶段/步骤（问题分析、矛盾构建、理想解陈述、资源分析、方案评估）并结合片段内容示例
- 问题转化：如何将模糊的实际问题转化为 ARIZ 可处理的矛盾模型
- 理想解：如何陈述最终理想解（IFR）与理想度分析
- 流程应用：以片段场景演示完整的 ARIZ 推理流程

每个样本是一个JSON对象，格式如下：
{
  "subset": "ariz_guidance",
  "angle": "step_guidance | problem_transform | ideal_solution | full_process 之一",
  "instruction": "用户向TRIZ专家提出的关于ARIZ的问题",
  "input": "可选的补充上下文（通常为空字符串）",
  "output": "基于材料片段生成的专业、完整、结构化的ARIZ回答"
}

质量要求：
1. subset 必须固定为 "ariz_guidance"，问题必须围绕 ARIZ 算法本身或其应用。
2. 问题紧密围绕材料片段内容，instruction 中应体现片段涉及的具体主题/方法/案例名称，禁止脱离片段的泛泛提问。
3. 回答基于材料片段，专业准确，必要时分步骤论述，不少于250字。
4. 不同样本的 instruction 必须有明显差异，禁止改写重复。
5. output 必须是纯文本字符串（可含 Markdown 分点），严禁把回答包在 {"answer": ...} 之类的嵌套JSON对象中。
6. 输出必须是严格的JSON数组，不要输出任何JSON之外的解释文字。
"""


def _unwrap_output(output: str) -> str:
    """修复模型把回答包成 {"answer": ...} 嵌套对象并被 str() 序列化的问题。"""
    import ast
    o = output.strip()
    if not o.startswith("{"):
        return output
    if "'answer'" not in o and '"answer"' not in o and "'output'" not in o and '"output"' not in o:
        return output
    d = None
    try:
        d = json.loads(o)
    except Exception:
        try:
            d = ast.literal_eval(o)
        except Exception:
            return output
    if isinstance(d, dict):
        for k in ("answer", "output", "response", "result"):
            if isinstance(d.get(k), str) and d[k].strip():
                return d[k].strip()
    return output


def generate_batch_ariz(gen, chunks, samples_per_chunk, max_tokens, temperature):
    """镜像 generate_batch_v2，但使用 ARIZ 专用 prompt 并强制子集标签。"""
    if not chunks:
        return []
    gen._rate_limit_sleep()
    prompt_body = _format_chunks_for_prompt(chunks)
    expected = len(chunks) * samples_per_chunk
    user_prompt = (
        f"请为以下 {len(chunks)} 个TRIZ材料片段各生成 {samples_per_chunk} 个 ARIZ 指导类微调样本"
        f"（每个片段使用不同角度: step_guidance/problem_transform/ideal_solution/full_process 轮换），"
        f"输出严格的JSON数组（共 {expected} 个JSON对象）:\n\n{prompt_body}"
    )
    try:
        response = gen.client.chat.completions.create(
            model=gen.model,
            messages=[
                {"role": "system", "content": ARIZ_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        gen.last_request_time = time.time()
        gen.total_requests += 1
        if response.usage:
            gen.total_tokens_used += response.usage.total_tokens
        content = response.choices[0].message.content
        samples = _parse_json_response(content, expected, fallback_subset="ariz_guidance")
        for s in samples:
            s["subset"] = "ariz_guidance"
            s["output"] = _unwrap_output(s["output"])
        return samples
    except RateLimitError:
        logger.warning("触发 Moonshot 速率限制，等待60秒后重试...")
        time.sleep(60)
        return generate_batch_ariz(gen, chunks, samples_per_chunk, max_tokens, temperature)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="moonshot-v1-8k")
    parser.add_argument("--rpm", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--samples-per-chunk", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-chunks", type=int, default=None, help="限制处理 chunk 数 (冒烟测试用)")
    args = parser.parse_args()

    t0 = datetime.now()
    logger.info(f"=== ARIZ boost 生成开始 {t0.isoformat()} ===")

    chunks = []
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c.get("metadata", {}).get("category") in ARIZ_CATEGORIES:
                chunks.append(c)
    if args.max_chunks:
        chunks = chunks[: args.max_chunks]
    logger.info(f"ARIZ 相关 chunk: {len(chunks)} 条 (类别: {sorted(ARIZ_CATEGORIES)})")

    # 既有 v2 样本的归一化 instruction (交叉去重)
    existing_keys = set()
    try:
        with open(EXISTING_CKPT, encoding="utf-8") as f:
            for s in json.load(f).get("samples", []):
                existing_keys.add(_normalize_instruction(s.get("instruction", "")))
        logger.info(f"既有 v2 样本去重键: {len(existing_keys)} 个")
    except Exception as e:
        logger.warning(f"既有 checkpoint 加载失败，跳过交叉去重: {e}")

    gen = CorpusSFTGenerator(
        model=args.model, rpm=args.rpm,
        output_dir=OUTPUT_DIR, checkpoint_dir=CHECKPOINT_DIR,
    )
    ckpt_file = Path(CHECKPOINT_DIR) / "corpus_sft_checkpoint.json"
    completed_ids, all_samples = set(), []
    if ckpt_file.exists():
        with open(ckpt_file, encoding="utf-8") as f:
            ck = json.load(f)
        completed_ids = set(ck.get("completed_ids", []))
        all_samples = ck.get("samples", [])
        logger.info(f"断点恢复: {len(completed_ids)}/{len(chunks)} chunks 已完成")

    total_batches = (len(chunks) + args.batch_size - 1) // args.batch_size
    for b in range(0, len(chunks), args.batch_size):
        batch = chunks[b: b + args.batch_size]
        bn = b // args.batch_size + 1
        if all(i in completed_ids for i in range(b, b + len(batch))):
            logger.info(f"批次 {bn}/{total_batches} 已完成，跳过")
            continue
        logger.info(f"批次 {bn}/{total_batches}: 处理 {len(batch)} 条 chunk")
        try:
            all_samples.extend(
                generate_batch_ariz(gen, batch, args.samples_per_chunk,
                                    args.max_tokens, args.temperature)
            )
            for i in range(b, b + len(batch)):
                completed_ids.add(i)
            gen._save_checkpoint(ckpt_file, completed_ids, all_samples)
        except BadRequestError as e:
            # 400 拒绝属提示词级确定性错误：标记完成以永久跳过，保存检查点后继续
            logger.warning(f"批次 {bn} 被 API 拒绝 (400)，跳过 {len(batch)} 条 chunk: {e}")
            for i in range(b, b + len(batch)):
                completed_ids.add(i)
            gen._save_checkpoint(ckpt_file, completed_ids, all_samples)
        except Exception as e:
            logger.error(f"批次 {bn} 失败: {e}")
            gen._save_checkpoint(ckpt_file, completed_ids, all_samples)
            raise

    # 质量门 + 与既有 v2 交叉去重
    all_samples, gate_stats = apply_v2_quality_gates(all_samples, min_output_chars=150)
    cross_dup = 0
    final = []
    for s in all_samples:
        if _normalize_instruction(s.get("instruction", "")) in existing_keys:
            cross_dup += 1
            continue
        final.append(s)
    logger.info(f"质量门: {gate_stats} | 与既有 v2 交叉重复剔除: {cross_dup}")

    grouped = {
        "ariz_guidance": [
            {"instruction": s["instruction"], "input": s.get("input", ""), "output": s["output"]}
            for s in final
        ]
    }
    saved = gen.save_subsets(grouped)
    logger.info(f"子集文件已保存: {saved}")

    stats = {
        "total_chunks": len(chunks),
        "completed_chunks": len(completed_ids),
        "total_samples": len(final),
        "quality_gates": gate_stats,
        "cross_duplicates_removed": cross_dup,
        "total_requests": gen.total_requests,
        "total_tokens_used": gen.total_tokens_used,
        "elapsed_minutes": round((datetime.now() - t0).total_seconds() / 60, 1),
        "args": vars(args),
    }
    stats_path = Path(OUTPUT_DIR) / f"ariz_boost_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"=== ARIZ boost 完成: {len(final)} 条新样本, 耗时 {stats['elapsed_minutes']} 分钟 ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
