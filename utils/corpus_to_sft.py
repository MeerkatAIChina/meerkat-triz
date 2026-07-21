"""
将 TRIZ-raw corpus (triz_corpus.jsonl) 转换为 SFT 微调数据集。

功能：
- 从 corpus 中采样文本片段
- 调用 Moonshot API 为每个片段生成 (instruction, output) 问答对
- 自动映射到项目定义的 6 个数据子集
- 支持断点续跑、速率限制、按子集保存
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

from openai import OpenAI, RateLimitError, APIError, BadRequestError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 项目定义的 6 个微调子集
VALID_SUBSETS = [
    "concept_explanation",
    "contradiction_analysis",
    "principle_recommendation",
    "case_generation",
    "ariz_guidance",
    "innovation_assessment",
]


SYSTEM_PROMPT = """你是一个TRIZ领域数据生成专家。你的任务是根据给定的TRIZ原始材料片段，生成高质量的中文指令微调样本。

每个输入片段会生成一个JSON对象，格式如下：
{
  "subset": "子集名称（必须是以下之一）",
  "instruction": "用户向TRIZ专家提出的问题",
  "input": "可选的补充上下文（通常为空字符串）",
  "output": "基于材料片段生成的专业、完整、准确的TRIZ回答"
}

子集必须是以下六个之一：
- concept_explanation: 概念解释（定义、原理、术语说明）
- contradiction_analysis: 矛盾分析（技术矛盾、物理矛盾、分离原理）
- principle_recommendation: 原理推荐（根据问题推荐发明原理）
- case_generation: 案例生成（基于场景生成创新方案/案例）
- ariz_guidance: ARIZ指导（ARIZ算法步骤、问题转化、理想解）
- innovation_assessment: 创新评估（方案评估、专利可行性、技术成熟度）

要求：
1. 问题必须紧密围绕材料片段内容，不能脱离材料泛泛而谈。
2. 回答必须基于材料片段，保持TRIZ方法论的专业性和准确性。
3. instruction 要自然、像是真实用户在咨询TRIZ专家。
4. 输出必须是严格的JSON数组，每个元素对应一个输入片段。
5. 不要输出任何JSON之外的解释文字。
"""


# ==================== V2: 多角度高质量生成 ====================

SYSTEM_PROMPT_V2 = """你是一个TRIZ领域数据生成专家。你的任务是根据给定的TRIZ原始材料片段，生成高质量的中文指令微调样本。

对每个输入片段，生成指定数量的样本，每个样本从不同角度切入（轮换使用）：
- 概念角度：解释片段中的核心概念/原理/术语
- 应用角度：如何将片段中的方法应用于实际工程问题
- 案例角度：基于片段场景构造具体的问题解决案例
- 对比角度：与相近概念/方法的辨析（如适用）

每个样本是一个JSON对象，格式如下：
{
  "subset": "子集名称（必须是以下六个之一）",
  "angle": "concept | application | case | comparison 之一",
  "instruction": "用户向TRIZ专家提出的问题",
  "input": "可选的补充上下文（通常为空字符串）",
  "output": "基于材料片段生成的专业、完整、结构化的TRIZ回答"
}

子集必须是以下六个之一：
- concept_explanation: 概念解释（定义、原理、术语说明）
- contradiction_analysis: 矛盾分析（技术矛盾、物理矛盾、分离原理）
- principle_recommendation: 原理推荐（根据问题推荐发明原理）
- case_generation: 案例生成（基于场景生成创新方案/案例）
- ariz_guidance: ARIZ指导（ARIZ算法步骤、问题转化、理想解）
- innovation_assessment: 创新评估（方案评估、专利可行性、技术成熟度）

质量要求：
1. 问题必须紧密围绕材料片段内容，语法正确、表述自然，像真实用户咨询。
2. 回答必须基于材料片段，专业准确，必要时分点论述，不少于250字。
3. 不同样本的 instruction 必须有明显差异，禁止换汤不换药的改写。
4. 输出必须是严格的JSON数组，不要输出任何JSON之外的解释文字。
5. 如果提供了来源类别与推荐子集，优先使用推荐子集。
"""

# 来源类别 → 推荐子集（作为提示与解析兜底，修正 V1 自由分类导致的失衡）
CATEGORY_SUBSET_HINTS = {
    "40个发明原理": ["principle_recommendation", "concept_explanation"],
    "TRIZ二级教材": ["concept_explanation", "contradiction_analysis"],
    "二级课件": ["concept_explanation", "contradiction_analysis"],
    "2019年创新方法培训班课件": ["concept_explanation", "ariz_guidance"],
    "零散TRIZ课件": ["concept_explanation", "principle_recommendation"],
    "TRIZ网课": ["concept_explanation", "ariz_guidance"],
    "教材和测试题": ["contradiction_analysis", "ariz_guidance"],
    "TRIZ面试官": ["contradiction_analysis", "innovation_assessment"],
    "四级案例": ["case_generation"],
    "阳光电源": ["case_generation", "innovation_assessment"],
    "4-大华-基于TRIZ的全局摄像机开发": ["case_generation"],
    "GEN-TRIZ": ["concept_explanation", "innovation_assessment"],
    "Simon Litvin": ["concept_explanation", "innovation_assessment"],
    "Jack Hipple": ["concept_explanation", "principle_recommendation"],
    "Proceedings": ["innovation_assessment", "concept_explanation"],
}

# 样本角度轮换
SAMPLE_ANGLES = ["concept", "application", "case", "comparison"]


def _format_chunks_for_prompt(chunks: List[Dict[str, Any]]) -> str:
    """将多个 corpus chunk 格式化为一个 prompt。"""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        category = meta.get("category", "未知")
        file_type = meta.get("file_type", "未知")
        text = chunk.get("text", "").strip()
        # 截断过长文本，控制 token 消耗
        if len(text) > 2000:
            text = text[:2000] + "\n...[内容已截断]"
        lines.append(f"片段 {i}:\n  来源类别: {category}\n  文件类型: {file_type}\n  内容:\n{text}\n")
    return "\n".join(lines)


def _parse_json_response(content: str, expected_count: int, fallback_subset: str = "concept_explanation") -> List[Dict[str, str]]:
    """从模型响应中解析 JSON 样本列表。"""
    content = content.strip()
    samples = []

    # 尝试直接解析 JSON 数组
    try:
        data = json.loads(content)
        if isinstance(data, list):
            samples = data
        elif isinstance(data, dict):
            # 模型可能把结果包在一个 dict 里
            for key in ("samples", "data", "results"):
                if key in data and isinstance(data[key], list):
                    samples = data[key]
                    break
            if not samples:
                samples = [data]
    except json.JSONDecodeError:
        # 尝试按行解析 JSONL
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, list):
                    samples.extend(obj)
                elif isinstance(obj, dict):
                    samples.append(obj)
            except json.JSONDecodeError:
                continue

    validated = []
    for s in samples:
        if not isinstance(s, dict):
            continue
        subset = s.get("subset", "").strip().lower()
        # 子集名称容错：去掉空格/下划线差异
        normalized = None
        for valid in VALID_SUBSETS:
            if subset == valid or subset.replace(" ", "_") == valid:
                normalized = valid
                break
        if normalized is None:
            normalized = fallback_subset  # 兜底 (V2: 按来源类别引导)

        instruction = str(s.get("instruction", "")).strip()
        output = str(s.get("output", "")).strip()
        if not instruction or not output:
            continue

        validated.append({
            "subset": normalized,
            "instruction": instruction,
            "input": str(s.get("input", "")).strip(),
            "output": output,
        })

    if len(validated) != expected_count:
        logger.warning(f"解析到 {len(validated)} 条有效样本，期望 {expected_count} 条")

    return validated


# ==================== V2: 质量门 ====================

def _strip_think_blocks(text: str) -> str:
    """移除空的 <think>...</think> 块 (V1 数据中发现 Qwen 模板注入的空思考块)。"""
    import re
    text = re.sub(r"<think>\s*</think>\s*", "", text)
    return text.strip()


def _normalize_instruction(instruction: str) -> str:
    """归一化 instruction 用于去重：去空白/标点/大小写。"""
    import re
    return re.sub(r"[\s，。、？！,.?!'\"“”'']", "", instruction.lower())


def apply_v2_quality_gates(
    samples: List[Dict[str, str]],
    min_output_chars: int = 150,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """
    V2 质量门：清洗 think 块、最短回答长度过滤、instruction 精确去重。

    Returns:
        (过滤后的样本列表, 统计信息)
    """
    stats = {"input": len(samples), "think_stripped": 0, "too_short": 0, "duplicate": 0}
    seen = set()
    cleaned = []
    for s in samples:
        output = _strip_think_blocks(s.get("output", ""))
        if output != s.get("output", ""):
            stats["think_stripped"] += 1
        if len(output) < min_output_chars:
            stats["too_short"] += 1
            continue
        key = _normalize_instruction(s.get("instruction", ""))
        if not key or key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)
        cleaned.append({**s, "output": output})
    stats["kept"] = len(cleaned)
    return cleaned, stats


class CorpusSFTGenerator:
    """从 TRIZ-raw corpus 生成 SFT 数据集的生成器。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "moonshot-v1-8k",
        rpm: int = 3,
        base_url: str = "https://api.moonshot.cn/v1",
        output_dir: str = "/home/meerkat/mongoose_ai/data/raw/corpus_sft",
        checkpoint_dir: str = "/home/meerkat/mongoose_ai/data/processed/checkpoint_corpus_sft",
    ):
        self.api_key = api_key or os.environ.get("MOONSHOT_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Moonshot API key 未设置。请设置环境变量 MOONSHOT_API_KEY "
                "或在初始化时传入 api_key 参数。"
            )

        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        self.model = model
        self.rpm = rpm
        self.min_interval = 60.0 / rpm
        self.last_request_time = 0.0
        self.total_requests = 0
        self.total_tokens_used = 0

        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _rate_limit_sleep(self):
        """按 RPM 限制等待。"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            logger.info(f"速率限制: 等待 {sleep_time:.1f} 秒")
            time.sleep(sleep_time)

    def generate_batch(
        self,
        chunks: List[Dict[str, Any]],
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ) -> List[Dict[str, str]]:
        """为一批 corpus chunk 生成 SFT 样本。"""
        if not chunks:
            return []

        self._rate_limit_sleep()

        user_prompt = (
            f"请为以下 {len(chunks)} 个TRIZ材料片段各生成一个微调样本，"
            f"输出严格的JSON数组（每个片段对应一个JSON对象）:\n\n"
            f"{_format_chunks_for_prompt(chunks)}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            self.last_request_time = time.time()
            self.total_requests += 1
            if response.usage:
                self.total_tokens_used += response.usage.total_tokens

            content = response.choices[0].message.content
            return _parse_json_response(content, len(chunks))

        except RateLimitError:
            logger.warning("触发 Moonshot 速率限制，等待60秒后重试...")
            time.sleep(60)
            return self.generate_batch(chunks, max_tokens, temperature)
        except APIError as e:
            logger.error(f"Moonshot API 错误: {e}")
            raise

    def generate_batch_v2(
        self,
        chunks: List[Dict[str, Any]],
        samples_per_chunk: int = 3,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        category_hints: bool = True,
    ) -> List[Dict[str, str]]:
        """V2: 为一批 chunk 各生成 samples_per_chunk 个多角度样本。"""
        if not chunks:
            return []

        self._rate_limit_sleep()

        prompt_body = _format_chunks_for_prompt(chunks)
        expected = len(chunks) * samples_per_chunk

        hints_text = ""
        fallback_subset = "concept_explanation"
        if category_hints:
            # 汇总本批 chunk 的来源类别，给出推荐子集
            batch_hints: List[str] = []
            for chunk in chunks:
                category = chunk.get("metadata", {}).get("category", "")
                hinted = CATEGORY_SUBSET_HINTS.get(category)
                if hinted:
                    batch_hints.append(f"来源类别「{category}」推荐子集: {', '.join(hinted)}")
            if batch_hints:
                hints_text = "\n\n" + "\n".join(sorted(set(batch_hints)))
                fallback_subset = CATEGORY_SUBSET_HINTS.get(
                    chunks[0].get("metadata", {}).get("category", ""),
                    [fallback_subset],
                )[0]

        user_prompt = (
            f"请为以下 {len(chunks)} 个TRIZ材料片段各生成 {samples_per_chunk} 个微调样本"
            f"（每个片段使用不同角度: concept/application/case/comparison 轮换），"
            f"输出严格的JSON数组（共 {expected} 个JSON对象）:\n\n"
            f"{prompt_body}{hints_text}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_V2},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            self.last_request_time = time.time()
            self.total_requests += 1
            if response.usage:
                self.total_tokens_used += response.usage.total_tokens

            content = response.choices[0].message.content
            return _parse_json_response(content, expected, fallback_subset=fallback_subset)

        except RateLimitError:
            logger.warning("触发 Moonshot 速率限制，等待60秒后重试...")
            time.sleep(60)
            return self.generate_batch_v2(
                chunks, samples_per_chunk, max_tokens, temperature, category_hints
            )
        except APIError as e:
            logger.error(f"Moonshot API 错误: {e}")
            raise

    def generate_from_corpus(
        self,
        corpus_path: str,
        max_samples: Optional[int] = None,
        batch_size: int = 5,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        seed: int = 42,
        samples_per_chunk: int = 1,
        min_output_chars: int = 0,
        dedup: bool = False,
        category_hints: bool = False,
    ) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, Any]]:
        """
        从 corpus 文件生成 SFT 样本，按子集分组返回。

        Args:
            corpus_path: triz_corpus.jsonl 路径
            max_samples: 最多处理的 chunk 数（None=全部）
            batch_size: 每个 API 请求的 chunk 数
            max_tokens: 每次生成的最大 token 数
            temperature: 采样温度
            seed: 随机种子（用于采样）
            samples_per_chunk: V2 — 每个 chunk 生成的样本数 (>1 启用多角度生成)
            min_output_chars: V2 — 最短回答字符数 (0=不过滤)
            dedup: V2 — 按归一化 instruction 精确去重
            category_hints: V2 — 按来源类别引导子集分类与解析兜底

        Returns:
            (按子集分组的样本字典, 统计信息字典)
        """
        corpus_path = Path(corpus_path)
        if not corpus_path.exists():
            raise FileNotFoundError(f"corpus 文件不存在: {corpus_path}")

        # 加载所有 chunks
        all_chunks = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_chunks.append(json.loads(line))

        logger.info(f"corpus 共 {len(all_chunks)} 条 chunk")

        # 采样
        if max_samples is not None and max_samples < len(all_chunks):
            import random
            rng = random.Random(seed)
            all_chunks = rng.sample(all_chunks, max_samples)
            logger.info(f"采样 {max_samples} 条 chunk 用于生成")

        # 加载检查点
        checkpoint_file = self.checkpoint_dir / "corpus_sft_checkpoint.json"
        completed_ids: Set[int] = set()
        all_samples: List[Dict[str, str]] = []

        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    checkpoint = json.load(f)
                completed_ids = set(checkpoint.get("completed_ids", []))
                all_samples = checkpoint.get("samples", [])
                logger.info(f"从检查点恢复: {len(completed_ids)}/{len(all_chunks)} 已完成")
            except Exception as e:
                logger.warning(f"检查点加载失败: {e}，从头开始")

        # 估算成本
        remaining_count = len(all_chunks) - len(completed_ids)
        est_batches = (remaining_count + batch_size - 1) // batch_size
        est_time = est_batches * (60.0 / self.rpm) / 60.0
        logger.info(
            f"预计剩余 {remaining_count} 条 chunk，约 {est_batches} 次 API 调用，"
            f"按 {self.rpm} RPM 约需 {est_time:.1f} 分钟"
        )

        # 批量生成
        total_batches = (len(all_chunks) + batch_size - 1) // batch_size
        for batch_idx in range(0, len(all_chunks), batch_size):
            batch_chunks = [
                all_chunks[i]
                for i in range(batch_idx, min(batch_idx + batch_size, len(all_chunks)))
            ]
            batch_num = batch_idx // batch_size + 1

            # 跳过已完成的批次
            if all(i in completed_ids for i in range(batch_idx, batch_idx + len(batch_chunks))):
                logger.info(f"批次 {batch_num}/{total_batches} 已完成，跳过")
                continue

            logger.info(f"批次 {batch_num}/{total_batches}: 处理 {len(batch_chunks)} 条 chunk")
            try:
                if samples_per_chunk > 1:
                    generated = self.generate_batch_v2(
                        batch_chunks,
                        samples_per_chunk=samples_per_chunk,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        category_hints=category_hints,
                    )
                else:
                    generated = self.generate_batch(
                        batch_chunks,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                all_samples.extend(generated)

                for i in range(batch_idx, batch_idx + len(batch_chunks)):
                    completed_ids.add(i)

                self._save_checkpoint(checkpoint_file, completed_ids, all_samples)

            except BadRequestError as e:
                # 400 拒绝（如内容过滤）属提示词级确定性错误，重试同一批无效：
                # 将该批 chunk 标记为已完成以永久跳过，保存检查点后继续后续批次
                logger.warning(
                    f"批次 {batch_num} 被 API 拒绝 (400 BadRequest，疑似内容过滤)，"
                    f"跳过 {len(batch_chunks)} 条 chunk 并继续: {e}"
                )
                for i in range(batch_idx, batch_idx + len(batch_chunks)):
                    completed_ids.add(i)
                self._save_checkpoint(checkpoint_file, completed_ids, all_samples)
                continue
            except Exception as e:
                logger.error(f"批次 {batch_num} 失败: {e}")
                self._save_checkpoint(checkpoint_file, completed_ids, all_samples)
                raise

        # V2: 质量门 (清洗 think 块 / 长度过滤 / 去重)
        gate_stats = None
        if dedup or min_output_chars > 0:
            all_samples, gate_stats = apply_v2_quality_gates(
                all_samples, min_output_chars=min_output_chars
            )
            logger.info(f"质量门: {gate_stats}")

        # 按子集分组
        grouped: Dict[str, List[Dict[str, str]]] = {s: [] for s in VALID_SUBSETS}
        for s in all_samples:
            grouped[s["subset"]].append({
                "instruction": s["instruction"],
                "input": s["input"],
                "output": s["output"],
            })

        stats = {
            "total_chunks": len(all_chunks),
            "completed_chunks": len(completed_ids),
            "total_samples": len(all_samples),
            "subset_distribution": {k: len(v) for k, v in grouped.items()},
            "total_requests": self.total_requests,
            "total_tokens_used": self.total_tokens_used,
        }
        if gate_stats:
            stats["quality_gates"] = gate_stats
        if samples_per_chunk > 1:
            stats["samples_per_chunk"] = samples_per_chunk

        logger.info(f"生成完成: {stats['total_samples']} 条样本")
        return grouped, stats

    def _save_checkpoint(
        self,
        checkpoint_file: Path,
        completed_ids: Set[int],
        samples: List[Dict[str, str]],
    ):
        """保存检查点。"""
        checkpoint = {
            "completed_ids": sorted(list(completed_ids)),
            "samples": samples,
            "saved_at": datetime.now().isoformat(),
        }
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        logger.debug(f"检查点已保存: {checkpoint_file}")

    def save_subsets(self, grouped: Dict[str, List[Dict[str, str]]]) -> List[str]:
        """按子集保存为 JSON 文件到 output_dir。"""
        saved = []
        for subset_name, samples in grouped.items():
            if not samples:
                continue
            output_file = self.output_dir / f"{subset_name}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
            logger.info(f"[{subset_name}] 保存 {len(samples)} 条样本 → {output_file}")
            saved.append(str(output_file))
        return saved

    @staticmethod
    def estimate_cost(chunks_count: int, batch_size: int = 5, rpm: int = 3) -> Dict[str, Any]:
        """
        估算生成成本（基于 Moonshot v1-8k 定价，仅供参考）。
        """
        num_batches = (chunks_count + batch_size - 1) // batch_size
        # 每个请求约 3K 输入 tokens + 1.5K 输出 tokens
        est_input = num_batches * 3000
        est_output = num_batches * 1500

        input_cost = (est_input / 1000) * 0.006
        output_cost = (est_output / 1000) * 0.006
        total_cny = input_cost + output_cost

        return {
            "chunks_count": chunks_count,
            "batch_size": batch_size,
            "num_batches": num_batches,
            "estimated_input_tokens": est_input,
            "estimated_output_tokens": est_output,
            "estimated_cost_cny": round(total_cny, 2),
            "estimated_cost_usd": round(total_cny / 7.2, 2),
            "estimated_time_minutes": round(num_batches * (60.0 / rpm) / 60.0, 1),
            "rpm": rpm,
        }
