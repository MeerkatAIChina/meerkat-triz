"""
合成数据生成流水线
使用Moonshot API通过OpenAI兼容接口生成语义多样化的TRIZ训练数据
支持：批量生成、速率限制、检查点恢复、输出验证
"""

import json
import logging
import os
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime

import numpy as np
from openai import OpenAI, RateLimitError, APIError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 生成策略提示词 ====================

STRATEGY_PROMPTS = {
    "rephrase": """你是一个TRIZ领域数据增强专家。请基于以下种子样本，改写每个问题的表述方式，生成新的训练样本。
要求：
1. 保持答案内容的事实准确性不变
2. 使用不同的提问角度、措辞风格或专业术语
3. 每个种子生成指定数量的变体
4. 输出必须是严格的JSON格式

对于每个生成的样本，格式如下：
{"instruction": "改写后的问题", "input": "", "output": "与种子相同的答案"}""",

    "generate_new": """你是一个TRIZ领域数据生成专家。请基于以下种子样本作为灵感，生成全新的TRIZ训练问答对。
要求：
1. 不要复制种子的具体问题或答案
2. 创造全新的场景、案例或问题
3. 保持TRIZ方法论的专业性和准确性
4. 输出必须是严格的JSON格式

对于每个生成的样本，格式如下：
{"instruction": "全新的TRIZ问题", "input": "", "output": "专业的TRIZ回答"}""",

    "mixed": """你是一个TRIZ领域数据增强专家。请基于以下种子样本，同时进行两种操作：
1. 改写部分种子的提问方式（保持答案不变）
2. 基于部分种子的主题，生成全新的问答对
要求：
- 改写和全新生成的比例约为1:1
- 所有输出必须保持TRIZ专业性
- 输出必须是严格的JSON格式

对于每个生成的样本，格式如下：
{"instruction": "问题", "input": "", "output": "回答"}""",
}


# ==================== 质量关卡函数 ====================

def compute_perplexity(
    text: str,
    model,
    tokenizer,
    device: Optional[str] = None,
    max_length: int = 2048,
) -> float:
    """
    使用基座模型计算文本的困惑度 (Perplexity)

    Args:
        text: 输入文本
        model: 已加载的语言模型 (需要支持 forward + labels)
        tokenizer: 分词器
        device: 计算设备 (None则自动选择)
        max_length: 最大截断长度

    Returns:
        困惑度值 (越低表示模型对文本越"熟悉")
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        encodings = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = encodings.input_ids.to(device)

        # 将模型移到对应设备
        model_device = next(model.parameters()).device
        if str(model_device) != device:
            model = model.to(device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

        perplexity = torch.exp(loss).item()
        return perplexity

    except Exception as e:
        logger.warning(f"困惑度计算失败: {e}")
        return float("inf")  # 失败时返回无穷大，不过滤


def filter_by_perplexity(
    samples: List[Dict[str, str]],
    model,
    tokenizer,
    percentile: int = 80,
    device: Optional[str] = None,
) -> Tuple[List[Dict[str, str]], float]:
    """
    按困惑度过滤样本，保留困惑度较低的样本

    Args:
        samples: 样本列表
        model: 已加载的语言模型 (可选，为None则跳过)
        tokenizer: 分词器
        percentile: 保留的百分位数 (默认80，保留困惑度最低的80%)
        device: 计算设备

    Returns:
        (过滤后的样本列表, 困惑度阈值)
    """
    if model is None or tokenizer is None:
        logger.info("困惑度过滤: 模型未加载，跳过")
        return samples, float("inf")

    logger.info(f"困惑度过滤: 计算 {len(samples)} 个样本的困惑度...")

    perplexities = []
    for sample in samples:
        text = f"{sample.get('instruction', '')}\n{sample.get('input', '')}\n{sample.get('output', '')}"
        ppl = compute_perplexity(text, model, tokenizer, device)
        perplexities.append(ppl)

    # 计算阈值 (percentile百分位数)
    threshold = np.percentile(perplexities, percentile)

    filtered = []
    removed = 0
    for sample, ppl in zip(samples, perplexities):
        if ppl <= threshold:
            sample["_perplexity"] = round(ppl, 2)
            filtered.append(sample)
        else:
            removed += 1

    logger.info(f"困惑度过滤完成: 保留 {len(filtered)}/{len(samples)} 个样本 "
               f"(阈值={threshold:.2f}, 移除 {removed} 个)")
    return filtered, threshold


def compute_diversity_score(
    samples: List[Dict[str, str]],
    n: int = 2,
    field: str = "instruction",
) -> float:
    """
    计算样本集的n-gram多样性分数 (Distinct-n)

    使用纯文本处理，不需要加载模型。
    Distinct-n = 唯一n-gram数 / 总n-gram数
    值域 [0, 1]，越高表示多样性越好。

    Args:
        samples: 样本列表
        n: n-gram长度 (1=unigram, 2=bigram)
        field: 用于计算的字段 (instruction/output)

    Returns:
        多样性分数 [0, 1]
    """
    all_ngrams = []
    for sample in samples:
        text = sample.get(field, "")
        # 简单分词: 按字符和空格分割 (适用于中英文混合)
        tokens = []
        for char in text:
            if char.strip():
                tokens.append(char)
        # 生成n-gram
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return 0.0

    return len(set(all_ngrams)) / len(all_ngrams)


def filter_by_diversity(
    samples: List[Dict[str, str]],
    min_distinct_1: float = 0.3,
    min_distinct_2: float = 0.15,
    field: str = "instruction",
) -> Tuple[List[Dict[str, str]], Dict[str, float]]:
    """
    按多样性过滤样本，移除与其他样本过于相似的重复项

    策略: 计算每个子集的distinct-1和distinct-2，如果低于阈值，
    则按instruction哈希去重，保留第一个出现的样本。

    Args:
        samples: 样本列表
        min_distinct_1: 最低unigram多样性阈值
        min_distinct_2: 最低bigram多样性阈值
        field: 用于计算的字段

    Returns:
        (过滤后的样本列表, 多样性统计字典)
    """
    if not samples:
        return [], {"distinct_1": 0.0, "distinct_2": 0.0}

    # 计算整体多样性
    d1 = compute_diversity_score(samples, n=1, field=field)
    d2 = compute_diversity_score(samples, n=2, field=field)

    logger.info(f"多样性评分: distinct-1={d1:.3f}, distinct-2={d2:.3f} "
               f"(阈值: {min_distinct_1}/{min_distinct_2})")

    stats = {"distinct_1": round(d1, 3), "distinct_2": round(d2, 3)}

    # 如果多样性已达标，直接返回
    if d1 >= min_distinct_1 and d2 >= min_distinct_2:
        return samples, stats

    # 多样性不足，进行instruction级别的去重
    logger.warning(f"多样性不足，执行instruction去重...")
    seen = set()
    unique = []
    for s in samples:
        instr = s.get(field, "").strip()
        # 简化: 取前30个字符作为指纹
        fingerprint = instr[:30]
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(s)

    removed = len(samples) - len(unique)
    if removed > 0:
        logger.info(f"去重后: {len(unique)}/{len(samples)} 个样本 (移除 {removed} 个重复)")
        # 重新计算多样性
        d1 = compute_diversity_score(unique, n=1, field=field)
        d2 = compute_diversity_score(unique, n=2, field=field)
        stats = {"distinct_1": round(d1, 3), "distinct_2": round(d2, 3)}

    return unique, stats


class MoonshotSyntheticClient:
    """Moonshot API客户端（OpenAI兼容）"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "moonshot-v1-8k",
        rpm: int = 3,
        base_url: str = "https://api.moonshot.cn/v1",
    ):
        self.api_key = api_key or os.environ.get("MOONSHOT_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Moonshot API key未设置。请设置环境变量 MOONSHOT_API_KEY "
                "或在初始化时传入api_key参数。"
            )

        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        self.model = model
        self.rpm = rpm
        self.min_interval = 60.0 / rpm
        self.last_request_time = 0.0
        self.total_tokens_used = 0
        self.total_requests = 0

    def _rate_limit_sleep(self):
        """根据RPM限制进行睡眠"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            logger.debug(f"速率限制: 等待 {sleep_time:.1f} 秒")
            time.sleep(sleep_time)

    def generate_variations(
        self,
        seeds: List[Dict[str, str]],
        strategy: str,
        subset_name: str,
        num_variations: int = 5,
        max_tokens: int = 2000,
        temperature: float = 0.8,
    ) -> List[Dict[str, str]]:
        """
        为一批种子样本生成合成变体

        Args:
            seeds: 种子样本列表，每个样本包含instruction, input, output
            strategy: 生成策略 (rephrase/generate_new/mixed)
            subset_name: 子集名称（用于日志）
            num_variations: 每个种子生成的变体数量
            max_tokens: 最大生成token数
            temperature: 采样温度

        Returns:
            生成的样本列表
        """
        if strategy not in STRATEGY_PROMPTS:
            raise ValueError(f"未知策略: {strategy}。可选: {list(STRATEGY_PROMPTS.keys())}")

        self._rate_limit_sleep()

        # 构建包含种子的提示词
        seeds_text = self._format_seeds_for_prompt(seeds, num_variations)
        system_prompt = STRATEGY_PROMPTS[strategy]
        user_prompt = f"子集: {subset_name}\n\n种子样本:\n{seeds_text}\n\n请生成JSON格式的输出。"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
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
            return self._parse_response(content, len(seeds), num_variations)

        except RateLimitError as e:
            logger.warning(f"触发速率限制: {e}。等待60秒后重试...")
            time.sleep(60)
            return self.generate_variations(
                seeds, strategy, subset_name, num_variations, max_tokens, temperature
            )
        except APIError as e:
            logger.error(f"API错误: {e}")
            raise

    def _format_seeds_for_prompt(
        self, seeds: List[Dict[str, str]], num_variations: int
    ) -> str:
        """将种子格式化为提示词文本"""
        lines = [f"共 {len(seeds)} 个种子，每个种子需要生成 {num_variations} 个变体。\n"]
        for i, seed in enumerate(seeds, 1):
            lines.append(f"种子 {i}:")
            lines.append(f"  问题: {seed.get('instruction', '')}")
            if seed.get('input'):
                lines.append(f"  输入: {seed['input']}")
            lines.append(f"  答案: {seed.get('output', '')[:200]}...")
            lines.append("")
        return "\n".join(lines)

    def _parse_response(
        self, content: str, seed_count: int, num_variations: int
    ) -> List[Dict[str, str]]:
        """解析API响应为样本列表"""
        samples = []

        # 尝试JSON解析
        try:
            # 有时模型返回JSON数组，有时返回JSONL
            content = content.strip()
            if content.startswith("["):
                data = json.loads(content)
                if isinstance(data, list):
                    samples = data
            elif content.startswith("{"):
                # 可能是单个JSON对象或JSONL
                for line in content.split("\n"):
                    line = line.strip()
                    if line:
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                samples.append(obj)
                        except json.JSONDecodeError:
                            pass
        except json.JSONDecodeError:
            logger.warning("JSON解析失败，尝试正则提取")

        # 验证每个样本
        validated = []
        for s in samples:
            if isinstance(s, dict) and "instruction" in s and "output" in s:
                validated.append({
                    "instruction": s["instruction"],
                    "input": s.get("input", ""),
                    "output": s["output"],
                })

        logger.info(f"解析结果: {len(validated)}/{len(samples)} 个有效样本")
        return validated

    def estimate_cost(self, seed_count: int, batch_size: int = 5) -> Dict[str, Any]:
        """
        估算生成成本

        Moonshot v1-8k 定价 (2026-05-27):
        - 输入: ￥0.006 / 1K tokens
        - 输出: ￥0.006 / 1K tokens
        """
        num_batches = (seed_count + batch_size - 1) // batch_size
        # 估算：每个请求约 2K 输入 tokens + 1.5K 输出 tokens
        est_input_tokens = num_batches * 2000
        est_output_tokens = num_batches * 1500

        # 人民币估算 (￥0.006 / 1K tokens)
        input_cost = (est_input_tokens / 1000) * 0.006
        output_cost = (est_output_tokens / 1000) * 0.006
        total_cost_cny = input_cost + output_cost

        # 时间估算 (按RPM)
        est_time_minutes = num_batches * (60.0 / self.rpm) / 60.0

        return {
            "seed_count": seed_count,
            "batch_size": batch_size,
            "num_batches": num_batches,
            "estimated_input_tokens": est_input_tokens,
            "estimated_output_tokens": est_output_tokens,
            "estimated_cost_cny": round(total_cost_cny, 2),
            "estimated_cost_usd": round(total_cost_cny / 7.2, 2),
            "estimated_time_minutes": round(est_time_minutes, 1),
            "rpm": self.rpm,
        }


class SyntheticPipeline:
    """合成数据生成流水线，支持检查点恢复"""

    def __init__(
        self,
        client: MoonshotSyntheticClient,
        output_dir: str = "/home/meerkat/mongoose_ai/data/processed/synthetic",
        checkpoint_dir: str = "/home/meerkat/mongoose_ai/data/processed/checkpoint",
    ):
        self.client = client
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def deduplicate_seeds(self, seeds: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """按(instruction, output)哈希去重种子数据"""
        seen = set()
        unique = []
        for s in seeds:
            key = hashlib.md5(
                f"{s.get('instruction', '')}|{s.get('output', '')}".encode()
            ).hexdigest()
            if key not in seen:
                seen.add(key)
                unique.append(s)

        removed = len(seeds) - len(unique)
        if removed > 0:
            logger.info(f"去重: 移除 {removed} 个重复种子，保留 {len(unique)} 个唯一种子")
        return unique

    def generate_subset(
        self,
        subset_name: str,
        seeds: List[Dict[str, str]],
        strategy: str,
        multiplier: int = 10,
        batch_size: int = 5,
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        为一个子集生成合成数据，支持检查点恢复

        Args:
            subset_name: 子集名称
            seeds: 种子样本列表
            strategy: 生成策略
            multiplier: 每个种子的扩展倍数
            batch_size: 每个API请求的种子数

        Returns:
            (生成的样本列表, 统计信息字典)
        """
        # 去重
        seeds = self.deduplicate_seeds(seeds)

        # 计算每个种子需要生成的变体数
        variations_per_seed = max(1, multiplier - 1)  # -1 because we keep original

        # 加载检查点
        checkpoint_file = self.checkpoint_dir / f"{subset_name}_checkpoint.json"
        completed_ids: Set[int] = set()
        results: List[Dict[str, str]] = []

        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    checkpoint = json.load(f)
                completed_ids = set(checkpoint.get("completed_ids", []))
                results = checkpoint.get("results", [])
                logger.info(
                    f"[{subset_name}] 从检查点恢复: "
                    f"{len(completed_ids)}/{len(seeds)} 个种子已完成"
                )
            except Exception as e:
                logger.warning(f"加载检查点失败: {e}，从头开始")

        # 过滤已完成的种子
        remaining = [s for i, s in enumerate(seeds) if i not in completed_ids]

        if not remaining:
            logger.info(f"[{subset_name}] 所有种子已完成，跳过生成")
            return results, self._build_stats(subset_name, seeds, results, True)

        # 显示成本估算
        cost_estimate = self.client.estimate_cost(len(remaining), batch_size)
        logger.info(f"[{subset_name}] 成本估算: ￥{cost_estimate['estimated_cost_cny']}, "
                   f"预计 {cost_estimate['estimated_time_minutes']} 分钟")

        # 批量生成
        total_batches = (len(remaining) + batch_size - 1) // batch_size
        for batch_idx in range(0, len(remaining), batch_size):
            batch = remaining[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1

            logger.info(
                f"[{subset_name}] 批次 {batch_num}/{total_batches}: "
                f"处理 {len(batch)} 个种子"
            )

            try:
                generated = self.client.generate_variations(
                    seeds=batch,
                    strategy=strategy,
                    subset_name=subset_name,
                    num_variations=variations_per_seed,
                )

                # 将原始种子也加入结果
                for s in batch:
                    results.append({
                        "instruction": s["instruction"],
                        "input": s.get("input", ""),
                        "output": s["output"],
                        "source": "seed",
                        "subset": subset_name,
                    })

                # 添加生成的样本
                for g in generated:
                    g["source"] = "synthetic"
                    g["subset"] = subset_name
                    results.append(g)

                # 标记完成
                for s in batch:
                    original_idx = seeds.index(s)
                    completed_ids.add(original_idx)

                # 保存检查点
                self._save_checkpoint(checkpoint_file, completed_ids, results)

            except Exception as e:
                logger.error(f"[{subset_name}] 批次 {batch_num} 失败: {e}")
                self._save_checkpoint(checkpoint_file, completed_ids, results)
                raise

        stats = self._build_stats(subset_name, seeds, results, True)
        logger.info(f"[{subset_name}] 生成完成: {stats['total_samples']} 条样本")
        return results, stats

    def _save_checkpoint(
        self, checkpoint_file: Path, completed_ids: Set[int], results: List[Dict]
    ):
        """保存检查点"""
        checkpoint = {
            "completed_ids": sorted(list(completed_ids)),
            "results": results,
            "saved_at": datetime.now().isoformat(),
        }
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        logger.debug(f"检查点已保存: {checkpoint_file}")

    def _build_stats(
        self,
        subset_name: str,
        seeds: List[Dict],
        results: List[Dict],
        completed: bool,
    ) -> Dict[str, Any]:
        """构建统计信息"""
        seed_count = len([r for r in results if r.get("source") == "seed"])
        synthetic_count = len([r for r in results if r.get("source") == "synthetic"])
        return {
            "subset": subset_name,
            "total_samples": len(results),
            "seed_samples": seed_count,
            "synthetic_samples": synthetic_count,
            "seed_count": len(seeds),
            "completed": completed,
        }

    def save_subset(self, subset_name: str, samples: List[Dict[str, str]]):
        """保存子集结果到JSON文件"""
        output_file = self.output_dir / f"{subset_name}_synthetic.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        logger.info(f"[{subset_name}] 保存 {len(samples)} 条样本到 {output_file}")
        return str(output_file)
