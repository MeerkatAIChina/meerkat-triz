"""
评测工具函数集
支持：Layer 1 通用能力基准、Layer 2 TRIZ定制评测、Layer 3 工程性能基准
"""

import json
import time
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== Layer 1: 通用能力基准 ====================

def run_lm_evaluation(
    model_path: str,
    tasks: List[str],
    output_dir: str,
    num_fewshot: Optional[int] = None,
    batch_size: int = 1,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    使用 lm-eval-harness 运行通用能力评测
    
    Args:
        model_path: 模型路径或HuggingFace ID
        tasks: 评测任务列表，如 ["mmlu_pro", "gpqa", "humaneval", "math", "bbh"]
        output_dir: 结果输出目录
        num_fewshot: few-shot样本数
        batch_size: 批大小
        device: 运行设备
    
    Returns:
        评测结果字典
    """
    try:
        import lm_eval
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        logger.error("lm-eval 未安装，请先运行: pip install lm-eval>=0.4.3")
        raise
    
    logger.info(f"开始通用能力评测: {tasks}")
    logger.info(f"模型: {model_path}")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 配置模型
    model_args = f"pretrained={model_path},dtype=float16,device={device}"
    
    # 运行评测
    results = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        write_out=True,
        output_path=str(output_path),
    )
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_path / f"lm_eval_results_{timestamp}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"评测结果已保存: {result_file}")
    
    # 打印关键指标摘要
    print_evaluation_summary(results, tasks)
    
    return results


def print_evaluation_summary(results: Dict[str, Any], tasks: List[str]):
    """打印评测结果摘要"""
    print("\n" + "=" * 60)
    print("通用能力评测结果摘要")
    print("=" * 60)
    
    for task in tasks:
        if task in results.get("results", {}):
            task_results = results["results"][task]
            # 提取主要指标
            for key, value in task_results.items():
                if "acc" in key or "score" in key:
                    print(f"  {task:20s} | {key:30s} | {value:.4f}")
    
    print("=" * 60 + "\n")


# ==================== Layer 2: TRIZ定制评测 ====================

class TRIZBenchmark:
    """TRIZ领域定制评测器"""
    
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
        # 40个发明原理列表
        self.principles = [
            "Segmentation", "Taking out", "Local quality", "Asymmetry",
            "Merging", "Universality", "Nested doll", "Anti-weight",
            "Preliminary anti-action", "Preliminary action", "Beforehand cushioning",
            "Equipotentiality", "The other way round", "Curved", "Dynamics",
            "Partial or excessive actions", "Another dimension", "Mechanical vibration",
            "Periodic action", "Continuity of useful action", "Skipping",
            "Blessing in disguise", "Feedback", "Intermediary", "Self-service",
            "Copying", "Cheap short-living objects", "Mechanics substitution",
            "Pneumatics and hydraulics", "Flexible shells and thin films",
            "Porous materials", "Color changes", "Homogeneity",
            "Discarding and recovering", "Parameter changes", "Phase transitions",
            "Thermal expansion", "Strong oxidants", "Inert atmosphere", "Composite materials"
        ]
        
        # 评测数据集（示例问题）
        self.test_questions = self._load_test_questions()
    
    def _load_test_questions(self) -> List[Dict]:
        """加载TRIZ评测问题集"""
        return [
            {
                "category": "principle_identification",
                "question": "一个系统需要在不改变整体结构的情况下增加功能模块，应该使用哪个发明原理？",
                "expected": "Nested doll",
                "type": "multiple_choice"
            },
            {
                "category": "principle_identification", 
                "question": "为了提高系统的灵活性，使其能够适应不同条件，应该应用哪个原理？",
                "expected": "Dynamics",
                "type": "multiple_choice"
            },
            {
                "category": "contradiction_resolution",
                "question": "一个机械设备需要既坚固又轻便，存在什么技术矛盾？如何化解？",
                "expected_keywords": ["strength", "weight", "composite materials", "porous materials"],
                "type": "open_ended"
            },
            {
                "category": "contradiction_resolution",
                "question": "飞机起落架需要在起飞降落时使用，但在飞行中会增加阻力。这是什么矛盾？请给出TRIZ解决方案。",
                "expected_keywords": ["retractable", "taking out", "dynamics"],
                "type": "open_ended"
            },
            {
                "category": "case_generation",
                "question": "请使用TRIZ方法，为'如何在不增加成本的情况下提高产品质量'生成一个创新解决方案。",
                "expected_keywords": ["principle", "contradiction", "solution"],
                "type": "generation"
            },
        ]
    
    def evaluate_principle_accuracy(self) -> Dict[str, float]:
        """评测40个发明原理识别准确率"""
        logger.info("评测: 发明原理识别准确率")
        
        correct = 0
        total = len(self.test_questions)
        
        for q in self.test_questions:
            if q["type"] == "multiple_choice":
                # 构建prompt
                prompt = self._build_prompt(q["question"])
                response = self._generate_response(prompt)
                
                # 检查是否包含正确答案
                if q["expected"].lower() in response.lower():
                    correct += 1
                    logger.debug(f"正确: {q['question'][:50]}...")
                else:
                    logger.debug(f"错误: 期望 '{q['expected']}', 得到 '{response[:100]}'")
        
        accuracy = correct / total if total > 0 else 0
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
    
    def evaluate_contradiction_resolution(self) -> Dict[str, float]:
        """评测矛盾解决能力"""
        logger.info("评测: 矛盾解决能力")
        
        scores = []
        
        for q in self.test_questions:
            if q["type"] == "open_ended":
                prompt = self._build_prompt(q["question"])
                response = self._generate_response(prompt)
                
                # 检查关键词覆盖
                keywords = q.get("expected_keywords", [])
                matched = sum(1 for kw in keywords if kw.lower() in response.lower())
                score = matched / len(keywords) if keywords else 0
                scores.append(score)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        return {
            "average_score": avg_score,
            "individual_scores": scores
        }
    
    def evaluate_case_quality(self) -> Dict[str, Any]:
        """评测创新案例生成质量 (使用BLEU/ROUGE)"""
        logger.info("评测: 案例生成质量")
        
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        except ImportError:
            logger.warning("rouge_score 未安装，跳过ROUGE评测")
            scorer = None
        
        rouge_scores = []
        
        for q in self.test_questions:
            if q["type"] == "generation":
                prompt = self._build_prompt(q["question"])
                response = self._generate_response(prompt)
                
                # 简单的关键词匹配作为质量指标
                keywords = q.get("expected_keywords", [])
                matched = sum(1 for kw in keywords if kw.lower() in response.lower())
                coverage = matched / len(keywords) if keywords else 0
                
                rouge_scores.append({
                    "coverage": coverage,
                    "response_length": len(response),
                })
        
        avg_coverage = sum(s["coverage"] for s in rouge_scores) / len(rouge_scores) if rouge_scores else 0
        return {
            "average_coverage": avg_coverage,
            "details": rouge_scores
        }
    
    def evaluate_ariz_completeness(self) -> Dict[str, float]:
        """评测ARIZ步骤完整性"""
        logger.info("评测: ARIZ步骤完整性")
        
        ariz_steps = [
            "problem analysis",
            "problem model",
            "ideal final result",
            "contradiction analysis", 
            "resource analysis",
            "solution evaluation"
        ]
        
        prompt = self._build_prompt(
            "请详细描述使用ARIZ算法解决一个技术问题的完整步骤。"
        )
        response = self._generate_response(prompt)
        
        # 检查ARIZ步骤覆盖
        matched_steps = sum(1 for step in ariz_steps if step.lower() in response.lower())
        completeness = matched_steps / len(ariz_steps)
        
        return {
            "completeness": completeness,
            "matched_steps": matched_steps,
            "total_steps": len(ariz_steps)
        }
    
    def run_all_evaluations(self) -> Dict[str, Any]:
        """运行全部TRIZ评测"""
        logger.info("开始TRIZ定制评测...")
        
        results = {
            "principle_accuracy": self.evaluate_principle_accuracy(),
            "contradiction_resolution": self.evaluate_contradiction_resolution(),
            "case_quality": self.evaluate_case_quality(),
            "ariz_completeness": self.evaluate_ariz_completeness(),
        }
        
        # 计算综合得分
        total_score = (
            results["principle_accuracy"]["accuracy"] * 0.3 +
            results["contradiction_resolution"]["average_score"] * 0.3 +
            results["case_quality"]["average_coverage"] * 0.2 +
            results["ariz_completeness"]["completeness"] * 0.2
        )
        results["overall_score"] = total_score
        
        logger.info(f"TRIZ综合得分: {total_score:.4f}")
        return results
    
    def _build_prompt(self, question: str) -> str:
        """构建评测prompt"""
        system_msg = (
            "You are TRIZ-Expert, a specialized AI for TRIZ (Theory of Inventive Problem Solving). "
            "Answer the following question with professional TRIZ knowledge."
        )
        return f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
    def _generate_response(self, prompt: str, max_new_tokens: int = 512) -> str:
        """生成模型回复"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 移除prompt部分
        response = response[len(prompt):].strip()
        return response


def run_triz_evaluation(
    model,
    tokenizer,
    output_dir: str,
    test_data_path: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    运行TRIZ定制评测的便捷函数

    Args:
        model: 已加载的模型
        tokenizer: 对应的分词器
        output_dir: 结果输出目录
        test_data_path: 测试数据路径 (可选，用于加载外部评测数据)
        max_new_tokens: 最大生成token数
        temperature: 生成温度

    Returns:
        TRIZ评测结果
    """
    benchmark = TRIZBenchmark(model, tokenizer)
    results = benchmark.run_all_evaluations()
    
    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_path / f"triz_eval_results_{timestamp}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"TRIZ评测结果已保存: {result_file}")
    return results


# ==================== Layer 3: 工程性能基准 ====================

def run_performance_benchmark(
    model,
    tokenizer,
    output_dir: str,
    test_prompts: Optional[List[str]] = None,
    max_tokens: int = 512
) -> Dict[str, Any]:
    """
    运行工程性能评测（吞吐量、延迟、内存）
    
    Args:
        model: 已加载的模型
        tokenizer: 对应的分词器
        output_dir: 结果输出目录
        test_prompts: 测试prompt列表
        max_tokens: 最大生成token数
    
    Returns:
        性能评测结果
    """
    logger.info("开始工程性能评测...")
    
    if test_prompts is None:
        test_prompts = [
            "请解释TRIZ的发明原理1（分割原理）及其应用场景。",
            "分析以下技术矛盾：汽车需要既安全又轻便。",
            "使用ARIZ算法指导解决'如何提高打印机墨水利用率'的问题。",
        ]
    
    model.eval()
    device = next(model.parameters()).device
    
    # 预热
    logger.info("模型预热中...")
    warmup_prompt = "Hello, this is a warmup prompt."
    inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=50)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # 性能测试
    latencies = []
    token_counts = []
    memory_before = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
    
    for i, prompt in enumerate(test_prompts):
        logger.info(f"性能测试 {i+1}/{len(test_prompts)}: {prompt[:40]}...")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_tokens = inputs["input_ids"].shape[1]
        
        # 计时
        start_time = time.time()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        output_tokens = outputs.shape[1] - input_tokens
        tokens_per_sec = output_tokens / (end_time - start_time)
        
        latencies.append(latency_ms)
        token_counts.append(output_tokens)
        
        logger.info(f"  延迟: {latency_ms:.1f}ms | 生成token: {output_tokens} | 吞吐量: {tokens_per_sec:.1f} tokens/s")
    
    # 内存使用
    memory_after = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
    memory_peak = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
    
    results = {
        "latency_p50_ms": sorted(latencies)[len(latencies)//2],
        "latency_avg_ms": sum(latencies) / len(latencies),
        "throughput_tokens_per_sec": sum(token_counts) / sum([l/1000 for l in latencies]),
        "memory_allocated_gb": memory_after,
        "memory_peak_gb": memory_peak,
        "test_count": len(test_prompts),
        "max_tokens": max_tokens,
        "device": str(device),
    }
    
    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_path / f"perf_benchmark_{timestamp}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"性能评测结果已保存: {result_file}")
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("工程性能评测摘要")
    print("=" * 60)
    print(f"  P50延迟:        {results['latency_p50_ms']:.1f} ms")
    print(f"  平均延迟:       {results['latency_avg_ms']:.1f} ms")
    print(f"  吞吐量:         {results['throughput_tokens_per_sec']:.1f} tokens/s")
    print(f"  峰值内存:       {results['memory_peak_gb']:.1f} GB")
    print(f"  设备:           {results['device']}")
    print("=" * 60 + "\n")
    
    return results


# ==================== 结果聚合 ====================

def aggregate_results(
    general_results: Optional[Dict] = None,
    triz_results: Optional[Dict] = None,
    perf_results: Optional[Dict] = None,
    output_dir: str = "./results"
) -> Dict[str, Any]:
    """
    聚合三层评测结果为综合报告
    
    Returns:
        综合评测报告
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {},
        "layer1_general": general_results or {},
        "layer2_triz": triz_results or {},
        "layer3_performance": perf_results or {},
    }
    
    # 计算综合评分
    scores = []
    
    if triz_results:
        triz_score = triz_results.get("overall_score", 0) * 100
        scores.append(triz_score)
        report["summary"]["triz_score"] = f"{triz_score:.1f}/100"
    
    if perf_results:
        # 性能评分 (吞吐量为主要指标)
        throughput = perf_results.get("throughput_tokens_per_sec", 0)
        perf_score = min(throughput / 2, 100)  # 100 tokens/s = 100分
        scores.append(perf_score)
        report["summary"]["performance_score"] = f"{perf_score:.1f}/100"
    
    if scores:
        report["summary"]["overall_score"] = f"{sum(scores)/len(scores):.1f}/100"
    
    # 保存报告
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_file = output_path / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"综合报告已保存: {report_file}")
    return report
