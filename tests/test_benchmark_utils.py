"""
Tests for benchmark_utils.py
Covers deterministic evaluation, Chinese keyword matching, and keyword helper logic.

Note: benchmark_utils.py imports torch at module level, which can fail in local
macOS test environments due to OpenMP conflicts. We mock torch (and numpy) before
importing the module under test.
"""

import sys
from unittest.mock import Mock, patch
import pathlib

# Prevent config.py from trying to create /home/meerkat directories at import time.
_original_mkdir = pathlib.Path.mkdir
pathlib.Path.mkdir = lambda self, *args, **kwargs: None

try:
    # Mock torch, numpy, and datasets before importing benchmark_utils.
    # patch.dict scopes the injection to this import block only — leaving the
    # mocks in sys.modules permanently made suite results order-dependent.
    _mock_torch = Mock()
    _mock_torch.tensor = Mock(return_value=Mock())
    _mock_torch.cuda = Mock()
    _mock_torch.cuda.memory_allocated = Mock(return_value=0)
    _mock_torch.cuda.max_memory_allocated = Mock(return_value=0)
    _mock_torch.cuda.is_available = Mock(return_value=False)
    _mock_torch.cuda.synchronize = Mock()
    _mock_torch.no_grad = Mock(return_value=Mock(__enter__=Mock(return_value=None), __exit__=Mock(return_value=None)))

    # Mock config before data_utils imports it
    _mock_config = Mock()
    _mock_config.DATA_CONFIG = {
        "chatml": {
            "system_message": "You are a TRIZ expert.",
            "max_length": 4096,
        }
    }

    sys.path.insert(0, ".")

    import importlib.util
    with patch.dict(sys.modules, {
        "torch": _mock_torch,
        "numpy": Mock(),
        "datasets": Mock(),
        "config": _mock_config,
    }):
        spec = importlib.util.spec_from_file_location(
            "benchmark_utils", "utils/benchmark_utils.py"
        )
        benchmark_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(benchmark_utils)
finally:
    pathlib.Path.mkdir = _original_mkdir

import pytest


class TestKeywordMatching:
    """Tests for the pure keyword matching helper."""

    def test_check_keywords_english_only(self):
        response = "This answer mentions problem analysis and ideal final result."
        keywords = ["problem analysis", "ideal final result"]
        matched, total = benchmark_utils._check_keywords(response, keywords)
        assert matched == 2
        assert total == 2

    def test_check_keywords_chinese_translation(self):
        response = "首先进行问题分析，然后确定理想最终解。"
        keywords = ["problem analysis", "ideal final result"]
        keyword_map = {
            "problem analysis": ["问题分析"],
            "ideal final result": ["理想最终解"],
        }
        matched, total = benchmark_utils._check_keywords(response, keywords, keyword_map)
        assert matched == 2
        assert total == 2

    def test_check_keywords_case_insensitive(self):
        response = "Problem Analysis is important."
        keywords = ["problem analysis"]
        matched, total = benchmark_utils._check_keywords(response, keywords)
        assert matched == 1

    def test_check_keywords_partial_no_match(self):
        response = "We should analyze the problem."
        keywords = ["problem analysis"]
        matched, total = benchmark_utils._check_keywords(response, keywords)
        assert matched == 0
    def test_check_keywords_chinese_and_english_consistency(self):
        """中英文关键词对等价回复应给出相同匹配数。"""
        english_response = (
            "ARIZ steps include problem analysis, problem model, ideal final result, "
            "contradiction analysis, resource analysis, and solution evaluation."
        )
        chinese_response = (
            "ARIZ步骤包括：问题分析、问题模型、理想最终解、"
            "矛盾分析、资源分析和方案评估。"
        )
        keywords = ["problem analysis", "problem model", "ideal final result",
                    "contradiction analysis", "resource analysis", "solution evaluation"]
        keyword_map = {
            "problem analysis": ["问题分析"],
            "problem model": ["问题模型"],
            "ideal final result": ["理想最终解"],
            "contradiction analysis": ["矛盾分析"],
            "resource analysis": ["资源分析"],
            "solution evaluation": ["方案评估"],
        }
        en_matched, en_total = benchmark_utils._check_keywords(english_response, keywords, keyword_map)
        zh_matched, zh_total = benchmark_utils._check_keywords(chinese_response, keywords, keyword_map)
        assert en_matched == zh_matched
        assert en_total == zh_total
        assert en_matched == 6


class TestTRIZBenchmarkKeywordMaps:
    """Tests that keyword maps support Chinese responses."""

    def test_ariz_step_map_has_chinese_translations(self):
        assert hasattr(benchmark_utils, "ARIZ_STEP_KEYWORD_MAP")
        mapping = benchmark_utils.ARIZ_STEP_KEYWORD_MAP
        assert "problem analysis" in mapping
        assert "ideal final result" in mapping
        for translations in mapping.values():
            assert isinstance(translations, list)
            assert len(translations) > 0

    def test_case_quality_keywords_include_chinese(self):
        assert hasattr(benchmark_utils, "CASE_QUALITY_KEYWORDS")
        keywords = benchmark_utils.CASE_QUALITY_KEYWORDS
        assert any("原理" in kw for kw in keywords)
        assert any("方案" in kw or "解决" in kw for kw in keywords)

    def test_principle_name_map_has_chinese(self):
        assert hasattr(benchmark_utils, "PRINCIPLE_NAME_MAP")
        mapping = benchmark_utils.PRINCIPLE_NAME_MAP
        assert "Nested doll" in mapping
        assert "Dynamics" in mapping
        assert any("嵌套" in t for t in mapping["Nested doll"])
        assert any("动态" in t for t in mapping["Dynamics"])


class TestTRIZBenchmarkScoring:
    """Tests for scoring methods with mocked responses."""

    def _make_benchmark(self, response_text):
        tokenizer = Mock()
        tokenizer.apply_chat_template = Mock(return_value="<prompt>")
        tokenizer.eos_token_id = 0
        tokenizer.decode = Mock(return_value="<prompt>" + response_text)

        # Model.generate receives **inputs; inputs must be a dict-like Mock
        encoding = Mock()
        encoding.keys = Mock(return_value=["input_ids"])
        encoding.__getitem__ = Mock(return_value=Mock())
        encoding.to = Mock(return_value=encoding)
        tokenizer.return_value = encoding

        model = Mock()
        model.generate = Mock(return_value=[Mock()])

        benchmark = benchmark_utils.TRIZBenchmark(model, tokenizer, device="cpu")
        # Bypass data_utils import chain (which pulls transformers/training_utils)
        benchmark._build_prompt = Mock(return_value="<prompt>")
        return benchmark

    def test_ariz_completeness_scores_chinese_response(self):
        response = (
            "ARIZ步骤包括：问题分析、问题模型、理想最终解、"
            "矛盾分析、资源分析和方案评估。"
        )
        benchmark = self._make_benchmark(response)
        result = benchmark.evaluate_ariz_completeness()
        assert result["completeness"] > 0
        assert result["matched_steps"] >= 4

    def test_case_quality_scores_chinese_response(self):
        response = (
            "基于TRIZ原理，本方案采用分割原理和动态化原理，"
            "提供了一个创新的解决方案。"
        )
        benchmark = self._make_benchmark(response)
        result = benchmark.evaluate_case_quality()
        assert result["average_coverage"] > 0

    def test_contradiction_resolution_scores_chinese_response(self):
        response = (
            "这个技术矛盾可以通过复合材料原理和多孔材料原理解决，"
            "使用碳纤维和泡沫金属。"
        )
        benchmark = self._make_benchmark(response)
        benchmark.test_questions = [
            {
                "category": "contradiction_resolution",
                "question": "如何既坚固又轻便？",
                "expected_keywords": ["strength", "weight", "composite materials", "porous materials"],
                "type": "open_ended",
            }
        ]
        result = benchmark.evaluate_contradiction_resolution()
        assert result["average_score"] > 0


class TestDeterministicGeneration:
    """Tests that temperature can be set to 0 for deterministic evaluation."""

    def test_default_temperature_is_zero(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template = Mock(return_value="<prompt>")
        tokenizer.eos_token_id = 0
        tokenizer.decode = Mock(return_value="response")

        encoding = Mock()
        encoding.keys = Mock(return_value=["input_ids"])
        encoding.__getitem__ = Mock(return_value=Mock())
        encoding.to = Mock(return_value=encoding)
        tokenizer.return_value = encoding

        model = Mock()
        model.generate = Mock(return_value=[Mock()])

        benchmark = benchmark_utils.TRIZBenchmark(model, tokenizer, device="cpu")
        assert benchmark.temperature == 0.0
        benchmark._generate_response("prompt")
        _, kwargs = model.generate.call_args
        assert kwargs["temperature"] == 0.0

    def test_run_triz_evaluation_default_temperature_is_zero(self):
        import inspect
        sig = inspect.signature(benchmark_utils.run_triz_evaluation)
        assert sig.parameters["temperature"].default == 0.0

    def test_temperature_propagates_to_generate(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template = Mock(return_value="<prompt>")
        tokenizer.eos_token_id = 0
        tokenizer.decode = Mock(return_value="response")

        encoding = Mock()
        encoding.keys = Mock(return_value=["input_ids"])
        encoding.__getitem__ = Mock(return_value=Mock())
        encoding.to = Mock(return_value=encoding)
        tokenizer.return_value = encoding

        model = Mock()
        model.generate = Mock(return_value=[Mock()])

        benchmark = benchmark_utils.TRIZBenchmark(model, tokenizer, device="cpu", temperature=0.3)
        benchmark._generate_response("prompt")
        _, kwargs = model.generate.call_args
        assert kwargs["temperature"] == 0.3


class TestAggregateResults:
    """Tests for aggregate_results report generation."""

    def test_aggregate_results_handles_missing_throughput(self):
        """Regression test: None/missing throughput should not crash scoring."""
        before_results = {
            "layer2_triz": {"overall_score": {"value": 0.30}},
            "layer3_performance": {"latency_p50_ms": 100.0},
        }
        after_results = {
            "layer2_triz": {"overall_score": {"value": 0.35}},
            "layer3_performance": {"latency_p50_ms": 90.0},
        }
        report = benchmark_utils.aggregate_results(
            before_results=before_results,
            after_results=after_results,
            output_dir="/tmp/test_eval",
        )
        assert "overall_score" in report["summary"]
        # performance_score should be absent because throughput is missing
        assert "performance_score" not in report["summary"]
        # overall should equal triz score only
        assert report["summary"]["overall_score"] == "35.0/100"

    def test_aggregate_results_computes_perf_score_with_throughput(self):
        before_results = {
            "layer2_triz": {"overall_score": {"value": 0.30}},
            "layer3_performance": {"throughput_tokens_per_sec": {"value": 50.0}},
        }
        after_results = {
            "layer2_triz": {"overall_score": {"value": 0.35}},
            "layer3_performance": {"throughput_tokens_per_sec": {"value": 60.0}},
        }
        report = benchmark_utils.aggregate_results(
            before_results=before_results,
            after_results=after_results,
            output_dir="/tmp/test_eval",
        )
        # perf_score = min(60 / 2, 100) = 30.0
        assert report["summary"]["performance_score"] == "30.0/100"
        # overall = (35.0 + 30.0) / 2 = 32.5
        assert report["summary"]["overall_score"] == "32.5/100"


def test_extract_layer1_metrics(layer1_results):
    """Layer 1 primary metric extraction prefers acc_norm over acc."""
    metrics = benchmark_utils._extract_layer1_metrics(layer1_results)
    assert isinstance(metrics, dict)
    assert "mmlu_pro" in metrics
    assert "gpqa" in metrics
    assert "humaneval" in metrics
    assert "math" in metrics
    assert "bbh" in metrics
    # acc_norm should be preferred over acc for mmlu_pro
    assert metrics["mmlu_pro"] == 0.42
    # humaneval uses pass_at_1
    assert metrics["humaneval"] == 0.25
    # Empty input returns empty dict
    assert benchmark_utils._extract_layer1_metrics({}) == {}


def test_aggregate_results_layer1_deltas():
    """aggregate_results produces Layer 1 delta metrics when before/after provided."""
    before_results = {
        "layer1_general": {"results": {"mmlu_pro": {"acc_norm": 0.40}}}
    }
    after_results = {
        "layer1_general": {"results": {"mmlu_pro": {"acc_norm": 0.42}}}
    }
    report = benchmark_utils.aggregate_results(
        before_results=before_results,
        after_results=after_results,
        output_dir="/tmp/test_eval",
    )
    assert report["layer1_general"]["source"] == "re-run_on_both_models"
    mmlu = report["layer1_general"]["metrics"]["mmlu_pro"]
    assert mmlu["before"] == 0.4
    assert mmlu["after"] == 0.42
    assert mmlu["delta"] == 0.02
    assert mmlu["delta_pct"] == 5.0


def test_compute_bleu_and_rouge_helpers():
    """EVAL-05 regression: BLEU/ROUGE helpers return numeric scores."""
    mock_bleu_result = Mock()
    mock_bleu_result.score = 12.34
    mock_bleu_result.signature = "mock"
    mock_sacrebleu = Mock()
    mock_sacrebleu.corpus_bleu = Mock(return_value=mock_bleu_result)

    mock_score = Mock()
    mock_score.fmeasure = 0.56
    mock_scorer = Mock()
    mock_scorer.score = Mock(return_value={
        "rouge1": mock_score,
        "rouge2": mock_score,
        "rougeL": mock_score,
    })
    mock_rouge_score = Mock()
    mock_rouge_score.rouge_scorer = Mock()
    mock_rouge_score.rouge_scorer.RougeScorer = Mock(return_value=mock_scorer)
    mock_jieba = Mock()
    mock_jieba.cut = Mock(side_effect=lambda x: iter(x.split()))

    with patch.dict(sys.modules, {
        "sacrebleu": mock_sacrebleu,
        "rouge_score": mock_rouge_score,
        "jieba": mock_jieba,
    }):
        bleu = benchmark_utils._compute_bleu(["一个测试预测"], ["一个测试参考"])
        assert isinstance(bleu, dict)
        assert "bleu" in bleu
        assert bleu["bleu"] >= 0

        rouge = benchmark_utils._compute_rouge(["一个测试预测"], ["一个测试参考"])
        assert isinstance(rouge, dict)
        assert "rouge1" in rouge
        assert "rouge2" in rouge
        assert "rougeL" in rouge
        assert rouge["rouge1"] >= 0
        assert rouge["rouge2"] >= 0
        assert rouge["rougeL"] >= 0


def test_evaluate_case_quality_calls_bleu_rouge():
    """EVAL-05 regression: TRIZBenchmark.evaluate_case_quality invokes helpers."""
    import inspect
    source = inspect.getsource(benchmark_utils.TRIZBenchmark.evaluate_case_quality)
    assert "_compute_bleu" in source
    assert "_compute_rouge" in source


def test_principle_accuracy_perfect_score_is_1():
    """分母 bug 回归：10 道选择题全对时 accuracy 必须等于 1.0。

    历史 bug（benchmark_utils.py 原理识别评分）：分母用全部题型题数而分子只
    统计 multiple_choice 题，导致即使选择题全对，accuracy 上限也只有 25-33%。
    """
    tokenizer = Mock()
    model = Mock()
    benchmark = benchmark_utils.TRIZBenchmark(model, tokenizer, device="cpu")

    mc_questions = [
        {
            "category": "principle_identification",
            "question": f"原理识别题 {i}",
            "expected": "Nested doll",
            "type": "multiple_choice",
        }
        for i in range(10)
    ]
    other_questions = [
        {
            "category": "contradiction_resolution",
            "question": "开放题",
            "expected_keywords": ["strength"],
            "type": "open_ended",
        }
        for _ in range(30)
    ]
    benchmark.test_questions = mc_questions + other_questions
    benchmark._build_prompt = Mock(side_effect=lambda q: q)
    benchmark._generate_response = Mock(return_value="Nested doll（嵌套原理）")

    result = benchmark.evaluate_principle_accuracy()
    assert result["correct"] == 10
    assert result["total"] == 10
    assert result["accuracy"] == 1.0


def test_evaluate_case_quality_pairs_predictions_and_references():
    """BLEU/ROUGE 对齐回归：只对带 reference 的 generation 题成对累计。

    历史 bug：predictions/references 长度判断恒为 False，BLEU/ROUGE 恒被跳过。
    """
    tokenizer = Mock()
    model = Mock()
    benchmark = benchmark_utils.TRIZBenchmark(model, tokenizer, device="cpu")
    benchmark.test_questions = [
        {
            "category": "case_generation",
            "question": "带参考的生成题",
            "reference": "参考答案",
            "expected_keywords": ["原理"],
            "type": "generation",
        },
        {
            "category": "case_generation",
            "question": "无参考的生成题",
            "expected_keywords": ["原理"],
            "type": "generation",
        },
    ]
    benchmark._build_prompt = Mock(side_effect=lambda q: q)
    benchmark._generate_response = Mock(return_value="基于分割原理的方案")

    captured = {}

    def _fake_bleu(predictions, references):
        captured["bleu"] = (list(predictions), list(references))
        return {"bleu": 10.0}

    def _fake_rouge(predictions, references):
        captured["rouge"] = (list(predictions), list(references))
        return {"rouge1": 0.5, "rouge2": 0.4, "rougeL": 0.45}

    with patch.object(benchmark_utils, "_compute_bleu", side_effect=_fake_bleu), \
         patch.object(benchmark_utils, "_compute_rouge", side_effect=_fake_rouge):
        result = benchmark.evaluate_case_quality()

    # 两题中只有 1 题带 reference → 只应累计 1 对，且两列表严格等长
    assert result["n_bleu_rouge"] == 1
    bleu_preds, bleu_refs = captured["bleu"]
    rouge_preds, rouge_refs = captured["rouge"]
    assert len(bleu_preds) == len(bleu_refs) == 1
    assert len(rouge_preds) == len(rouge_refs) == 1
    assert result["bleu"] == {"bleu": 10.0}
    assert result["rouge"]["rouge1"] == 0.5
