"""
Tests for benchmark_utils.py
Covers deterministic evaluation, Chinese keyword matching, and keyword helper logic.

Note: benchmark_utils.py imports torch at module level, which can fail in local
macOS test environments due to OpenMP conflicts. We mock torch (and numpy) before
importing the module under test.
"""

import sys
from unittest.mock import Mock
import pathlib

# Prevent config.py from trying to create /home/meerkat directories at import time.
_original_mkdir = pathlib.Path.mkdir
pathlib.Path.mkdir = lambda self, *args, **kwargs: None

try:
    # Mock torch, numpy, and datasets before importing benchmark_utils
    _mock_torch = Mock()
    _mock_torch.tensor = Mock(return_value=Mock())
    _mock_torch.cuda = Mock()
    _mock_torch.cuda.memory_allocated = Mock(return_value=0)
    _mock_torch.cuda.max_memory_allocated = Mock(return_value=0)
    _mock_torch.cuda.is_available = Mock(return_value=False)
    _mock_torch.cuda.synchronize = Mock()
    _mock_torch.no_grad = Mock(return_value=Mock(__enter__=Mock(return_value=None), __exit__=Mock(return_value=None)))
    sys.modules["torch"] = _mock_torch
    sys.modules["numpy"] = Mock()
    sys.modules["datasets"] = Mock()

    # Mock config before data_utils imports it
    _mock_config = Mock()
    _mock_config.DATA_CONFIG = {
        "chatml": {
            "system_message": "You are a TRIZ expert.",
            "max_length": 4096,
        }
    }
    sys.modules["config"] = _mock_config

    sys.path.insert(0, "ref/mongoose_ai_dgx")

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "benchmark_utils", "utils/benchmark_utils.py"
    )
    benchmark_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark_utils)
finally:
    pathlib.Path.mkdir = _original_mkdir

import pytest
from unittest.mock import patch


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
