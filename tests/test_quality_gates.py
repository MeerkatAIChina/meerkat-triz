"""
Quality gate tests
Covers DATA-02: perplexity filtering and diversity scoring
All tests use mocked model/tokenizer — no real model loading.
"""

import sys
sys.path.insert(0, ".")

# Import synthetic_pipeline directly to avoid utils/__init__.py torch dependency
import importlib.util
spec = importlib.util.spec_from_file_location("synthetic_pipeline", "utils/synthetic_pipeline.py")
synthetic_pipeline_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(synthetic_pipeline_mod)
compute_perplexity = synthetic_pipeline_mod.compute_perplexity
filter_by_perplexity = synthetic_pipeline_mod.filter_by_perplexity
compute_diversity_score = synthetic_pipeline_mod.compute_diversity_score
filter_by_diversity = synthetic_pipeline_mod.filter_by_diversity


def test_token_length_filter():
    """Test token length filtering logic (standalone, no tokenizer needed)"""
    samples = [
        {"instruction": "Short question?", "input": "", "output": "Short answer."},
        {"instruction": "Q " * 2000, "input": "", "output": "A " * 2000},  # would exceed 3500 tokens
    ]

    # Simple char-based proxy for token length (4 chars ~ 1 token for CJK)
    max_tokens = 3500
    filtered = []
    for s in samples:
        text = f"{s.get('instruction', '')}\n{s.get('input', '')}\n{s.get('output', '')}"
        # Approximate: 1 token ~ 1.5 chars for mixed CJK/English
        estimated_tokens = len(text) / 1.5
        if estimated_tokens <= max_tokens:
            filtered.append(s)

    assert len(filtered) == 1, f"Expected 1 sample after length filter, got {len(filtered)}"
    assert filtered[0]["instruction"] == "Short question?"


def test_diversity_score_computation():
    """Test n-gram diversity scoring (no model needed)"""
    # High diversity: all different
    diverse_samples = [
        {"instruction": "什么是分离原理？"},
        {"instruction": "如何应用局部质量原理？"},
        {"instruction": "解释嵌套原理的应用场景"},
    ]

    d1 = compute_diversity_score(diverse_samples, n=1, field="instruction")
    d2 = compute_diversity_score(diverse_samples, n=2, field="instruction")

    assert 0.0 <= d1 <= 1.0, f"distinct-1 out of range: {d1}"
    assert 0.0 <= d2 <= 1.0, f"distinct-2 out of range: {d2}"
    # With diverse Chinese text, diversity should be reasonably high
    assert d1 > 0.5, f"Expected high distinct-1 for diverse samples, got {d1}"

    # Low diversity: nearly identical
    similar_samples = [
        {"instruction": "什么是TRIZ原理？"},
        {"instruction": "什么是TRIZ原理？"},
        {"instruction": "什么是TRIZ原理？"},
    ]

    d1_low = compute_diversity_score(similar_samples, n=1, field="instruction")
    d2_low = compute_diversity_score(similar_samples, n=2, field="instruction")

    # Identical text should have lower diversity
    assert d1_low < d1, f"Similar samples should have lower distinct-1: {d1_low} vs {d1}"


def test_perplexity_filter_mocked(mock_model_and_tokenizer):
    """Test perplexity filtering with mocked model (no real model loading)"""
    model, tokenizer = mock_model_and_tokenizer

    samples = [
        {"instruction": "Q1", "input": "", "output": "A1"},
        {"instruction": "Q2", "input": "", "output": "A2"},
        {"instruction": "Q3", "input": "", "output": "A3"},
    ]

    # Test compute_perplexity directly
    ppl = compute_perplexity("test text", model, tokenizer, device="cpu")
    assert ppl > 0, f"Perplexity should be positive, got {ppl}"
    assert ppl != float("inf"), "Perplexity should not be infinity with mocked model"

    # Test filter_by_perplexity with mocked model
    filtered, threshold = filter_by_perplexity(
        samples, model, tokenizer, percentile=80, device="cpu"
    )

    # With identical mocked loss, all samples should pass
    assert len(filtered) == len(samples), f"Expected all samples to pass, got {len(filtered)}/{len(samples)}"
    assert threshold > 0, f"Threshold should be positive, got {threshold}"

    # Test skip when model is None
    filtered_skip, _ = filter_by_perplexity(samples, None, tokenizer)
    assert len(filtered_skip) == len(samples), "Should skip filtering when model is None"


def test_filter_by_diversity():
    """Test diversity-based filtering"""
    # Samples with decent diversity
    samples = [
        {"instruction": "什么是分离原理？", "output": "分离原理是..."},
        {"instruction": "如何应用局部质量原理？", "output": "局部质量原理..."},
        {"instruction": "解释嵌套原理的应用场景", "output": "嵌套原理可以..."},
    ]

    filtered, stats = filter_by_diversity(
        samples,
        min_distinct_1=0.3,
        min_distinct_2=0.15,
        field="instruction",
    )

    assert "distinct_1" in stats
    assert "distinct_2" in stats
    assert 0.0 <= stats["distinct_1"] <= 1.0
    assert 0.0 <= stats["distinct_2"] <= 1.0

    # With diverse samples, all should pass
    assert len(filtered) == len(samples), f"Expected all samples to pass diversity filter"

    # Test with empty list
    empty_filtered, empty_stats = filter_by_diversity([], min_distinct_1=0.3)
    assert len(empty_filtered) == 0
    assert empty_stats["distinct_1"] == 0.0
