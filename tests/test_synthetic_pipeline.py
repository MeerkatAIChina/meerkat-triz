"""
Synthetic pipeline tests
Covers DATA-01 (synthetic generation), INFRA-05 (Moonshot client with batching/rate limiting)
All tests use mocked API client — no real network calls.
"""

import sys
sys.path.insert(0, ".")

import os
import json

import pytest

# synthetic_pipeline 模块级依赖 numpy/openai；裸环境下跳过而非收集期报错
pytest.importorskip("numpy", reason="synthetic_pipeline 依赖 numpy")
pytest.importorskip("openai", reason="synthetic_pipeline 依赖 openai")

# Import synthetic_pipeline directly to avoid utils/__init__.py torch dependency
import importlib.util
spec = importlib.util.spec_from_file_location("synthetic_pipeline", "utils/synthetic_pipeline.py")
synthetic_pipeline_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(synthetic_pipeline_mod)
SyntheticPipeline = synthetic_pipeline_mod.SyntheticPipeline
MoonshotSyntheticClient = synthetic_pipeline_mod.MoonshotSyntheticClient


def test_deduplicate_seeds(mock_moonshot_client):
    """Test seed deduplication by instruction+output hash"""
    pipeline = SyntheticPipeline(
        client=mock_moonshot_client,
        output_dir="/tmp/test_out",
        checkpoint_dir="/tmp/test_chk",
    )

    seeds = [
        {"instruction": "What is TRIZ?", "output": "TRIZ is..."},
        {"instruction": "What is TRIZ?", "output": "TRIZ is..."},  # duplicate
        {"instruction": "How to use ARIZ?", "output": "ARIZ steps..."},
    ]

    unique = pipeline.deduplicate_seeds(seeds)
    assert len(unique) == 2, f"Expected 2 unique seeds, got {len(unique)}"


def test_estimate_cost(mock_moonshot_client):
    """Test cost estimation math"""
    cost = mock_moonshot_client.estimate_cost(100, batch_size=5)

    assert cost["seed_count"] == 100
    assert cost["batch_size"] == 5
    assert cost["num_batches"] == 20  # 100 / 5 = 20
    assert cost["estimated_cost_cny"] > 0
    assert cost["estimated_time_minutes"] > 0
    assert cost["rpm"] == 3


def test_generate_variations_mocked(mock_moonshot_client):
    """Test synthetic generation with mocked API responses"""
    seeds = [
        {"instruction": "Q1", "input": "", "output": "A1"},
        {"instruction": "Q2", "input": "", "output": "A2"},
    ]

    results = mock_moonshot_client.generate_variations(
        seeds=seeds,
        strategy="rephrase",
        subset_name="concept_explanation",
        num_variations=3,
    )

    # Should get 2 seeds * 3 variations = 6 results
    assert len(results) == 6, f"Expected 6 results, got {len(results)}"

    # Each result should have required fields
    for r in results:
        assert "instruction" in r
        assert "output" in r
        assert len(r["instruction"]) > 0
        assert len(r["output"]) > 0


def test_checkpoint_save_load(mock_moonshot_client, temp_checkpoint_dir):
    """Test checkpoint save and resume"""
    pipeline = SyntheticPipeline(
        client=mock_moonshot_client,
        output_dir="/tmp/test_out",
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Simulate saving a checkpoint
    checkpoint_file = os.path.join(temp_checkpoint_dir, "test_subset_checkpoint.json")
    completed_ids = {0, 1}
    results = [
        {"instruction": "Q1", "output": "A1", "source": "seed", "subset": "test_subset"},
        {"instruction": "Q2", "output": "A2", "source": "synthetic", "subset": "test_subset"},
    ]

    # Use the internal save method
    from pathlib import Path
    pipeline._save_checkpoint(Path(checkpoint_file), completed_ids, results)

    # Verify checkpoint file exists
    assert os.path.exists(checkpoint_file), "Checkpoint file not created"

    # Verify checkpoint content
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    assert "completed_ids" in checkpoint
    assert checkpoint["completed_ids"] == [0, 1]
    assert "results" in checkpoint
    assert len(checkpoint["results"]) == 2
    assert "saved_at" in checkpoint
