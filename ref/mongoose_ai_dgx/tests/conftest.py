"""
Pytest shared fixtures for Meerkat AI tests
"""

import json
import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path

import sys
sys.path.insert(0, "ref/mongoose_ai_dgx")


@pytest.fixture
def mock_moonshot_client():
    """Mock Moonshot API client that returns predefined responses"""
    from utils.synthetic_pipeline import MoonshotSyntheticClient

    client = Mock(spec=MoonshotSyntheticClient)
    client.model = "moonshot-v1-8k"
    client.rpm = 3
    client.total_tokens_used = 0
    client.total_requests = 0

    # Mock estimate_cost to return predictable values
    def _estimate_cost(seed_count, batch_size=5):
        num_batches = (seed_count + batch_size - 1) // batch_size
        return {
            "seed_count": seed_count,
            "batch_size": batch_size,
            "num_batches": num_batches,
            "estimated_input_tokens": num_batches * 2000,
            "estimated_output_tokens": num_batches * 1500,
            "estimated_cost_cny": round(num_batches * 0.012, 2),
            "estimated_cost_usd": round(num_batches * 0.012 / 7.2, 2),
            "estimated_time_minutes": round(num_batches * 20 / 60, 1),
            "rpm": 3,
        }

    client.estimate_cost.side_effect = _estimate_cost

    # Mock generate_variations to return synthetic samples
    def _generate_variations(seeds, strategy, subset_name, num_variations=5, max_tokens=2000, temperature=0.8):
        results = []
        for seed in seeds:
            for i in range(num_variations):
                results.append({
                    "instruction": f"[{subset_name}] 合成问题 {i+1} 基于: {seed.get('instruction', '')[:30]}",
                    "input": "",
                    "output": f"[{subset_name}] 合成答案 {i+1} 基于: {seed.get('output', '')[:30]}",
                })
        return results

    client.generate_variations.side_effect = _generate_variations

    return client


@pytest.fixture
def temp_state_file(tmp_path):
    """Provide a temporary state file path"""
    return str(tmp_path / "test_pipeline_state.json")


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Provide a temporary checkpoint directory"""
    checkpoint_dir = tmp_path / "test_checkpoints"
    checkpoint_dir.mkdir()
    return str(checkpoint_dir)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide a temporary output directory"""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir()
    return str(output_dir)


@pytest.fixture
def sample_seeds():
    """Provide sample seed data for testing"""
    return [
        {
            "instruction": "什么是TRIZ的发明原理？",
            "input": "",
            "output": "TRIZ的发明原理是...",
        },
        {
            "instruction": "如何分析技术矛盾？",
            "input": "",
            "output": "分析技术矛盾需要...",
        },
        {
            "instruction": "什么是TRIZ的发明原理？",  # duplicate
            "input": "",
            "output": "TRIZ的发明原理是...",  # duplicate
        },
        {
            "instruction": "ARIZ算法的步骤是什么？",
            "input": "",
            "output": "ARIZ算法包括以下步骤...",
        },
    ]


@pytest.fixture
def mock_model_and_tokenizer():
    """Mock model and tokenizer for perplexity computation tests"""
    import torch

    # Mock tokenizer
    tokenizer = Mock()

    def _tokenizer_call(text, return_tensors="pt", truncation=True, max_length=2048):
        # Return a mock encoding with input_ids
        mock_encoding = Mock()
        mock_encoding.input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        return mock_encoding

    tokenizer.side_effect = _tokenizer_call

    # Mock model
    model = Mock()
    mock_loss = Mock()
    mock_loss.item = Mock(return_value=2.5)  # loss = 2.5 -> ppl = exp(2.5) ~ 12.18
    mock_output = Mock()
    mock_output.loss = mock_loss
    model.forward = Mock(return_value=mock_output)

    # Mock parameters for device detection
    mock_param = Mock()
    mock_param.device = "cpu"
    model.parameters = Mock(return_value=iter([mock_param]))

    return model, tokenizer
