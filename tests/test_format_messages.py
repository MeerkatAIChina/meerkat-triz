import sys
from unittest.mock import MagicMock

# Mock heavy dependencies before import
sys.modules['datasets'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['transformers'].AutoTokenizer = MagicMock()
sys.modules['transformers'].AutoModelForCausalLM = MagicMock()
sys.modules['peft'] = MagicMock()
sys.modules['peft'].AutoPeftModelForCausalLM = MagicMock()
sys.modules['peft'].LoraConfig = MagicMock()
sys.modules['trl'] = MagicMock()
sys.modules['trl'].SFTTrainer = MagicMock()
sys.modules['bitsandbytes'] = MagicMock()

# Mock pathlib.Path.mkdir to prevent directory creation errors on import
import pathlib
original_mkdir = pathlib.Path.mkdir
pathlib.Path.mkdir = lambda self, **kwargs: None

sys.path.insert(0, 'ref/mongoose_ai_dgx')

from utils.data_utils import format_messages

# Keep mkdir mocked — format_messages does 'from config import DATA_CONFIG' at call time
# which triggers config.py's module-level directory creation
# NOTE: do NOT restore original_mkdir here


class MockTokenizer:
    """Mock tokenizer with apply_chat_template support."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        result = "\n".join(parts)
        if add_generation_prompt:
            result += "\n<|im_start|>assistant\n"
        return result


def test_format_messages_inference():
    """Test inference prompt with add_generation_prompt=True."""
    tokenizer = MockTokenizer()
    result = format_messages(tokenizer, "Explain segmentation.", add_generation_prompt=True)
    assert "<|im_start|>system" in result
    assert "<|im_start|>user" in result
    assert "Explain segmentation." in result
    assert "<|im_start|>assistant\n" in result
    assert result.endswith("assistant\n")


def test_format_messages_training():
    """Test training data with assistant_content and add_generation_prompt=False."""
    tokenizer = MockTokenizer()
    result = format_messages(
        tokenizer,
        "Explain segmentation.",
        assistant_content="Segmentation divides a system...",
        add_generation_prompt=False,
    )
    assert "<|im_start|>system" in result
    assert "<|im_start|>user" in result
    assert "<|im_start|>assistant" in result
    assert "Segmentation divides a system..." in result
    assert not result.endswith("assistant\n")


def test_format_messages_custom_system():
    """Test custom system message override."""
    tokenizer = MockTokenizer()
    custom_system = "You are a test assistant."
    result = format_messages(tokenizer, "Hello", system_message=custom_system, add_generation_prompt=True)
    assert "You are a test assistant." in result
    assert "Meerkat-AI" not in result


def test_format_messages_default_system():
    """Test default system message from DATA_CONFIG."""
    tokenizer = MockTokenizer()
    result = format_messages(tokenizer, "Hello", add_generation_prompt=True)
    assert "Meerkat-AI" in result or "TRIZ" in result
