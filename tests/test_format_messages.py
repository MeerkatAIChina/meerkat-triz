import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, '.')

# Prevent config.py from trying to create /home/meerkat directories at import time.
# Import the real config now (cached in sys.modules) so format_messages' call-time
# 'from config import DATA_CONFIG' hits the cache after mkdir is restored.
import pathlib
_original_mkdir = pathlib.Path.mkdir
pathlib.Path.mkdir = lambda self, *args, **kwargs: None
try:
    import config  # noqa: F401
finally:
    pathlib.Path.mkdir = _original_mkdir

# Import data_utils directly to avoid utils/__init__.py heavy dependencies
# (torch/transformers/numpy/openai). The 'datasets' mock is scoped with
# patch.dict so it cannot leak into other test modules.
import importlib.util
with patch.dict(sys.modules, {"datasets": MagicMock()}):
    spec = importlib.util.spec_from_file_location("data_utils", "utils/data_utils.py")
    data_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_utils)

format_messages = data_utils.format_messages


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
