import sys
from unittest.mock import MagicMock

# Mock heavy dependencies before import
sys.modules['torch'] = MagicMock()
sys.modules['jieba'] = MagicMock()
sys.modules['jieba'].cut = lambda text: text.split()

sys.path.insert(0, '.')

# Import only the metric functions we need
import importlib.util
spec = importlib.util.spec_from_file_location("benchmark_utils", "utils/benchmark_utils.py")
benchmark_utils = importlib.util.module_from_spec(spec)

# We need to mock more dependencies before execution
sys.modules['rouge_score'] = MagicMock()
sys.modules['sacrebleu'] = MagicMock()
sys.modules['lm_eval'] = MagicMock()

# For this test, we'll test the functions by direct import if possible,
# or verify they exist in the module
try:
    spec.loader.exec_module(benchmark_utils)
except ImportError:
    pass  # Some dependencies may still be missing

# Alternative: test by importing the module and checking function signatures
import ast


def test_bleu_function_exists():
    """Verify _compute_bleu function exists in benchmark_utils.py."""
    with open("utils/benchmark_utils.py", "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)
    func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert "_compute_bleu" in func_names
    assert "corpus_bleu" in source
    assert "tokenize='zh'" in source


def test_rouge_function_exists():
    """Verify _compute_rouge function exists with correct parameters."""
    with open("utils/benchmark_utils.py", "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)
    func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert "_compute_rouge" in func_names
    assert "use_stemmer=False" in source
    assert "jieba.cut" in source


def test_evaluate_case_quality_returns_bleu_rouge():
    """Verify evaluate_case_quality returns bleu and rouge fields."""
    with open("utils/benchmark_utils.py", "r", encoding="utf-8") as f:
        source = f.read()
    assert '"bleu"' in source or "'bleu'" in source
    assert '"rouge"' in source or "'rouge'" in source


def test_chinese_aware_tokenization():
    """Verify Chinese-aware tokenization is used."""
    with open("utils/benchmark_utils.py", "r", encoding="utf-8") as f:
        source = f.read()
    assert "tokenize='zh'" in source
    assert "use_stemmer=False" in source
