import sys

sys.path.insert(0, '.')

import ast


def test_aggregate_results_accepts_before_after():
    """Verify aggregate_results signature includes before_results and after_results."""
    with open("utils/benchmark_utils.py", "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "aggregate_results":
            args = [arg.arg for arg in node.args.args]
            assert "before_results" in args
            assert "after_results" in args
            assert "model_info" in args
            return
    raise AssertionError("aggregate_results not found or missing before_results/after_results")


def test_compute_deltas_exists():
    """Verify _compute_deltas helper function exists."""
    with open("utils/benchmark_utils.py", "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)
    func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert "_compute_deltas" in func_names


def test_delta_structure_in_code():
    """Verify the code constructs before/after/delta/delta_pct structure."""
    with open("utils/benchmark_utils.py", "r", encoding="utf-8") as f:
        source = f.read()
    assert "\"before\"" in source or "'before'" in source
    assert "\"after\"" in source or "'after'" in source
    assert "\"delta\"" in source or "'delta'" in source
    assert "\"delta_pct\"" in source or "'delta_pct'" in source


def test_evaluation_report_filename():
    """Verify report saves to evaluation_report_YYYYMMDD_HHMMSS.json."""
    with open("utils/benchmark_utils.py", "r", encoding="utf-8") as f:
        source = f.read()
    assert "evaluation_report_" in source


def test_report_layer_structure():
    """Verify report contains layer2_triz and layer3_performance keys."""
    with open("utils/benchmark_utils.py", "r", encoding="utf-8") as f:
        source = f.read()
    assert "layer2_triz" in source
    assert "layer3_performance" in source
    assert "layer1_general" in source
