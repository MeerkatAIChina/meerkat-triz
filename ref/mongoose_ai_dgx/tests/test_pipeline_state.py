"""
Pipeline state registry tests
Covers INFRA-06: JSON artifact registry for cross-notebook state tracking
"""

import sys
sys.path.insert(0, "ref/mongoose_ai_dgx")

import os

# Import pipeline_state directly to avoid triggering utils/__init__.py
# which imports benchmark_utils -> torch (not available in test env)
import importlib.util
spec = importlib.util.spec_from_file_location("pipeline_state", "ref/mongoose_ai_dgx/utils/pipeline_state.py")
pipeline_state_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_state_mod)
PipelineState = pipeline_state_mod.PipelineState


def test_register(temp_state_file):
    """Test artifact registration"""
    state = PipelineState(temp_state_file)
    state.register("test_artifact", "/tmp/test.txt", "data", {"key": "value"})

    # Verify file was created
    assert os.path.exists(temp_state_file), "State file not created"

    # Verify artifact is registered
    artifact = state.get("test_artifact")
    assert artifact is not None, "Artifact not found after registration"
    assert artifact["name"] == "test_artifact"
    assert artifact["path"] == "/tmp/test.txt"
    assert artifact["type"] == "data"
    assert artifact["metadata"]["key"] == "value"


def test_get(temp_state_file):
    """Test artifact retrieval"""
    state = PipelineState(temp_state_file)
    state.register("artifact_a", "/path/a", "model")
    state.register("artifact_b", "/path/b", "dataset")

    a = state.get("artifact_a")
    assert a is not None
    assert a["path"] == "/path/a"

    b = state.get("artifact_b")
    assert b is not None
    assert b["type"] == "dataset"

    missing = state.get("nonexistent")
    assert missing is None


def test_verify(temp_state_file, tmp_path):
    """Test artifact verification (checks filesystem existence)"""
    state = PipelineState(temp_state_file)

    # Create an actual file
    real_file = tmp_path / "real.txt"
    real_file.write_text("test")
    state.register("real", str(real_file), "data")

    # Register a non-existent file
    state.register("fake", "/nonexistent/path/file.txt", "data")

    assert state.verify("real") is True, "Should verify existing file"
    assert state.verify("fake") is False, "Should not verify non-existent file"
    assert state.verify("unregistered") is False, "Should not verify unregistered artifact"


def test_preflight(temp_state_file):
    """Test preflight checks"""
    state = PipelineState(temp_state_file)
    state.register("required_1", "/tmp", "data")

    # Test with missing artifact
    errors = state.preflight(required_artifacts=["required_1", "missing"])
    assert len(errors) == 1, f"Expected 1 error, got {len(errors)}: {errors}"
    assert "missing" in errors[0]

    # Test with all artifacts present
    errors = state.preflight(required_artifacts=["required_1"])
    assert len(errors) == 0, f"Expected 0 errors, got {len(errors)}"

    # Test with package version check (using packaging which has __version__)
    errors = state.preflight(required_packages={"packaging": "0.0.1"})
    assert len(errors) == 0, "packaging module should pass version check"


def test_summary(temp_state_file):
    """Test state summary"""
    state = PipelineState(temp_state_file)
    state.register("model_1", "/path/m1", "model")
    state.register("dataset_1", "/path/d1", "dataset")
    state.register("dataset_2", "/path/d2", "dataset")

    summary = state.summary()
    assert summary["total_artifacts"] == 3
    assert "model" in summary["type_counts"]
    assert "dataset" in summary["type_counts"]
    assert summary["type_counts"]["model"] == 1
    assert summary["type_counts"]["dataset"] == 2
    assert "model_1" in summary["artifact_names"]
    assert "dataset_1" in summary["artifact_names"]


def test_layer1_path_in_metadata(temp_state_file):
    """Baseline artifact metadata round-trips layer1_path and summary."""
    state = PipelineState(temp_state_file)
    state.register(
        "baseline_results",
        "/tmp/results",
        "benchmark",
        metadata={
            "layer1_path": "/tmp/results/lm_eval_results_20260101_000000.json",
            "layer1_summary": {"mmlu_pro": {"acc_norm": 0.42}},
        },
    )

    artifact = state.get("baseline_results")
    assert artifact is not None
    assert artifact["metadata"]["layer1_path"] == "/tmp/results/lm_eval_results_20260101_000000.json"
    assert artifact["metadata"]["layer1_summary"]["mmlu_pro"]["acc_norm"] == 0.42
