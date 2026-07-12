"""
Config validation tests
Covers TRAIN-02 (target_modules), TRAIN-03 (lora_dropout), and SYNTHETIC_CONFIG
"""

import sys
sys.path.insert(0, ".")

# Prevent config.py from trying to create /home/meerkat directories at import time.
# On macOS /home is a symlink to /System/Volumes/Data/home which may not exist,
# so we monkey-patch pathlib.Path.mkdir before importing config.
import pathlib
_original_mkdir = pathlib.Path.mkdir

def _noop_mkdir(self, *args, **kwargs):
    pass

pathlib.Path.mkdir = _noop_mkdir

try:
    import config
finally:
    pathlib.Path.mkdir = _original_mkdir


def test_target_modules_count_is_12():
    """TRAIN-02: target_modules must have exactly 12 explicit modules"""
    target_modules = config.QLORA_CONFIG["lora"]["target_modules"]
    assert isinstance(target_modules, list), "target_modules must be a list"
    assert len(target_modules) == 12, f"Expected 12 modules, got {len(target_modules)}: {target_modules}"

    # Verify all expected modules are present
    expected = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    for mod in expected:
        assert mod in target_modules, f"Missing module: {mod}"


def test_lora_dropout_is_zero():
    """TRAIN-03: lora_dropout must be 0.0 for MoE architecture compatibility"""
    dropout = config.QLORA_CONFIG["lora"]["lora_dropout"]
    assert dropout == 0.0, f"lora_dropout must be 0.0, got {dropout}"


def test_synthetic_config_exists():
    """SYNTHETIC_CONFIG must exist with required keys"""
    assert hasattr(config, "SYNTHETIC_CONFIG"), "SYNTHETIC_CONFIG not found in config.py"

    sc = config.SYNTHETIC_CONFIG
    assert "api" in sc, "SYNTHETIC_CONFIG missing 'api' section"
    assert "multipliers" in sc, "SYNTHETIC_CONFIG missing 'multipliers' section"
    assert "strategies" in sc, "SYNTHETIC_CONFIG missing 'strategies' section"
    assert "quality_gates" in sc, "SYNTHETIC_CONFIG missing 'quality_gates' section"

    # Check API config
    api = sc["api"]
    assert api.get("model") == "moonshot-v1-8k", f"Unexpected model: {api.get('model')}"
    assert api.get("rpm") == 3, f"Unexpected RPM: {api.get('rpm')}"
    assert api.get("batch_size") == 5, f"Unexpected batch_size: {api.get('batch_size')}"

    # Check all 6 subsets have multipliers and strategies
    subsets = ["concept_explanation", "ariz_guidance", "principle_recommendation",
               "innovation_assessment", "case_generation", "contradiction_analysis"]
    for subset in subsets:
        assert subset in sc["multipliers"], f"Missing multiplier for {subset}"
        assert subset in sc["strategies"], f"Missing strategy for {subset}"
        assert sc["strategies"][subset] in ["rephrase", "generate_new", "mixed"], \
            f"Invalid strategy for {subset}: {sc['strategies'][subset]}"


def test_quality_gates_config():
    """Quality gates config must have expected thresholds"""
    qg = config.SYNTHETIC_CONFIG["quality_gates"]
    assert "max_tokens" in qg, "Missing max_tokens in quality_gates"
    assert qg["max_tokens"] == 3500, f"Expected max_tokens=3500, got {qg['max_tokens']}"
    assert "deduplicate" in qg, "Missing deduplicate in quality_gates"
    assert qg["deduplicate"] is True, "deduplicate should be True"
