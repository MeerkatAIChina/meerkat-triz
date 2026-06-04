"""
猫鼬AI DGX Spark 项目工具包
"""

# Monkey-patch flash-linear-attention device context bug.
# fla caches torch.cpu as device_torch_lib when triton backend reports 'cpu',
# then tries to call torch.cpu.device(index) which doesn't exist.
# This breaks model.generate() on CUDA. We intercept the error and fall back
# to torch.cuda.device or a no-op context.
try:
    import contextlib
    import torch
    import fla.utils as _fla_utils

    _fla_orig_custom_device_ctx = _fla_utils.custom_device_ctx

    def _fla_patched_custom_device_ctx(index: int):
        try:
            return _fla_orig_custom_device_ctx(index)
        except AttributeError:
            if torch.cuda.is_available() and index is not None:
                return torch.cuda.device(index)
            return contextlib.nullcontext()

    _fla_utils.custom_device_ctx = _fla_patched_custom_device_ctx
except Exception:
    pass  # fla not installed or already fixed

from .benchmark_utils import (
    run_lm_evaluation,
    run_triz_evaluation,
    run_performance_benchmark,
    aggregate_results,
)

from .data_utils import (
    load_raw_data,
    convert_to_chatml,
    create_synthetic_data,
    split_dataset,
    save_dataset,
    validate_chatml_format,
)

from .training_utils import (
    load_model_and_tokenizer,
    setup_qlora_config,
    prepare_qlora_model,
    setup_training_arguments,
    create_trainer,
    merge_and_save_model,
    save_adapter_only,
    find_all_linear_names,
    get_qwen36_target_modules,
    CheckpointValidationCallback,
    resume_from_checkpoint,
)

from .pipeline_state import PipelineState

from .synthetic_pipeline import (
    MoonshotSyntheticClient,
    SyntheticPipeline,
)

__all__ = [
    "run_lm_evaluation",
    "run_triz_evaluation",
    "run_performance_benchmark",
    "aggregate_results",
    "load_raw_data",
    "convert_to_chatml",
    "create_synthetic_data",
    "split_dataset",
    "save_dataset",
    "validate_chatml_format",
    "load_model_and_tokenizer",
    "setup_qlora_config",
    "prepare_qlora_model",
    "setup_training_arguments",
    "create_trainer",
    "merge_and_save_model",
    "save_adapter_only",
    "find_all_linear_names",
    "get_qwen36_target_modules",
    "CheckpointValidationCallback",
    "resume_from_checkpoint",
    # Pipeline (new)
    "PipelineState",
    "MoonshotSyntheticClient",
    "SyntheticPipeline",
]
