"""
猫鼬AI DGX Spark 项目工具包
"""

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
    # Pipeline (new)
    "PipelineState",
    "MoonshotSyntheticClient",
    "SyntheticPipeline",
]
