"""
猫鼬AI DGX Spark 项目工具包

轻量包入口 (复盘整改): 所有子模块均通过 PEP 562 __getattr__ 惰性加载，
`import utils` 不再拉入 torch / transformers / datasets / openai / fla 等重依赖，
使无 GPU 的开发机与 CI 环境也能直接 import 本包。首次访问具体名称时才导入
对应子模块。

原位于此处的 fla device-ctx monkey-patch 已移入 utils/training_utils.py
(_apply_fla_device_ctx_patch)，在训练入口 load_model_and_tokenizer() 调用时执行。
"""

import importlib as _importlib

# 导出名 -> 子模块名 的惰性映射
_LAZY_EXPORTS = {
    # benchmark_utils (顶层依赖 torch)
    "run_lm_evaluation": "benchmark_utils",
    "run_triz_evaluation": "benchmark_utils",
    "run_performance_benchmark": "benchmark_utils",
    "aggregate_results": "benchmark_utils",
    # data_utils (顶层依赖 datasets —— WARN-04: 故 format_messages 只能惰性导出)
    "load_raw_data": "data_utils",
    "convert_to_chatml": "data_utils",
    "create_synthetic_data": "data_utils",
    "split_dataset": "data_utils",
    "save_dataset": "data_utils",
    "validate_chatml_format": "data_utils",
    "format_messages": "data_utils",
    # training_utils (torch/transformers/peft/trl 已在模块内做条件惰性处理)
    "load_model_and_tokenizer": "training_utils",
    "setup_qlora_config": "training_utils",
    "prepare_qlora_model": "training_utils",
    "setup_training_arguments": "training_utils",
    "create_trainer": "training_utils",
    "merge_and_save_model": "training_utils",
    "save_adapter_only": "training_utils",
    "find_all_linear_names": "training_utils",
    "get_qwen36_target_modules": "training_utils",
    "get_final_train_loss": "training_utils",
    "CheckpointValidationCallback": "training_utils",
    "resume_from_checkpoint": "training_utils",
    # pipeline_state (轻量，但统一走惰性加载保持 import 最小化)
    "PipelineState": "pipeline_state",
    # synthetic_pipeline (顶层依赖 numpy/openai)
    "MoonshotSyntheticClient": "synthetic_pipeline",
    "SyntheticPipeline": "synthetic_pipeline",
}


def __getattr__(name):
    """PEP 562 惰性导出: 首次访问时才导入对应子模块。"""
    if name in _LAZY_EXPORTS:
        module = _importlib.import_module(f".{_LAZY_EXPORTS[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value  # 缓存，避免重复触发 __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


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
    "format_messages",
    "load_model_and_tokenizer",
    "setup_qlora_config",
    "prepare_qlora_model",
    "setup_training_arguments",
    "create_trainer",
    "merge_and_save_model",
    "save_adapter_only",
    "find_all_linear_names",
    "get_qwen36_target_modules",
    "get_final_train_loss",
    "CheckpointValidationCallback",
    "resume_from_checkpoint",
    # Pipeline
    "PipelineState",
    "MoonshotSyntheticClient",
    "SyntheticPipeline",
]
