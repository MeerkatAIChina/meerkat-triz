"""
peft 0.19.1 × transformers 5.10.1 WeightConverter 兼容补丁 —— import 即生效。

事故背景 (v3 lora_B 全零事故):
    peft 0.19.1 的 convert_peft_adapter_state_dict_for_transformers() 调用了
    transformers 5.10.1 的 WeightConverter, 但传入了该版本不认识的
    `distributed_operation` 关键字参数, 抛:
        TypeError: WeightConverter.__init__() got an unexpected keyword
        argument 'distributed_operation'
    更糟的是旧代码 `PeftModel.from_pretrained(model.unload(), best_ckpt)` 中
    Python 先求值 model.unload() 再抛异常, fallback 于是保存了被 unload
    污染的内存态 → 310 个 lora_B 全零、dtype 变 F32。

绕过方法 (提取自 scripts/eval_adapter_vs_base.py 第 26-33 行):
    将 peft 的转换入口替换为恒等函数, 跳过与 transformers v5 WeightConverter
    不兼容的转换路径。LoRA 适配器权重在 peft 内部格式与 transformers v5
    期望格式之间无需转换 (本模型 trust_remote_code 的混合架构层名保持一致)。

使用约束:
    必须在任何 peft/transformers 模型加载 (from_pretrained / PeftModel)
    之前 import 本模块。train.py 在 main() 内、重依赖导入之前首先 import 本模块。

v4 备注:
    v4 发货流程不再从内存加载/保存适配器 (改为复制 Trainer 落盘文件),
    本补丁属于防御性兜底: 防止任何后续代码路径 (评测、合并) 踩到同一地雷。
"""

import peft.utils.transformers_weight_conversion as _twc


def _skip_weight_conversion(model, peft_config, adapter_state_dict, adapter_name):
    """恒等绕过: 不做任何权重名/格式转换, 原样返回适配器 state_dict。"""
    return adapter_state_dict


_twc.convert_peft_adapter_state_dict_for_transformers = _skip_weight_conversion
