#!/usr/bin/env python3
"""
FLUX.1-schnell transformer BF16 -> FP8 (e4m3fn) 量化转换 (optimum-quanto)。

产物: models/FLUX.1-schnell-transformer-fp8/
  - transformer 权重 (FP8 + scale)
  - quanto_qmap.json (重载所需量化映射)

用法:
  source venv_v5/bin/activate
  python3 convert_flux_fp8.py
"""
import json
import os
import torch
from diffusers import FluxTransformer2DModel
from optimum.quanto import QuantizedDiffusersModel, qfloat8_e4m3fn

SRC = "/home/chinux/jupyterlab/meerkatai/models/FLUX.1-schnell"
DST = "/home/chinux/jupyterlab/meerkatai/models/FLUX.1-schnell-transformer-fp8"


class QuantizedFluxTransformer2DModel(QuantizedDiffusersModel):
    base_class = FluxTransformer2DModel


def main():
    print("[1/3] 加载 transformer (BF16, 23G) ...", flush=True)
    transformer = FluxTransformer2DModel.from_pretrained(
        SRC,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    print("[2/3] 量化 FP8 (e4m3fn) ...", flush=True)
    # quantize 会 in-place 把 nn.Linear 替换为 QLinear, freeze 后权重转为 FP8
    qmodel = QuantizedFluxTransformer2DModel.quantize(
        transformer, weights=qfloat8_e4m3fn
    )
    print("[3/3] 保存到", DST, flush=True)
    qmodel.save_pretrained(DST)
    print("转换完成:", DST, flush=True)


if __name__ == "__main__":
    main()
