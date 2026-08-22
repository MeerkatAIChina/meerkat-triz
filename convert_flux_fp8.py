#!/usr/bin/env python3
"""
FLUX transformer BF16 -> FP8 (e4m3fn) 量化转换 (optimum-quanto)。

产物: <dst>/ 目录
  - transformer 权重 (FP8 + scale)
  - quanto_qmap.json (重载所需量化映射)

用法:
  source venv_v5/bin/activate
  python3 convert_flux_fp8.py --src <pipeline目录> --dst <输出目录>
"""
import argparse
import json
import os
import torch
from diffusers import FluxTransformer2DModel
from optimum.quanto import QuantizedDiffusersModel, qfloat8_e4m3fn

DEFAULT_SRC = "/home/chinux/jupyterlab/meerkatai/models/FLUX.1-schnell"
DEFAULT_DST = "/home/chinux/jupyterlab/meerkatai/models/FLUX.1-schnell-transformer-fp8"


class QuantizedFluxTransformer2DModel(QuantizedDiffusersModel):
    base_class = FluxTransformer2DModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=DEFAULT_SRC)
    parser.add_argument("--dst", default=DEFAULT_DST)
    args = parser.parse_args()
    src, dst = args.src, args.dst

    print("[1/3] 加载 transformer (BF16, 23G) ...", flush=True)
    transformer = FluxTransformer2DModel.from_pretrained(
        src,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    print("[2/3] 量化 FP8 (e4m3fn) ...", flush=True)
    # quantize 会 in-place 把 nn.Linear 替换为 QLinear, freeze 后权重转为 FP8
    qmodel = QuantizedFluxTransformer2DModel.quantize(
        transformer, weights=qfloat8_e4m3fn
    )
    print("[3/3] 保存到", dst, flush=True)
    qmodel.save_pretrained(dst)
    print("转换完成:", dst, flush=True)


if __name__ == "__main__":
    main()
