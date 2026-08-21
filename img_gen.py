#!/usr/bin/env python3
"""
Meerkat-TRIZ 文生图 (FLUX.1-schnell) — 本地推理脚本。
用法:
  python3 img_gen.py "一只在 TRIZ 实验室思考的猫鼬" -o out.png --steps 4

默认 FLUX.1-schnell: guidance_scale=0.0, num_inference_steps=4 (快速高质量)。
"""

import argparse
import torch
from diffusers import FluxPipeline


def generate(prompt, output_path, steps=4, width=1024, height=1024,
             seed=None, model_path="models/FLUX.1-schnell"):
    print(f"[加载] FLUX.1-schnell from {model_path} ...", flush=True)
    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    pipe.to("cuda")
    print("[就绪] 开始生成 ...", flush=True)

    generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None

    image = pipe(
        prompt,
        guidance_scale=0.0,          # schnell 用 0
        num_inference_steps=steps,   # 默认 4 步
        width=width,
        height=height,
        max_sequence_length=256,
        generator=generator,
    ).images[0]

    image.save(output_path)
    print(f"[done] 图片已保存: {output_path}")
    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FLUX.1-schnell 文生图")
    ap.add_argument("prompt", help="图片描述 (中文或英文)")
    ap.add_argument("-o", "--output", default="output.png")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--model-path", default="models/FLUX.1-schnell")
    args = ap.parse_args()
    generate(args.prompt, args.output, args.steps, args.width, args.height,
             args.seed, args.model_path)
