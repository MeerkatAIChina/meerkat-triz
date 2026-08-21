#!/bin/bash
# 重启两个 vLLM 容器: 启用 function calling + 降显存到 0.2 (给 FLUX 腾空间)
set -e

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "停止两个 vLLM 容器"
docker stop meerkat-vllm-qwen38 meerkat-vllm-qwen36 2>/dev/null || true
docker rm meerkat-vllm-qwen38 meerkat-vllm-qwen36 2>/dev/null || true
sleep 5

log "启动 v6 (Qwen3.8-27B) @ 0.25 + tool calling"
docker run -d --name meerkat-vllm-qwen38 \
  --network host --ipc host --gpus all \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v /home/chinux/.cache/huggingface:/root/.cache/huggingface \
  -v /home/chinux/jupyterlab/meerkatai/models/Qwen3.8-27B-NVFP4:/model \
  -v /home/chinux/.cache/vllm:/root/.cache/vllm \
  qwen36-dgx-spark:v0.1.1 \
  /model \
    --served-model-name Qwen3.8-27B-NVFP4 \
    --quantization compressed-tensors \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.25 \
    --max-num-seqs 2 \
    --limit-mm-per-prompt '{"image":3,"video":1}' \
    --mm-processor-cache-gb 2 \
    --enable-lora \
    --lora-modules Meerkat-TRIZ-v1-Qwen3.8-27B=/root/.cache/vllm/loras/meerkat-triz-v1-qwen3.8 \
    --max-lora-rank 64 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --override-generation-config '{"temperature":0.7,"top_p":0.8,"top_k":20,"max_new_tokens":8192}' \
    --host 0.0.0.0 --port 8000

log "等待 v6 启动完成 ..."
for i in $(seq 1 120); do
  if docker logs meerkat-vllm-qwen38 2>&1 | grep -qiE "startup complete|Uvicorn running|Application startup complete"; then
    log "v6 就绪 (第 ${i} 次检查)"
    break
  fi
  # 若容器崩了提前退出
  if ! docker inspect -f '{{.State.Running}}' meerkat-vllm-qwen38 2>/dev/null | grep -q true; then
    log "!! v6 容器已退出, 日志:"
    docker logs meerkat-vllm-qwen38 2>&1 | tail -30
    exit 1
  fi
  sleep 5
done

log "启动 v1 (Qwen3.6-35B-A3B) @ 0.3 + tool calling"
docker run -d --name meerkat-vllm-qwen36 \
  --network host --ipc host --gpus all \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v /home/chinux/.cache/huggingface:/root/.cache/huggingface \
  -v /home/chinux/jupyterlab/meerkatai/models/Qwen3.6-35B-A3B-NVFP4-Fast:/model36 \
  -v /home/chinux/.cache/vllm:/root/.cache/vllm \
  qwen36-dgx-spark:v0.1.1 \
  /model36 \
    --served-model-name Qwen3.6-35B-A3B-NVFP4 \
    --quantization compressed-tensors \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.3 \
    --max-num-seqs 8 \
    --limit-mm-per-prompt '{"image":3,"video":1}' \
    --mm-processor-cache-gb 4 \
    --enable-lora \
    --lora-modules Meerkat-TRIZ-v1-Qwen3.6-35B-A3B=/root/.cache/vllm/loras/meerkat-triz-v1 \
    --max-lora-rank 64 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --host 0.0.0.0 --port 8001

log "等待 v1 启动完成 ..."
for i in $(seq 1 120); do
  if docker logs meerkat-vllm-qwen36 2>&1 | grep -qiE "startup complete|Uvicorn running|Application startup complete"; then
    log "v1 就绪 (第 ${i} 次检查)"
    break
  fi
  if ! docker inspect -f '{{.State.Running}}' meerkat-vllm-qwen36 2>/dev/null | grep -q true; then
    log "!! v1 容器已退出, 日志:"
    docker logs meerkat-vllm-qwen36 2>&1 | tail -30
    exit 1
  fi
  sleep 5
done

log "=== 重启完成 ==="
docker ps --filter name=meerkat-vllm --format '{{.Names}} {{.Status}}'
