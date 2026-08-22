#!/bin/bash
# 起 SGLang + DFlash 实验实例 (v1: Qwen3.6-35B-A3B) — pip 方式 (venv_sglang)
# ⚠️ GB10/aarch64 关键: 必须 --attention-backend triton, 禁用默认 flashinfer
#     (flashinfer 在 GB10 有已知 bug, 曾导致系统挂起)
# ⚠️ LoRA + NVFP4 MoE 是 SGLang 功能缺口 (get_triton_quant_info 缺失), 当前不支持
set -e
log() { echo "[$(date '+%H:%M:%S')] $*"; }

MEERKAT=/home/chinux/jupyterlab/meerkatai
VENV=$MEERKAT/venv_sglang
MODEL=$MEERKAT/models/Qwen3.6-35B-A3B-NVFP4-Fast
DRAFT=/home/chinux/.cache/huggingface/hub/models--z-lab--Qwen3.6-35B-A3B-DFlash
LOG=$MEERKAT/sglang_dflash_v1.log

log "清理旧 SGLang 进程"
pkill -f "sglang.launch_server" 2>/dev/null || true
sleep 2

log "启动 SGLang v1 + DFlash (端口 30000, 无 LoRA, triton backend)"
source "$VENV/bin/activate"
nohup python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name Qwen3.6-35B-A3B-NVFP4 \
  --trust-remote-code \
  --dtype bfloat16 \
  --context-length 32768 \
  --mem-fraction-static 0.35 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend triton \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DRAFT" \
  --speculative-draft-attention-backend triton \
  --speculative-dflash-block-size 8 \
  --host 0.0.0.0 \
  --port 30000 \
  > "$LOG" 2>&1 &

log "SGLang 进程 PID: $!"
log "等待启动 (最长 10 分钟) ..."
for i in $(seq 1 120); do
  if curl -s --max-time 3 http://127.0.0.1:30000/v1/models >/dev/null 2>&1; then
    log "SGLang v1 就绪 (第 ${i} 次检查)"
    break
  fi
  if ! pgrep -f "sglang.launch_server" >/dev/null 2>&1; then
    log "!! SGLang 进程已退出, 日志尾部:"
    tail -40 "$LOG"
    exit 1
  fi
  sleep 5
done

log "=== 启动完成 ==="
curl -s --max-time 5 http://127.0.0.1:30000/v1/models
echo ""
