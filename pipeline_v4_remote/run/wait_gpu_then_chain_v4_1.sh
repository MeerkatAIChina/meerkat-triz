#!/usr/bin/env bash
# wait_gpu_then_chain_v4_1.sh — 等 v5a 训练进程退出后, 自动接力启动 chain_v4_1.sh
# (2026-07-27 主人决策: v4_1 排队在 v5a 之后续跑, 避免 GPU 并发 OOM)
# 用法: tmux new-session -d -s v4_1_queue 'bash pipeline_v4/run/wait_gpu_then_chain_v4_1.sh'
set -uo pipefail

cd "$(dirname "$0")/../.."   # 项目根目录
LOG=data/processed/v4_1_queue.log
mkdir -p data/processed

qlog() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

qlog "看守启动: 等待 pipeline_v5 train.py (v5a) 结束 (每 5 分钟轮询)..."
while pgrep -f "pipeline_v5/src/train.py" >/dev/null 2>&1; do
  sleep 300
done
qlog "v5a 训练进程已退出; 120s 落盘缓冲"
sleep 120

if pgrep -f "pipeline_v5/src/train.py" >/dev/null 2>&1; then
  qlog "缓冲期内 v5 训练进程重现, 放弃本次接力, 需人工处理"
  exit 1
fi

tmux new-session -d -s v4_1 "cd $PWD && bash pipeline_v4/run/chain_v4_1.sh"
qlog "chain_v4_1 已在 tmux 会话 v4_1 中启动 (断点自动续训), 看守退出"
