#!/usr/bin/env bash
# pipeline_v4 训练单入口
# 用法: bash pipeline_v4/run/train_v4.sh [--dry-run] [--resume <ckpt_dir>] ...
set -euo pipefail
set -o pipefail

cd "$(dirname "$0")/../.."   # 项目根目录
mkdir -p checkpoints

LOG=checkpoints/train_v4.log
echo "===== train_v4 启动 $(date '+%Y-%m-%d %H:%M:%S') =====" | tee -a "$LOG"

venv_v5/bin/python pipeline_v4/src/train.py \
  --config pipeline_v4/configs/train_v4.json \
  "$@" 2>&1 | tee -a "$LOG"

exit_code=${PIPESTATUS[0]}
echo "===== train_v4 结束 $(date '+%Y-%m-%d %H:%M:%S') exit=$exit_code =====" | tee -a "$LOG"
exit "$exit_code"
