#!/usr/bin/env bash
# 等待旧评测 triz_eval_v3 结束后启动 eval2 流水线
echo "[watcher] $(date '+%F %T') 等待 triz_eval_v3 结束..."
while tmux has-session -t triz_eval_v3 2>/dev/null; do
  sleep 60
done
echo "[watcher] $(date '+%F %T') triz_eval_v3 已结束，启动 run_all.sh"
exec /tmp/eval_pipeline_v2/run_all.sh
