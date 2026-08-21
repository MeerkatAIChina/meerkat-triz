#!/bin/bash
# v6 (Qwen3.8-27B) 训练监控器 —— 每 N 分钟 SSH 抓一次进度。
# 能区分三种终态: 正常完成(COMPLETED) / 崩溃(CRASH) / 仍在跑(ALIVE)。
# 用法: bash monitor_v6_qwen38.sh [间隔秒=600] [日志文件=v6_monitor.log]
set -u

HOST=spark-855a
BASE=/home/chinux/jupyterlab/meerkatai/checkpoints/qlora_triz_v6_qwen38
RS=/home/chinux/jupyterlab/meerkatai/results/run_summary_v6_qwen38_main.json
INTERVAL=${1:-600}
OUT=${2:-v6_monitor.log}
PID_TO_WATCH=518663

remote_status() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" 'bash -s' <<'REMOTE'
    BASE=/home/chinux/jupyterlab/meerkatai/checkpoints/qlora_triz_v6_qwen38
    RS=/home/chinux/jupyterlab/meerkatai/results/run_summary_v6_qwen38_main.json
    if ps -o pid= -p 518663 >/dev/null 2>&1; then
      echo -n "ALIVE"
    elif grep -q "训练流程结束" "$BASE/train.log" 2>/dev/null || [ -f "$RS" ]; then
      echo -n "COMPLETED"
    else
      echo -n "CRASH"
    fi
    step=$(grep -oE '[0-9]+/5548 \[[0-9:]+<' "$BASE/train.log" 2>/dev/null | tail -1 | sed 's/ \[[0-9:]*<.*//')
    ev=$(grep -oE "'eval_loss': '[0-9.]+'" "$BASE/train.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+')
    nb=$(grep -oE 'NEW BEST: eval_loss=[0-9.]+ @ eval_step=[0-9]+' "$BASE/train.log" 2>/dev/null | tail -1)
    echo -n " | step=${step:-?}"
    echo -n " | eval_loss=${ev:-?}"
    [ -n "$nb" ] && echo -n " | $nb"
    echo ""
REMOTE
}

echo "[$(date '+%F %T')] monitor started (interval=${INTERVAL}s, pid=${PID_TO_WATCH})" >> "$OUT"

while true; do
  status=$(remote_status 2>/dev/null)
  echo "[$(date '+%F %T')] $status" >> "$OUT"
  case "$status" in
    CRASH*)
      echo "[$(date '+%F %T')] ★★★ 训练进程死亡且无完成标记 → 需续训 ★★★" >> "$OUT"
      echo "[$(date '+%F %T')] 续训命令见 V6_RUNBOOK.md「崩溃续训」章节" >> "$OUT"
      break
      ;;
    COMPLETED*)
      echo "[$(date '+%F %T')] ★★★ 训练正常结束 → 进入评测阶段 ★★★" >> "$OUT"
      echo "[$(date '+%F %T')] 评测命令见 V6_RUNBOOK.md「训练完成后的评测」章节" >> "$OUT"
      break
      ;;
  esac
  sleep "$INTERVAL"
done
