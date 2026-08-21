#!/bin/bash
# v6 评测监控器 —— 每 N 分钟 SSH 抓一次评测进度，区分完成/崩溃/运行中。
# 用法: bash monitor_v6_eval.sh [间隔秒=600] [日志文件=v6_eval_monitor.log]
set -u

HOST=spark-855a
EVLOG=/home/chinux/jupyterlab/meerkatai/results/v5/eval_v6_gold.log
RESDIR=/home/chinux/jupyterlab/meerkatai/results/v5
INTERVAL=${1:-600}
OUT=${2:-v6_eval_monitor.log}
PID_TO_WATCH=1304634

remote_status() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" 'bash -s' <<'REMOTE'
    EVLOG=/home/chinux/jupyterlab/meerkatai/results/v5/eval_v6_gold.log
    RESDIR=/home/chinux/jupyterlab/meerkatai/results/v5
    if ps -o pid= -p 1304634 >/dev/null 2>&1; then
      echo -n "RUNNING"
    elif ls "$RESDIR"/eval_v5_v6_gold_*.json >/dev/null 2>&1; then
      echo -n "COMPLETED"
    else
      echo -n "CRASH"
    fi
    # 最后一条带时间戳的进度日志
    last=$(grep -oE '^\[[0-9:]+\] [^]]*$' "$EVLOG" 2>/dev/null | tail -1)
    # 生成进度 (若 tqdm 生成条)
    gen=$(grep -oE '[0-9]+/300 \[[0-9:]+<' "$EVLOG" 2>/dev/null | tail -1 | sed 's/ \[.*//')
    echo -n " | last=${last:-?}"
    [ -n "$gen" ] && echo -n " | gen=${gen}/300"
    echo ""
REMOTE
}

echo "[$(date '+%F %T')] eval monitor started (interval=${INTERVAL}s, pid=${PID_TO_WATCH})" >> "$OUT"

while true; do
  status=$(remote_status 2>/dev/null)
  echo "[$(date '+%F %T')] $status" >> "$OUT"
  case "$status" in
    CRASH*)
      echo "[$(date '+%F %T')] ★★★ 评测进程死亡且无报告 → 需排查 ★★★" >> "$OUT"
      echo "[$(date '+%F %T')] 查 eval_v6_gold.log 尾部定位失败原因" >> "$OUT"
      break
      ;;
    COMPLETED*)
      echo "[$(date '+%F %T')] ★★★ 评测完成 → 进入台账登记 ★★★" >> "$OUT"
      break
      ;;
  esac
  sleep "$INTERVAL"
done
