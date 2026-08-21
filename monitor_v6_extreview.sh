#!/bin/bash
# v6 异族评委评测监控器 —— 每 N 分钟 SSH 抓进度，区分完成/崩溃/运行中。
set -u
HOST=spark-855a
EVLOG=/home/chinux/jupyterlab/meerkatai/results/v5/ext_review_v6.log
BRIEF=/home/chinux/jupyterlab/meerkatai/results/ext_review_v6/external_review_brief.md
INTERVAL=${1:-600}
OUT=${2:-v6_extreview_monitor.log}
PID_TO_WATCH=2183134

remote_status() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" 'bash -s' <<'REMOTE'
    EVLOG=/home/chinux/jupyterlab/meerkatai/results/v5/ext_review_v6.log
    BRIEF=/home/chinux/jupyterlab/meerkatai/results/ext_review_v6/external_review_brief.md
    if ps -o pid= -p 2183134 >/dev/null 2>&1; then
      echo -n "RUNNING"
    elif [ -f "$BRIEF" ]; then
      echo -n "COMPLETED"
    else
      echo -n "CRASH"
    fi
    last=$(grep -oE '^\[[ 0-9.]+s\] .*' "$EVLOG" 2>/dev/null | tail -1)
    echo -n " | ${last:0:90}"
    echo ""
REMOTE
}

echo "[$(date '+%F %T')] extreview monitor started (pid=${PID_TO_WATCH})" >> "$OUT"
while true; do
  status=$(remote_status 2>/dev/null)
  echo "[$(date '+%F %T')] $status" >> "$OUT"
  case "$status" in
    CRASH*) echo "[$(date '+%F %T')] ★★★ 异族评测崩溃, 需排查 ext_review_v6.log ★★★" >> "$OUT"; break ;;
    COMPLETED*) echo "[$(date '+%F %T')] ★★★ 异族评测完成, 进入台账登记 ★★★" >> "$OUT"; break ;;
  esac
  sleep "$INTERVAL"
done
