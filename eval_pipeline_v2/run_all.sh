#!/usr/bin/env bash
# eval2 一键执行：generate -> score -> report
# 用法: /tmp/eval_pipeline_v2/run_all.sh
set -uo pipefail

cd /home/meerkat/mongoose_ai
eval "$(grep '^export MOONSHOT_API_KEY' ~/.bashrc)" || echo "[run_all] 警告: MOONSHOT_API_KEY 未加载，judge 轨将失败"

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p results/eval2
LOG=results/eval2/eval2_${TS}.log

{
  echo "[run_all] start $(date '+%F %T')"

  # 防御性二次确认：GPU 阶段前确保旧评测已结束（watcher 之外的双保险）
  while tmux has-session -t triz_eval_v3 2>/dev/null; do
    echo "[run_all] triz_eval_v3 仍在运行，60s 后重查..."
    sleep 60
  done
  echo "[run_all] GPU 空闲确认，开始 generate"

  venv_v5/bin/python /tmp/eval_pipeline_v2/eval2.py --phase generate --models base,v1,v2,v3
  GEN_RC=$?
  echo "[run_all] generate rc=${GEN_RC}"

  venv_v5/bin/python /tmp/eval_pipeline_v2/eval2.py --phase score --models base,v1,v2,v3
  SCORE_RC=$?
  echo "[run_all] score rc=${SCORE_RC}"

  venv_v5/bin/python /tmp/eval_pipeline_v2/eval2.py --phase report --models base,v1,v2,v3
  REPORT_RC=$?
  echo "[run_all] report rc=${REPORT_RC}"

  if [ "${GEN_RC}" -eq 0 ] && [ "${SCORE_RC}" -eq 0 ] && [ "${REPORT_RC}" -eq 0 ]; then
    echo "EVAL2_ALL_DONE"
  else
    echo "EVAL2_FINISHED_WITH_ERRORS gen=${GEN_RC} score=${SCORE_RC} report=${REPORT_RC}"
  fi
} 2>&1 | tee "${LOG}"
