#!/usr/bin/env bash
# v5 主训练 (Worker M): 预算帽 7h; 续跑: ./run_main_v5.sh --resume
set -u
cd /home/meerkat/mongoose_ai
PY=venv_v5/bin/python
CFG=pipeline_v5/configs/train_v5.json
CKPT_DIR=checkpoints/qlora_triz_v5
LOG=$CKPT_DIR/train.log
mkdir -p "$CKPT_DIR"
RESUME_ARG=""
[ "${1:-}" = "--resume" ] && RESUME_ARG="--resume $CKPT_DIR"
echo "[main] $(date -Is) v5 主训练启动 (config=$CFG resume=${RESUME_ARG:-no})"
timeout 25200 $PY pipeline_v5/src/train.py --config "$CFG" $RESUME_ARG > "$LOG" 2>&1
rc=$?
echo "[main] $(date -Is) v5 主训练结束 rc=$rc (124=预算帽超时)"
exit $rc
