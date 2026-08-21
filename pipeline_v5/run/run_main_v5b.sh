#!/usr/bin/env bash
# v5b 主训练 (Worker M): 预算帽 7h; 续跑: ./run_main_v5b.sh --resume
# 修复(2026-07-31): --resume 必须指向具体 checkpoint-XXXX 子目录(含 trainer_state.json),
# 传父目录会 FileNotFoundError: trainer_state.json 崩溃循环。
set -u
cd /home/meerkat/mongoose_ai
PY=venv_v5/bin/python
CFG=pipeline_v5/configs/train_v5b.json
CKPT_DIR=checkpoints/qlora_triz_v5b
LOG=$CKPT_DIR/train.log
mkdir -p "$CKPT_DIR"
RESUME_ARG=""
if [ "${1:-}" = "--resume" ]; then
  LAST_CKPT=$(ls "$CKPT_DIR" 2>/dev/null | grep -oE 'checkpoint-[0-9]+' | sort -t- -k2 -n | tail -1)
  if [ -n "$LAST_CKPT" ] && [ -f "$CKPT_DIR/$LAST_CKPT/trainer_state.json" ]; then
    RESUME_ARG="--resume $CKPT_DIR/$LAST_CKPT"
  fi
fi
echo "[main] $(date -Is) v5b 主训练启动 (config=$CFG resume=${RESUME_ARG:-no})"
timeout 25200 $PY pipeline_v5/src/train.py --config "$CFG" $RESUME_ARG > "$LOG" 2>&1
rc=$?
echo "[main] $(date -Is) v5b 主训练结束 rc=$rc (124=预算帽超时)"
exit $rc
