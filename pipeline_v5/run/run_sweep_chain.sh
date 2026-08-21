#!/usr/bin/env bash
# v5 Day2 P0 六组小扫 GPU 串行链 (tmux 会话 v5sweep 内运行)
# 纪律: 单组失败不中断后续组 (记录 .failed 继续);
#       断言/冒烟类失败 (退出码 3) 立即停止全链 (CHAIN_ABORTED);
#       每组完成写 checkpoints/sweep_status/<arm>.done (断点续跑跳过)。
set -u
cd /home/meerkat/mongoose_ai
PY=venv_v5/bin/python
SWEEP_DIR=pipeline_v5/configs/sweep
STATUS_DIR=checkpoints/sweep_status
mkdir -p "$STATUS_DIR"

# 顺序: 默认臂 2e-4/False 第一 (验证链健康), 其后逐组
ARMS=(
  sweep_lr2e-4_rsFalse
  sweep_lr2e-4_rsTrue
  sweep_lr1e-4_rsFalse
  sweep_lr1e-4_rsTrue
  sweep_lr5e-4_rsFalse
  sweep_lr5e-4_rsTrue
)

echo "[chain] $(date -Is) P0 小扫链启动, 共 ${#ARMS[@]} 组"
for arm in "${ARMS[@]}"; do
  done_f="$STATUS_DIR/$arm.done"
  if [ -f "$done_f" ]; then
    echo "[chain] $arm 已完成 (.done 存在), 跳过"
    continue
  fi
  cfg="$SWEEP_DIR/$arm.json"
  log="checkpoints/$arm/train.log"
  mkdir -p "checkpoints/$arm"
  echo "[chain] $(date -Is) 启动 $arm (config=$cfg)"
  start=$(date +%s)
  $PY pipeline_v5/src/train.py --config "$cfg" > "$log" 2>&1
  rc=$?
  dur=$(( $(date +%s) - start ))
  if [ $rc -eq 0 ]; then
    touch "$done_f"
    echo "[chain] $(date -Is) $arm 完成, 用时 ${dur}s"
  else
    printf 'exit=%s duration_s=%s time=%s\n' "$rc" "$dur" "$(date -Is)" > "$STATUS_DIR/$arm.failed"
    echo "[chain] $(date -Is) $arm 失败 rc=$rc, 用时 ${dur}s (详见 $log)"
    if [ $rc -eq 3 ]; then
      echo "[chain] $(date -Is) 断言/冒烟类失败 (rc=3), 按纪律停止全链" | tee "$STATUS_DIR/CHAIN_ABORTED"
      exit 3
    fi
  fi
done
echo "[chain] $(date -Is) 全部组处理完毕"
touch "$STATUS_DIR/CHAIN_COMPLETE"
