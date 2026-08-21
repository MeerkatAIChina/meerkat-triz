#!/usr/bin/env bash
# pipeline_v4 串行总链: 等金标 → 数据构建 → base/v2/v3 评测 → v4 训练 → v4 评测 → 汇总
# 可检查点续跑: 每步完成 touch data/processed/v4_chain_state/<step>.done, 已 done 则跳过。
# 用法: bash pipeline_v4/run/chain_v4.sh    (建议 tmux 内运行)
set -euo pipefail

cd "$(dirname "$0")/../.."   # 项目根目录
STATE=data/processed/v4_chain_state
LOG=data/processed/v4_chain.log
mkdir -p "$STATE" results

# 非交互 shell 加载 Moonshot API key (judge 轨需要)
eval "$(grep '^export MOONSHOT_API_KEY' ~/.bashrc)"

step_log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

run_step() {
  # run_step <step_name> <log_file> <cmd...>
  local name="$1"; local logf="$2"; shift 2
  if [ -f "$STATE/$name.done" ]; then
    step_log "== $name: 已完成 (.done 存在), 跳过 =="
    return 0
  fi
  step_log "== $name: 开始 =="
  if "$@" 2>&1 | tee -a "$logf" "$LOG"; then
    touch "$STATE/$name.done"
    step_log "== $name: 完成 =="
  else
    local rc=${PIPESTATUS[0]}
    step_log "== $name: 失败 (exit=$rc), 链条中止; 修复后重跑本脚本将从此步续跑 =="
    exit "$rc"
  fi
}

# ---------- 1. 等待金标集满 100 题 (轮询, 超时 2h 报错退出) ----------
wait_gold() {
  local gold=data/processed/v4_gold.jsonl
  local deadline=$(( $(date +%s) + 7200 ))
  while true; do
    local n=0
    [ -f "$gold" ] && n=$(wc -l < "$gold")
    if [ "$n" -ge 100 ]; then
      step_log "wait_gold: $gold 已有 $n 题, 继续"
      return 0
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      step_log "wait_gold: 超时 2h, 金标集只有 $n/100 题, 链条中止"
      return 1
    fi
    step_log "wait_gold: $n/100 题, 60s 后重查..."
    sleep 60
  done
}

step_log "===== chain_v4 启动 ====="

run_step wait_gold data/processed/v4_chain_wait_gold.log wait_gold

# ---------- 2. 数据构建 ----------
run_step data_build data/processed/v4_chain_data_build.log \
  venv_v5/bin/python pipeline_v4/src/data_build.py \
    --config pipeline_v4/configs/data_v4.json

# ---------- 3. 基线评测三连: 纯 base / v2 / v3 ----------
run_step eval_base data/processed/v4_chain_eval_base.log \
  venv_v5/bin/python pipeline_v4/src/eval_harness.py \
    --config pipeline_v4/configs/eval_v4.json --tag base_gold

run_step eval_v2 data/processed/v4_chain_eval_v2.log \
  venv_v5/bin/python pipeline_v4/src/eval_harness.py \
    --config pipeline_v4/configs/eval_v4.json \
    --adapter-path models/meerkat_triz_adapter_v2 --tag v2_gold

run_step eval_v3 data/processed/v4_chain_eval_v3.log \
  venv_v5/bin/python pipeline_v4/src/eval_harness.py \
    --config pipeline_v4/configs/eval_v4.json \
    --adapter-path models/meerkat_triz_adapter_v3 --tag v3_gold

# ---------- 4. v4 训练 ----------
run_step train_v4 checkpoints/train_v4_chain.log \
  bash pipeline_v4/run/train_v4.sh

# ---------- 5. v4 评测 ----------
run_step eval_v4 data/processed/v4_chain_eval_v4.log \
  venv_v5/bin/python pipeline_v4/src/eval_harness.py \
    --config pipeline_v4/configs/eval_v4.json \
    --adapter-path models/meerkat_triz_adapter_v4 --tag v4_gold

# ---------- 6. 汇总报告 + 决策门 ----------
run_step final_report data/processed/v4_chain_final_report.log \
  venv_v5/bin/python pipeline_v4/src/final_report.py

step_log "===== chain_v4 全部完成 → results/v4_final_report.md ====="
