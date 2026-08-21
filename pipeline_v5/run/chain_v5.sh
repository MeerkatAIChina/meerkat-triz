#!/usr/bin/env bash
# pipeline_v5 串行总链: 等金标 → 合并 → base/v2 评测 → v5 评测 → 决策门
# 断点续跑: 每步 touch data/processed/v5_chain_state/<step>.done
# 用法: bash pipeline_v5/run/chain_v5.sh (建议 tmux 内运行)
set -euo pipefail

cd "$(dirname "$0")/../.."
STATE=data/processed/v5_chain_state
LOG=data/processed/v5_chain.log
mkdir -p "$STATE" results/v5

eval "$(grep '^export MOONSHOT_API_KEY' ~/.bashrc)"

step_log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

run_step() {
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

# ---------- 1. 等待金标集满 300 题 ----------
wait_gold() {
  local gold=data/processed/v5_data/v5_gold.jsonl
  local deadline=$(( $(date +%s) + 7200 ))
  while true; do
    local n=0
    [ -f "$gold" ] && n=$(wc -l < "$gold")
    if [ "$n" -ge 300 ]; then
      step_log "wait_gold: $gold 已有 $n 题, 继续"
      return 0
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      step_log "wait_gold: 超时 2h, 金标集只有 $n/300 题, 链条中止"
      return 1
    fi
    step_log "wait_gold: $n/300 题, 60s 后重查..."
    sleep 60
  done
}

step_log "===== chain_v5 启动 ====="

run_step wait_gold data/processed/v5_chain_wait_gold.log wait_gold

# ---------- 2. 合并 300 题金标 (idempotent) ----------
run_step assemble_gold data/processed/v5_chain_assemble_gold.log \
  bash -c 'cd /home/meerkat/mongoose_ai && \
    cat data/processed/v4_gold.jsonl \
        data/processed/v5_data/v5_gold_new100.jsonl \
        data/processed/v5_data/v5_gold_201_300.jsonl \
      > data/processed/v5_data/v5_gold.jsonl && \
    echo "金标合并完成: $(wc -l < data/processed/v5_data/v5_gold.jsonl) 题"'

# ---------- 3. 基线评测: base + v2 ----------
run_step eval_base data/processed/v5_chain_eval_base.log \
  venv_v5/bin/python pipeline_v5/eval/eval_harness_v5.py \
    --config pipeline_v5/eval/configs/eval_v5.json \
    --eval-file data/processed/v5_data/v5_gold.jsonl \
    --tag base_goldfix_v5

run_step eval_v2 data/processed/v5_chain_eval_v2.log \
  venv_v5/bin/python pipeline_v5/eval/eval_harness_v5.py \
    --config pipeline_v5/eval/configs/eval_v5.json \
    --eval-file data/processed/v5_data/v5_gold.jsonl \
    --adapter-path models/meerkat_triz_adapter_v2 \
    --tag v2_gold_v5

# ---------- 4. v5 评测 ----------
run_step eval_v5 data/processed/v5_chain_eval_v5.log \
  venv_v5/bin/python pipeline_v5/eval/eval_harness_v5.py \
    --config pipeline_v5/eval/configs/eval_v5.json \
    --eval-file data/processed/v5_data/v5_gold.jsonl \
    --adapter-path models/meerkat_triz_adapter_v5 \
    --tag v5_gold

# ---------- 5. 决策门终审 ----------
run_step decision_gate data/processed/v5_chain_decision_gate.log \
  venv_v5/bin/python pipeline_v5/eval/decision_gate.py \
    results/v5/v5_scores.json

step_log "===== chain_v5 全部完成 ====="
