#!/usr/bin/env bash
# pipeline_v4.1 串行链: 数据构建(修复版 rebalance) → v4.1 训练 → v4.1 评测 → 五方汇总 + 决策门
# 与 chain_v4.sh 的差异:
#   - 复用 v4 链的金标集 (v4_gold.jsonl) 与 base/v2/v3/v4 评测结果, 不重跑;
#   - v4.1 评测锚点用 v2 (干净锚点; stats_review §2.3 证实 base 锚点被 think 污染,
#     vs-base 提升幅度全部高估, 不可用于判读);
#   - 成功判读重点: keyword/concept_explanation 差值 CI 是否回到包含 0 (相对 v2 不再
#     显著为负, 即 v4 的 -0.083 [-0.148,-0.023] 被修复), 而非必须显著超过 v2。
# 可检查点续跑: 每步完成 touch data/processed/v4_1_chain_state/<step>.done。
# 用法: bash pipeline_v4/run/chain_v4_1.sh    (建议 tmux 内运行)
set -euo pipefail
set -o pipefail

cd "$(dirname "$0")/../.."   # 项目根目录
STATE=data/processed/v4_1_chain_state
LOG=data/processed/v4_1_chain.log
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

step_log "===== chain_v4_1 启动 ====="

# ---------- 0. 前置检查: 金标集与 base/v2/v3/v4 评测结果必须已存在 (来自 v4 链) ----------
GOLD=data/processed/v4_gold.jsonl
N_GOLD=0; [ -f "$GOLD" ] && N_GOLD=$(wc -l < "$GOLD")
if [ "$N_GOLD" -lt 100 ]; then
  step_log "前置检查失败: 金标集 $GOLD 只有 $N_GOLD/100 题; 先跑 chain_v4.sh"
  exit 1
fi
for t in base_gold v2_gold v3_gold v4_gold; do
  if ! ls results/eval_v4_"$t"_*.json >/dev/null 2>&1; then
    step_log "前置检查失败: 缺 results/eval_v4_${t}_*.json; 先跑 chain_v4.sh"
    exit 1
  fi
done
step_log "前置检查通过: 金标 $N_GOLD 题, base/v2/v3/v4 评测结果齐备"

# ---------- 1. 数据构建 (v4.1: term_coverage_random, cap 2500 → data/processed/v4_1_*.jsonl) ----------
run_step data_build data/processed/v4_1_chain_data_build.log \
  venv_v5/bin/python pipeline_v4/src/data_build.py \
    --config pipeline_v4/configs/data_v4.json

# ---------- 2. v4.1 训练 (train_v4.json: run_name=v4.1 → checkpoints/qlora_triz_v4_1) ----------
# 断点自动续训: 存在 checkpoint-* 时从最新一个 --resume (2026-07-27 v4_1 曾在 step~200
# 被外部 kill, 无错误现场; checkpoint-100 + best/ 在盘可续)
LAST_CKPT=$(ls -d checkpoints/qlora_triz_v4_1/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
RESUME_ARG=()
if [ -n "$LAST_CKPT" ]; then
  step_log "发现断点 $LAST_CKPT, 训练将从该 checkpoint 续跑"
  RESUME_ARG=(--resume "$LAST_CKPT")
fi
run_step train_v4_1 checkpoints/train_v4_1_chain.log \
  bash pipeline_v4/run/train_v4.sh "${RESUME_ARG[@]}"

# ---------- 3. v4.1 评测 (干净锚点 v2) ----------
V2_BASELINE=$(ls results/eval_v4_v2_gold_*.json | sort | tail -1)
step_log "v4.1 评测基线: $V2_BASELINE"
run_step eval_v4_1 data/processed/v4_1_chain_eval_v4_1.log \
  venv_v5/bin/python pipeline_v4/src/eval_harness.py \
    --config pipeline_v4/configs/eval_v4.json \
    --adapter-path models/meerkat_triz_adapter_v4_1 --tag v4_1_gold \
    --baseline-results "$V2_BASELINE"

# ---------- 4. 五方汇总 + 决策门 (candidate=v4_1_gold) ----------
run_step final_report data/processed/v4_1_chain_final_report.log \
  venv_v5/bin/python pipeline_v4/src/final_report.py \
    --tags base_gold v2_gold v3_gold v4_gold v4_1_gold \
    --candidate v4_1_gold --out results/v4_1_final_report.md

step_log "===== chain_v4_1 全部完成 → results/v4_1_final_report.md ====="
