#!/usr/bin/env bash
# eval_v5a_then_chain_v4_1.sh — 先跑 v5a 金标评测, 结束后(无论成败)接回 v4_1 训练链
# 2026-07-28 主人指令: v5a 金标评测优先, v4_1 训练排在其后 (v4_1 从最新 checkpoint 自动续训)。
# 注意: tag 必须用 v5a_gold —— 旧 tag v5_gold 的生成/judge 缓存属于 backup 适配器, 复用会张冠李戴。
# 用法: tmux new-session -d -s v5a_eval 'bash pipeline_v4/run/eval_v5a_then_chain_v4_1.sh'
set -uo pipefail

cd "$(dirname "$0")/../.."   # 项目根目录
LOG=data/processed/v5a_eval_then_v4_1.log
mkdir -p data/processed

qlog() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# judge 轨需要 Moonshot API key
eval "$(grep '^export MOONSHOT_API_KEY' ~/.bashrc)"

qlog "== v5a 金标评测开始 (adapter=meerkat_triz_adapter_v5, tag=v5a_gold, baseline=base_goldfix_v5) =="
venv_v5/bin/python pipeline_v5/eval/eval_harness_v5.py \
  --config pipeline_v5/eval/configs/eval_v5.json \
  --eval-file data/processed/v5_data/v5_gold.jsonl \
  --adapter-path models/meerkat_triz_adapter_v5 \
  --tag v5a_gold 2>&1 | tee -a data/processed/v5a_eval.log "$LOG"
rc=${PIPESTATUS[0]}
if [ "$rc" -eq 0 ]; then
  qlog "== v5a 评测完成 → results/v5/eval_v5_v5a_gold_*.json =="
else
  qlog "== v5a 评测失败 (exit=$rc); 仍按计划接回 v4_1 训练, 评测可稍后重跑 (生成/judge 缓存续跑) =="
fi

qlog "== 启动 chain_v4_1 (断点自动续训, 从最新 checkpoint) =="
tmux new-session -d -s v4_1 "cd $PWD && bash pipeline_v4/run/chain_v4_1.sh"
qlog "chain_v4_1 已在 tmux 会话 v4_1 中启动, 本脚本退出 (rc=$rc)"
exit "$rc"
