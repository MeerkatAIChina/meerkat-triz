#!/usr/bin/env bash
# v5 Day2 阶段2: top-2 臂 40 题金标双轨冒烟链 (Worker M)
# base/v2 走完整生成缓存(不碰 GPU); 两 sweep 臂 GPU 生成 + judge; 全程串行
set -u
cd /home/meerkat/mongoose_ai
eval "$(grep "^export MOONSHOT_API_KEY" ~/.bashrc)"
PY=venv_v5/bin/python
CFG=pipeline_v5/eval/configs/eval_v5.json
GOLD=data/processed/v5_data/v5_gold_smoke40.jsonl
LOG=results/v5/smoke_chain.log
echo "=== SMOKE CHAIN START $(date) ===" >> $LOG

run() {
  local tag=$1; shift
  echo "--- RUN $tag $(date) ---" >> $LOG
  $PY pipeline_v5/eval/eval_harness_v5.py --config $CFG --eval-file $GOLD --tag $tag "$@" >> $LOG 2>&1
  local rc=$?
  echo "--- END $tag rc=$rc $(date) ---" >> $LOG
  return $rc
}

run base_goldfix_smoke40 || { echo "CHAIN_ABORTED at base" >> $LOG; exit 1; }
BASE_JSON=$(ls -t results/v5/eval_v5_base_goldfix_smoke40_*.json | head -1)
echo "baseline: $BASE_JSON" >> $LOG

run v2_smoke40 --adapter-path models/meerkat_triz_adapter_v2 --baseline-results "$BASE_JSON" || { echo "CHAIN_ABORTED at v2" >> $LOG; exit 1; }

run sweep2e4_smoke40 --adapter-path models/sweep_adapters/sweep_lr2e-4_rsFalse --baseline-results "$BASE_JSON" || { echo "CHAIN_ABORTED at sweep2e4" >> $LOG; exit 1; }

run sweep5e4_smoke40 --adapter-path models/sweep_adapters/sweep_lr5e-4_rsFalse --baseline-results "$BASE_JSON" || { echo "CHAIN_ABORTED at sweep5e4" >> $LOG; exit 1; }

echo "=== SMOKE_CHAIN_COMPLETE $(date) ===" >> $LOG
touch results/v5/smoke_chain.done
