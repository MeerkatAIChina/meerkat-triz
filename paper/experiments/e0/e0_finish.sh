#!/bin/bash
# E0 自动收尾: 等生成满 100 → harness 汇总 → e0_stats 配对统计
set -u
cd /home/meerkat/mongoose_ai
GEN=results/v4_gen_base_goldfix.jsonl
echo "[waiter] 等待生成满 100 ..."
while [ "$(wc -l < $GEN)" -lt 100 ]; do sleep 30; done
echo "[waiter] 生成满 100, 等 judge 驱动收尾 (最多 10 分钟) ..."
for i in $(seq 1 20); do
  if grep -q JUDGE_ALL_DONE results/e0_basefix/judge_driver.log 2>/dev/null; then break; fi
  sleep 30
done
eval "$(grep '^export MOONSHOT_API_KEY' ~/.bashrc)"
echo "[waiter] harness 汇总 (全缓存命中) ..."
venv_v5/bin/python pipeline_v4/src/eval_harness.py \
  --config pipeline_v4/configs/eval_v4.json --tag base_goldfix \
  2>&1 | tee results/e0_basefix/harness_basefix.log
TSJSON=$(ls -t results/eval_v4_base_goldfix_*.json | head -1)
echo "[waiter] 统计: baseline=$TSJSON"
venv_v5/bin/python results/e0_basefix/e0_stats.py \
  --base "$TSJSON" \
  --v2 results/eval_v4_v2_gold_20260723_124807.json \
  --v4 results/eval_v4_v4_gold_20260724_004355.json \
  --base-polluted results/eval_v4_base_gold_20260723_105438.json \
  --out-json results/e0_basefix/e0_stats.json \
  --out-md results/e0_basefix/e0_stats_report.md \
  2>&1 | tee results/e0_basefix/stats.log
echo "[waiter] E0_FINISH_DONE"
