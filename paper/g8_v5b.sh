#!/usr/bin/env bash
# v5b G8 异源评委复核（决策门 G8 数据源）
# 前置: 候选评测 json 已从远端取回至 paper/external_review_v5b/
# 用法: TENSORIS_API_KEY=sk-... bash g8_v5b.sh <candidate_eval_json文件名>
# 密钥只走环境变量，不落盘、不入库。
set -euo pipefail
cd "$(dirname "$0")"
: "${TENSORIS_API_KEY:?需要 TENSORIS_API_KEY 环境变量}"

CAND="${1:?用法: bash g8_v5b.sh <candidate_eval_json文件名>}"
ROOT="/Volumes/2nd-HD/claude/Meerkat-AI"

python3 "$ROOT/pipeline_v5/eval/external_judge_track.py" \
  --candidate-json "external_review_v5b/$CAND" \
  --anchor-json    "external_review_v5b/eval_v5_base_goldfix_v5_20260726_234434.json" \
  --gold-jsonl     "external_review_v5b/v5_gold.jsonl" \
  --cmp-name v5_vs_base \
  --workdir external_review_v5b \
  --judges claude-sonnet-4-6 gpt-5.4 gemini-3.5-flash \
  --time-budget 7200

echo "== G8 复核完成, 碎片在 external_review_v5b/, 并入 scores.json 后跑 decision_gate =="
