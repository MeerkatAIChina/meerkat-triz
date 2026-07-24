#!/bin/bash
# v5 Day1 生成链: base -> v2 -> v4 (GPU 串行), 每段后过 5% 质量门
set -u
cd /home/meerkat/mongoose_ai
PY=venv_v5/bin/python
G=results/v5/gen

gate_check () {
  local tag=$1
  $PY - "$G/raw_${tag}.jsonl" <<'EOF'
import json, sys
raws = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
n = len(raws)
inv = sum(1 for r in raws if r["mode"] == "badwords_invalid")
rate = inv / n if n else 1.0
print(f"{sys.argv[1]}: n={n} invalid={inv} rate={rate:.3f}")
sys.exit(0 if rate <= 0.05 else 1)
EOF
}

echo "=== [$(date +%H:%M:%S)] stage base_v5gold ==="
$PY $G/v5_gen.py --tag base_v5gold \
  --out $G/responses_base_v5gold.jsonl --raw-out $G/raw_base_v5gold.jsonl \
  > $G/gen_base.log 2>&1
gate_check base_v5gold || { echo "ABORT: base >5% invalid" > $G/chain.abort; exit 1; }

echo "=== [$(date +%H:%M:%S)] stage v2_v5gold ==="
$PY $G/v5_gen.py --tag v2_v5gold --adapter models/meerkat_triz_adapter_v2 \
  --base-cache $G/responses_base_v5gold.jsonl \
  --out $G/responses_v2_v5gold.jsonl --raw-out $G/raw_v2_v5gold.jsonl \
  > $G/gen_v2.log 2>&1
gate_check v2_v5gold || { echo "ABORT: v2 >5% invalid" > $G/chain.abort; exit 1; }

echo "=== [$(date +%H:%M:%S)] stage v4_v5gold ==="
$PY $G/v5_gen.py --tag v4_v5gold --adapter models/meerkat_triz_adapter_v4 \
  --base-cache $G/responses_base_v5gold.jsonl \
  --out $G/responses_v4_v5gold.jsonl --raw-out $G/raw_v4_v5gold.jsonl \
  > $G/gen_v4.log 2>&1
gate_check v4_v5gold || { echo "ABORT: v4 >5% invalid" > $G/chain.abort; exit 1; }

$PY $G/v5_gen_report.py > $G/gen_report.log 2>&1
echo "ALL_DONE $(date +%H:%M:%S)" > $G/chain.done
