#!/bin/bash
# v3 流水线接力器：等待 ariz boost 完成 → 构建 v3 数据+扩充评测集 → 启动 v3 训练
set -u
cd /home/meerkat/mongoose_ai || exit 1

echo "[chain] 等待 triz_ariz_boost 完成..."
while tmux has-session -t triz_ariz_boost 2>/dev/null; do sleep 30; done
echo "[chain] boost 已结束，开始构建 v3 数据"

if ! venv_v5/bin/python /tmp/build_v3_and_evalset.py 2>&1 | tee data/processed/build_v3.log; then
    echo "[chain] 构建失败，中止（不启动训练）"
    exit 1
fi

# 校验 v3_train.jsonl 行数
N=$(wc -l < data/processed/v3_train.jsonl)
if [ "$N" -le 8000 ]; then
    echo "[chain] v3_train.jsonl 行数异常 ($N)，中止"
    exit 1
fi

echo "[chain] 构建完成 (v3_train=$N 行)，启动 v3 训练"
tmux new-session -d -s triz_train_v3 "cd /home/meerkat/mongoose_ai && venv_v5/bin/python scripts/train_qlora.py --run-name v3 --train-file data/processed/v3_train.jsonl --val-file data/processed/v3_validation.jsonl 2>&1 | tee checkpoints/train_v3.log"
echo "[chain] triz_train_v3 已启动"
