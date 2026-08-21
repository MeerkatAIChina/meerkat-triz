#!/bin/bash
# Worker H 种子扩写: 等 Worker E (styleC 长答, tmux v5gen) 释放 API 额度后再启动
cd /home/meerkat/mongoose_ai
TARGET=3445
echo "=== [$(date +%H:%M:%S)] Worker H 等待 Worker E 完成 (目标 $TARGET 行) ==="
while true; do
  n=$(wc -l < data/processed/v5_data/styleC_long_answers.jsonl 2>/dev/null || echo 0)
  if [ "$n" -ge "$TARGET" ]; then
    echo "=== [$(date +%H:%M:%S)] E 完成: lines=$n >= $TARGET ==="; break
  fi
  if tail -20 data/processed/v5_data/workerE_gen.log 2>/dev/null | grep -q "完成:"; then
    echo "=== [$(date +%H:%M:%S)] E 日志出现完成字样 (lines=$n) ==="; break
  fi
  if ! pgrep -f "styleC_gen_v5.py" >/dev/null && ! pgrep -f "safety_gen_v5.py" >/dev/null; then
    echo "=== [$(date +%H:%M:%S)] E 进程已退出 (lines=$n) ==="; break
  fi
  sleep 300
done
sleep 30   # 尾批落盘宽限
eval "$(grep '^export MOONSHOT_API_KEY' ~/.bashrc)"
echo "=== [$(date +%H:%M:%S)] Worker H 种子扩写启动 ==="
venv_v5/bin/python pipeline_v5/src/seed_expand_v5.py 2>&1 | tee -a data/processed/v5_data/seed_expand_gen.log
echo "=== [$(date +%H:%M:%S)] 扩写结束, 合并与质量门终检 ==="
venv_v5/bin/python pipeline_v5/src/seed_expand_finalize.py 2>&1 | tee data/processed/v5_data/seed_expand_finalize.log
echo "=== [$(date +%H:%M:%S)] Worker H 全部完成 ==="
