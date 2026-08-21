#!/bin/bash
# 干净 base 补跑链: E1a' -> E1b补臂 -> E3' -> 分析
set -u
cd /home/meerkat/mongoose_ai
eval "$(grep '^export MOONSHOT_API_KEY' ~/.bashrc)"
PY=venv_v5/bin/python
E1=results/e1

run_step() {
  local name="$1"; local script="$2"; local done="$3"
  if [ -f "$done" ]; then echo "[chain] $name 已完成, 跳过"; return 0; fi
  echo "[chain] === $name 开始 $(date '+%F %T') ==="
  $PY "$script"
  echo "[chain] === $name 结束 rc=$? $(date '+%F %T') ==="
}

run_step E1a_goldfix $E1/e1a_goldfix.py $E1/e1a_goldfix.done
run_step E1b_goldfix $E1/e1b_goldfix.py $E1/e1b_goldfix.done
run_step E3_goldfix results/e3/e3_goldfix.py results/e3/e3_goldfix.done

echo "[chain] === goldfix 分析 $(date '+%F %T') ==="
$PY $E1/e_goldfix_analyze.py || true
echo "[chain] goldfix 全部结束 $(date '+%F %T')"
touch results/goldfix_chain.done
