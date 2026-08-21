#!/bin/bash
# P0 实验包总链: E1a -> E1b -> E1c -> E3 -> E2, 每步 .done 断点续跑
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
  local rc=$?
  echo "[chain] === $name 结束 rc=$rc $(date '+%F %T') ==="
  return $rc
}

# E1a 位置交换双跑
run_step E1a $E1/e1a_position_swap.py $E1/e1a.done
# E1b 多评委交叉
run_step E1b $E1/e1b_rejudge.py $E1/e1b.done
# E1c 翻转率
run_step E1c $E1/e1c_flip.py $E1/e1c.done
# E3 ARIZ rubric
run_step E3 results/e3/e3_ariz_rubric.py results/e3/e3.done
# E2 concept 归因
run_step E2 results/e2/e2_concept.py results/e2/e2.done

# 分析 (各 .done 到位后)
echo "[chain] === 分析 $(date '+%F %T') ==="
$PY $E1/e1_analyze.py || true
$PY results/e3/e3_analyze.py || true
echo "[chain] 全部结束 $(date '+%F %T')"
touch results/p0_chain.done
