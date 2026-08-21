#!/bin/bash
# Meerkat tool bridge 守护脚本: 崩溃自动重启
# 配合 crontab @reboot 实现开机自启 (无需 sudo)
cd /home/chinux/jupyterlab/meerkatai || exit 1
source /home/chinux/jupyterlab/meerkatai/venv_v5/bin/activate

LOG=/home/chinux/jupyterlab/meerkatai/tool_bridge.log

while true; do
  echo "[$(date '+%F %T')] 启动 tool_bridge.py" >> "$LOG"
  python3 /home/chinux/jupyterlab/meerkatai/tool_bridge.py >> "$LOG" 2>&1
  echo "[$(date '+%F %T')] tool_bridge.py 退出(码=$?)，2 秒后自动重启" >> "$LOG"
  sleep 2
done
