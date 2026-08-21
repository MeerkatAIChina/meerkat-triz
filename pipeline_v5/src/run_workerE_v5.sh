#!/bin/bash
cd /home/meerkat/mongoose_ai
eval "$(grep "^export MOONSHOT_API_KEY" ~/.bashrc)"
echo "=== [$(date +%H:%M:%S)] Task1 Safety-Refusal 300 start ==="
venv_v5/bin/python pipeline_v5/src/safety_gen_v5.py
echo "=== [$(date +%H:%M:%S)] Task1 done, Task2 styleC long answers start ==="
venv_v5/bin/python pipeline_v5/src/styleC_gen_v5.py
echo "=== [$(date +%H:%M:%S)] Task2 done ==="
