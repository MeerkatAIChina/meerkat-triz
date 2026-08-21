#!/bin/bash
# ARIZ boost 生成启动器：从 ~/.bashrc 提取 MOONSHOT_API_KEY（bashrc 有交互守卫，非交互 shell 不会加载）
eval "$(grep '^export MOONSHOT_API_KEY' ~/.bashrc)"
cd /home/meerkat/mongoose_ai || exit 1
mkdir -p data/processed/corpus_sft_v2_ariz_boost
venv_v5/bin/python /tmp/run_ariz_boost.py "$@" 2>&1 | tee -a data/processed/corpus_sft_v2_ariz_boost/ariz_boost.log
