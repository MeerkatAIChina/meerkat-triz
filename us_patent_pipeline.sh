#!/bin/bash
# DGX 上解压美国专利 + 提取 RLVR 样本 (按需解压, 不全部解压 1.4TB)
# 用法: bash us_patent_pipeline.sh [起始年] [结束年]
set -u

ZIP="/home/chinux/jupyterlab/meerkatai/data/us_patents/output_US.zip"
DIR="/home/chinux/jupyterlab/meerkatai/data/us_patents"
PY="/home/chinux/jupyterlab/meerkatai/venv_v5/bin/python3"
BUILD="$DIR/build_us_patent_rlvr.py"

START="${1:-2020}"
END="${2:-2024}"

mkdir -p "$DIR/extracted"

for year in $(seq "$START" "$END"); do
    csv="$DIR/extracted/${year}.csv"
    out="$DIR/us_${year}.jsonl"
    if [ ! -f "$out" ]; then
        echo "[$(date '+%H:%M:%S')] 解压 ${year}.csv ..."
        unzip -o "$ZIP" "${year}.csv" -d "$DIR/extracted" > /dev/null 2>&1
        echo "[$(date '+%H:%M:%S')] 提取 ${year} RLVR 样本 ..."
        "$PY" "$BUILD" "$csv" "$out"
        # 提取完删除解压的 CSV (省空间)
        rm -f "$csv"
    else
        echo "[skip] ${year} 已处理"
    fi
done
echo "[pipeline] ${START}-${END} 完成"
