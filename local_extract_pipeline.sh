#!/bin/bash
# 本地解压+提取+传样本 (美国专利, 按需年份)
# 用法: bash local_extract_pipeline.sh [起始年] [结束年]
set -u

ZIP="/Volumes/2nd-HD/Dataset/US-Patents-1790-2024/full/output_US.zip"
EXTRACT="/tmp/uspat_extract"
BUILD="/Volumes/2nd-HD/claude/Meerkat-AI/build_us_patent_rlvr.py"
DGX="spark-855a:/home/chinux/jupyterlab/meerkatai/data/us_patents/"

START="${1:-2021}"
END="${2:-2024}"

mkdir -p "$EXTRACT"

for year in $(seq "$START" "$END"); do
    # 跳过已处理(DGX 已有 parquet)
    if ssh spark-855a "[ -f /home/chinux/jupyterlab/meerkatai/data/us_patents/us_${year}.parquet ]" 2>/dev/null; then
        echo "[skip] ${year} 已处理"
        continue
    fi
    echo "[$(date '+%H:%M:%S')] 解压 ${year}.csv ..."
    unzip -o "$ZIP" "${year}.csv" -d "$EXTRACT" > /dev/null 2>&1
    echo "[$(date '+%H:%M:%S')] 提取 ${year} ..."
    python3 "$BUILD" "$EXTRACT/${year}.csv" "/tmp/us_${year}.jsonl" 2>&1 | tail -1
    echo "[$(date '+%H:%M:%S')] 传 ${year} 样本 ..."
    scp -q "/tmp/us_${year}.jsonl" "$DGX"
    rm -f "$EXTRACT/${year}.csv" "/tmp/us_${year}.jsonl"
    echo "[$(date '+%H:%M:%S')] ${year} 完成"
done
echo "[pipeline] ${START}-${END} 完成"
