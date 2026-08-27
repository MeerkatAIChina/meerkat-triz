#!/bin/bash
# =================================================================
# Meerkat-TRIZ-v1 生产环境停止脚本
# =================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "== 停止 Meerkat-TRIZ-v1 生产环境 =="

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

docker compose down --remove-orphans

echo "✅ 已停止"
