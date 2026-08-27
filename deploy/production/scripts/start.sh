#!/bin/bash
# =================================================================
# Meerkat-TRIZ-v1 生产环境启动脚本
# =================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# ── 检查 .env ──
if [[ ! -f .env ]]; then
    echo "❌ .env 文件不存在，请复制 .env.example 并修改："
    echo "   cp .env.example .env"
    exit 1
fi

# ── 加载环境变量 ──
set -a
source .env
set +a

# ── 检查必要条件 ──
echo "== 检查必要条件 =="

if ! docker info &>/dev/null; then
    echo "❌ Docker 未运行"
    exit 1
fi

echo "✅ Docker 运行中"

if ! command -v nvidia-smi &>/dev/null; then
    echo "⚠️ nvidia-smi 未找到，请确保 NVIDIA Container Toolkit 已安装"
else
    echo "✅ NVIDIA 驱动: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
fi

# ── 检查模型文件存在 ──
LORA_PATH="${LORA_DIR/#\~/$HOME}"
if [[ ! -d "$LORA_PATH/adapter_config.json" && ! -f "$LORA_PATH/adapter_config.json" ]]; then
    echo "⚠️ LoRA 目录可能未就绪: $LORA_PATH"
    echo "   如需从 HuggingFace 下载："
    echo "   huggingface-cli download Meerkat-AI/Meerkat-TRIZ-v1 --local-dir $LORA_PATH"
fi

# ── 清理旧容器（避免冲突）──
if docker ps -a --format '{{.Names}}' | grep -q '^meerkat-vllm$'; then
    echo "== 清理旧 vLLM 容器 =="
    docker stop meerkat-vllm 2>/dev/null || true
fi

if docker ps -a --format '{{.Names}}' | grep -q '^meerkat-webui$'; then
    echo "== 清理旧 WebUI 容器 =="
    docker stop meerkat-webui 2>/dev/null || true
    docker rm meerkat-webui 2>/dev/null || true
fi

# ── 启动服务 ──
echo ""
echo "== 启动 Meerkat-TRIZ-v1 生产环境 =="
echo "   vLLM 镜像: $VLLM_IMAGE"
echo "   WebUI 镜像: $WEBUI_IMAGE"
echo "   WebUI 端口: $WEBUI_PORT"
echo ""

docker compose up -d

# ── 等待就绪 ──
echo ""
echo "== 等待 vLLM 就绪（首次启动约 2-5 分钟，含内核编译）=="
for i in $(seq 1 60); do
    if curl -sf -o /dev/null "http://127.0.0.1:8000/v1/models" 2>/dev/null; then
        echo "✅ vLLM 就绪"
        break
    fi
    if [[ $i -eq 60 ]]; then
        echo "❌ vLLM 启动超时，请检查日志: docker logs -f meerkat-vllm"
        exit 1
    fi
    echo -n "."
    sleep 5
done

echo ""
echo "== 等待 WebUI 就绪 =="
for i in $(seq 1 30); do
    if curl -sf -o /dev/null "http://127.0.0.1:${WEBUI_PORT}/" 2>/dev/null; then
        echo "✅ WebUI 就绪"
        break
    fi
    if [[ $i -eq 30 ]]; then
        echo "⚠️ WebUI 启动超时，请检查日志: docker logs -f meerkat-webui"
    fi
    echo -n "."
    sleep 2
done

# ── 打印状态 ──
echo ""
echo "============================================================"
echo "🎉 Meerkat-TRIZ-v1 生产环境已启动"
echo "============================================================"
echo ""
echo "📡 服务状态："
docker compose ps

echo ""
echo "🔗 访问地址："
echo "   Web UI:      http://$(hostname -I | awk '{print $1}'):${WEBUI_PORT}/"
echo "   API (vLLM):  http://$(hostname -I | awk '{print $1}'):8000/v1"
echo ""
echo "📋 可用模型："
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool 2>/dev/null || true

echo ""
echo "📖 常用命令："
echo "   查看日志:    docker logs -f meerkat-vllm"
echo "   查看日志:    docker logs -f meerkat-webui"
echo "   停止服务:    ./scripts/stop.sh"
echo "   重启服务:    ./scripts/start.sh"
echo ""
