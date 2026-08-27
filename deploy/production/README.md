# Meerkat-TRIZ-v1 生产环境部署包

> 面向 **Meerkat-TRIZ-v1 (Qwen3.6-35B-A3B)** 的标准化 Docker 部署配置。
> 基于 vLLM 0.25.0 + Open WebUI，支持运行时 LoRA 热加载。

---

## 架构

```
浏览器 ──> Open WebUI (:12001)
              │ OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1
              ▼
          vLLM 0.25.0 (:8000, --network host)
              ├─ 基座: Qwen3.6-35B-A3B-NVFP4 (unsloth, NVFP4 量化)
              └─ LoRA: Meerkat-TRIZ-v1 (运行时热挂)
```

| 组件 | 说明 |
|------|------|
| **基座模型** | `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` — 35B MoE, NVFP4 量化 |
| **LoRA 适配器** | `Meerkat-TRIZ-v1` — r=64, α=128, BF16 训练 |
| **推理后端** | vLLM 0.25.0 + MARLIN MoE 后端（支持 fused MoE LoRA）|
| **前端** | Open WebUI (ollama tag) |
| **部署方式** | Docker Compose |

---

## 前置条件

| 要求 | 版本/规格 |
|------|----------|
| Docker + Compose | v2.x+ |
| NVIDIA Container Toolkit | 已安装并配置 `default-runtime: nvidia` |
| GPU | NVIDIA GPU（推荐 ≥80GB 显存/统一内存）|
| 模型权重 | 基座 + LoRA 已下载到宿主机 |

---

## 快速开始

### 1. 准备模型文件

```bash
# 基座模型（NVFP4 量化版，约 20GB）
huggingface-cli download unsloth/Qwen3.6-35B-A3B-NVFP4-Fast \
  --local-dir ~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast

# LoRA 适配器（约 180MB）
huggingface-cli download Meerkat-AI/Meerkat-TRIZ-v1 \
  --local-dir ~/.cache/vllm/loras/meerkat-triz-v1
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，确认 HF_HOME 和 LORA_DIR 路径与你的环境一致
```

### 3. 启动服务

```bash
chmod +x scripts/start.sh scripts/stop.sh
./scripts/start.sh
```

首次启动约 **2-5 分钟**（含 SM121 内核编译）。

### 4. 验证

```bash
# 查看可用模型
curl http://127.0.0.1:8000/v1/models | python3 -m json.tool

# 简单对话测试
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Meerkat-TRIZ-v1",
    "messages": [{"role": "user", "content": "什么是技术矛盾？"}],
    "max_tokens": 2048,
    "temperature": 0.6
  }'
```

---

## 关键配置说明

### 生成参数默认值（`OVERRIDE_GENERATION_CONFIG`）

```json
{"temperature":0.6,"top_p":0.95,"repetition_penalty":1.05,"max_new_tokens":16384}
```

| 参数 | 取值 | 原因 |
|------|------|------|
| `temperature` | **0.6** | 避免 `temperature=0` 贪心解码导致的无限重复循环 |
| `top_p` | 0.95 | nucleus sampling |
| `repetition_penalty` | 1.05 | 轻微抑制重复 |
| `max_new_tokens` | **16384** | 限制最长生成，防止客户端不传 `max_tokens` 时无限生成（上下文 262k 最坏情况约 23 分钟）|

> 评测 harness 需要确定性时，在请求中显式传 `temperature=0` 即可覆盖。

### MoE 后端选择

| 后端 | 是否可用 | 说明 |
|------|---------|------|
| `flashinfer_b12x` | ❌ **不可用** | vLLM 0.25.0 中该 NVFP4 MoE 内核不支持 LoRA，会直接拒绝启动 |
| `MARLIN` (auto) | ✅ **当前使用** | 自动选择，支持 fused MoE LoRA，性能无损（C1 77.3 tok/s，开销约 13%）|

### 用户注册

`.env` 中控制：

```bash
ENABLE_SIGNUP=true          # 开启自助注册
DEFAULT_USER_ROLE=user      # 新用户直接可用（改为 pending 则审批制）
```

---

## 文件结构

```
deploy/production/
├── docker-compose.yml       # 编排定义
├── .env.example             # 环境变量模板
├── scripts/
│   ├── start.sh             # 一键启动（含健康检查）
│   └── stop.sh              # 停止服务
└── README.md                # 本文件
```

---

## 运维命令

```bash
# 查看日志
docker logs -f meerkat-vllm
docker logs -f meerkat-webui

# 进入容器
docker exec -it meerkat-vllm bash

# 重启单服务
docker compose restart meerkat-vllm

# 查看资源使用
docker stats meerkat-vllm meerkat-webui

# 更新 LoRA 权重（不重启 vLLM）
# 1. 替换宿主机 LORA_DIR 下的文件
# 2. vLLM 会自动检测并热加载（vLLM ≥0.25.0 支持）
```

---

## 注意事项

1. **主机重启后需手动启动**：vLLM 容器不带 `--restart`，需执行 `./scripts/start.sh`
2. **API 无认证**：8000 端口仅限 Tailscale/局域网访问，不要暴露公网
3. **推理长度**：reasoning 模型思维链较长，`max_tokens` 建议 ≥ 4096，否则可能 `finish_reason=length`
4. **量化口径**：部署为「NVFP4 基座 + BF16 LoRA」，与训练评测口径不同；如需严格复现评测结果，应在相同量化配置下补测

---

## 性能参考（DGX Spark GB10, 2026-08-01）

| 模型 | 并发 | 吞吐 |
|------|------|------|
| Qwen3.6-35B-A3B-NVFP4 | C1 | 88.8 tok/s |
| Meerkat-TRIZ-v1 (LoRA) | C1 | 77.3 tok/s |
| Meerkat-TRIZ-v1 (LoRA) | C4 | 196.7 tok/s (聚合) |

---

## 关联文档

- [DGX Spark 运维手册](../MEERKAT-DGX-SPARK-OPS.md) — 原始部署记录
- [Pi 运维手册](../PI-DGX-SPARK-OPS.md) — pi.dev coding agent 接入配置
- [HF Model Card](../../publish_v6/README.md) — 模型卡与评测结果
