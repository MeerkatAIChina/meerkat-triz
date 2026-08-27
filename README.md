# 猫鼬AI（Meerkat-AI）—— TRIZ 创新咨询智能体系统

面向细分行业头部企业的 TRIZ 创新咨询 AI，基于 Qwen3.6-35B-A3B 微调，运行于 NVIDIA DGX Spark（GB10, 128GB 统一内存）。

## 当前生产状态（2026-08）

| 项 | 值 |
|---|---|
| 模型 | `Meerkat-TRIZ-v1-Qwen3.6-35B-A3B`（Qwen3.6-35B-A3B + LoRA r=64，judge 3.42） |
| 基座量化 | NVFP4（Unsloth，compressed-tensors，23GB） |
| 推理 | vLLM 0.25，端口 8888 |
| 上下文 | **1,010,000 tokens**（YaRN factor 4.0） |
| 并发 | max-num-seqs 64，KV cache 5.2M tokens |
| thinking | 中文思考 + reasoning-parser qwen3 |
| 采样 | temperature 0.7 / top_p 0.9 / max_tokens 8192 |

## 核心能力

- **28 个专业 skill**（Open WebUI 0.11）：TRIZ 方法论 9 + 新品全流程 14 + 眼镜端 5，按业务流程分类（问题定义/求解/评估前瞻/教学辅助/洞察/技术/概念/立项/销售/眼镜端），高频核心 10 个常驻、低频进阶 18 个按需
- **文档工具链**：Markdown → Word/PDF/Excel/PPT/HTML（5 格式，纯 Python 不占 GPU），Excel 含表头样式/列宽自适应/冻结首行
- **文生图**：已下线（FLUX，见 V7B_RUNBOOK.md）
- **运维**：KV cache 监控脚本 + FLUX 空闲自动卸载 + tool_bridge 守护

## 关键文档

| 文档 | 内容 |
|---|---|
| [V7B_RUNBOOK.md](V7B_RUNBOOK.md) | **主文档**：完整决策链（Ornith 微调失败→回 Qwen）、生产配置、踩坑教训 |
| [V6_RUNBOOK.md](V6_RUNBOOK.md) | v6（Qwen3.8）历史 runbook |
| [DGX_Spark_恢复步骤.md](DGX_Spark_恢复步骤.md) | DGX Spark 系统恢复步骤 |
| [SGLang_DFlash_迁移评估.md](SGLang_DFlash_迁移评估.md) | SGLang/DFlash 迁移评估（未采用） |

## 目录结构

```
Meerkat-AI/
├── README.md                    # 本文件
├── V7B_RUNBOOK.md               # 主 runbook（决策链+生产配置+教训）
├── config.py                    # 全局配置（训练超参数）
├── requirements.txt             # Python 依赖
│
├── doc_tools.py                 # 文档工具链（md→docx/pdf/xlsx/pptx/html）
├── tool_bridge.py               # 工具桥接服务（文档转换，端口 8090）
├── bridge_daemon.sh             # tool_bridge 守护脚本
├── restart_vllm_tools.sh        # vLLM 重启脚本
├── openwebui_tool_doc.py        # Open WebUI 文档工具定义
├── monitor_kv_cache.sh          # KV cache 监控脚本
│
├── deploy/production/           # 生产部署配置（docker-compose + env 模板 + 启停）
├── pipeline_v5/                 # 训练与评测 pipeline（v5）
├── utils/                       # 工具函数
├── notebooks/                   # Jupyter Notebook
├── data/                        # 数据目录
├── results/                     # 评测结果
├── docs/                        # 文档
├── ref/                         # 参考文档
│
├── meerkat-skills-*.zip         # skill 源包（28 个 skill）
└── *_skills.py                  # skill 解析/导入/优化脚本
```

## 模型演进史

| 阶段 | 模型 | judge_armA | 结论 |
|---|---|---|---|
| v5a | Qwen3.6-35B-A3B + LoRA | 3.42 | ✅ 当前生产 |
| v6 | Qwen3.8-27B + LoRA | 3.53 | 备用（已退役） |
| v7/v7b/diag1 | Ornith-1.5-35B-A3B + LoRA | 2.99/2.91/2.93 | ❌ 微调无效（基座不响应） |

完整决策链见 [V7B_RUNBOOK.md](V7B_RUNBOOK.md)。

## 训练参考

训练与评测 pipeline 在 `pipeline_v5/`（`src/train.py` + `eval/eval_harness_v5.py`），LoRA 训练关键点：

- Qwen3.6 混合架构（Gated DeltaNet + Gated Attention + MoE）的 `target_modules` 须显式指定，**不能用 `all-linear`**（会误含 lm_head）
- completion-only loss（prompt/completion 数据集格式）
- 训练环境 `TORCH_OPTIM_FOREACH=0`（否则 OOM）

## 硬件环境

| 组件 | 规格 |
|------|------|
| GPU | NVIDIA GB10 Grace Blackwell |
| 统一内存 | 128 GB（CPU+GPU 共享） |
| 内存带宽 | 273 GB/s |
| FP4 算力 | 1 PFLOPS |
| CPU | 20-core Grace CPU |
