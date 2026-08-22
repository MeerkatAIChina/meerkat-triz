# SGLang + DFlash 迁移评估报告

> 目标：评估用 DFlash 投机解码加速 v6 (Qwen3.8-27B) / v1 (Qwen3.6-35B-A3B) 的可行性、迁移成本与风险。
> **最终结论（2026-08-23 实测后）：不可行。** SGLang 在 DGX Spark (GB10/aarch64) 上对 Qwen3.6-35B-A3B 做 CUDA graph capture 触发系统级挂起（两次，均需物理重启），且 LoRA + NVFP4 MoE 存在功能缺口。此路线终止。

---

## 一、结论摘要

| 维度 | 结论 |
|------|------|
| 可行性 | ❌ **不可行**（两次系统挂起实测）|
| 框架 | ❌ 必须从 vLLM 0.25 切到 SGLang（vLLM 的 DFlash 仍在 bring-up）|
| 硬件 | ⚠️ DGX Spark (GB10) 是问题所在：SGLang CUDA graph capture 系统级 bug |
| 收益 | 理论上 1.5–3.6×，但**无法在 DGX Spark 上实测**（挂起）|
| 挂起根因 | ❌ SGLang 在 GB10/aarch64 对 hybrid MoE 的 CUDA graph capture（非 flashinfer、非内存不足）|
| LoRA 缺口 | ❌ `CompressedTensorsW4A4Nvfp4MoE` 缺 `get_triton_quant_info`，flashinfer 也不支持 LoRA |
| 结论 | 终止此路线，等 SGLang 修复 GB10 CUDA graph bug 后再评估 |

> 实测历程：SGLang 经 pip 装好（main 源码 0.0.0.dev1+g3c69a4c74，含 DFLASH+LoRA 白名单）、draft 下载完成（0.77GB）、无 LoRA 的 DFLASH 初始化全部通过，但两次进入 CUDA graph capture 阶段即挂死整个用户空间（ping 通、所有 TCP 不可达），只能物理断电重启。

---

## 二、DFlash 是什么

- **DFlash = Block Diffusion for Flash Speculative Decoding**（块扩散投机解码），Z-Lab + Modal 联合训练。
- 用**轻量块扩散 draft 模型并行提议多个 token**（block size 8–16），target 一次性验证。
- 与之前查的「原生小 draft 模型」（qwen3_5 无 0.6B/1.7B）**不同**：DFlash draft 是独立训练的产物，你的两个模型都有现成版本：

| 目标模型 | DFlash draft | draft 大小 |
|---------|-------------|-----------|
| v6 Qwen3.8-27B | `z-lab/Qwen3.8-27B-DFlash2`（第二代） | 1.92B (~4GB) |
| v1 Qwen3.6-35B-A3B | `z-lab/Qwen3.6-35B-A3B-DFlash`（第一代） | 0.77GB (~400M) |

---

## 三、可行性确认（关键）

### 3.1 LoRA + DFLASH 官方支持 ✅

SGLang 源码 `server_args.py`：

```python
_LORA_SPEC_ALGORITHMS = ("EAGLE", "EAGLE3", "DFLASH", "DSPARK")
```

`_check_lora_speculative_compatibility()` 明确：**LoRA 与 DFLASH 兼容**，设计语义是：

> "Adapters apply to the target only; a shared draft runs unadapted."
> （LoRA 只挂在 target 验证模型上，draft 草稿模型不挂 LoRA，用 base 分布。）

### 3.2 已知限制（LoRA + DFLASH 组合下不可用）

| 限制项 | 说明 |
|--------|------|
| `--speculative-adaptive` | 自适应投机不可用 |
| `experimental_sgl_trtllm` MoE runner | 该 MoE 后端读 LoRA 配置时机冲突 |
| `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` | overlap plan stream 与 LoRA 批准备冲突 |

> 注意：v1 官方示例里用了 `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`，但那是**无 LoRA** 场景；你的模型带 LoRA，需去掉该 env。

### 3.3 版本与硬件 ✅

- SGLang v0.5.x 主线已含 DFlash（PR #35371 已合并）；vLLM 对应 PR #40898 未合并。
- NVFP4 支持：`compressed-tensors` / `nvfp4_online` 等量化路径齐全。
- DGX Spark (GB10/SM121) 有同款现成 recipe：`Weschera/Qwen3.8-27B-NVFP4-DFlash2-DGX-Spark`。

---

## 四、参数映射（vLLM 0.25 → SGLang）

| 用途 | vLLM (当前) | SGLang (对应) | 备注 |
|------|------------|--------------|------|
| 模型名 | `--served-model-name` | `--served-model-name` | 同名 |
| 量化 | `--quantization compressed-tensors` | 自动检测 / `--load-format` | NVFP4 自动识别 |
| dtype | `--dtype bfloat16` | `--dtype bfloat16` | 同名 |
| 上下文 | `--max-model-len 32768` | `--context-length 32768` | 改名 |
| 内存占用 | `--gpu-memory-utilization` | `--mem-fraction-static` | 语义近似 |
| 并发 | `--max-num-seqs` | `--max-running-requests` | 改名 |
| KV cache | `--kv-cache-dtype fp8` | `--kv-cache-dtype fp8_e4m3` | 值不同 |
| LoRA 挂载 | `--lora-modules name=path` | `--lora-paths name=path` | 改名 |
| LoRA rank | `--max-lora-rank 64` | `--max-lora-rank` | 同名 |
| 工具 parser | `--tool-call-parser qwen3_xml` | `--tool-call-parser qwen3_coder` | ⚠️ 无 qwen3_xml，需验证 |
| auto tool choice | `--enable-auto-tool-choice` | 无需 flag（OpenAI `tool_choice=auto` 默认支持） | Open WebUI 侧不变 |
| 图片输入 | `--limit-mm-per-prompt '{"image":3,"video":1}'` | `--limit-mm-data-per-request '{"image":3,"video":1}'` | 改名 |
| 采样默认 | `--override-generation-config {...}` | `--sampling-defaults model`（读 generation_config.json） | 机制不同 |
| reasoning | (thinking 关闭) | `--reasoning-parser` | 需按模型配置 |
| 投机解码 | (无) | `--speculative-algorithm DFLASH` + `--speculative-draft-model-path` | 新增 |
| 草稿 block | (无) | `--speculative-dflash-block-size 8/16` | 新增 |
| 草稿 KV | (无) | `--speculative-draft-kv-cache-dtype fp8_e4m3` | 可减半 draft 内存 |

---

## 五、关键风险与未知

### 5.1 接受率折扣（最高风险，必须实测）

draft 基于 **base 模型** 分布训练，你的 target 是 **LoRA 微调的 TRIZ 专家**。target 分布被 LoRA 改偏后，base draft 的提议命中率会下降 → 接受长度缩短 → 加速打折。

- 理论上加速可能从 3.6× 掉到接近 1×（甚至负收益，draft 开销白费）。
- 但 block diffusion 并行草稿对分布偏移比串行 draft 更鲁棒，实际折扣未知。
- **唯一解法：用你的 LoRA 模型实测接受长度（accept length）**。

### 5.2 工具调用 parser 迁移

- vLLM 用 `qwen3_xml`，SGLang 枚举里**没有** `qwen3_xml`，只有 `qwen3_coder` / `qwen`。
- Qwen3.8 社区 SGLang recipe 用 `qwen3_coder`，但你的工具是 qwen3_xml 格式训练的，需实测工具调用是否还能正确触发。

### 5.3 输出非逐字节等价

Weschera recipe 明确警告：DFlash2 输出 hash 与关投机不完全一致（但 GSM8K/HumanEval 等质量评分等价）。对 TRIZ 咨询业务，需确认输出质量无明显劣化。

### 5.4 DGX Spark「boot lottery」

DGX Spark 的 AutoTuner 在每次启动时竞速 fp8_gemm kernel，可能落到慢档（33 vs 42 tok/s），且不持久化。社区用 `verify_boot.sh` 门控重启（约 1/3–1/6 概率慢档）。

### 5.5 draft 保持 BF16

draft FP8 量化会破坏高接受率路径（Weschera 实测 42→35 tok/s）。draft 必须 BF16（但 draft 很小，v1 仅 0.77GB，影响可忽略）。

---

## 六、迁移方案（分阶段，不碰生产）

### 阶段 1：并行实验实例（半天内）

1. 安装 SGLang（新 venv 或容器，不动现有 vLLM 环境）
2. 下载 draft：`z-lab/Qwen3.8-27B-DFlash2`（1.92B）或 `z-lab/Qwen3.6-35B-A3B-DFlash`（0.77GB）
3. 起 SGLang 实例在 **30000 端口**（生产 8000/8001 不动），挂 LoRA + DFLASH
4. 关键参数（v1 示例，去 overlap plan stream）：
   ```bash
   python -m sglang.launch_server \
     --model-path <你的 NVFP4 模型> \
     --speculative-algorithm DFLASH \
     --speculative-draft-model-path z-lab/Qwen3.6-35B-A3B-DFlash \
     --speculative-dflash-block-size 8 \
     --lora-paths Meerkat-TRIZ-v1=<lora路径> \
     --max-lora-rank 64 \
     --kv-cache-dtype fp8_e4m3 \
     --tool-call-parser qwen3_coder \
     --limit-mm-data-per-request '{"image":3,"video":1}' \
     --host 0.0.0.0 --port 30000
   ```

### 阶段 2：实测验收（决定是否切换）

- **接受长度**：`completion_tokens / spec_verify_ct`，对比社区无 LoRA 基线（block 8 ~3.24 tokens）
- **吞吐**：你的 TRIZ 中文 workload，单并发 + 并发 8/16，对比当前 vLLM（v6 11 / v1 57 tok/s）
- **质量**：TRIZ 输出规则是否仍触发、工具调用是否正常、图片输入是否正常

### 阶段 3：切换（仅当实测收益达标）

- 达标标准（建议）：单并发 ≥1.3× 且质量无劣化
- 切换后 tool_bridge 的 vLLM 地址 8000/8001 → SGLang 30000
- Open WebUI 的 model base_url 同步切换

---

## 七、实测结果（2026-08-22 ~ 08-23）

### 7.1 成功完成的部分

| 步骤 | 结果 |
|------|------|
| SGLang 安装 | ✅ pip 装 0.5.18 → 源码升 main（`0.0.0.dev1+g3c69a4c74`，含 DFLASH+LoRA 白名单）|
| 依赖升级 | ✅ torch 2.13 + flashinfer 0.6.17 + cutlass-dsl 4.6.2 等全部满足 |
| draft 下载 | ✅ `z-lab/Qwen3.6-35B-A3B-DFlash`（0.77GB）|
| 无 LoRA DFlash 初始化 | ✅ 主模型(NVFP4 MoE) + draft + DFLASH draft runner 全部通过 |
| 生产恢复 | ✅ 两次挂起后均完整恢复 v6+v1+WebUI+桥接 |

### 7.2 两次系统挂起（根因定位）

| | 第一次 | 第二次 |
|---|---|---|
| attention backend | flashinfer（默认）| **triton（显式指定）** |
| 内存 | 充足（avail 53GB）| 充足（avail 53GB）|
| 挂起时机 | CUDA graph capture | **CUDA graph capture（同阶段）** |

**结论：根因是 SGLang 在 GB10/aarch64 上对 Qwen3.6-35B-A3B（hybrid：Gated DeltaNet + Gated Attention + MoE）做 CUDA graph capture 的系统级 bug**，与 flashinfer、内存无关。表现为：ping 通、所有 TCP 服务（SSH/8000/8001/12001/8090/30000）不可达，只能物理断电重启。

### 7.3 关键教训

1. **GB10 上 SGLang 必须 `--attention-backend triton`**（论坛已警告，但即使 triton 也逃不过 CUDA graph capture 挂起）。
2. **DGX Spark 异常关机后启动极慢**：重启后 docker 自动恢复 v6 容器 + kswapd0 回收 swap，load average 飙到 42/61，SSH 被压住 70+ 分钟才恢复。
3. **Docker Hub CloudFront 国内不可达**：镜像拉取走不通（i/o timeout），pip 装 SGLang 是正确替代（PyPI + 清华镜像 aarch64 wheel 齐全）。
4. **手动 `docker stop` 的容器重启后不自动恢复**（v1 需 `docker start`），而自动崩溃的容器（v6）会恢复。

---

## 八、最终决策（已定）

**此路线终止。** DFlash 投机解码在 DGX Spark + Qwen3.6-35B-A3B 上不可落地：上游有 CUDA graph capture 系统挂起 bug，下游有 LoRA + NVFP4 MoE 功能缺口。

当前生产已稳定运行的全部加速（KV FP8 + 并发调优 + thinking-off + TRIZ 输出规则 + FLUX FP8 + max_tokens 16384）是 DGX Spark 上稳定可得的优化。DFlash/SGLang 等 SGLang 官方修复 GB10 CUDA graph bug 后再评估。
