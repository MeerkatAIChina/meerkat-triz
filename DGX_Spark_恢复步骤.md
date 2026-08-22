# DGX Spark 恢复步骤（2026-08-22 系统挂起后）

## 背景

起 SGLang + DFlash 实验实例时，漏加 `--attention-backend triton`，SGLang 默认用 flashinfer
（GB10/aarch64 已知 bug），导致系统挂起：ping 通、所有 TCP 服务（SSH/8000/8001/12001/8090/30000）不可达。

## 重启后恢复顺序

### 1. 确认系统起来 + 清理残留 SGLang 进程

```bash
ssh spark-855a
# 清理可能残留的 SGLang 进程（上次挂起前的）
pkill -f "sglang.launch_server" 2>/dev/null || true
free -h
```

### 2. 恢复生产 vLLM（v6 + v1，带 LoRA + KV FP8）

```bash
bash /home/chinux/jupyterlab/meerkatai/restart_vllm_tools.sh
# 顺序: v6 先 (~6min), v1 后 (~6min)
```

### 3. 确认桥接 + WebUI

```bash
curl -s -X POST http://127.0.0.1:8090/health    # {"ok":true,...}
docker ps --filter name=meerkat-webui            # Up (healthy)
```

### 4. 确认生产服务恢复

```bash
curl -s http://127.0.0.1:8000/v1/models   # v6
curl -s http://127.0.0.1:8001/v1/models   # v1
```

## SGLang DFlash 实验（生产恢复后，可选）

修正后的脚本（已加 `--attention-backend triton`）：

```bash
scp start_sglang_dflash_v1.sh spark-855a:/home/chinux/jupyterlab/meerkatai/
ssh spark-855a "bash /home/chinux/jupyterlab/meerkatai/start_sglang_dflash_v1.sh"
```

## 关键结论（本次实验已确认）

1. ✅ SGLang 已装好（main 源码 0.0.0.dev1+g3c69a4c74，含 DFLASH+LoRA 支持）
2. ✅ draft 已下载（0.77GB）
3. ✅ 无 LoRA 的 DFlash 能跑通（主模型 + draft + DFLASH draft runner 初始化成功）
4. ❌ LoRA + NVFP4 MoE 是 SGLang 功能缺口（get_triton_quant_info 缺失）— 需合并 LoRA 权重或等修复
5. ⚠️ GB10 必须 `--attention-backend triton`（不能默认 flashinfer）

## 教训

- 每次 SGLang 启动前核对 GB10 参数：`--attention-backend triton`
- 社区已知 GB10 "boot lottery"（AutoTuner 竞速 fp8_gemm，可能落慢档）
- 大镜像拉取：Docker Hub CloudFront 国内不可达，用 pip 装 SGLang 更可靠
