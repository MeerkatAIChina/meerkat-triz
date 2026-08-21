# Meerkat-TRIZ-v1 @ DGX Spark 部署运维手册

部署日期：2026-08-01。主机：DGX Spark（GB10, 121GB 统一内存, aarch64, CUDA 13）。

## 架构

```
浏览器 ──> Open WebUI (meerkat-webui, :12001)
              │ OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1
              ▼
          vLLM 0.25.0 (meerkat-vllm, :8000, --network host)
              ├─ served model: Qwen3.6-35B-A3B-NVFP4   (基座, unsloth NVFP4)
              └─ served model: Meerkat-TRIZ-v1          (运行时 LoRA, 正式发布 v5 权重)
```

- 镜像：`qwen36-dgx-spark:v0.1.1`（= vllm/vllm-openai v0.25.0 arm64 + PyAV，
  构建自 NNNtrance/Qwen3.6-35B-A3B-NVFP4-Fast-DGX-Spark 的 Dockerfile）
- 基座快照：`~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots/24ccf90e45a8f7e84e6251f4a19648104949c9f1`
- LoRA：`~/.cache/vllm/loras/meerkat-triz-v1/`，权重 sha256
  `e006d4c8e6d8a54d856dc001382f68657ba8b334f0e5e5c155e27aecfafec0b7`（与正式发布版一致）

## 关键配置（与参考仓库的差异）

- `--moe-backend auto`：**不能用参考仓库的 `flashinfer_b12x`**，该 NVFP4 MoE 内核
  不支持 LoRA（vLLM 0.25.0 会直接拒绝启动）。自动选择落到 `MARLIN` 后端，
  支持 fused MoE LoRA，实测性能无损（见下）。
- `--enable-lora --lora-modules Meerkat-TRIZ-v1=... --max-lora-rank 64`
- `ENABLE_MULTIMODAL=0`（纯文本部署），其余沿用验证过的 profile：
  gpu-mem 0.83 / max-model-len 262144 / batched-tokens 8192 / seqs 64 / MTP 3 tokens
- **生成默认值已改为 `{"temperature":0.6,"top_p":0.95,"repetition_penalty":1.05,"max_new_tokens":16384}`**
  （`.env` 的 `OVERRIDE_GENERATION_CONFIG`，注意值必须带单引号否则 bash source 后丢引号）。
  原因有二：
  1. 参考仓库默认 `temperature:0` 贪心解码会让模型在闲聊/身份类问题上陷入
     无限重复循环（实测 Meerkat-TRIZ-v1 同一 60 字块重复 41 次）；
     改为 Qwen 官方推荐的思考模式采样参数后全部正常收尾。
  2. 客户端不传 `max_tokens` 时 vLLM 默认按 262144 上下文放开生成——
     实测一次 WebUI 代码解释器多轮调用让单请求连续生成 23 分钟（约 12 万 token）
     仍不停止。`max_new_tokens:16384` 把最坏情况约束在约 3.5 分钟内。
  评测 harness 需要确定性时，在请求里显式传 `temperature=0` / 自定义 `max_tokens`
  即可覆盖服务器默认（已验证兼容）。

## 常用命令

```bash
cd ~/jupyterlab/meerkatai/deploy/qwen36-vllm

# 启动（首次/重启后约 5 分钟就绪，含 SM121 内核编译）
DETACH=1 ./scripts/run-container-meerkat.sh

# 状态与日志
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
docker logs -f meerkat-vllm

# 停止 / 重启
docker stop meerkat-vllm          # 容器带 --rm，停止即删除，重启用上面的启动命令

# Web UI
docker logs meerkat-webui
docker restart meerkat-webui      # 已配置 --restart unless-stopped
```

## 访问地址

- Web UI：http://100.115.91.8:12001/ （Tailscale）或 http://192.168.5.246:12001/ （局域网）
- OpenAI 兼容 API：http://100.115.91.8:8000/v1 ，模型名 `Meerkat-TRIZ-v1` / `Qwen3.6-35B-A3B-NVFP4`

## 实测性能（2026-08-01, 本机 smoke, 365 tok 输入 / 512 tok 输出, 非流式）

| 模型 | 并发 | 吞吐 |
|---|---|---|
| Qwen3.6-35B-A3B-NVFP4 | C1 | 88.8 tok/s |
| Meerkat-TRIZ-v1 | C1 | 77.3 tok/s（LoRA 开销约 13%） |
| Meerkat-TRIZ-v1 | C4 | 196.7 tok/s（聚合） |

参考：上游仓库在相同硬件用 flashinfer_b12x（无 LoRA）实测 C1 91.5 tok/s，
本部署 MARLIN 后端基座 88.8 tok/s 与之基本持平。

## 注意事项

1. **推理长度**：这是 reasoning 模型且 LoRA 版本思维链更长，`max_tokens` 建议 ≥ 4096，
   否则思考耗尽额度会导致 `content` 为空（`finish_reason=length`）。
2. **API 无认证**：8000 端口不要暴露公网，仅 Tailscale/局域网使用。
3. **Open WebUI 首个注册账号即管理员**。
4. **量化口径**：部署为「BF16 训练、NVFP4 基座 + BF16 LoRA 热挂」，与训练评测口径不同；
   发布前建议在本地金标上补一轮 NVFP4+LoRA 评测，确认无量化漂移。
5. 未改动 `gsplat` 与原有 `open-webui`（:12000）容器；本部署未写入任何 Git 仓库。
6. 主机重启后需手动执行启动命令（vLLM 容器）；Web UI 会自动恢复。

## 2026-08-02 开启自助注册

- 问题：登录页无注册入口，仅 admin 可用。根因是 DB 持久化配置 `ui.enable_signup=False`
  （PersistentConfig：DB 值覆盖环境变量，单改 env 无效）。
- 处理：`deploy/enable_signup.sh`（注意 docker exec 需 `-i` 才能接 heredoc）——
  ① DB 直改 `enable_signup=True, default_user_role=user`；
  ② 重建容器补 `-e ENABLE_SIGNUP=true -e DEFAULT_USER_ROLE=user -e ENABLE_LOGIN_FORM=true`；
  ③ 重启后 `/api/config` 验证 `features.enable_signup=true`。
- 新注册用户角色为 `user`（即刻可用）；如需审批制改为 `pending`（DB + env 同步改）。
- 测试账号已注册-登录验证后清理；现存账号仅 admin（chinux@live.com）。

## 2026-08-02 (2) 审批制 + Pi WebUI 多用户网关

- Open WebUI 新用户默认角色改为 `pending`（注册后需 admin 在设置里批准），
  DB + 容器 env（`DEFAULT_USER_ROLE=pending`）双写，注册实测角色为 pending。
- Pi WebUI（pi-web-ui 扩展）设计上是单会话镜像、作者已移除认证，无法原生多用户。
  采用 nginx 认证网关方案：容器 `piweb-gate`（nginx:alpine，:12003 → host:12002，
  Basic Auth + WebSocket 反代），账号文件 `~/jupyterlab/meerkatai/deploy/piweb-gate/htpasswd`。
- 账号管理：`bash ~/jupyterlab/meerkatai/deploy/piweb-gate/piweb-users.sh add|del|list`
  （改密后 `nginx -s reload` 热加载，已实测生效；挂载必须是目录而非单文件，
  否则 sed -i 换 inode 后容器看不到更新）。
- 初始账号 demo / Meerkat2026!（建议尽快改密）。所有用户共享同一 pi 会话；
  原 12002 端口未加锁，保留给 admin 直连/录屏。
- 注意：pi 是有工作区写权限的 coding agent，共享会话模式下任何登录用户都可
  操控会话，账号只发给可信人员。

## 2026-08-02 (3) 取消 Pi WebUI 多用户网关

- 应要求撤销：piweb-gate 容器与远端配置目录已删除，本地配置同步删除。
- Pi WebUI 维持单会话、无认证，仅 :12002 直连使用（Tailscale/局域网）。

## 2026-08-03 修复新用户无模型可见

- 现象：新注册用户（Zamir Zhang）登录后模型选择器为空，"未选择模型"。
- 根因（Open WebUI 0.9.6 源码确认，utils/models.py get_filtered_models）：
  vLLM 连接模型无 DB 条目 → 仅 admin 可见（"only admins can see unconfigured models"）；
  唯一的工作区模型 Meerkat-TRIZ-v1 为 admin 私有（access_grant 表为空）。
- 修复：access_grant 插入公共读授权（principal_type=user, principal_id=*）；
  另注册 qwen36-base 工作区模型（base_model_id=Qwen3.6-35B-A3B-NVFP4）并公开，
  供普通用户 A/B 对比。
- 验证：临时 user 注册→提权→/api/models 见双模型→对话实测通过→清理。
- 注意：今后新增工作区模型默认私有，需在 UI 设置可见性或插入 access_grant。
