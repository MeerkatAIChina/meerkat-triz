# V7B RUNBOOK — Ornith v7b LoRA 训练与评测

> 记录时间：2026-08-24 11:50（训练进行中）；最终结果 2026-08-24 23:02 落盘。目标：验证「系统提示 mismatch 是否为 v7 提升微弱的根因」。

## 最终结论（2026-08-24 23:02）

**系统提示 mismatch 假设被证伪。Ornith 基座 LoRA 微调无效，建议放弃 Ornith 回 Qwen。**

| 模型 | judge_armA | keyword | judge Δ vs base | 判定 |
|---|---|---|---|---|
| base Ornith | 2.9666 | 0.6067 | — | — |
| v7（system mismatch） | 2.9933 | 0.5877 | +0.0268 不显著 | — |
| **v7b（system 对齐）** | **2.9133** | **0.6053** | **-0.0535 不显著** | 对齐无改善，略降 |
| Qwen v5a | 3.4233 | 0.638 | +0.39 | 有效 |
| Qwen v6 | 3.5333 | 0.624 | +0.50 | 有效 |

- v7b 对齐系统提示后 judge 反而从 +0.027 降到 -0.053（均不显著），证明 **mismatch 非根因**。
- v7b 唯一改善：keyword 从 v7 的 0.5877（显著恶化）恢复到 0.6053（≈ base 0.6067），说明对齐修复了 v7 的 keyword 恶化，但 judge 无对应提升。
- 两次独立 LoRA 尝试（v7 rank64 mismatch / v7b rank64 aligned）judge 都在 base 噪声范围，远低于 Qwen 的 +0.39~+0.50。**Ornith（qwen3_5_moe）基座 LoRA 微调无效**。

**决策**：放弃 Ornith 基座。候选：① 回 Qwen3.6（已验证 +0.39~+0.50）；② rank 64→128 重试（但两次 rank64 均无效，边际收益存疑，不推荐优先）。

---

## 方案 D 子集诊断（2026-08-25 进行中）

**零成本子集分析**（base/v7/v7b 三份评测结果的 per-subset Δ）：

| 子集 | v7Δ | v7bΔ | 诊断 |
|---|---|---|---|
| contradiction_analysis | +0.174 | +0.024 | v7 唯一涨但 v7b 回落 → 噪声 |
| ariz_guidance | -0.100 | -0.083 | **两次都跌 → 负迁移源** |
| innovation_assessment | -0.033 | -0.067 | **两次都跌 → 负迁移源** |
| case_generation | +0.022 | -0.111 | 不稳定 |
| concept_explanation | +0.022 | -0.111 | 不稳定 |
| principle_recommendation | +0.050 | 0.000 | 归零 |

**第 2 步 leave-out 实验**（2026-08-25 完成）：剔除 ariz+innovation，剩余 5 子集 6942 条训练，best eval_loss 1.6327 @ 800 步（epoch 1.267），评测 tag v7b_diag1。

**leave-out 结果**：整体 judge **2.9300**（base 2.9666，Δ -0.0334 不显著）；keyword 0.6026（Δ -0.0041 不显著）。

**负迁移假设证伪**：被剔除的 ariz/innovation 在评测时**依然下跌甚至更甚**（ariz: v7b -0.083 → diag1 -0.100；innovation: v7b -0.067 → diag1 -0.133）。说明它们的下跌不是"数据负迁移"，而是 **LoRA 微调本身的灾难性遗忘**——无论训练什么数据，都会损害这两类能力。

**最终结论（三次实验汇总）**：

| 实验 | 数据 | judge | Δ vs base |
|---|---|---|---|
| base | 无 | 2.9666 | — |
| v7（mismatch） | 全量 11421 | 2.9933 | +0.0268 |
| v7b（aligned） | 全量 11421 | 2.9133 | -0.0535 |
| diag1（剔负迁移） | 6942 | 2.9300 | -0.0334 |

三次 LoRA 微调，judge 全部在 base 噪声范围内，无一显著提升。contradiction_analysis 是唯一有微弱正信号的子集（v7 +0.17、diag1 +0.08），但 CI 均含 0 且 v7b 接近 0，不稳定。

**最终决策：回 Qwen。** Ornith（qwen3_5_moe）基座对 LoRA 微调不响应——无论数据组合（全量/剔负迁移）、系统提示（mismatch/aligned）如何变化，都无法获得 Qwen 的 +0.39~+0.50 提升。方案 D 排除"数据负迁移"这一最后可能的解释后，基座问题成为唯一结论。

---

## 回 Qwen 生产部署（2026-08-25 完成）

生产模型从 Ornith 切回 **Qwen v6（Qwen3.8-27B，judge 3.5333 最高）**：

- 推理容器 `meerkat-vllm-qwen38`：`Qwen3.8-27B-NVFP4` + lora `Meerkat-TRIZ-v1-Qwen3.8-27B`，端口 8888（对齐 webui `openai.api_base_urls`）
- lora name = webui model.id（工具关联关键），`base_model_id=None`，`capabilities.builtin_tools=true`，`toolIds=[meerkat-doc-tools, meerkat-image-gen]`（v6 原有配置完整保留）
- model 表：`Meerkat-TRIZ-v1-Qwen3.8-27B` active=1，`Meerkat-TRIZ-v1-Ornith-35B-A3B` active=0
- 推理测试通过（TRIZ 40 原理回答正确），thinking default-off（chat_template `enable_thinking is defined and true` 才开，容器无 `--reasoning-parser`）
- tool_bridge（8090）+ webui（12001 healthy）运行正常

**生产状态**：Qwen v6 已恢复服务，Ornith 全系（v7/v7b/diag1）退役。

---

## thinking/reasoning 打开 + 并发优化（2026-08-25）

用户要求打开 thinking/reasoning 并支持并发多用户：

- **thinking default-on**：改 `chat_template.jinja` 第 58 行 `enable_thinking is defined and true` → `enable_thinking is not defined or true`（enable_thinking 未定义时进入 reasoning 分支，reasoning_effort 默认 xhigh）；备份 `.jinja.bak2`
- **加 `--reasoning-parser qwen3`**：vLLM 将 `<think>` 思考分离到 `reasoning` 字段
- **并发优化**：`--max-num-seqs 8→24`，`--gpu-memory-utilization 0.5→0.6`（更多 KV cache）
- 验证：reasoning 660 字思考 + content 1971 字回答；8 并发全 200 成功（23-55s）

**关键权衡**（thinking 打开的代价）：单请求 ~2600 token（reasoning+answer），响应 20-50s；max_tokens 需 ≥4096（Open WebUI 默认够，若设 <1500 会 content=None）。

---

## H20 兼容部署验证：FP8 + LoRA（2026-08-25）

**核心结论**：H20（Hopper）不支持 NVFP4，需用 FP8。验证成功：FP8 base + LoRA 正确加载 + 推理正常。

- **FP8 base 来源**：ModelScope `Qwen/Qwen3.6-35B-A3B-FP8`（Qwen 官方 FP8，37.49GB，40 layers 分片）。**HF CDN 限速 57KB/s 不可用，ModelScope 镜像 39.6MB/s（~700 倍）**，国内下载必须走 ModelScope。
- **量化格式**：`quant_method: fp8`，`fmt: e4m3`，`activation_scheme: dynamic`，lm_head 未量化（BF16）
- **部署**：`--quantization fp8 --enable-lora --lora-modules Meerkat-TRIZ-v1-Qwen3.6-35B-A3B=/root/.cache/vllm/loras/meerkat-triz-v1`
- **验证结果**：`Loaded new LoRA adapter: Meerkat-TRIZ-v1-Qwen3.6-35B-A3B` ✅，模型加载 41.31 GiB，推理正常
- **次要问题**：FP8 容器未加 `--reasoning-parser qwen3`，thinking 混入 content（英文思考泄漏）。H20 部署时加 `--reasoning-parser qwen3` 即可分离。

**H20 部署要点**（供移植）：
1. 用 ModelScope 下载 FP8（`Qwen/Qwen3.6-35B-A3B-FP8`），勿用 HF CDN
2. vLLM `--quantization fp8`（不是 modelopt/compressed-tensors）
3. LoRA 适配器直接复用（BF16 独立于 base 量化格式）
4. H20 96GB：FP8 模型 ~41GB + KV cache ~50GB，并发可到 32+

---

## 生产最终态：NVFP4 + ARIZ 聚焦 + 深度优化（2026-08-25）

### 部署（生产切回 NVFP4，DGX 原生格式）

| 项 | 值 |
|---|---|
| 容器 | `meerkat-vllm-qwen36-nvfp4`（端口 8888） |
| 基座 | `Qwen3.6-35B-A3B-NVFP4-Fast`（Unsloth NVFP4，compressed-tensors，23GB） |
| LoRA | `Meerkat-TRIZ-v1-Qwen3.6-35B-A3B`（r=64，169MB） |
| served-model-name | `Qwen3.6-35B-A3B-NVFP4` |
| 并发/KV | max-num-seqs **48**，gpu-memory **0.7**（KV cache 50GB，理论并发 132x） |
| 采样 | temperature **0.7**，top_p **0.9**，max_tokens **8192** |
| thinking | 打开（`--reasoning-parser qwen3`）+ 中文引导 |

### thinking 中文输出结论

Qwen3.6 的 thinking 英文是**基座特性**（CoT 训练数据英文），LoRA 无法改变。三种方案：
- system prompt "用中文思考" → 中文占比仅 18%（效果弱）
- **think 块注入中文引导（采用）** → thinking 减少、全中文直接回答、更完整
- 无干预 → thinking 纯英文

> 对比 Qwen3.8 基座 thinking 是中文（基座差异）。

### 质量深度优化（token 无成本）

- max_tokens 4096→**8192**（避免 thinking+回答截断）
- temperature 1.0→**0.7**（TRIZ 输出稳定专业）
- 放宽输出规则（"可展开定义背景 + 追求深度完整性"）→ content 1467→**4061 字**（+177%）

### ARIZ 聚焦

system prompt 聚焦 ARIZ-85C 九步流程（IFR/物理矛盾/物场资源/知识库/方案评估）。验证：无人机续航 vs 轻量化矛盾，输出完整 ARIZ 引导（3005 字，含 IFR + 技术矛盾 + 矛盾矩阵原理推荐）。

### 关键教训

1. **每次重启推理容器前先卸载 FLUX**：`curl -X POST http://127.0.0.1:8090/unload`（FLUX 会反复自动加载 35GB，导致 gpu-memory 0.7 启动 OOM）
2. **并发/速度权衡**：深度长回答（3000-4000 字）单请求 40-90s；48 路并发下超载会排队但 KV cache 足够不 OOM

---

## 多 subagent 体系 + tools calling（2026-08-25）

### 9 个 TRIZ 专业 Skill（subagent）

Open WebUI 0.11 的 subagent 用 **Skill** 实现（`model.meta.skillIds` 关联 + @提及触发 + content 注入 system prompt）。已创建 9 个专业 skill，覆盖 TRIZ 方法论全貌：

| Skill ID | 名称 | 核心能力 |
|---|---|---|
| ariz-solver | ARIZ 算法求解引导 | ARIZ-85C 九步流程 |
| contradiction-analyst | TRIZ 矛盾分析 | 技术/物理矛盾 + 矛盾矩阵 + 分离原理 |
| sufield-analysis | 物场分析（Su-Field） | S1/S2/F 模型 + 76 标准解 |
| ifr-expert | 理想最终结果（IFR） | IFR 定义 + 理想度分析 |
| technology-evolution | 技术系统进化分析 | S 曲线 + 进化法则 + 技术预见 |
| principle-advisor | 发明原理推荐 | 40 原理 + 矛盾矩阵推荐 |
| innovation-assessment | 创新方案评估 | 理想度/资源/可行性量化评估 |
| case-generator | TRIZ 案例生成 | 教学/启发式案例 |
| concept-explanation | TRIZ 概念解释 | 概念/原理/术语深入解释 |

### 触发机制（代码确认）

- **@提及的 skill**：content 直接注入 system prompt（`<skill name="...">content</skill>`），立即生效
- **关联但未提及的 skill**：以 `<available_skills>` manifest 让模型可见，模型可推荐
- skill 关联在 `model.meta.skillIds`（JSON 数组）；`builtin_tools=true` 时两种模式配合正常

### ⚠️ access_grant 教训（skills 未加载的根因）

Open WebUI 的资源（model/tool/skill）都需 `access_grant` 记录才能被用户访问。**创建 skill 时必须同时插入 access_grant**：

```
INSERT INTO access_grant (resource_type, resource_id, principal_type, principal_id, permission)
VALUES ('skill', <skill_id>, 'user', '*', 'read')
```

漏插入 → `get_skills_by_user_id(user.id, 'read')` 返回空 → skill 不加载。已为 9 个 skill 补全。

### tools calling

| 工具 | 功能 | 验证 |
|---|---|---|
| meerkat-doc-tools | Markdown→Word/PDF/Excel/PPT（含图片嵌入） | ✅ 模型正确返回 tool_call |
| meerkat-image-gen | 文生图（FLUX.1-dev，中文标题+图例） | ✅ 已配置 |

- 工具关联：`model.meta.toolIds` + `capabilities.builtin_tools=true` + `base_model_id=None`
- tool_bridge：`http://192.168.5.246:8090`（/convert 文档、/image 图片）
- 工具兜底：format 空时默认 docx，不会失败

---

## 输出质量修复（2026-08-25）

### 问题 1：TRIZ 应用不够详细 specific

**根因**：system prompt「避免空泛套话」约束太抽象，模型仍倾向"罗列原理名不展开应用"（如"可用分割、局部质量、嵌套原理"）。

**修复**：强化【输出规则】：
1. 每个推荐原理必须展开"如何应用到具体问题"（参数/结构/工艺/数值/操作步骤），**禁止罗列式**
2. 给出量化预期效果和可检验判断标准（"重量降 15%"等）
3. 多方案逐个展开对比

**验证**：笔记本散热噪音问题，输出 3371 字，从罗列变为逐个展开（多风扇分区控制、可变几何风扇等具体方案 + TRIZ 解释 + 效果）。

### 问题 2：sandbox:/mnt/data/ 文件路径打不开

**根因**：Qwen3.6 基座训练数据里 sandbox:/mnt/data/ 是常见沙盒路径，模型"惯性"输出编造路径（实际文件通过 webui 文件系统附加对话）。

**修复**：强化【文件生成规则】，明确"文件自动附加对话、用户点击下载"，**严禁输出任何路径/链接**（sandbox:/mnt/data/、/tmp/、/home/、file://、http://localhost 等），只告知"文件已生成，请在对话中点击下载"。

**验证**：多轮工具调用测试（工具返回后模型回复）→ "文件已生成，请查看附件"，含 sandbox=False、含路径=False。

### 问题 2 深化：否定提示陷阱（真正根因）

初版修复失败——用户仍看到 sandbox:/mnt/data/output.docx。**真正根因是"否定提示陷阱"**：system prompt 和工具返回里写了"不要提及路径（**例如 sandbox:/mnt/data/ 之类**）"，反而把 sandbox:/mnt/data/ 字符串**植入模型上下文**，模型"模仿"输出它。

**彻底修复**：从 tool 返回 + 所有 model params 中**删除 sandbox:/mnt/data/ 等具体路径字符串**，改用抽象表述"不要提及任何文件路径"。全库 sandbox 出现次数 = 0。

> **教训**：LLM 的否定指令里不要写具体示例（"不要提 X"会反向植入 X），用抽象表述（"不要提路径"）。

---

## 内存优化：FLUX 空闲自动卸载（2026-08-25）

### 问题

FLUX（35GB 文生图）与 VLLM（82.6GB 推理）同时常驻 → 内存 100% 占满（available 仅 168Mi）+ swap 10.7GB，极度危险。

内存账本：VLLM 模型 27.7GB + KV cache 50.5GB + CUDA graph 4.4GB = 82.6GB；FLUX 35.2GB；合计 ~121GB。

### 根因

tool_bridge 的 FLUX 是"懒加载 + 常驻"（`get_pipe()` 全局缓存，加载后不自动释放），只有手动 `POST /unload` 才释放。一旦有图像请求，FLUX 35GB 永久占用。

### 修复

给 tool_bridge.py 加**空闲自动卸载**（已备份 `tool_bridge.py.bak_idle`）：

- 新增 `FLUX_IDLE_TIMEOUT = 300`（5 分钟）、`_idle_timer`、`_idle_timer_lock`
- 新增 `_do_unload()`（统一释放逻辑）和 `_schedule_idle_unload()`（threading.Timer 延迟卸载）
- `_image()` 生成后调用 `_schedule_idle_unload()` 重置计时器
- `_unload()` 复用 `_do_unload()` + 取消计时器

**效果**：FLUX 按需加载，空闲 5 分钟自动释放，与 VLLM 不再冲突。内存 available 从 168Mi 恢复到 35Gi。

**部署**：kill tool_bridge 进程（PID 2104），bridge_daemon.sh 守护脚本自动重启加载新代码。

---

## 性能优化：1M 上下文 + 64 并发（2026-08-25）

### 性能基线（统计结果）

| 指标 | 数值 |
|---|---|
| TTFT（首 token） | 0.11s |
| 单请求吞吐 | 57.5 tokens/s |
| 8 并发吞吐 | 217.5 tokens/s |
| decode 瓶颈 | GB10 显存带宽（~273GB/s），硬件上限 |

### 1M 上下文（YaRN factor 4.0）

Qwen3.6-35B-A3B 原生 262,144，用 YaRN factor 4.0 扩展到 1,010,000。官方 vLLM 配置：

```bash
-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
--max-model-len 1010000 \
--hf-overrides '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}'
```

### 并发优化：max-num-seqs 32 → 64

**关键认知**：多 subagent 并发（短/中上下文）的瓶颈是 `max-num-seqs`（调度槽位），而非 KV cache。提 max-num-seqs 不额外吃内存。

并发能力（按上下文长度）：

| 场景 | 并发 | 瓶颈 |
|---|---|---|
| 短对话（2K）/ 中上下文（32K） | 64 | max-num-seqs |
| 长文档（256K） | ~20 | KV cache |
| 超长文档（1M） | 5 | KV cache |

### 未做的方案及原因

1. **KV cache 0.85（71GB）**：需 FLUX 换 FP8（35→17GB），但 FLUX FP8 之前有质量风险（生成噪声），且收益有限（1M 并发 5→7），跳过。
2. **speculative decoding（MTP）**：NVFP4-Fast 无 MTP 权重（FP8 版本才有），需换模型；与 1M 上下文 + 大 KV cache 内存冲突，暂缓。

### 最终生产配置

| 参数 | 值 |
|---|---|
| 上下文 | 1,010,000（1M）+ YaRN factor 4.0 |
| 并发槽位 | max-num-seqs 64 |
| KV cache | 5.17M tokens（gpu-memory 0.7） |
| 量化 | NVFP4（compressed-tensors）+ LoRA r=64 |
| thinking | 中文引导 + reasoning-parser qwen3 |
| 采样 | temperature 0.7 / top_p 0.9 / max_tokens 8192 |

## 背景与决策线

- v7（Ornith 首个 LoRA，系统提示用「简洁模式 ≤300字/详细模式」）评测 judge 提升 **+0.0268（不显著）**，keyword **-0.0190（显著恶化）**，远低于 Qwen 基座 LoRA 的 +0.39~+0.50。
- 根因假设：训练 system prompt 与评测 system（「你是 TRIZ 创新方法论专家助手…」）不一致，导致模型学到的行为与评测期望错位。
- v7b 对策：用 `align_triz_system.py` 重写 v5b 数据 system 为评测 system，重新训练，隔离 mismatch 变量。**其余（rank=64、lr=2e-4、数据 11421 条）全冻结。**

## 当前训练状态（2026-08-24 11:50）

- **进程 PID 406079**（nohup 后台，独立于 SSH 会话）
- 日志：`/tmp/train_v7b4.log`
- 进度：约 59/5712 步（1428 步/epoch × 4 epoch），每步 ~10.5s，**ETA ~16h**
- 配置：`pipeline_v5/configs/train_v7b_ornith.json`（run_name `v7b_ornith_trizsys`）
- 输出：checkpoint `checkpoints/qlora_triz_v7b_ornith/`（首存于 100 步），适配器导出 `models/meerkat_triz_adapter_v7b_ornith/`
- 关键信号点：100 步首次 eval_loss + checkpoint；early stopping patience=3 threshold=0.002

## OOM 根因（已解决，重要教训）

前 3 次启动都在权重加载 46-47% (322-325/693) 处**静默死亡**，内核日志实锤是 **OOM-killer SIGKILL**（非 CUDA OOM）：

```
oom-kill: task=python3 pid=400507 total-vm:160267048kB
gsd-housekeepin invoked oom-killer  ← GNOME 后台进程都申请不到内存
```

**真凶**：`tool_bridge.py`（PID 2104）的 FLUX **一直占着 35GB GPU 内存没真正卸载**。`free -h` 显示「84Gi 可用」是误导——DGX Spark 统一内存下，这 35GB 压在物理内存里。加载 70GB Ornith + 35GB FLUX + 系统峰值 > 121Gi 上限 → 内核 OOM。

**修复**：`curl -X POST http://127.0.0.1:8090/unload` 真正释放 FLUX（`used 37Gi → 3.1Gi`，`free 84Gi → 119Gi`）后再启动。

**铁律**：加载 70G BF16 前，必须确认 `nvidia-smi --query-compute-apps` 里无 35GB 级残留（`free -h` 的 used 不可信，要看 GPU 计算进程）。FLUX 卸载靠 `/unload` 端点，不是停容器。

## 训练完成后操作（一键）

> ⚠️ **自动评测守护曾失效（pgrep 自匹配陷阱，2026-08-24 已修复）**：初版守护用 `pgrep -f 'pipeline_v5/src/train.py'` 轮询，但守护进程自己的 bash -c 命令行含该字符串，导致 while 永不退出；`run_v7b_eval.sh` 前置检查同样自匹配误判"训练仍在运行" exit 1。**修复**：改为 `pgrep -f '[p]ipeline_v5/src/train.py'`（`[p]` 字符类打破自匹配）。已手动触发评测。

> ✅ **评测已启动**（2026-08-24 18:56，PID 677769，日志 `/tmp/eval_v7b.log`）：训练 18:48 完成 → 适配器导出 → 部署 v7b lora → ornith-v7b 推理就绪 → 评测 `tag=v7b_ornith` 300 题，基线正确加载 Ornith 锚点。

手动等效命令：
```bash
ssh spark-855a
cd /home/chinux/jupyterlab/meerkatai
./run_v7b_eval.sh   # 已含前置检查 + key 注入 + 部署 + 评测
```

脚本 `run_v7b_eval.sh` 执行流：
1. 前置检查（训练进程已退出 + 适配器已导出）
2. 复制 `adapter_config.json` + `adapter_model.safetensors` 到 `~/.cache/vllm/loras/meerkat-triz-v7b-ornith/`
3. 重启 `ornith-v7b` 容器（`--lora-modules Meerkat-TRIZ-v1-Ornith-35B-A3B=...v7b-ornith`，其余参数与 v7 一致，端口 8888）
4. 等待就绪 → 后台跑评测 `--tag v7b_ornith`（日志 `/tmp/eval_v7b.log`）

评测命令（手动等效）：
```bash
export MOONSHOT_API_KEY="$(cat .env_moonshot | tr -d '[:space:]')"
venv_v5/bin/python pipeline_v5/eval/eval_harness_v5.py \
  --config pipeline_v5/eval/configs/eval_ornith.json \
  --adapter-path models/meerkat_triz_adapter_v7b_ornith \
  --baseline-results results/v5/eval_v5_base_goldfix_ornith_20260824_045124.json \
  --tag v7b_ornith
```

> ⚠️ **`--baseline-results` 必须显式指定 Ornith 锚点**（2026-08-24 教训）：`find_baseline()` 只匹配 `eval_v5_base_goldfix_v5_*.json` 模式，Ornith 锚点 tag 是 `base_goldfix_ornith` 不匹配，漏参数会自动配到 qwen38 base（judge 2.930 而非 2.9666），使 judge Δ 被低估 ~0.037。v7 评测当时也是显式指定此文件（meta 实锤）。

## 决策基线（对比用）

| 模型 | judge_armA | keyword | judge Δ vs base |
|---|---|---|---|
| base 锚点（纯 Ornith NVFP4） | 2.9666 | 0.6067 | — |
| v7（system mismatch） | 2.9933 | 0.5877 | +0.0268 不显著 |
| （参照）Qwen v5a | 3.4233 | 0.638 | +0.39 |
| （参照）Qwen v6 | 3.5333 | 0.624 | +0.50 |

**判定标准**：v7b 若 judge Δ 达到 Qwen 量级（+0.3 以上且显著）→ 系统提示 mismatch 是根因，Ornith 保留；若仍 +0.03 量级 → mismatch 非根因，问题在 Ornith 基座本身（可训练性/架构），需重新评估 Ornith 去留（候选：rank 64→128 或放弃 Ornith 回 Qwen）。

## 关键文件索引

- 训练配置：`pipeline_v5/configs/train_v7b_ornith.json`
- 评测配置：`pipeline_v5/eval/configs/eval_ornith.json`（含 api 块）
- 一键脚本：`run_v7b_eval.sh`（DGX 上）
- 评测 harness：`pipeline_v5/eval/eval_harness_v5.py`（已 patch `api_generate`）
- 数据重写：本地 `align_triz_system.py` → `data/processed/v5b_data/final/v5_train_v5b_trizsys.jsonl`（11421 条）
- Moonshot key：`/home/chinux/jupyterlab/meerkatai/.env_moonshot`（裸 key，无变量名，600 权限）
