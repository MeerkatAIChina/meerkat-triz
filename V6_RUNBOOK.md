# V6 (Qwen3.8-27B) 续训运行手册

> 2026-08-17 建立。目标：把 v6 换基座（Qwen3.8-27B）训练**干净地跑完并交付**。
> 远程 DGX：`ssh spark-855a`，项目根 `/home/chinux/jupyterlab/meerkatai`，venv `venv_v5`。

## 一、当前状态（2026-08-17 晚）

- **正在运行的训练**：PID 518663，19:07:50 开训，`--config pipeline_v5/configs/train_v6_qwen38.json`，**`resume=None`（全新干净跑）**。
- 输出目录 `checkpoints/qlora_triz_v6_qwen38/`，日志 `train.log`。
- 进度：~step 269/5548，~15s/步；eval@200 = 1.587（与 resume_v2 同点 1.585 一致，可复现性 OK）。
- 超参：LR 2e-4 / dropout 0.0 / optim adamw_torch / r=64 α=128 / completion-only loss / BF16 无量化。
- 计划：cosine horizon **2774 步（2-epoch，~12h）**；`num_train_epochs=4` 是安全帽（5548 步 ~24h）；early-stopping patience=3 / threshold=0.002 大概率提前收。
- 监控：`bash monitor_v6_qwen38.sh`（每 10 分钟 SSH 抓进度，日志 `v6_monitor.log`）。

## 二、崩溃史与根因（续训必读）

v6 反复崩溃，三类根因：

| 根因 | 表现 | 正确应对 |
|---|---|---|
| **BF16 optimizer + torch foreach**（仅续训触发） | `RuntimeError: expected dtype float for 'end' but got c10::BFloat16` | 续训前 `export TORCH_OPTIM_FOREACH=0`；必要时把 `checkpoint-XXX/optimizer.pt` 里 BF16 张量转 FP32（原文件备份） |
| **caching_allocator_warmup OOM** | `torch.OutOfMemoryError ... 50.10 GiB`（模型加载期预分配） | 内存充足时重试即可；加载成功则无碍 |
| **无声死亡**（一次，step 500） | GPU 利用率归 0、ps 无进程、日志无 OOM | 未定位，疑似人工停止/信号 |

**铁律：续训必须用原配置 `train_v6_qwen38.json`，禁用 `train_v6_qwen38_resume.json`** —— 后者把 LR 从 2e-4 砍到 1e-4、dropout 从 0.0 改成 0.05，且实测续训时 cosine 调度器被重置从头 warmup（steps 1400–1700 全程近零 LR 空转），正是已发货适配器 provenance 混乱的根源。

**已确认（2026-08-17）：BF16 optimizer 污染在全新 run 的 checkpoint 里同样存在**（checkpoint-200/optimizer.pt = 992 FP32 + 1984 BF16）。它是 LoRA 参数 BF16 化的固有副作用，与是否续训无关。因此**任何从 checkpoint 续训都必须 `export TORCH_OPTIM_FOREACH=0`**，不是防御性措施，是硬性要求。

## 三、崩溃续训（若 PID 518663 死亡）

```bash
cd /home/chinux/jupyterlab/meerkatai
export TORCH_OPTIM_FOREACH=0   # 防 BF16 optimizer foreach bug（关键）

# 找到最新 checkpoint（用 sort -V，勿用 ls 顺序）
LATEST=$(ls -d checkpoints/qlora_triz_v6_qwen38/checkpoint-* 2>/dev/null | sort -V | tail -1)
echo "续训源: $LATEST"

nohup venv_v5/bin/python pipeline_v5/src/train.py \
  --config pipeline_v5/configs/train_v6_qwen38.json \
  --resume "$LATEST" \
  > checkpoints/qlora_triz_v6_qwen38/train.log 2>&1 &
echo $!
```

若报 BF16 dtype 错误（`optimizer.pt` 含 BF16 张量），用 venv 修复后重跑：

```bash
venv_v5/bin/python - <<'PY'
import torch, os
p="checkpoints/qlora_triz_v6_qwen38/<最新>/optimizer.pt"
opt=torch.load(p, map_location="cpu", weights_only=False)
for st in opt.get("state",{}).values():
    for k,v in list(st.items()):
        if isinstance(v, torch.Tensor) and v.dtype==torch.bfloat16:
            st[k]=v.float()
os.rename(p, p+".backup"); torch.save(opt, p)
print("optimizer.pt BF16→FP32 已修复")
PY
```

**注意**：全新 run 的 EarlyStopping 状态与 checkpoint 一致（best_metric 来自同一 run），续训不必重置 ES 计数——这与之前 resume_v2 从「另一段已饱和的 ES 状态」续训是两回事。

## 四、训练完成后的评测（与锚点对照）

> **预检已通过（2026-08-17）**：`MOONSHOT_API_KEY` 已配置（~/.bashrc，51 字符）；锚点产物完整（judge armA=2.93，0/300 invalid）；`eval_harness_v5.py --dry-run` 用 `eval_v5_qwen38_anchor.json` 跑通（exit 0，12/12 质量门，正确解析到锚点基线）。评测随时可发。

锚点产物（裸基座 Qwen3.8-27B，judge armA overall = 2.93，三决策门已通过）：
`results/v5/eval_v5_base_goldfix_v5_qwen38_20260815_023643.json`

训练完成后对**发货适配器**（`models/meerkat_triz_adapter_v6_qwen38`）跑评测：

```bash
cd /home/chinux/jupyterlab/meerkatai
export MOONSHOT_API_KEY=...   # judge 用
venv_v5/bin/python pipeline_v5/eval/eval_harness_v5.py \
  --config pipeline_v5/eval/configs/eval_v5_qwen38_anchor.json \
  --adapter-path models/meerkat_triz_adapter_v6_qwen38 \
  --baseline-results results/v5/eval_v5_base_goldfix_v5_qwen38_20260815_023643.json \
  --tag v6_gold
```

判读：跨基座对比以**各自锚点算差值**，不直接比绝对分。v5a 对照基线报告在 `results/v5/eval_v5_v5a_gold_20260729_065141.{json,md}`（不是 train config 里写的 `paper/v5_execution/` 旧路径）。v5a 关键值：keyword overall **0.6383**、judge armA **3.4233**、overrefusal **0%**、漏判率 25.0%。注意 `eval_v5_qwen38_anchor.json` 里 `keep_empty_think_block=true` 是待验证项，若质量门①成片失败需改 false 并记录。

## 五、异族评委评测（tensoris，决策门 G8 数据源）

> 前置：v6 主评测（第四节）跑完，产出 `results/v5/eval_v5_v6_gold_<ts>.json`。
> `TENSORIS_API_KEY` 已配置到远端 `~/.bashrc`（2026-08-18 验证通过，tensoris 面板 24 模型可用）。

对主评测产出的**同一批 300 条回答**（`records[].response`，不重新生成），用三个异族评委重打分：

```bash
cd /home/chinux/jupyterlab/meerkatai
eval "$(grep '^export TENSORIS_API_KEY' ~/.bashrc)"
venv_v5/bin/python pipeline_v5/eval/external_judge_track.py \
  --candidate-json results/v5/eval_v5_v6_gold_<ts>.json \
  --anchor-json    results/v5/eval_v5_base_goldfix_v5_qwen38_20260815_023643.json \
  --gold-jsonl     data/processed/v5_data/v5_gold.jsonl \
  --cmp-name v6_vs_base \
  --workdir results/ext_review_v6 \
  --judges claude-sonnet-4-6 gpt-5.4 gemini-3.5-flash \
  --time-budget 7200
```

- **最终三席（2026-08-18 定稿）**：`claude-sonnet-4-6`（Anthropic，0.180）/ `gpt-5.4`（OpenAI，0.680）/ `gemini-3.5-flash`（Google，0.780）。**与 v5b 异源终审评委完全一致**，跨版本可比。
- **翻转率实测（`flip_probe_v6.py`，50 题 × 3 次重复，臂 A 协议逐字一致，2026-08-18）**：
  | 评委 | 翻转率 | D6 门限 ≤0.02 |
  |---|---|---|
  | **claude-sonnet-4-6**（最终采用） | **0.180** | ❌（全面板最确定的外部评委） |
  | **gpt-5.4**（最终采用） | **0.680** | ❌（历史实测） |
  | **gemini-3.5-flash**（最终采用） | **0.780** | ❌（本轮实测；历史 0.800） |
  | claude-opus-4-8（已弃用） | 1.000 | ❌ 全部翻转，无重复性 |
  | gpt-5.6-terra（已弃用） | 0.860 | ❌ |
- 产物：`results/ext_review_v6/external_review_detail.json` + `external_review_fragment.json` + `external_review_brief.md`。
- **噪声账（D6 铁律）**：三评委翻转率均远超 0.02 门限，**关键结论必须 N=3 复跑取均值或附噪声传播合成 CI**（3× API 成本 + 宽 CI）。
- 判读：异族评委下 v6−base 差值若缩水 >50%（相对 Moonshot 同族读数），按 v5a 先例重述结论（v5a 同族 +0.39 → 异源 +0.09~+0.10）。
- 注：`--cmp-name` 需先在 `external_judge_track.py` 的 choices 里加 `v6_vs_base`（原只有 `v5_vs_base`/`v5_vs_v2`）。

## 六、登记台账

训练+评测完成后，在 `results/METRICS_LEDGER.md` 追加 v6 行（当前只到 corpus_sft_v2）：run 名、日期、数据集版本、epochs、best eval_loss、Layer1/2/3 差值、适配器 SHA-256、备注（换基座 Qwen3.8-27B + 崩溃/续训史 + 异族评委结论）。

## 七、已知遗留

- `auto_resume.sh`（checkpoint 目录内）有 typo（`TORCH_OPTIM_FOREATH`）且 resume 逻辑用错 config，**勿依赖**；已确认它当前不在运行。
- `train_v6_qwen38_resume.json` 是错误超参版本，归档勿再用。
- 已发货的 `meerkat_triz_adapter_v6_qwen38`（13:34 来自 resume_v2，step 1700 early-stop）provenance 有瑕疵（超参不一致 + LR 调度重置 + 丢两次 optimizer 状态），**将被本次全新 run 的干净产物覆盖**。
