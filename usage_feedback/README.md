# usage_feedback — 真实使用数据回流管线

把 DGX Spark 上 Meerkat-TRIZ-v1 的真实使用数据周期性回流为 v5c 训练候选料。

## 架构

```
DGX Spark                                    本机 workspace
├─ Open WebUI (webui.db: chat+feedback) ──┐
└─ pi (~/.pi/agent/sessions/*.jsonl)    ──┤ ssh/scp
                                          ▼
                    usage_feedback/raw/<date>/         ← fetch_usage_bundle.sh
                                          ▼
                          分诊：本地失败启发式 + moonshot 评委 rubric
                                          ▼
                    usage_feedback/triage/<date>/      ← triage_usage.py
                    ├─ candidate_sft.jsonl    (judge 双维度 ≥4，无失败标记)
                    ├─ candidate_prefs.jsonl  (👍/👎 反馈 → DPO 偏好对)
                    ├─ failures.md            (identity_denial / repetition_loop / runaway / empty)
                    └─ summary.md             (总量/分布/主题)
```

## 手动运行

```bash
# 1. 采集（增量，水位线在 state/last_harvest.json）
bash usage_feedback/scripts/fetch_usage_bundle.sh

# 2. 分诊（密钥从 spark 的 ~/.bashrc 读取，不落盘到仓库）
eval "$(ssh spark-855a "grep '^export MOONSHOT_API_KEY' ~/.bashrc")"
python3 usage_feedback/scripts/triage_usage.py usage_feedback/raw/<最新目录>

# 只跑本地失败检测、不调评委：加 --no-judge
```

## 筛选规则

- **SFT 候选**：无失败标记 且 `triz_correctness ≥ 4` 且 `data_value ≥ 4`（双门槛，防止评委对通用闲聊给人情分）
- **偏好对**：Open WebUI feedback 表的 👍(+1)/👎(−1)，供 DPO 启用后配对
- **失败案例**：四类本地启发式（身份否认 / 重复循环 / 失控超长 / 空输出），不进语料，进 failures.md 做归因

## 每周定时任务

Automation 每周日 09:17（Asia/Shanghai）自动跑采集+分诊并在会话里汇报摘要。
人工动作：看 `triage/<date>/summary.md`，把认可的数据归档进 v5c 数据池。

## 原则

- 原始日志**不直接进训练**：全部过 去重 → 评委 → 人工抽审 三道筛
- 密钥不落仓库：moonshot key 运行时从 spark 读取
- 真实使用数据的价值在分布发现与失败归因，不在体量
