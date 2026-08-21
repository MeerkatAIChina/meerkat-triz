# eval2 评测流水线

base / v1 / v2 / v3 四方对比评测，分三阶段。代码在 `/tmp/eval_pipeline_v2/`，输出在 `results/eval2/`。

## 用法

```bash
cd /home/meerkat/mongoose_ai
eval "$(grep '^export MOONSHOT_API_KEY' ~/.bashrc)"   # judge 轨需要

# 分阶段（推荐，便于断点续跑）
venv_v5/bin/python /tmp/eval_pipeline_v2/eval2.py --phase generate --models base,v1,v2,v3
venv_v5/bin/python /tmp/eval_pipeline_v2/eval2.py --phase score    --models base,v1,v2,v3
venv_v5/bin/python /tmp/eval_pipeline_v2/eval2.py --phase report   --models base,v1,v2,v3

# 调试
venv_v5/bin/python /tmp/eval_pipeline_v2/eval2.py --phase generate --models base --limit 5
venv_v5/bin/python /tmp/eval_pipeline_v2/eval2.py --phase score --models base --limit 5 --skip-judge
venv_v5/bin/python /tmp/eval_pipeline_v2/eval2.py --phase judge_smoke   # judge API 冒烟测试

# 一键（含等待旧评测结束的 watcher）
tmux new-session -d -s eval2_chain /tmp/eval_pipeline_v2/watch_and_run.sh
tmux attach -t eval2_chain        # 观察进度；日志同时 tee 到 results/eval2/eval2_<ts>.log
```

## 输出文件（results/eval2/）

| 文件 | 内容 |
|---|---|
| `responses_<tag>.json` | 每题原始回答（断点续跑：完整则跳过该 tag） |
| `judge_<tag>.json` | LLM judge 缓存（断点续跑） |
| `scores_<tag>.json` | 每题关键词轨 + judge 轨分数 |
| `report_<ts>.json/.md` | 配对 bootstrap / McNemar / Wilson 统计报告 |

## 指标

- 关键词轨：principle_accuracy（期望原理全覆盖 0/1）、contradiction_coverage、case_coverage、ariz_step_coverage（6 步骤）、general_probe_coverage（30 题通用回归探针）
- judge 轨（moonshot-v1-8k，批量 10 条，RPM=3）：ariz 6 步骤语义命中、contradiction 概念语义覆盖
- 综合：overall_kw = 0.3·principle + 0.3·contradiction + 0.2·case + 0.2·ariz（与旧公式一致）；overall_judge 为 judge 替换变体
- 统计：配对 bootstrap（10000 次，分层重抽样，seed=42）、McNemar 精确检验（principle）、Wilson CI

## 与旧评测的差异 / 降级说明

1. 扩充评测集是 SFT 格式（instruction/input/output），无 expected_keywords 字段，期望关键词由 reference 自动抽取（原理名、矛盾参数名）；数据集中无选择题，principle 轨改为"期望原理全覆盖 0/1 + 覆盖率"。
2. judge 输入的 response 截断前 500 字符（moonshot-v1-8k 上下文限制，10 条/批）。
3. 未使用 BERTScore（venv 无 bert_score，且不允许 pip install）；以 sacrebleu(zh) + ROUGE 作参考轨。
4. concept_explanation / innovation_assessment 仅作参考轨（concept_coverage / 原始响应），不进入 overall。
