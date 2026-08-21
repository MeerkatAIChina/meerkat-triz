# STATE_WorkerM — v5 Day2 阶段 2 冒烟终判 + v5 主训练

> 角色:Worker M 主训练 ｜ 最近更新:2026-07-25 19:10 (远端 +08) ｜ 状态:**✅ 冒烟终判完成(胜出家 2e-4/rsF);主 run 训练中(tmux v5main)**

## 冒烟终判结论(§5.3)

| 模型 | 质量门 | kw 别名表后 | judge 臂A | overrefusal |
|---|---|---|---|---|
| base (goldfix) | 40/40 | 0.6106 | 3.250 | — |
| v2 | 40/40 | 0.5712 | 2.775 | 0/40 |
| 2e-4/rsF | 40/40 | **0.6632** | 3.475 | 0/40 |
| 5e-4/rsF | 40/40 | 0.6403 | **3.625** | 0/40 |

- 配对 bootstrap(10000,seed42):judge(5e4−2e4)=+0.150 [−0.050,+0.425];kw=−0.023 [−0.071,+0.018] — CI 均跨 0,噪声内平局。
- 无质量灾难(两臂门失败率 0%,judge 均超 base/v2)→ 平局取默认:**lr=2e-4 / rsLoRA=False**(可比性优先,亦小扫 loss 第 1)。
- 报告:远端 `results/v5/smoke_report.md` + 本地 `paper/v5_execution/day2/smoke_report.md`(已回传,含 4 模型 eval json 与冒烟集)。
- harness 修复记录:`eval_harness_v5.py` GPU 路径 `import compat` ModuleNotFoundError → sys.path 追加 pipeline_v5/src(只修不绕,断点续跑完成)。

## 主 run 状态(tmux v5main,19:04 启动)

- 配置 `pipeline_v5/configs/train_v5.json`:lr=2e-4/rsF、horizon 2676(1338×2)/warmup 133、epochs=4 安全帽、patience=3/threshold=0.002、eval=save=100、max_length=2048、adamw_torch 显式。
- 启动断言全绿:completion_only_loss=True ✓;loss 冒烟 PASSED(prompt 52 token 全 −100,无 ChatML 泄漏)✓;LoRA 620 张量 BF16、可训练 84.66M/34.7B=0.2437% ✓。
- 日志 `checkpoints/qlora_triz_v5/train.log`;预算帽 timeout 25200(7h);预计 2.5-4h 早停(~step 1000-1500)。
- 完成后 train.py 自动:best 磁盘发货 → `models/meerkat_triz_adapter_v5/` + adapter_info.json + `results/train_log_v5_main.json` + `results/run_summary_v5_main.json`。

## 下一窗口待办

1. 查训练进度:`ssh -o BatchMode=yes chinux@spark-855a 'grep -v "it/s\|s/it" /home/meerkat/mongoose_ai/checkpoints/qlora_triz_v5/train.log | tail -5'`
2. 完成后:适配器验证(lora_B 全非零/BF16/sha256,跑 pipeline_v5 验证器)、adapter_info.json 核对、
   `cp results/train_log_v5_main.json results/train_log_v5.json`、登记 `results/METRICS_LEDGER.md`(未评测字段标"待 Day 3")。
3. 回传本地 `paper/v5_execution/day2/`(adapter_info/train_log/run_summary/METRICS_LEDGER 行)+ git commit(本地 paper/v5_execution/)。
4. 中断续跑:`tmux new-session -d -s v5main "bash pipeline_v5/run/run_main_v5.sh --resume"`。
