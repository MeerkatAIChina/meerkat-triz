# STATE_WorkerS — v5 Day2 阶段 1 P0 超参小扫

> 角色:Worker S 超参小扫 ｜ 最近更新:2026-07-25 17:10 (远端 +08) ｜ 状态:**✅ 小扫完成 (6/6,CHAIN_COMPLETE),待 Orchestrator 审查进阶段 2**

## 终稿 (17:10,6/6 组)
| 排名 | 臂 | 终点 eval_loss | best eval_loss@step | 步数 | 早停 | 时长 h | 显存 GB | best/ 验证 |
|---|---|---|---|---|---|---|---|---|
| 1 | 2e-4/rsF | **1.558633** | 1.558633@669 | 669 | 否 | 2.912 | 69.72 | PASSED |
| 2 | 5e-4/rsF | **1.559034** | 1.559034@669 | 669 | 否 | 2.927 | 69.72 | PASSED |
| 3 | 1e-4/rsT | 1.566711 | 1.566545@600 | 669 | 否 | 2.911 | 69.72 | PASSED |
| 4 | 1e-4/rsF | 1.573316 | 1.573316@669 | 669 | 否 | 2.909 | 69.72 | PASSED |
| 5 | 2e-4/rsT | 1.611503 | 1.611503@669 | 669 | 否 | 2.921 | 69.72 | PASSED |
| 6 | 5e-4/rsT | 6.294009 | 2.906696@100 | 400 | **是(发散)** | 1.216 | 69.72 | PASSED |

- **rsLoRA 最终裁决:不开启**。配对差(正=rsT 更差):2e-4:**+0.0529**;1e-4:−0.0066(平局带内);
  5e-4:**+4.735,训练发散**(2.9067→7.28→6.65→6.29,早停@400,有效更新 ×8 所致)。
  config.py:65 配置债正式回答:r=64/α=128 下 rsLoRA 无收益且高 lr 下不稳定。
- **lr 主效应(rsF)**:2e-4 1.5586 ≈ 5e-4 1.5590(平局)> 1e-4 1.5733。**支持默认 2e-4**。
- **阶段 1 初筛 top-2:2e-4/rsF + 5e-4/rsF**(差 0.0004 互为平局);平局带内还有 1e-4/rsT。
- **"极差<0.01 取默认"条款:未触发**(6 组极差 4.735,含发散臂)。
- **短跑外推风险声明**:各正常组末段仍在降、均未早停,0.5 epoch 系统性偏好高 lr(R-04);
  5e-4/rsF 与 2e-4/rsF 的 0.0004 差在噪声内,终判须阶段 2 金标双轨冒烟;冒烟平局取 2e-4/False。
- 终稿 + 6 组 run_summary + 6 组 train_log 已全部回传本地 `paper/v5_execution/day2/`。

## 阶段 2 交接:top-2 臂 checkpoint 路径 (远端 /home/meerkat/mongoose_ai)
| 臂 | 发货适配器(已验证) | best 源 | adapter_info |
|---|---|---|---|
| 2e-4/rsF | `models/sweep_adapters/sweep_lr2e-4_rsFalse/` | `checkpoints/sweep_lr2e-4_rsFalse/best/` (best@669) | 同目录 adapter_info.json |
| 5e-4/rsF | `models/sweep_adapters/sweep_lr5e-4_rsFalse/` | `checkpoints/sweep_lr5e-4_rsFalse/best/` (best@669) | 同目录 adapter_info.json |
- 两臂均 669 步跑满、best=终点、验证 PASSED(620 tensors 全 BF16 非零)。
- 阶段 2 = 两臂 40 题金标双轨冒烟(judge=moonshot-v1-32k,T=0,AB/BA 双序),选 judge 更高且 kw 无显著退化者;仍平局取 2e-4/False。**本 Worker 不执行,由 Orchestrator 安排。**
- tmux v5sweep 已退出(CHAIN_COMPLETE);GPU 已空闲。

## 窗口 4 结果 (15:55,5/6 组)
| 排名 | 臂 | best eval_loss@step | 终点 eval_loss | 时长 h | 显存 GB | best/ 验证 |
|---|---|---|---|---|---|---|
| 1 | 2e-4/rsF | **1.558633** @669 | 1.558633 | 2.912 | 69.72 | PASSED |
| 2 | 5e-4/rsF | **1.559034** @669 | 1.559034 | 2.927 | 69.72 | PASSED |
| 3 | 1e-4/rsT | **1.566545** @600 | 1.566711 | 2.911 | 69.72 | PASSED |
| 4 | 1e-4/rsF | **1.573316** @669 | 1.573316 | 2.909 | 69.72 | PASSED |
| 5 | 2e-4/rsT | **1.611503** @669 | 1.611503 | 2.921 | 69.72 | PASSED |

- **rsLoRA 配对(终点差,正=rsT 更差)**:2e-4:**+0.0529**;1e-4:**−0.0066**(rsT 略优但在 0.01 平局带内);5e-4 待组 6。
  **rsLoRA 阶段结论:不应开启**——最优 rsT 臂(1e-4/T 1.5667)仍输两个 rsF 臂 ~0.008;
  2e-4 下大幅更差;rsLoRA 只是把它推向更低 lr 才追平,无收益。待组 6 终裁(预期 5e-4/rsT 更差)。
- **lr 主效应(rsF 三点)**:1e-4:1.5733 / 2e-4:1.5586 / 5e-4:1.5590——2e-4 与 5e-4 平局(差 0.0004),
  均优于 1e-4 ~0.014;**支持默认 2e-4**。
- 1e-4/rsT 是唯一 best≠终点的组(best@600,669 微升 0.0002,噪声级)。
- 当前初筛 top-2:2e-4/rsF、5e-4/rsF(与基线平局带内含 1e-4/rsT);组 6 须 <1.5586 才撼动格局。
- 组 6(5e-4/rsT)15:14 启动,窗口检查时 step ~184/669,预计 18:18 完成。
- 5/6 中期报告已回传本地。

## 窗口 3 中期结果 (09:45,3/6 组)
| 排名 | 臂 | best eval_loss@step | 时长 h | 显存 GB | best/ 验证 |
|---|---|---|---|---|---|
| 1 | 2e-4/rsF | **1.558633** @669 | 2.912 | 69.72 | PASSED |
| 2 | 1e-4/rsF | **1.573316** @669 | 2.909 | 69.72 | PASSED |
| 3 | 2e-4/rsT | **1.611503** @669 | 2.921 | 69.72 | PASSED |

- **rsLoRA 配对 @2e-4:True 显著更差 +0.0529**(1.6115 vs 1.5586)——同 lr 下 rsLoRA 有效更新放大 8 倍,2e-4 对它过高;轨迹全程高悬(100:1.8089 → 669:1.6115)。关键看组 4(1e-4/rsT)能否翻盘——理论预测 rsLoRA 最优 lr 更低。
- **lr 配对 @rsF:2e-4 优于 1e-4 −0.0147**,差 > 0.01 平局阈值;但两组末段仍下降未早停(短跑偏好高 lr,R-04 风险在案)。
- 三组均 669 步跑满、无早停、best=终点;单组实际 ~3.0h(含加载/eval/发货)。
- 中期报告已回传 `paper/v5_execution/day2/sweep_report.{md,json}` + 3 组 run_summary。
- 组 4(1e-4/rsT)09:08 启动,训练中(eval 阶段观测正常)。

## 窗口 2 检查结论 (00:50)
1. **checkpoint 验证**:组 1 `best/` 与 `checkpoint-100` 均 PASSED(620 tensors,A/B 各 310,全 BF16,全部 lora_B 非零,sha256 11 文件)——pipeline_v5 验证器远端直跑。
2. **eval 轨迹对照**(同 completion-only 口径,可与 v4 比):
   - v5 组1(2e-4/F):step100 = **1.635336**
   - v4 主 run:100:1.64724 → 200:1.62610 → 300:1.60921 → 400:1.59401 → 500:1.57834 → 600:1.57078 → 700:1.55923(best)→ 800:1.58327(回升)
   - 组 1 起点低于 v4 同点 0.012,方向健康;单点尚不能下结论,待 200-400 点确认下降趋势。
   - 机制确认:eval 先于 save 触发,best 回调 pending→on_save 补存按设计工作(日志可见 deferred 促销)。
3. **链可靠性**(读脚本+md5 核对 2c3cfc02 两端一致,tmux 存活):
   - 组崩溃(rc≠0,≠3)→ 写 .failed 续下组 ✓;rc=3 → CHAIN_ABORTED 停链 ✓;
   - .done 跳过已完成组 ✓;链整体重启时被中断组从头重训(无 --resume,checkpoint 轮转覆盖,可接受)✓。

## 任务锚点
- 远端:`ssh -o BatchMode=yes chinux@spark-855a`,项目 `/home/meerkat/mongoose_ai`(实为 /home/chinux/jupyterlab/meerkatai 的软链)
- tmux 会话:**v5sweep**;链日志:`checkpoints/sweep_chain.log`
- 数据:v5_train.jsonl 10,698 条 / v5_validation.jsonl 1,050 条(已核对)
- 小扫矩阵:lr {1e-4,2e-4,5e-4} × rsLoRA {F,T} 共 6 组,每组 max_steps=669(0.5 epoch),eval/save=100

## 已完成
1. pipeline_v5/src/train.py 已按 §11.2 落实并推送远端(不覆盖 v4;compat/checkpointing 原样复制):
   cosine horizon=⌈10698/8⌉×2=2676(create_optimizer_and_scheduler 重建,max_steps 不缩 horizon)、
   early_stopping_threshold=0.002、max_length=2048 断言(退出码 3)、optim=adamw_torch 显式、
   loss 冒烟(首 batch labels:prompt 全 −100 且 completion 无 ChatML 泄漏,失败退出码 3)、
   completion_only_loss 断言(退出码 3)、末步 checkpoint 落盘+终点 eval、时长/显存峰值记录。
2. 6 组配置生成:`pipeline_v5/configs/sweep/sweep_lr{1e-4,2e-4,5e-4}_rs{False,True}.json`,
   独立输出 `checkpoints/<arm>/` + `models/sweep_adapters/<arm>/`。
3. 链脚本 `pipeline_v5/run/run_sweep_chain.sh`:GPU 串行,单组失败写 .failed 续跑,
   rc=3(断言/冒烟)立即停链(CHAIN_ABORTED),每组完成写 `checkpoints/sweep_status/<arm>.done`。
4. **首组 sweep_lr2e-4_rsFalse 健康验证全部通过**(00:09):
   completion_only_loss=True ✓;loss 冒烟 PASSED(prompt 52 token 全 −100,监督 1121 token)✓;
   horizon 重建 2676 步/warmup 133 ✓;EarlyStopping patience=3 threshold=0.002 ✓;
   LoRA 620 张量 BF16 校正、可训练 84.66M/34.7B=0.2437%(与 v4 一致)✓;
   步速 ~11-12.7 s/step。

## 运行顺序与进度
| # | 臂 | 状态 |
|---|---|---|
| 1 | sweep_lr2e-4_rsFalse | ✅ best 1.558633@669 |
| 2 | sweep_lr2e-4_rsTrue | ✅ best 1.611503@669 |
| 3 | sweep_lr1e-4_rsFalse | ✅ best 1.573316@669 |
| 4 | sweep_lr1e-4_rsTrue | ✅ best 1.566545@600 |
| 5 | sweep_lr5e-4_rsFalse | ✅ best 1.559034@669 |
| 6 | sweep_lr5e-4_rsTrue | 训练中(15:14 起,预计 18:18) |

## 组 6 完成后剩余步骤 (下一窗口)
1. 远端:`venv_v5/bin/python pipeline_v5/run/summarize_sweep.py`(6/6 自动转终稿,去"中期"标注)。
2. 回传:`scp chinux@spark-855a:/home/meerkat/mongoose_ai/results/{sweep_report.json,sweep_report.md,run_summary_sweep_lr5e-4_rsTrue.json,train_log_sweep_*.json} paper/v5_execution/day2/`
3. 复核组 6 best/ 验证 + CHAIN_COMPLETE 存在;终稿结论:top-2 臂 + 是否触发"极差<0.01 取默认"条款
   (当前 5 组极差 0.0529,已不可能触发;除非组 6 爆冷,top-2 = 2e-4/rsF + 5e-4/rsF)。
4. 汇报后本任务交付;阶段 2(top-2 金标冒烟)由 Orchestrator 安排。

实测步速 ~11-12.7 s/step → 单组 ≈ 2.2h 训练 + ~9min 加载 ≈ **2.4h/组,6 组 ≈ 14-15h**
(比任务预估 7-9h 长:0.5 epoch=669 步 × 实测步速,原估 1.2-1.5h/组偏乐观)。

## Resume 操作手册
1. 查进度:`ssh -o BatchMode=yes chinux@spark-855a 'tail -5 /home/meerkat/mongoose_ai/checkpoints/sweep_chain.log; ls /home/meerkat/mongoose_ai/checkpoints/sweep_status/'`
2. 查当前组:`grep -v "examples/s\|it/s\]" .../checkpoints/<当前臂>/train.log | tail -20`
3. 单组完成产物:`results/run_summary_<arm>.json`(轨迹/best/时长/显存)+ `results/train_log_<arm>.json` + adapter_info.json
4. 全 6 组完成(CHAIN_COMPLETE)后:远端跑 `venv_v5/bin/python pipeline_v5/run/summarize_sweep.py`
   → `results/sweep_report.{json,md}`;回传本地 `paper/v5_execution/day2/`:
   `scp chinux@spark-855a:/home/meerkat/mongoose_ai/results/{sweep_report.json,sweep_report.md,run_summary_sweep_*.json,train_log_sweep_*.json} paper/v5_execution/day2/`
5. 若 CHAIN_ABORTED(rc=3):读失败组 train.log 尾部,修复后删对应 .failed 重启 tmux 链(断点续跑)。
6. 不做 top-2 金标冒烟(阶段 2,Orchestrator 审查后另行安排)。

## 关键纪律
- 汇报 ≤250 字;窗口将尽先更新本文件再汇报。
- 冒烟/断言失败 = 停链报告,不得跳过。
