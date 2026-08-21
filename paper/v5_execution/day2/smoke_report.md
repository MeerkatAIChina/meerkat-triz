# v5 Day2 阶段 2: top-2 臂 40 题金标双轨冒烟终判 (Worker M)

> 时间: 2026-07-25 17:12–18:59 (+08) ｜ 冒烟集: v5_gold_smoke40.jsonl (200 题金标分层等比抽 40, seed=42, 子集 8/8/8/6/6/4, v4/v5 源各 20)
> judge: moonshot-v1-32k, T=0, 臂A(反冗长+不截断); kw: 子串+别名表(9 条 confirmed); base/v2 用缓存响应(goldfix/v5 harness 谱系), 两臂现场 GPU 生成(E0 干净协议+四道质量门)

## 结果总表 (n=40)

| 模型 | 质量门 | kw 别名表后 | kw 别名表前 | judge 臂A | overrefusal |
|---|---|---|---|---|---|
| base (goldfix) | 40/40 | 0.6106 | 0.5541 | 3.250 | — |
| v2 | 40/40 | 0.5712 | 0.5627 | 2.775 | 0/40 过门 |
| **2e-4/rsF** (sweep#1) | 40/40 | **0.6632** | 0.6534 | 3.475 | 0/40 过门 |
| **5e-4/rsF** (sweep#2) | 40/40 | 0.6403 | 0.6242 | **3.625** | 0/40 过门 |

## 两臂配对检验 (stdlib Random(42), bootstrap 10000)

- judge (5e-4 − 2e-4): **+0.150, 95%CI [−0.050, +0.425]** — CI 跨 0, 噪声内
- kw (5e-4 − 2e-4): **−0.023, 95%CI [−0.071, +0.018]** — CI 跨 0, 无显著退化

## 终判 (§5.3 规则)

1. 质量灾难检查: 两臂质量门失败率 0% (<5%), overrefusal 均过门, judge 均高于 base/v2,
   与其 loss 排名(小扫 #1/#2)相符 → **无淘汰**。
2. 两臂 judge/kw 差异 CI 均跨 0 → **噪声内平局**。
3. 平局取默认: **胜出家 = lr 2e-4 / rsLoRA=False**(与 v4 可比性优先; 亦为小扫 loss 第 1 名 1.558633)。

## harness 修复记录 (只修不绕)

- `eval_harness_v5.py` GPU 生成路径 `import compat` 失败 (ModuleNotFoundError: compat 在 pipeline_v5/src,
  未入 sys.path) → 已修(sys.path 追加 pipeline_v5/src 并注释), 断点重启后续跑完成。

## 产物 (远端 results/v5/)

- eval_v5_{base_goldfix_smoke40,v2_smoke40,sweep2e4_smoke40,sweep5e4_smoke40}_*.json/.md
- 生成缓存 v5_gen_{base_goldfix,v2,sweep2e4,sweep5e4}_smoke40.jsonl; 漏判审计 v5_miss_audit_*_smoke40.jsonl
- 冒烟集 data/processed/v5_data/v5_gold_smoke40.jsonl; 链日志 smoke_chain.log; smoke_chain.done
