# kimi-k3 评委适配性探针报告（2026-07-31）

探针：`paper/bridge/k3_probe.py`（远端 spark-855a 执行，可重跑）

## 结论（先行）

**kimi-k3 在当前 API 形态下不能担任钉版同族评委**——D6 门规第 2 步
（T=0 翻转率 ≤0.02）硬性不通过。建议：moonshot-v1-32k 继续作为钉版
同族评委；k3 转作 Phase 1 专利语料抽取的长上下文工人（该场景不要求
确定性）。外部面板升级（gpt-5.5 等）不受此阻，可按桥接协议推进。

## 实测数据

### 1. 温度参数（协议破坏性）
| 参数 | 结果 |
|---|---|
| temperature=0 / 0.0 | **HTTP 400: "invalid temperature: only 1 is allowed for this model"** |
| 省略（默认） / =1 | accepted |

项目的钉版评委协议（T=0 硬断言）与 k3 直接冲突。

### 2. 确定性翻转率（评委 prompt ×5，默认温度）
- 唯一输出 2/5，**存在翻转**（run0/run4 同为 overall=3 但 JSON 文本不同；
  run1–3 返回**空 content**，仅 reasoning）
- 空 content 率 3/5——judge batch 会记录为 missing，属严重集成隐患
- 对照：moonshot-v1-32k 同 prompt 翻转率 0.000（历史实测）

### 3. 性能与成本
- k3 单次评分 ~27–35s（含 3.2k–4.6k 字符 reasoning），v1-32k 同任务 1.0s
  ——**约 30× 延迟 +  reasoning token 成本**
- 32k 字符长文档输入正常（19k prompt tokens，8.8s）——长上下文能力真实可用

### 4. tensoris 面板（对照可达性实验，本地直连）
- 25 模型全部可达；gpt-5.5 / gpt-5.6-sol **接受 T=0**，响应正常
- claude 最新 sonnet 仍为 4-6（现评委已是）；gemini-3.5-flash 已是最新 flash
- 外部面板升级候选：gpt-5.4 → gpt-5.5（或 5.6-sol），claude/gemini 无需动

## 对方案的修订（v6_redesign_plan §2b 增补）

1. 同族评委：v1-32k 继续钉版；k3 评委化冻结，直至 API 开放 T=0 或
   提供确定性推理模式（seed/确定性开关）后重新探针。
2. k3 的正当用途：Phase 1 专利全文 → 矛盾描述抽取（长上下文、
   非确定性可容忍、产出过数据契约门 D1）。
3. 外部面板桥接按 §2b 七步推进：gpt-5.5（+备用 5.6-sol）先过
   T=0 翻转率（50 题×3），再三臂桥接重评分。
4. 若 moonshot 未来停服 v1-32k：启动应急预案——k3 评委化需改为
   "同 prompt N=5 多数投票"协议并重新校准全部门限（成本 5×，且需
   在论文中作为协议变更披露）。
