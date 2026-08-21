# v5 完成后工作项(Owner 已确认,2026-07-24)

## W-E5+:E5 模块消融扩展档(含 shared_expert 臂)

**来源**:Owner 在 MoE 讨论后确认"v5做完后需要"(2026-07-24 19:01 对话)。
**启动前提**:v5 主训练完成 + Day 3 评测与决策门 2.0 判定完成;GPU 排期空闲。

### 实验设计(草案,v5 主 run 后定稿)
- **科学问题**:在 Gated DeltaNet(30层)+ Gated Attention(10层)+ MoE 混合架构上,领域能力沉积在哪条通路?shared_expert 通路在 routed experts 零覆盖(实测 0/620 tensors)的前提下能承载多少领域能力?
- **臂**(同数据 v5、同超参=小扫胜出家、同预算短 run):
  1. 仅 Attention(10 层 q/k/v/o_proj)
  2. 仅 DeltaNet(30 层 in_proj_qkv/z/b/a + out_proj)
  3. **仅 shared_expert(40 层 mlp.shared_expert gate/up/down_proj)——本次新增**
  4. 全 12 模块同预算短 run 对照(不能与 v5 完整 run 直接比,步数混淆须注明)
- **评测**:40 题金标双轨冒烟(与阶段 2 同口径)+ ARIZ rubric 语义轨(该维度是 v4 唯一显著正面发现,通路归因最有价值的子集)。
- **预算**:4 臂 × ~2.4-3h GPU(0.5-1 epoch 待定)+ 评测 ~2h ≈ 12-14h。
- **产出**:通路归因报告 + 论文贡献点 A 升级材料;若 shared_expert 臂显著强于预期,追加讨论专家级 LoRA(MoLE 式)的可行性边界。
- **已知混淆**(报告须声明):DeltaNet 层投影维度小,r=64 相对过参数化;短 run 偏好高 lr;routed expert 零覆盖下 shared_expert 臂测的是"共享通路承载上限"而非全部 MLP 贡献。

### 状态
- [ ] v5 主 run 完成
- [ ] Day 3 评测与决策门完成
- [ ] E5+ 设计定稿(预算/epoch/判据)
- [ ] E5+ 执行与归因报告
