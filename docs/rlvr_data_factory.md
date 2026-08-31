# RLVR 数据工厂设计

> 电商系统的第一定位：为 RLVR（Reinforcement Learning with Verifiable Rewards）提供**可验证奖励 / 训练数据**。决策质量与业务收益是第二层衍生价值，不是首要目标。

## 一、定位反转：先造"数据工厂"，再造"决策系统"

传统思路把电商系统当成"给团队用的决策工具"，RLVR 视角下它是：

```
电商闭环 = 可验证奖励发生器
   ↓
产出 (state, action, verifiable_reward) 样本
   ↓
喂给 RLVR 训练 pipeline（verl / OpenRLHF）
   ↓
训练出的模型反过来服务 600 人团队 → 数据飞轮
```

**第一指标**：单位时间能稳定产出多少条高质量、奖励有区分度的样本。决策烂没关系——每个错误决策 + 真实结果就是一个高价值负样本。

## 二、真实数据源画像（order.csv）

| 维度 | 值 |
|---|---|
| 订单量 | 382,287 条 |
| 时间跨度 | 2024-01-01 → 2025-01-02（368 天）|
| 商品数 | 1,000（订单数 279~773，**完全不稀疏**）|
| 品类 | 5（食品生鲜/3C电子/美妆护肤/家居日用/服装鞋帽）|
| 品牌 | 15（小米/三只松鼠/花西子/UNIQLO/Apple…）|
| 促销类型 | None / Coupon / FlashSale / FullDiscount |
| 订单状态 | Delivered 60% / Shipped / Paid / Pending / Cancelled |
| 日销量 | 天均 1,299 件 / 1,039 单 / GMV 178 万 |

字段：`order_time, quantity, amount, price, brand, category, promotion_type, order_status, ...`

## 三、可验证性分层（RLVR 任务选型）

RLVR 的命门：奖励必须**自动、确定、无人判断**地算出。

| 层级 | 任务 | 奖励 | 反事实问题 | 首期 |
|---|---|---|---|---|
| **Tier 1** | 销量预测、库存预测 | `-clip(\|pred−actual\|/actual)` | **零反事实** | ✅ 首选 |
| Tier 2 | 补货、调价 | 缺货率 / 收益 delta | 有（基准不可观测）| ⏸ 二期 |
| Tier 3 | 选品、文案、品牌调性 | 转化率 | 噪声大、难量化 | ❌ 剔除 |

**结论：RLVR 第一批燃料必须是纯预测任务（Tier 1）**，因为它奖励确定、可事后核对、无需"如果不这样会怎样"的假设。

## 四、首期任务：商品-天销量预测

- **粒度**：商品 × 天（1,000 商品 × 368 天）
- **state**：过去 30 天销量序列 + 品牌 + 品类 + 价格 + 促销类型
- **action**：预测未来 7 天总销量
- **样本量**：约 `1000 × (368-30-7) ≈ 331,000` 条
- **ground_truth**：未来 7 天真实销量（事后核对）

### 粒度选型的硬证据（naive baseline = 最近7天销量预测未来7天）

| 粒度 | 样本数 | reward 均值 | 标准差 | 结论 |
|---|---|---|---|---|
| 品类-天 | 1,650 | -0.069 | 0.058 | ❌ naive 已准(7%误差)，无学习空间 |
| 品牌-天 | 4,950 | -0.084 | 0.073 | ❌ 同上 |
| **商品-天** | **330,000** | **-0.485** | **0.348** | ✅ naive 误差48%，学习空间大 |

**选型黄金标准**：RLVR 任务必须满足「baseline 有改善空间 + ground_truth 确定」。聚合粒度（品类/品牌）销量太平滑，naive 预测几乎完美（reward≈0），模型无从学起；商品粒度 naive 有 48% 误差，模型有真实学习空间——即使单样本噪声大（日销 0-1 泊松），33 万样本平均下来能学到稳定的跨商品/跨时间模式。

## 五、数据契约（先行锁定）

对齐 verl / OpenRLHF 主流格式：

```json
{
  "prompt": "商品【小米 蓝牙耳机】(3C电子, 价格199元): 过去30天销量 [2,0,1,3,...]。请预测未来7天总销量。",
  "response": "预测未来7天总销量: 14 件",
  "reward": -0.15,
  "ground_truth": 12,
  "verifier": "verify_sales_forecast",
  "metadata": {
    "product_id": "200262", "brand": "小米", "category": "3C电子",
    "window_start": "2024-06-01", "window_end": "2024-06-07"
  }
}
```

**关键约定**：
- `prompt` 只含 state（历史可观测信息），**绝不含未来信息**（防泄漏）
- `ground_truth` 是事后核对的事实，训练时 verifier 用它算 reward
- `reward` 由**确定性函数**计算，不依赖任何人工判断
- 样本生成器输出 `(prompt, ground_truth, metadata)`，`response`/`reward` 在 RLVR 训练时由模型生成 + verifier 计算

## 六、奖励函数

```python
def verify_sales_forecast(pred: float, actual: float) -> float:
    """销量预测奖励: 相对误差的负值, clip 到 [-1, 0]"""
    if actual <= 0:
        return 0.0 if pred <= 0 else -1.0   # 实际为0, 预测>0 判错
    ape = abs(pred - actual) / actual        # 绝对百分比误差
    return -min(ape, 1.0)                    # 误差≥100% 都记 -1
```

- 预测准（误差 0%）→ reward 0；误差 100%+ → reward -1
- 奖励**单调、有区分度**，无人工阈值

## 七、样本生成器（离线回测）

```
order.csv → 按 (product_id, day) 聚合销量 → 滑动窗口切分
   → 30 天历史 = state, 未来 7 天 = ground_truth
   → 输出 JSONL 样本
```

**完全离线、无需平台 API、无需真实执行**——这是今天就能跑通、直击"可验证奖励"的最小路径。

### 防泄漏三检查（硬约束）

1. `prompt` 只含 `window_start` 之前的数据
2. `ground_truth` 严格是 `window_end` 之后的销量
3. 时间戳不进入 prompt（避免模型学时间作弊）

## 八、数据飞轮（三步）

```
Step 1  离线回测（今天）: 33 万样本, 冷启动, 无业务依赖
Step 2  RLVR 训练 v0: 喂 verl/OpenRLHF, 训出销量预测模型
Step 3  真实闭环: 模型接生产预测 → 真实结果回流 → 样本持续增长
```

## 九、一个关键推论

**可验证性 = 系统的硬约束**。凡无法量化为"事后可核对的数字"的决策，对 RLVR 零价值，应从首期范围剔除。这反过来精简了电商系统的首期范围——**不是"先做完整电商系统"，而是"先做销量预测这一个可验证闭环"**。慢即是快。
