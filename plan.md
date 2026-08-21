# 任务计划 — 东方智感《智能体及AI最新发展》全员培训 PPT

## 背景
- 客户:东方智感(insentek.com)——物联网环境传感器公司,业务:智慧农业(智墒/耘曜)、智慧水利、自然灾害监测(山境/天圻)、物质检测(方物);自研传感器+物联网平台+大数据AI引擎。
- 受众:全体员工,已完成前两期培训(机器学习基础、Prompt Engineering)。
- 目标:提升**智能体思维**,掌握 **Harness / Loop Engineering**,并与各岗位工作结合。
- 交付:PPTX(约 28 页),中文,16:9。

## 工作流

### Stage 0 — 准备(Orchestrator 亲自完成)
- 读取 kimi-slides 技能:pptd.md、cli.md、slides_categories.md + education-training.md ✅
- 确定场景:教育培训(主)+ 科技工程(辅,借图形语法)
- 产出:plan.md(本文件)、design.md、outline.md、deck 目录下 insentek-agent-training.pptd 主文件

### Stage 1 — 视觉设计与内容大纲(Orchestrator 亲自完成)
- `design.md`:学习任务定义 + 视觉简报(暖纸底、墨绿结构色、赭石强调色;无卡片、线框留白分层;网格/字号/组件规范)
- `outline.md`:28 页逐页大纲(章节/页型/学习动作/标题/主展品/支撑层/行动检查/来源/密度)
- `insentek-agent-training/insentek-agent-training.pptd`:主题色板、文字样式、页面清单

### Stage 2 — 页面制作(分派给 worker,coder 类型)
将 28 页 .page 制作按章节分给 4 个并行 worker,互不冲突:
- Worker_P1:页面 1–7(封面、路线图、第一章 5 页:为何是现在、演进时间线、智能体定义、谱系、Agent循环)
- Worker_P2:页面 8–14(第二、三章前半:智能体思维 4 页 + Harness 定义、Loop Engineering)
- Worker_P3:页面 15–21(工具设计、验证评估、案例演示、岗位结合 3 页)
- Worker_P4:页面 22–28(职能岗、机会地图、风险边界、30天行动计划、结尾)
每个 worker 必须先读:design.md、outline.md、insentek-agent-training.pptd、技能 pptd.md/shapes.md/fonts.md 相关部分,再写各自 pages/*.page 文件。

### Stage 3 — 校验与修复(Orchestrator)
- `kimi-slides check` 多轮修复 → `kimi-slides screenshot` 拼接总览 + 关键页单独审阅 → 修复问题页

### Stage 4 — 交付(Orchestrator)
- `kimi-slides package` 输出 insentek-agent-training.pptx 到工作区,附文件链接交付
