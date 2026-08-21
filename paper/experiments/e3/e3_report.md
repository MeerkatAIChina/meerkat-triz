# E3 ARIZ rubric 重判报告

- 题数 20, judge 裁决 60/60
## 模型均值 (rubric 覆盖 vs 关键词轨)
- base: rubric=0.675 kw=0.444 per_step={"step1": 0.95, "step2": 0.95, "step3": 0.8, "step4": 0.9, "step5": 0.45, "step6": 0.0}
- v2: rubric=0.875 kw=0.644 per_step={"step1": 0.95, "step2": 0.95, "step3": 0.9, "step4": 0.95, "step5": 0.9, "step6": 0.6}
- v4: rubric=0.883 kw=0.749 per_step={"step1": 1.0, "step2": 1.0, "step3": 1.0, "step4": 1.0, "step5": 1.0, "step6": 0.3}

## 漏判率
- kw<0.5 且 rubric>=0.5 (关键词低估): **19/60 = 0.317**
- kw>=0.5 且 rubric<0.5 (关键词高估): 1/60

## 配对 bootstrap 95%CI
- v4_vs_v2: rubric [0.0083, -0.0833, 0.125] | kw [0.1055, -0.0124, 0.2363]
- v4_vs_base: rubric [0.2083, 0.1333, 0.3] | kw [0.3057, 0.1612, 0.4515]
- v2_vs_base: rubric [0.2, 0.0667, 0.3333] | kw [0.2002, 0.08, 0.3198]

## 漏判表述清单 (19 条)
- v4_gold_002/base kw=0.00 rubric=0.83 期望词=['问题分析', '理想解IFR', '资源分析'] 证据={"step1": "Understand User Input: - Context: A Chinese company expanded production scale but realized that without independent core technology, further development is impossible.", "step2": "Construct Contradictions - Technical Contradiction: Improving one parameter worsens another.", "step3": "Ideal Final Result (IFR) Definition", "step4": "Resource Analysis", "step5": "Solution Generation & Evaluation (using TRIZ tools like 40 Principles, Separation Principles, etc.)"}
- v4_gold_005/base kw=0.33 rubric=0.67 期望词=['问题分析', '资源分析', '方案评估'] 证据={"step1": "Problem Context: Hollow bricks (hollow blocks) have cavities.", "step2": "Contradiction: Technical Contradiction: Improving anchor fixation (parameter: strength of connection/stability) worsens another parameter (e.g., ease of installation, cost, or causes damage to the bri", "step3": "IFR (Ideal Final Result): The anchor fixes itself perfectly without additional parts, tools, or energy, while maintaining the brick's integrity and being easy to install/remove.", "step4": "Resource Analysis: Internal (brick material, cavity shape, anchor material), External (drill, user force, environment), Systemic (tool-anchoring-brick interaction)."}
- v4_gold_005/v2 kw=0.33 rubric=1.00 期望词=['问题分析', '资源分析', '方案评估'] 证据={"step1": "面对的问题是锚固件在空心砖中的钻孔不能紧密固定", "step2": "根本原因是缺乏对冲击能量的控制", "step3": "理想解 IFR（理想最终结果），即实现一种钻孔设备，它能够在不破坏砖块的前提下，提供足够的穿透力，并能精确控制冲击能量", "step4": "识别已有的资源和潜在的解决方案，例如考虑使用可调节冲击能量的钻头、改进钻孔技术或使用新型材料来减少冲击", "step5": "评估这些潜在解决方案的可行性、成本效益和实施难度，选择最佳的解决方案进行实施", "step6": "最终可以形成一套有效的解决方案来解决锚固件不能紧密固定在空心砖中的问题"}
- v4_gold_008/base kw=0.00 rubric=0.67 期望词=['ARIZ算法', '燃料电池系统', '物理矛盾', '文氏管', 'Venturi effect'] 证据={"step1": "Problem Formulation (Ideal Final Result, Contradiction)", "step2": "Technical Contradiction: Parameter improving: Pressure (leads to higher current/power) Parameter worsening: Seal durability/reliability (or stress on seal)", "step3": "IFR: The seal should automatically adapt to high pressure without being damaged, while maintaining perfect gas tightness, without adding complex external systems.", "step4": "Resource Analysis"}
- v4_gold_008/v4 kw=0.40 rubric=1.00 期望词=['ARIZ算法', '燃料电池系统', '物理矛盾', '文氏管', 'Venturi effect'] 证据={"step1": "首先进行问题分析，该问题涉及燃料电池系统中的密封件保护和高电流产生两个相互矛盾的需求", "step2": "接着进行矛盾构建，确定物理矛盾为气体压力需要同时低和高", "step3": "理想解（IFR）是消除有害作用（气体损害密封件）的同时允许有用的功能（高电流产生）发生，而不增加系统的复杂性或产生任何有害后果", "step4": "通过资源分析，利用系统的Substance and Field Resources (SFR)，考虑可能的解决方案", "step5": "提出使用文氏管（Venturi effect）来实现低气体压力在密封区域和高气体压力在燃料电池区域的效果", "step6": "通过这种方法，即使密封件受损，也能在密封区域产生“文氏效应”，从而在不增加系统复杂性的情况下实现保护密封件和产生高电流的双重目的"}
- v4_gold_024/base kw=0.40 rubric=0.67 期望词=['问题分析', '资源分析', 'RCA+图', 'Altshuller矩阵', '发明原理'] 证据={"step1": "Step 1: 问题分析", "step2": "Step 2: 矛盾构建", "step4": "Resource Analysis", "step5": "Generate & Evaluate Solutions"}
- v4_gold_024/v2 kw=0.40 rubric=0.83 期望词=['问题分析', '资源分析', 'RCA+图', 'Altshuller矩阵', '发明原理'] 证据={"step1": "进行问题分析，识别出'悬崖'与布料之间的物理矛盾", "step2": "明确'悬崖'需要移动以完成功能，但又不能移动以避免卡住布料", "step3": "定义理想解IFR，即'悬崖'在不卡住布料的情况下能够移动", "step4": "识别可用的资源，如布料、'悬崖'、机械部件等", "step5": "评估可能的解决方案，如改变'悬崖'的设计，使其能够在不卡住布料的情况下移动，或者引入新的机制来辅助'悬崖'的移动"}
- v4_gold_024/v4 kw=0.40 rubric=1.00 期望词=['问题分析', '资源分析', 'RCA+图', 'Altshuller矩阵', '发明原理'] 证据={"step1": "问题分析阶段", "step2": "矛盾构建阶段", "step3": "理想解IFR阶段", "step4": "资源分析阶段", "step5": "方案评估阶段", "step6": "评估不同解决方案的可行性和效果，选择最佳方案实施。"}
- v4_gold_034/base kw=0.33 rubric=0.67 期望词=['垂直文化', '开放性问题', '定义问题', '解决方案', 'STC操作', '故事轮'] 证据={"step1": "问题分析", "step2": "矛盾构建", "step3": "理想解IFR", "step4": "资源分析"}
- v4_gold_034/v2 kw=0.33 rubric=0.67 期望词=['垂直文化', '开放性问题', '定义问题', '解决方案', 'STC操作', '故事轮'] 证据={"step1": "设计一个教育方案，旨在培养学生在遵循韩国垂直文化的同时，提高其定义和解决开放性问题的能力", "step2": "将TRIZ理论融入课程，教授学生如何识别和解决技术矛盾", "step4": "通过分析实际案例，让学生了解如何在尊重传统的同时，运用TRIZ工具解决复杂问题", "step5": "设计一些需要学生定义和解决开放性问题的实践项目，让他们在真实情境中应用TRIZ理论"}
- v4_gold_057/base kw=0.33 rubric=0.67 期望词=['ARIZ', '问题分析', '资源分析', '方案评估', 'TRIZ', '管理支持'] 证据={"step1": "问题分析", "step2": "构建矛盾", "step3": "定义理想解（IFR）", "step4": "进行资源分析"}
- v4_gold_057/v4 kw=0.33 rubric=0.83 期望词=['ARIZ', '问题分析', '资源分析', '方案评估', 'TRIZ', '管理支持'] 证据={"step1": "首先，进行问题分析，明确公司当前面临的挑战", "step2": "接着，构建矛盾，识别实施TRIZ的障碍", "step3": "然后，定义理想解（IFR）", "step4": "接下来，进行资源分析", "step5": "最后，进行方案评估"}
- v4_gold_063/base kw=0.43 rubric=0.67 期望词=['问题分析', '矛盾构建', '理想解IFR', '资源分析', '方案评估', '螺旋套筒', '挤出模具'] 证据={"step1": "问题分析", "step2": "矛盾构建", "step3": "提出理想解IFR", "step4": "资源分析"}
- v4_gold_075/base kw=0.43 rubric=0.83 期望词=['ARIZ 算法', '问题分析', '技术矛盾', '理想解 IFR', '资源分析', '方案评估', '生产线效率'] 证据={"step1": "Understand User Request", "step2": "Current Problems: Equipment aging, unreasonable layout, frequent breakdowns, production delays.", "step3": "*IFR Statement:* 生产线在无需额外增加设备投资的前提下，自动维持高可靠性与最优布局，故障率趋近于零，生产延误自动消除，且系统能自我优化。", "step4": "*Resource Analysis*", "step5": "*Step 5: Apply Invention Principles / Generate Solutions"}
- v4_gold_081/base kw=0.40 rubric=0.83 期望词=['问题分析', '矛盾构建', '资源分析', '方案评估', '分割'] 证据={"step1": "*Original Problem:* 高温下玻璃因重力下垂 → 表面凹凸不平 → 需大量打磨。", "step2": "*Technical Contradiction (TC):* 改善参数: 形状精度/表面质量 (TRIZ Parameter #11 或 #29 制造精度)", "step3": "*IFR (理想最终结果):* 玻璃在高温状态下自动保持完美平整表面，无需外部支撑或后续打磨，且生产线能耗/复杂度不增加。", "step4": "*资源分析 (Resources Mobilization)*", "step5": "*发明原理与效应应用 (Invention Principles & Effects)* - 匹配TRIZ 40个发明原理:"}
- v4_gold_081/v2 kw=0.00 rubric=0.83 期望词=['问题分析', '矛盾构建', '资源分析', '方案评估', '分割'] 证据={"step1": "首先定义问题：玻璃在高温下因重力下垂导致表面凹凸不平，需要后续打磨。", "step2": "识别技术矛盾：需要保持玻璃表面光滑，同时减少打磨工作量。", "step3": "将技术矛盾转化为物理矛盾：玻璃需要在高温下保持形状不变，同时减少后续处理。", "step4": "使用分离原理，提出解决方案：设计一种新型玻璃加工生产线，在玻璃成型过程中使用支撑结构或冷却系统，以减少玻璃因重力下垂。", "step5": "评估解决方案的可行性和效果，进行必要的调整和优化。"}
- v4_gold_093/base kw=0.17 rubric=0.67 期望词=['问题分析', '资源分析', '方案评估', '塑料板', '混凝土轨枕', '重复高强度冲击'] 证据={"step1": "*Step 1: Problem Formulation & IFR (Ideal Final Result)*", "step2": "*Contradiction:* Technical: To withstand impact, the plate needs high strength/durability, but high strength materials often lack elasticity/damping or are too stiff, causing other issues (e.g., rail ", "step3": "*IFR: The system performs its function (shock absorption, load distribution, insulation, protection) without the plastic plate itself degrading, aging, or requiring replacement.", "step4": "*Step 2: Resource Analysis*"}
- v4_gold_099/base kw=0.00 rubric=0.67 期望词=['产品生态扩展', '成本控制', 'ARIZ 算法', '资源分析', '方案评估'] 证据={"step1": "*System:* Mobile phone packaging (box, inserts, manuals, accessories, outer sleeve, eco-materials)", "step2": "*Technical Contradiction:* Improving one parameter worsens another.", "step3": "*Goal:* Design packaging that supports ecosystem expansion, controls costs, maintains brand premium, meets sustainability goals.", "step4": "*Resource Analysis* (Identify available resources: materials, energy, space, time, information, etc.)"}
- v4_gold_099/v2 kw=0.20 rubric=0.83 期望词=['产品生态扩展', '成本控制', 'ARIZ 算法', '资源分析', '方案评估'] 证据={"step1": "首先进行问题分析，识别出需要减少的组件数量、降低的运输成本以及提升的环保性。", "step2": "接着构建矛盾，即如何在减少组件的同时保持产品的保护性和用户体验。", "step3": "然后定义理想解IFR，即一个无需额外组件就能保护产品并提升用户体验的包装。", "step4": "资源分析阶段，识别出可以利用的资源，如手机本身的形状、材料特性等。", "step5": "评估可能的解决方案，如利用手机壳作为缓冲材料，或者设计一种可折叠的环保包装材料，既能保护产品又能减少运输成本。"}
---
# 干净 base 锚点 (base_goldfix) 补跑 — 上文 base rubric=0.675 (草稿分) 作废

## E3' rubric 轨三方 (干净锚点)
- base_goldfix: rubric=0.800 kw=0.605 per_step={"step1": 1.0, "step2": 1.0, "step3": 0.95, "step4": 1.0, "step5": 0.75, "step6": 0.1}
- v2: rubric=0.875 | v4: rubric=0.883 (不变)
- **v4 vs base_goldfix rubric: [0.0833, 0.0333, 0.1417]**
- **v2 vs base_goldfix rubric: [0.075, -0.025, 0.15]**
- base_goldfix 关键词漏判 (kw<0.5 且 rubric>=0.5): 3/20
