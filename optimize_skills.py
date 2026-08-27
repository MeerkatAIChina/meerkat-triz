import json

# 28 个 skill 的精简 description + 分层标记
# core=True 表示高频核心（常驻 model.skillIds），False 表示低频进阶（仅 @提及）
OPTIMIZE = {
    # ===== TRIZ 高频核心 =====
    'ariz-solver': ('core', '引导用户走 ARIZ-85C 九步算法解决复杂发明问题。当用户要系统化求解技术难题时用。'),
    'contradiction-analyst': ('core', '识别技术矛盾/物理矛盾，用矛盾矩阵和分离原理求解。当用户描述工程矛盾冲突时用。'),
    'principle-advisor': ('core', '根据工程问题推荐 40 发明原理和矛盾矩阵解。当用户要创新点子或原理启发时用。'),
    'sufield-analysis': ('core', '用物场模型（S1/S2/F）和 76 个标准解分析技术系统。当用户要分析系统相互作用时用。'),
    # ===== TRIZ 低频进阶 =====
    'ifr-expert': ('adv', '定义理想最终结果（IFR）和理想度，引导方案向理想解收敛。当用户追求最优解时用。'),
    'technology-evolution': ('adv', '用 S 曲线和技术进化法则预测技术发展方向。当用户问技术趋势或下一代产品时用。'),
    'innovation-assessment': ('adv', '评估创新方案的理想度、资源利用和可行性。当用户要评判方案优劣时用。'),
    'case-generator': ('adv', '生成 TRIZ 应用案例用于教学和启发。当用户要学习案例或找参考时用。'),
    'concept-explanation': ('adv', '深入解释 TRIZ 概念、原理和术语。当用户问某个 TRIZ 名词的含义时用。'),
    # ===== 新品流程 高频核心 =====
    'interview-transcript-processor': ('core', '把原始访谈笔录加工成结构化洞察和用户心智地图。当用户贴访谈记录要整理时用。'),
    'product-concept-house-generator': ('core', '多角色 8 步流程生成产品概念屋。当用户要产生产品概念时用。'),
    'charter-writing-suite': ('core', '完整 Charter 产品规划撰写（从市场洞察到落地）。当用户要立项文档时用。'),
    'sales-pitch-sop': ('core', '按四步法把卖点转成销售流程 SOP。当用户要销售话术或流程时用。'),
    'selling-point-packaging': ('core', '四步卖点包装流程（发现→分层→包装→验证）。当用户要包装卖点时用。'),
    # ===== 新品流程 低频进阶 =====
    'scenario-mining-benefit-extraction': ('adv', '从品类价值拆解痛点场景，再把痛点转化为产品利益点。当用户要挖掘用户需求时用。'),
    'perceptible-experience-extractor': ('adv', '从访谈笔录挖掘深层真相，输出可感知体验参数表。当用户要提炼体验指标时用。'),
    'differentiated-value-polishing': ('adv', '把零散灵感转化为结构化产品战略数据并校验因果链。当用户要打磨差异化价值时用。'),
    'minimum-controllable-technology-point': ('adv', '结合 TRIZ 识别最小可控技术点（MCTP）规划技术突破。当用户要做技术规划时用。'),
    'business-success-canvas-generator': ('adv', '用 5×3 矩阵生成商业成功画布。当用户要做商业规划时用。'),
    'product-roadmap-excel-generator': ('adv', '把需求/产品路标转成结构化 Excel 规划。当用户要路标规划表格时用。'),
    'industrial-design-brief-writer': ('adv', '撰写工业设计 Brief（课题/项目任务书）。当用户要设计任务书时用。'),
    'sales-scenario-video-script': ('adv', '把卖点转成场景化视频拍摄脚本。当用户要视频脚本时用。'),
    'minimum-conversion-action': ('adv', '把销售流程拆成最小转化动作。当用户要提升转化时用。'),
    # ===== 眼镜端 高频核心 =====
    'lawaken-memory': ('core', '查询 lawaken 记忆库（AI 总结/原文片段/记忆包）。当用户提到查眼镜记忆或录音时用。'),
    # ===== 眼镜端 低频进阶 =====
    'review-criteria-capture': ('adv', '从文字提炼评审标准入档。当用户要沉淀评审标准时用。'),
    'product-idea-refinery': ('adv', '创意提取→评审打分→打磨→三件套存档。当用户要整理产品创意时用。'),
    'todo-task-executor': ('adv', '提取待办→转任务→按风险分档执行。当用户要执行待办时用。'),
    'kb-question-resolver': ('adv', '查知识库作答并沉淀问答卡片。当用户要查知识库问答时用。'),
}

core_ids = [sid for sid, (lvl, _) in OPTIMIZE.items() if lvl == 'core']
adv_ids = [sid for sid, (lvl, _) in OPTIMIZE.items() if lvl == 'adv']

print(f'高频核心: {len(core_ids)} 个')
print(f'低频进阶: {len(adv_ids)} 个')
print(f'总计: {len(OPTIMIZE)} 个')

with open('/tmp/skills_optimize.json', 'w', encoding='utf-8') as f:
    json.dump({'desc': {sid: d for sid, (_, d) in OPTIMIZE.items()},
               'core_ids': core_ids}, f, ensure_ascii=False)
print('已保存 /tmp/skills_optimize.json')
