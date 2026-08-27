import re, json, glob

# 分类前缀 + 简洁显示名映射
MAPPING = {
    # zip1 洞察层
    'interview-transcript-processor': ('【洞察】', '访谈笔录结构化'),
    'scenario-mining-benefit-extraction': ('【洞察】', '场景挖掘与利益点提炼'),
    'perceptible-experience-extractor': ('【洞察】', '可感知体验参数整理'),
    'differentiated-value-polishing': ('【洞察】', '差异化价值打磨'),
    # zip1 技术层
    'minimum-controllable-technology-point': ('【技术】', '最小可控技术点（MCTP）'),
    # zip1 概念层
    'product-concept-house-generator': ('【概念】', '产品概念屋生成'),
    # zip1 立项层
    'business-success-canvas-generator': ('【立项】', '商业成功画布'),
    'charter-writing-suite': ('【立项】', 'Charter 撰写套件'),
    'product-roadmap-excel-generator': ('【立项】', '产品路标 Excel 生成'),
    'industrial-design-brief-writer': ('【立项】', '工业设计 Brief 撰写'),
    # zip1 销售层
    'sales-pitch-sop': ('【销售】', '销售话术 SOP'),
    'sales-scenario-video-script': ('【销售】', '场景化视频脚本'),
    'selling-point-packaging': ('【销售】', '卖点包装方法论'),
    'minimum-conversion-action': ('【销售】', '最小转化动作'),
    # zip2 眼镜端
    'lawaken-memory': ('【眼镜端】', '记忆库查询'),
    'review-criteria-capture': ('【眼镜端】', '评审标准捕获'),
    'product-idea-refinery': ('【眼镜端】', '产品创意精炼'),
    'todo-task-executor': ('【眼镜端】', '待办任务执行'),
    'kb-question-resolver': ('【眼镜端】', '知识库问答沉淀'),
}

def parse_skill(fpath):
    txt = open(fpath, encoding='utf-8').read()
    m = re.match(r'^---\n(.*?)\n---\n', txt, re.DOTALL)
    fm = {}
    if m:
        for line in m.group(1).split('\n'):
            if ':' in line:
                k, v = line.split(':', 1)
                fm[k.strip()] = v.strip().strip("'\"")
    content = txt[m.end():].strip() if m else txt.strip()
    return fm, content

skills = []
for d in sorted(glob.glob('/tmp/skills_extract/zip1/skills/*/SKILL.md')) + \
         sorted(glob.glob('/tmp/skills_extract/zip2/meerkat-skills-lawaken/skills/*/SKILL.md')):
    fm, content = parse_skill(d)
    name = fm.get('name', '')
    if name not in MAPPING:
        print(f'警告: {name} 未在映射中')
        continue
    prefix, display = MAPPING[name]
    skills.append({
        'id': name,
        'name': prefix + display,
        'description': fm.get('description', ''),
        'content': content,
    })

print(f'解析完成: {len(skills)} 个 skill')
for s in skills:
    print(f"  {s['id']} -> {s['name']} (content {len(s['content'])} 字)")

with open('/tmp/skills_manifest.json', 'w', encoding='utf-8') as f:
    json.dump(skills, f, ensure_ascii=False)
print('\n清单已保存 /tmp/skills_manifest.json')
