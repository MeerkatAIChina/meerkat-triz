"""把 Open WebUI db 里的 skill 导出为 SKILL.md 格式（MAPID 可加载）。

用法：在 DGX 上 docker exec 或直接跑（需访问 webui.db）
输出：skills/<name>/SKILL.md
"""

import sqlite3
import json
import os
import re

DB = '/app/backend/data/webui.db'
OUT_DIR = '/tmp/skills_export'


def strip_prefix(name):
    """去掉【分类】前缀，恢复纯中文名。"""
    return re.sub(r'^【[^】]+】', '', name)


def export():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    skills = c.execute('SELECT * FROM skill WHERE is_active=1 ORDER BY name').fetchall()

    os.makedirs(OUT_DIR, exist_ok=True)
    exported = []

    for s in skills:
        sid = s['id']
        display_name = strip_prefix(s['name'])
        desc = s['description']
        content = s['content']

        # frontmatter
        fm = f"""---
name: {sid}
title: {display_name}
description: {desc}
type: skill
date: '2026-08-30'
---

"""
        skill_md = fm + content

        skill_dir = os.path.join(OUT_DIR, sid)
        os.makedirs(skill_dir, exist_ok=True)
        with open(os.path.join(skill_dir, 'SKILL.md'), 'w', encoding='utf-8') as f:
            f.write(skill_md)
        exported.append((sid, display_name))

    print(f'导出 {len(exported)} 个 skill 到 {OUT_DIR}')
    for sid, name in exported:
        print(f'  {sid} -> {name}')
    return exported


if __name__ == '__main__':
    export()
