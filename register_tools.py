#!/usr/bin/env python3
"""在 Open WebUI 容器内运行，注册文档转换工具并给模型启用；同时删除已废弃的文生图工具。"""
import json
import sqlite3
import time
import uuid

DB = "/app/backend/data/webui.db"
ADMIN_ID = "2336e02f-77df-407d-857a-8b6c0154fc84"  # Siyuan Huang (admin)

DOC_TOOL_ID = "meerkat-doc-tools"
IMAGE_TOOL_ID = "meerkat-image-gen"  # 已废弃(FLUX 下线), 会被删除

TOOLS = [
    {
        "id": DOC_TOOL_ID,
        "name": "Markdown 文档转换",
        "content_file": "/tmp/openwebui_tool_doc.py",
        "description": "将 Markdown 文本转换为 HTML/Word/PDF/Excel/PPT 文件（业界级排版 + 数据图表）",
        "specs": [
            {
                "name": "convert_markdown_to_file",
                "description": "将 Markdown 文本转换成 HTML/Word/PDF/Excel/PPT 文件。支持在 Markdown 中用 ```chart 代码块输出 JSON 生成数据图表（type: bar/barh/line/area/pie/donut/radar/scatter/funnel/gauge），自动渲染为矢量图表或原生图表对象。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "format": {"type": "string", "description": "目标格式: html/docx/pdf/xlsx/pptx"},
                        "markdown_text": {"type": "string", "description": "要转换的 Markdown 文本内容（可留空，留空时自动使用对话历史里刚生成的内容）"},
                    },
                    "required": ["format"],
                },
            }
        ],
    },
]

db = sqlite3.connect(DB)
cur = db.cursor()
now = int(time.time())

# 0. 删除已废弃的文生图工具 (FLUX 已下线)
cur.execute("DELETE FROM tool WHERE id=?", (IMAGE_TOOL_ID,))
cur.execute("DELETE FROM access_grant WHERE resource_type='tool' AND resource_id=?", (IMAGE_TOOL_ID,))
print(f"[cleanup] 已删除废弃工具 {IMAGE_TOOL_ID}")

for t in TOOLS:
    with open(t["content_file"], encoding="utf-8") as f:
        content = f.read()
    meta = json.dumps({"description": t["description"], "manifest": {}})
    specs = json.dumps(t["specs"], ensure_ascii=False)
    valves = json.dumps({})

    cur.execute("SELECT id FROM tool WHERE id=?", (t["id"],))
    if cur.fetchone():
        cur.execute(
            "UPDATE tool SET user_id=?, name=?, content=?, specs=?, meta=?, valves=?, updated_at=? WHERE id=?",
            (ADMIN_ID, t["name"], content, specs, meta, valves, now, t["id"]),
        )
        print(f"[tool] {t['id']} 已更新")
    else:
        cur.execute(
            "INSERT INTO tool (id, user_id, name, content, specs, meta, valves, updated_at, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (t["id"], ADMIN_ID, t["name"], content, specs, meta, valves, now, now),
        )
        print(f"[tool] {t['id']} 已插入")

    cur.execute(
        "SELECT id FROM access_grant WHERE resource_type='tool' AND resource_id=? "
        "AND principal_type='user' AND principal_id='*' AND permission='read'",
        (t["id"],),
    )
    if not cur.fetchone():
        cur.execute(
            "INSERT INTO access_grant (id, resource_type, resource_id, principal_type, principal_id, permission, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), "tool", t["id"], "user", "*", "read", now),
        )
        print(f"[grant] {t['id']} -> 公开 read")

# 给所有 workspace 模型启用文档工具 (并移除已废弃的文生图工具)
tool_ids = [DOC_TOOL_ID]
cur.execute("SELECT id, meta FROM model")
for mid, meta_raw in cur.fetchall():
    meta = json.loads(meta_raw) if meta_raw else {}
    meta["toolIds"] = tool_ids
    cur.execute(
        "UPDATE model SET meta=?, updated_at=? WHERE id=?",
        (json.dumps(meta, ensure_ascii=False), now, mid),
    )
    print(f"[model] {mid}: toolIds = {tool_ids}")

db.commit()
db.close()
print("[done] 工具注册完成 (文生图已下线)")
