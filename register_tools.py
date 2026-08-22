#!/usr/bin/env python3
"""在 Open WebUI 容器内运行，把两个工具注册进 webui.db 并给模型启用。"""
import json
import sqlite3
import time
import uuid

DB = "/app/backend/data/webui.db"
ADMIN_ID = "2336e02f-77df-407d-857a-8b6c0154fc84"  # Siyuan Huang (admin)

DOC_TOOL_ID = "meerkat-doc-tools"
IMAGE_TOOL_ID = "meerkat-image-gen"

TOOLS = [
    {
        "id": DOC_TOOL_ID,
        "name": "Markdown 文档转换",
        "content_file": "/tmp/openwebui_tool_doc.py",
        "description": "将 Markdown 文本转换为 Word/PDF/Excel/PPT 文件",
        "specs": [
            {
                "name": "convert_markdown_to_file",
                "description": "将 Markdown 文本转换成 Word/PDF/Excel/PPT 文件",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "format": {"type": "string", "description": "目标格式: docx/pdf/xlsx/pptx"},
                        "markdown_text": {"type": "string", "description": "要转换的 Markdown 文本内容（可留空，留空时自动使用对话历史里刚生成的内容）"},
                    },
                    "required": ["format"],
                },
            }
        ],
    },
    {
        "id": IMAGE_TOOL_ID,
        "name": "文生图 (FLUX.1-dev)",
        "content_file": "/tmp/openwebui_tool_image.py",
        "description": "根据文字描述生成图片（本地 FLUX.1-dev，支持标题+图例中文标注）",
        "specs": [
            {
                "name": "generate_image",
                "description": "根据文字描述生成一张图片；生成结构图/原理图时用 title（顶部标题）和 legend（底部图例）添加中文标注，不要要求模型在图中直接渲染中文",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "图片描述，中文或英文（结构图/工程图建议英文，构图更准）"},
                        "title": {"type": "string", "description": "可选，图片顶部居中的中文标题，如「齿轮传动系统结构示意图」"},
                        "legend": {
                            "type": "array",
                            "description": "可选，底部图例的中文文字列表，自动编号，如 ['主动轮','从动轮','传动轴']",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["prompt"],
                },
            }
        ],
    },
]

db = sqlite3.connect(DB)
cur = db.cursor()
now = int(time.time())

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

# 给所有 workspace 模型启用这两个工具
tool_ids = [DOC_TOOL_ID, IMAGE_TOOL_ID]
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
print("[done] 工具注册完成")
