import re

# ============ 1. doc_tools.py: 加 import html ============
p = '/home/chinux/jupyterlab/meerkatai/doc_tools.py'
s = open(p).read()

old_import = 'import base64\nimport io\nimport re\nimport sys'
new_import = 'import base64\nimport html\nimport io\nimport re\nimport sys'
assert old_import in s, 'doc_tools import 未找到'
s = s.replace(old_import, new_import, 1)

# ============ 2. 插入 md_to_html 函数 ============
html_func = '''

# ============ HTML ============

_HTML_CSS = """
    <style>
      :root { --text:#1f2328; --muted:#57606a; --border:#d0d7de; --bg:#f6f8fa; --accent:#0969da; }
      * { box-sizing: border-box; }
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; max-width: 900px; margin: 0 auto; padding: 40px 28px; line-height: 1.75; color: var(--text); font-size: 16px; background: #fff; }
      h1 { font-size: 2em; font-weight: 700; margin: 0 0 .6em; padding-bottom: .35em; border-bottom: 2px solid var(--border); }
      h2 { font-size: 1.5em; font-weight: 650; margin: 1.6em 0 .6em; padding-bottom: .25em; border-bottom: 1px solid var(--border); }
      h3 { font-size: 1.25em; font-weight: 600; margin: 1.3em 0 .5em; }
      h4 { font-size: 1.1em; font-weight: 600; margin: 1.1em 0 .4em; }
      h5, h6 { font-size: 1em; font-weight: 600; margin: 1em 0 .3em; }
      p { margin: .7em 0; }
      strong { font-weight: 650; }
      ul, ol { padding-left: 1.8em; margin: .6em 0; }
      li { margin: .25em 0; }
      table { border-collapse: collapse; width: 100%; margin: 1.1em 0; font-size: .95em; }
      th, td { border: 1px solid var(--border); padding: 9px 14px; text-align: left; }
      th { background: var(--bg); font-weight: 650; white-space: nowrap; }
      tr:nth-child(even) td { background: #fafbfc; }
      img { max-width: 100%; height: auto; border-radius: 6px; margin: .6em 0; }
      code { font-family: "SF Mono", Consolas, "Liberation Mono", monospace; background: var(--bg); padding: .2em .45em; border-radius: 4px; font-size: .88em; }
      blockquote { border-left: 4px solid var(--border); margin: 1em 0; padding: .4em 1.2em; color: var(--muted); background: var(--bg); border-radius: 0 6px 6px 0; }
      hr { border: none; border-top: 1px solid var(--border); margin: 1.6em 0; }
      a { color: var(--accent); text-decoration: none; }
      a:hover { text-decoration: underline; }
      @media print { body { max-width: none; padding: 0; } }
    </style>
"""


def _inline_html(text):
    """行内格式: `code` -> <code>, **bold** -> <strong>, *italic* -> <em>。"""
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\\1</code>", text)
    text = re.sub(r"\\*\\*([^*]+)\\*\\*", r"<strong>\\1</strong>", text)
    text = re.sub(r"(?<!\\*)\\*([^*]+)\\*(?!\\*)", r"<em>\\1</em>", text)
    return text


def md_to_html(md_text, output_path):
    """Markdown → 自包含 HTML（内联 CSS，GitHub 风格 + 中文优化）。"""
    parts = []
    for typ, payload in _parse_blocks(md_text):
        if typ == "heading":
            level, text = payload
            level = min(level, 6)
            parts.append(f'<h{level}>{_inline_html(text)}</h{level}>')
        elif typ == "para":
            parts.append(f'<p>{_inline_html(payload)}</p>')
        elif typ == "image":
            alt, fmt, b64 = payload
            parts.append(f'<img src="data:image/{fmt};base64,{b64}" alt="{html.escape(alt)}" />')
        elif typ == "list":
            items = "".join(f'<li>{_inline_html(item)}</li>' for item in payload)
            parts.append(f'<ul>{items}</ul>')
        elif typ == "table":
            header, data = _parse_table_rows(payload)
            if not header and not data:
                continue
            thead = ''
            if header:
                thead = '<thead><tr>' + ''.join(f'<th>{_inline_html(c)}</th>' for c in header) + '</tr></thead>'
            tbody_rows = []
            for row in data:
                tbody_rows.append('<tr>' + ''.join(f'<td>{_inline_html(c)}</td>' for c in row) + '</tr>')
            parts.append(f'<table>{thead}<tbody>{"".join(tbody_rows)}</tbody></table>')
    body = "\\n".join(parts)
    doc = (f'<!DOCTYPE html>\\n<html lang="zh-CN">\\n<head>\\n<meta charset="utf-8">\\n'
           f'<meta name="viewport" content="width=device-width, initial-scale=1">\\n'
           f'<title>Meerkat 文档</title>\\n{_HTML_CSS}\\n</head>\\n'
           f'<body>\\n{body}\\n</body>\\n</html>\\n')
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return output_path

'''

old_marker = '# ============ 统一入口 ============\n\n_FORMATS = {"docx": md_to_docx, "pdf": md_to_pdf, "xlsx": md_to_xlsx, "pptx": md_to_pptx}'
new_marker = html_func + '\n# ============ 统一入口 ============\n\n_FORMATS = {"docx": md_to_docx, "pdf": md_to_pdf, "xlsx": md_to_xlsx, "pptx": md_to_pptx, "html": md_to_html}'
assert old_marker in s, 'doc_tools 统一入口未找到'
s = s.replace(old_marker, new_marker, 1)

open(p, 'w').write(s)
print('doc_tools.py 修改成功')

# ============ 3. tool_bridge.py: MIME 加 html ============
p2 = '/home/chinux/jupyterlab/meerkatai/tool_bridge.py'
s2 = open(p2).read()

old_mime = '"pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",'
new_mime = '"pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",\n    "html": "text/html",'
assert old_mime in s2, 'tool_bridge MIME 未找到'
s2 = s2.replace(old_mime, new_mime, 1)

old_ext = 'EXT = {"docx": ".docx", "pdf": ".pdf", "xlsx": ".xlsx", "pptx": ".pptx"}'
new_ext = 'EXT = {"docx": ".docx", "pdf": ".pdf", "xlsx": ".xlsx", "pptx": ".pptx", "html": ".html"}'
assert old_ext in s2, 'tool_bridge EXT 未找到'
s2 = s2.replace(old_ext, new_ext, 1)

open(p2, 'w').write(s2)
print('tool_bridge.py 修改成功')
