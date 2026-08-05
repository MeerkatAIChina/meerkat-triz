#!/usr/bin/env python
"""paper_zh.md → 学术排版 HTML（供 Chrome 打印 PDF）。"""
from pathlib import Path
from markdown_it import MarkdownIt

SRC = Path("/Volumes/2nd-HD/claude/Meerkat-AI/paper/arxiv/paper_zh.md")
OUT = Path("/Volumes/2nd-HD/claude/Meerkat-AI/paper/arxiv/paper_zh.html")

md = MarkdownIt("commonmark", {"html": True}).enable("table")
body = md.render(SRC.read_text(encoding="utf-8"))

CSS = """
@page { size: A4; margin: 22mm 20mm; }
* { box-sizing: border-box; }
body {
  font-family: "Songti SC", "Noto Serif CJK SC", "SimSun", "STSong", serif;
  font-size: 10.5pt; line-height: 1.78; color: #1a1a1a;
  max-width: 170mm; margin: 0 auto; padding: 10mm 0;
  -webkit-font-smoothing: antialiased;
}
h1 { font-size: 17pt; line-height: 1.4; text-align: center; margin: 0 0 6pt; }
h2 { font-size: 13pt; margin: 18pt 0 6pt; border-bottom: 1px solid #999; padding-bottom: 3pt; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; }
p { margin: 5pt 0; text-align: justify; }
strong { font-weight: 700; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9pt;
        font-family: "PingFang SC", "Noto Sans CJK SC", sans-serif; }
th, td { padding: 4pt 6pt; border-top: 1px solid #bbb; text-align: left; vertical-align: top; }
th { border-top: 1.5px solid #333; border-bottom: 1px solid #333; font-weight: 600; }
tr:last-child td { border-bottom: 1.5px solid #333; }
img { display: block; max-width: 88%; margin: 10pt auto 2pt; }
code { font-family: "SF Mono", Menlo, monospace; font-size: 9pt; background: #f3f3f0; padding: 0 3px; border-radius: 3px; }
hr { border: none; border-top: 1px solid #ccc; margin: 14pt 0; }
sup { font-size: 7.5pt; }
/* 首页头部（标题/作者/摘要）区域 */
h1 + p, h1 + p + p, h1 + p + p + p, h1 + p + p + p + p { text-align: center; }
/* 表格、图片跨页控制 */
table, img { break-inside: avoid; }
h2, h3 { break-after: avoid; }
/* 参考文献段悬挂缩进 */
ol, ul { padding-left: 18pt; }
li { margin: 2pt 0; }
"""

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<title>Meerkat-TRIZ 论文（中文版）</title>
<style>{CSS}</style></head><body>
{body}
</body></html>"""
OUT.write_text(html, encoding="utf-8")
print("HTML 已生成:", OUT, len(html), "bytes")
