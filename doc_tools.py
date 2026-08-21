#!/usr/bin/env python3
"""
Meerkat-TRIZ 文档工具链: Markdown → Word / PDF / Excel / PPT
纯 Python 实现, 不占 GPU。供 LLM function calling 调用。

用法 (命令行):
  python3 doc_tools.py md2docx <input.md> <output.docx>
  python3 doc_tools.py md2pdf  <input.md> <output.pdf>
  python3 doc_tools.py md2xlsx <input.md> <output.xlsx>
  python3 doc_tools.py md2pptx <input.md> <output.pptx>

用法 (函数调用, 供 LLM):
  from doc_tools import convert
  convert("docx", markdown_text, "/path/out.docx")
"""

import base64
import io
import re
import sys


# ============ Markdown 解析 ============

def _parse_blocks(md_text):
    """把 markdown 解析成结构化块: (type, payload)。"""
    blocks = []
    lines = md_text.split("\n")
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip()
        stripped = line.strip()

        # 标题
        m = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if m:
            blocks.append(("heading", (len(m.group(1)), m.group(2).strip())))
            i += 1
            continue

        # 图片 (Markdown 内联图: ![alt](data:image/png;base64,XXX))
        m = re.match(r"!\[([^\]]*)\]\(data:image/([a-zA-Z]+);base64,([A-Za-z0-9+/=]+)\)", stripped)
        if m:
            blocks.append(("image", (m.group(1), m.group(2), m.group(3))))
            i += 1
            continue

        # 表格
        if stripped.startswith("|"):
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            blocks.append(("table", rows))
            continue

        # 列表
        if re.match(r"^\s*([-*+]|\d+\.)\s+", stripped):
            items = []
            while i < n and re.match(r"^\s*([-*+]|\d+\.)\s+", lines[i].strip()):
                items.append(re.sub(r"^\s*([-*+]|\d+\.)\s+", "", lines[i].strip()))
                i += 1
            blocks.append(("list", items))
            continue

        # 空行
        if not stripped:
            i += 1
            continue

        # 段落
        para = []
        while i < n and lines[i].strip() and not re.match(
                r"^(#{1,6}\s|\||\s*[-*+]\s|\s*\d+\.\s|!\[)", lines[i]):
            para.append(lines[i].strip())
            i += 1
        blocks.append(("para", " ".join(para)))

    return blocks


def _parse_table_rows(rows):
    """解析 markdown 表格行为 header + data 行 (跳过 |---| 分隔行)。"""
    header, data = None, []
    for r in rows:
        cells = [c.strip() for c in r.strip("|").split("|")]
        if all(re.match(r"^:?-{2,}:?$", c) for c in cells):
            continue  # 分隔行
        if header is None:
            header = cells
        else:
            data.append(cells)
    return header, data


# ============ Word ============

def md_to_docx(md_text, output_path):
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    for typ, payload in _parse_blocks(md_text):
        if typ == "heading":
            level, text = payload
            doc.add_heading(text, level=min(level, 4))
        elif typ == "para":
            doc.add_paragraph(payload)
        elif typ == "image":
            from docx.shared import Inches
            _alt, _fmt, b64 = payload
            try:
                doc.add_picture(io.BytesIO(base64.b64decode(b64)), width=Inches(5.5))
            except Exception:
                doc.add_paragraph(_alt or "[图片]")
        elif typ == "list":
            for item in payload:
                doc.add_paragraph(item, style="List Bullet")
        elif typ == "table":
            header, data = _parse_table_rows(payload)
            rows = ([header] if header else []) + data
            if rows:
                table = doc.add_table(rows=len(rows), cols=max(len(r) for r in rows))
                for ri, row in enumerate(rows):
                    for ci in range(len(table.columns)):
                        table.cell(ri, ci).text = row[ci] if ci < len(row) else ""
                # 表头加粗
                if header:
                    for cell in table.rows[0].cells:
                        for p in cell.paragraphs:
                            for run in p.runs:
                                run.bold = True
    doc.save(output_path)
    return output_path


# ============ PDF ============

def md_to_pdf(md_text, output_path):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    # 注册内置简体中文字体 (STSong-Light), 否则中文会乱码
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        _FONT = "STSong-Light"
    except Exception:
        _FONT = "Helvetica"

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    # 所有文本样式改用中文字体
    for sname in ("Heading1", "Heading2", "Heading3", "Heading4", "BodyText", "Title"):
        if sname in styles:
            styles[sname].fontName = _FONT
    story = []

    for typ, payload in _parse_blocks(md_text):
        if typ == "heading":
            level, text = payload
            style = styles["Heading1"] if level <= 2 else styles["Heading2"]
            story.append(Paragraph(_escape(text), style))
        elif typ == "para":
            story.append(Paragraph(_escape(payload), styles["BodyText"]))
        elif typ == "image":
            from reportlab.platypus import Image as RLImage
            _alt, _fmt, b64 = payload
            try:
                img_data = base64.b64decode(b64)
                img = RLImage(io.BytesIO(img_data), width=14 * cm, height=10 * cm, kind="proportional")
                story.append(img)
            except Exception:
                story.append(Paragraph(_escape(_alt or "[图片]"), styles["BodyText"]))
        elif typ == "list":
            for item in payload:
                story.append(Paragraph("• " + _escape(item), styles["BodyText"]))
        elif typ == "table":
            header, data = _parse_table_rows(payload)
            rows = ([header] if header else []) + data
            if rows:
                t = Table([[Paragraph(_escape(c), styles["BodyText"]) for c in r]
                           for r in rows])
                t.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey) if header else
                    ("BACKGROUND", (0, 0), (-1, 0), colors.white),
                ]))
                story.append(t)
        story.append(Spacer(1, 6))

    doc.build(story)
    return output_path


def _escape(text):
    """转义 reportlab 的 XML 特殊字符。"""
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# ============ Excel ============

def md_to_xlsx(md_text, output_path):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"

    row_idx = 1
    for typ, payload in _parse_blocks(md_text):
        if typ == "table":
            header, data = _parse_table_rows(payload)
            rows = ([header] if header else []) + data
            for r in rows:
                for ci, cell in enumerate(r, start=1):
                    ws.cell(row=row_idx, column=ci, value=cell)
                row_idx += 1
            row_idx += 1  # 表格之间空一行
        elif typ == "heading":
            level, text = payload
            ws.cell(row=row_idx, column=1, value=text)
            row_idx += 1
        elif typ == "image":
            from openpyxl.drawing.image import Image as XLImage
            _alt, _fmt, b64 = payload
            try:
                img = XLImage(io.BytesIO(base64.b64decode(b64)))
                img.width = 360
                img.height = 360
                ws.add_image(img, f"A{row_idx}")
                row_idx += 22  # 预留图片占用的行
            except Exception:
                ws.cell(row=row_idx, column=1, value=_alt or "[图片]")
                row_idx += 1

    wb.save(output_path)
    return output_path


# ============ PPT ============

def md_to_pptx(md_text, output_path):
    from pptx import Presentation
    from pptx.util import Inches, Pt

    prs = Presentation()
    title_slide = False

    for typ, payload in _parse_blocks(md_text):
        if typ == "heading":
            level, text = payload
            if level <= 1 and not title_slide:
                # 第一个 H1 用标题版式
                slide = prs.slides.add_slide(prs.slide_layouts[0])
                slide.shapes.title.text = text
                title_slide = True
            else:
                # 其余标题用"标题+内容"版式
                slide = prs.slides.add_slide(prs.slide_layouts[1])
                slide.shapes.title.text = text
                # 预留给后续段落/列表填到内容占位符
                _pending_heading = text
        elif typ == "image":
            _alt, _fmt, b64 = payload
            try:
                slide = prs.slides.add_slide(prs.slide_layouts[6])  # 空白版式
                slide.shapes.add_picture(
                    io.BytesIO(base64.b64decode(b64)), Inches(0.8), Inches(0.8), width=Inches(8.5)
                )
            except Exception:
                pass
        elif typ == "para" and not title_slide:
            slide = prs.slides.add_slide(prs.slide_layouts[1])
            slide.shapes.title.text = "内容"
            body = slide.placeholders[1]
            body.text = payload
        elif typ == "list" and not title_slide:
            slide = prs.slides.add_slide(prs.slide_layouts[1])
            slide.shapes.title.text = "要点"
            body = slide.placeholders[1]
            tf = body.text_frame
            tf.text = payload[0] if payload else ""
            for item in payload[1:]:
                p = tf.add_paragraph()
                p.text = item
                p.level = 1

    if len(prs.slides._sldIdLst) == 0:
        slide = prs.slides.add_slide(prs.slide_layouts[0])
        slide.shapes.title.text = "文档"

    prs.save(output_path)
    return output_path


# ============ 统一入口 ============

_FORMATS = {"docx": md_to_docx, "pdf": md_to_pdf, "xlsx": md_to_xlsx, "pptx": md_to_pptx}


def convert(fmt, md_text, output_path):
    """统一转换入口: convert('docx'|'pdf'|'xlsx'|'pptx', markdown_text, output_path)。"""
    fmt = fmt.lower().lstrip(".")
    if fmt not in _FORMATS:
        raise ValueError(f"不支持的格式: {fmt}, 支持: {list(_FORMATS)}")
    return _FORMATS[fmt](md_text, output_path)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python3 doc_tools.py <md2docx|md2pdf|md2xlsx|md2pptx> <input.md> <output>")
        sys.exit(1)
    cmd, inp, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    fmt = cmd.replace("md2", "")
    with open(inp, encoding="utf-8") as f:
        md = f.read()
    result = convert(fmt, md, outp)
    print(f"[done] {fmt} → {result}")
