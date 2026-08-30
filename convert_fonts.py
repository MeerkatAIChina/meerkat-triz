#!/usr/bin/env python3
"""把 Noto CJK (CFF/PostScript) 字体转成 TrueType 轮廓, 供 reportlab 使用。一次性操作。"""
import os
import sys
import time

from fontTools.ttLib import TTFont, newTable
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen


def otf2ttf(src, dst, max_err=1.0):
    t0 = time.time()
    font = TTFont(src)
    glyph_order = font.getGlyphOrder()
    glyph_set = font.getGlyphSet()
    quad = {}
    max_pts = max_contours = max_cpts = max_ccontours = 0
    for gname in glyph_order:
        glyph = glyph_set[gname]
        ttPen = TTGlyphPen(glyph_set)
        cu2quPen = Cu2QuPen(ttPen, max_err, reverse_direction=True)
        glyph.draw(cu2quPen)
        g = ttPen.glyph()
        quad[gname] = g
        if g.isComposite():
            for comp in g.components:
                cpts = len(getattr(comp, "coordinates", ()) or ())
                max_cpts = max(max_cpts, cpts)
            max_ccontours = max(max_ccontours, len(getattr(g, "endPtsOfContours", ()) or ()))
        else:
            max_pts = max(max_pts, len(g.coordinates))
            max_contours = max(max_contours, len(g.endPtsOfContours))
    glyf = newTable("glyf")
    glyf.glyphOrder = glyph_order
    glyf.glyphs = quad
    font["glyf"] = glyf
    # 关键: loca 表作为空容器, save 时 glyf.compile() 会自动填充 offsets
    font["loca"] = newTable("loca")
    for tag in ("CFF ", "CFF2", "VORG"):
        if tag in font:
            del font[tag]
    maxp = newTable("maxp")
    maxp.tableVersion = 0x00010000
    maxp.numGlyphs = len(glyph_order)
    maxp.maxPoints = max_pts
    maxp.maxContours = max_contours
    maxp.maxCompositePoints = max_cpts
    maxp.maxCompositeContours = max_ccontours
    maxp.maxZones = 1
    maxp.maxTwilightPoints = 0
    maxp.maxStorage = 0
    maxp.maxFunctionDefs = 0
    maxp.maxInstructionDefs = 0
    maxp.maxStackElements = 0
    maxp.maxSizeOfInstructions = 0
    maxp.maxComponentElements = 0
    maxp.maxComponentDepth = 0
    font["maxp"] = maxp
    # 关键: 把 sfntVersion 从 "OTTO" 改为 TrueType, 否则 reportlab 误判为 PostScript 字体
    font.sfntVersion = "\x00\x01\x00\x00"
    font.save(dst)
    print(f"[{time.time()-t0:.1f}s] {os.path.basename(src)} -> {os.path.basename(dst)} ({len(glyph_order)} glyphs)", flush=True)


if __name__ == "__main__":
    outdir = "/home/chinux/jupyterlab/meerkatai/fonts_ttf"
    os.makedirs(outdir, exist_ok=True)
    src_dir = "/home/chinux/jupyterlab/meerkatai/fonts"
    for name in ["NotoSansCJKsc-Regular", "NotoSansCJKsc-Bold",
                 "NotoSerifCJKsc-Regular", "NotoSerifCJKsc-Bold"]:
        src = os.path.join(src_dir, name + ".ttf")
        dst = os.path.join(outdir, name + ".ttf")
        if os.path.exists(dst) and os.path.getsize(dst) > 1000000:
            print(f"[skip] {name} 已存在", flush=True)
            continue
        otf2ttf(src, dst)
    print("ALL DONE", flush=True)
