#!/usr/bin/env python3
"""测试生产 bridge 的 /convert 接口 (html + 图表)。"""
import requests

md = """# 测试报告

## 销售数据

```chart
{"type": "bar", "title": "月度销量", "categories": ["1月","2月","3月"], "series": [{"name": "销量", "data": [120, 150, 180]}]}
```
"""

for fmt in ["html", "pdf", "xlsx", "pptx", "docx"]:
    r = requests.post("http://127.0.0.1:8090/convert",
                      json={"fmt": fmt, "md_text": md}, timeout=60)
    ct = r.headers.get("Content-Type", "")
    extra = ""
    if fmt == "html":
        extra = f" 含SVG={'是' if b'<svg' in r.content else '否'} 含封面={'是' if b'cover' in r.content else '否'}"
    print(f"{fmt:5s} HTTP {r.status_code} 大小 {len(r.content):7d} {ct}{extra}")
