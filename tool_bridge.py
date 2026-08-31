#!/usr/bin/env python3
"""
Meerkat 工具桥接服务：供 Open WebUI function calling 调用的宿主机 HTTP 服务。

端点:
  POST /health  健康检查
  POST /convert {fmt, md_text}  -> 返回 HTML/Word/PDF/Excel/PPT 文件

(文生图 FLUX 已于 2026-08 彻底下线, 相关代码与模型已移除)

运行:
  source venv_v5/bin/activate
  python3 tool_bridge.py     # 监听 0.0.0.0:8090
"""
import json
import os
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, "/home/chinux/jupyterlab/meerkatai")
from doc_tools import convert as doc_convert  # noqa: E402

BRIDGE_PORT = 8090

MIME = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pdf": "application/pdf",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "html": "text/html; charset=utf-8",
}
EXT = {"docx": ".docx", "pdf": ".pdf", "xlsx": ".xlsx", "pptx": ".pptx", "html": ".html"}


class Handler(BaseHTTPRequestHandler):
    def _read_json(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def _send(self, status, data, ctype, filename=None):
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass  # 客户端已断开, 忽略

    def _send_json(self, obj, status=200):
        self._send(status, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            if path == "/health":
                return self._send_json({"ok": True})
            if path == "/convert":
                return self._convert()
            return self._send_json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass  # 客户端断开
        except Exception as e:
            try:
                return self._send_json({"error": f"{type(e).__name__}: {e}"}, 500)
            except Exception:
                pass

    def _convert(self):
        body = self._read_json()
        fmt = str(body.get("fmt", "docx")).lower().lstrip(".")
        md = str(body.get("md_text", ""))
        if fmt not in MIME:
            return self._send_json({"error": f"unsupported fmt: {fmt}", "supported": list(MIME)}, 400)
        if not md.strip():
            return self._send_json({"error": "md_text is empty"}, 400)

        with tempfile.NamedTemporaryFile(suffix=EXT[fmt], delete=False) as f:
            out_path = f.name
        try:
            doc_convert(fmt, md, out_path)
            with open(out_path, "rb") as f:
                data = f.read()
            return self._send(200, data, MIME[fmt], f"output{EXT[fmt]}")
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass

    def log_message(self, *args):
        pass  # 静默


if __name__ == "__main__":
    # 0.0.0.0: 同时接受本机 + docker 网桥网关(host.docker.internal)访问
    print(f"[bridge] listening on 0.0.0.0:{BRIDGE_PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", BRIDGE_PORT), Handler).serve_forever()
