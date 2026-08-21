#!/usr/bin/env python3
"""
Meerkat 工具桥接服务：供 Open WebUI function calling 调用的宿主机 HTTP 服务。

端点:
  POST /health  健康检查
  POST /convert {fmt, md_text}                 -> 返回 Word/PDF/Excel/PPT 文件
  POST /image   {prompt, steps?, width?, height?} -> 返回 PNG 图片
  POST /unload  释放 FLUX 显存

运行:
  source venv_v5/bin/activate
  python3 tool_bridge.py     # 监听 127.0.0.1:8090
"""
import gc
import io
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, "/home/chinux/jupyterlab/meerkatai")
from doc_tools import convert as doc_convert  # noqa: E402

BRIDGE_PORT = 8090
FLUX_PATH = "/home/chinux/jupyterlab/meerkatai/models/FLUX.1-schnell"

MIME = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pdf": "application/pdf",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}
EXT = {"docx": ".docx", "pdf": ".pdf", "xlsx": ".xlsx", "pptx": ".pptx"}

# FLUX 懒加载 + 常驻缓存（单进程）
_pipe = None
_pipe_lock = threading.Lock()


def get_pipe():
    global _pipe
    with _pipe_lock:
        if _pipe is None:
            import torch
            from diffusers import FluxPipeline
            print("[bridge] 加载 FLUX.1-schnell ...", flush=True)
            # device_map="cuda" 直接载入 GPU, 避免统一内存下 CPU+GPU 双副本
            _pipe = FluxPipeline.from_pretrained(
                FLUX_PATH,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="cuda",
            )
            print("[bridge] FLUX 就绪", flush=True)
        return _pipe


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
                return self._send_json({"ok": True, "flux_loaded": _pipe is not None})
            if path == "/convert":
                return self._convert()
            if path == "/image":
                return self._image()
            if path == "/unload":
                return self._unload()
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

    def _image(self):
        body = self._read_json()
        prompt = str(body.get("prompt", "")).strip()
        if not prompt:
            return self._send_json({"error": "prompt is empty"}, 400)
        steps = int(body.get("steps", 4))
        width = int(body.get("width", 768))
        height = int(body.get("height", 768))

        pipe = get_pipe()
        image = pipe(
            prompt,
            guidance_scale=0.0,
            num_inference_steps=steps,
            width=width,
            height=height,
            max_sequence_length=256,
        ).images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return self._send(200, buf.getvalue(), "image/png", "generated.png")

    def _unload(self):
        global _pipe
        with _pipe_lock:
            if _pipe is not None:
                del _pipe
                _pipe = None
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return self._send_json({"ok": True})

    def log_message(self, *args):
        pass  # 静默


if __name__ == "__main__":
    # 0.0.0.0: 同时接受本机 + docker 网桥网关(host.docker.internal)访问
    print(f"[bridge] listening on 0.0.0.0:{BRIDGE_PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", BRIDGE_PORT), Handler).serve_forever()
