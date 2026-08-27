p = '/home/chinux/jupyterlab/meerkatai/tool_bridge.py'
s = open(p).read()

# 1. 全局变量：加空闲计时器
old1 = '_pipe_lock = threading.Lock()\n\n\ndef get_pipe():'
new1 = '_pipe_lock = threading.Lock()\n_idle_timer = None\n_idle_timer_lock = threading.Lock()\nFLUX_IDLE_TIMEOUT = 300  # 空闲 5 分钟自动卸载 FLUX\n\n\ndef get_pipe():'
assert old1 in s, '1 未找到'
s = s.replace(old1, new1, 1)

# 2. 加 _do_unload 和 _schedule_idle_unload 函数
old2 = '        return _pipe\n\n\nclass Handler(BaseHTTPRequestHandler):'
new2 = '''        return _pipe


def _do_unload():
    """释放 FLUX 显存（空闲计时器或 /unload 调用）。"""
    global _pipe
    with _pipe_lock:
        if _pipe is not None:
            del _pipe
            _pipe = None
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[bridge] FLUX 已自动卸载（空闲超时）", flush=True)


def _schedule_idle_unload():
    """每次图像生成后调用：重置空闲计时器，超时自动卸载。"""
    global _idle_timer
    with _idle_timer_lock:
        if _idle_timer is not None:
            _idle_timer.cancel()
        _idle_timer = threading.Timer(FLUX_IDLE_TIMEOUT, _do_unload)
        _idle_timer.daemon = True
        _idle_timer.start()


class Handler(BaseHTTPRequestHandler):'''
assert old2 in s, '2 未找到'
s = s.replace(old2, new2, 1)

# 3. _image 生成后重置空闲计时器
old3 = 'return self._send(200, buf.getvalue(), "image/png", "generated.png")'
new3 = '_schedule_idle_unload()  # 生成后重置空闲计时器\n        return self._send(200, buf.getvalue(), "image/png", "generated.png")'
assert old3 in s, '3 未找到'
s = s.replace(old3, new3, 1)

# 4. _unload 复用 _do_unload + 取消计时器
old4 = '''    def _unload(self):
        global _pipe
        with _pipe_lock:
            if _pipe is not None:
                del _pipe
                _pipe = None
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return self._send_json({"ok": True})'''
new4 = '''    def _unload(self):
        with _idle_timer_lock:
            if _idle_timer is not None:
                _idle_timer.cancel()
        _do_unload()
        return self._send_json({"ok": True})'''
assert old4 in s, '4 未找到'
s = s.replace(old4, new4, 1)

open(p, 'w').write(s)
print('修改成功')
