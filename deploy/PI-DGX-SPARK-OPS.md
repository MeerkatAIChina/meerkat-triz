# Pi (pi.dev) @ DGX Spark 运维说明

安装日期：2026-08-01。Pi coding agent v0.83.0 + pi-web-ui 扩展，接入本机 vLLM 的 Meerkat-TRIZ-v1。

## 组成

- Node.js v22.20.0（用户级）：`~/.local/node22/bin`
- Pi CLI：全局 npm 包 `@earendil-works/pi-coding-agent`（npm registry 已设为 npmmirror）
- 自定义 provider：`~/.pi/agent/models.json` → provider `meerkat`
  （baseUrl `http://127.0.0.1:8000/v1`，含 `Meerkat-TRIZ-v1` 和基座 `Qwen3.6-35B-A3B-NVFP4`）
- Web UI 扩展源码：`~/jupyterlab/meerkatai/deploy/pi-web-ui`（kkkiio/pi-web-ui，本地构建后注册）

## 访问

- pi Web UI：http://100.115.91.8:12002/ （Tailscale）或 http://192.168.5.246:12002/ （局域网）
- 与 Open WebUI（:12001）并存：12001 是聊天界面，12002 是带文件/bash 工具的 coding agent 界面

## 启动 / 重启

Web UI 依赖 tmux 里的 pi 会话（扩展随会话启动，随会话结束）：

```bash
tmux kill-session -t piweb 2>/dev/null
tmux new-session -d -s piweb -x 220 -y 50 \
  "cd ~/jupyterlab/meerkatai && PATH=\$HOME/.local/node22/bin:\$PATH \
   PI_WEB_UI_HOST=0.0.0.0 PI_WEB_UI_PORT=12002 \
   pi --provider meerkat --model Meerkat-TRIZ-v1 --api-key local"

# 查看会话 / 直接在里面操作
tmux attach -t piweb
```

前提：`meerkat-vllm` 容器在运行（见 MEERKAT-OPS.md）。主机重启后两个服务都需手动拉起。

## 中文化配置（2026-08-01）

- 全局指令 `~/.pi/agent/AGENTS.md`：强制简体中文回答、术语中英对照、文档默认中文、TRIZ 领域身份
- `settings.json` 默认模型：`defaultProvider: meerkat` + `defaultModel: Meerkat-TRIZ-v1`
  （启动命令可省略 `--provider/--model/--api-key` 参数，直接 `pi` 即可）
- 注意：pi TUI 与 pi-web-ui 的**界面文字**（按钮、菜单）无官方中文本地化，
  如需界面中文化须 fork 扩展源码修改

## 已知事项

1. 首次启动时 pi 尝试从 GitHub 下载 fd/ripgrep 失败（API 403 限流），文件搜索走内置 fallback，不影响对话；如需补全可手动放置二进制到 PATH。
2. `~/.tmux.conf` 已加 `set -g extended-keys on`（否则 TUI 里组合 Enter 键可能失效）。
3. pi 的 bash 工具在 `~/jupyterlab/meerkatai` 项目目录下拥有读写执行权限——这是 coding agent 的设计行为，注意安全边界。
4. 模型选择：Web UI 顶部可切换 `Meerkat-TRIZ-v1` / 基座对照；thinking level 默认 medium。
