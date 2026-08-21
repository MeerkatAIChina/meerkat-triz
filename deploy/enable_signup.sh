#!/bin/bash
# 开启 Open WebUI 自助注册: DB 配置 + 容器环境变量双保险
set -e

echo "== 1. 更新 DB 配置 =="
docker exec meerkat-webui python3 - <<'PYEOF'
import sqlite3, json
c = sqlite3.connect("/app/backend/data/webui.db")
row = c.execute("SELECT data FROM config WHERE id=1").fetchone()
d = json.loads(row[0])
ui = d.setdefault("ui", {})
ui["enable_signup"] = True
ui["default_user_role"] = "user"
c.execute("UPDATE config SET data=? WHERE id=1", (json.dumps(d),))
c.commit()
print("DB 已更新: enable_signup=True, default_user_role=user")
PYEOF

echo "== 2. 重建容器(保留卷/端口/原有环境) =="
docker stop meerkat-webui
docker rm meerkat-webui
docker run -d --name meerkat-webui \
  --restart unless-stopped \
  -p 12001:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v meerkat-webui-data:/app/backend/data \
  -e ENABLE_OLLAMA_API=False \
  -e WEBUI_NAME=Meerkat-TRIZ \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=local \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e HF_HUB_ENDPOINT=https://hf-mirror.com \
  -e ENABLE_SIGNUP=true \
  -e DEFAULT_USER_ROLE=user \
  -e ENABLE_LOGIN_FORM=true \
  ghcr.io/open-webui/open-webui:ollama

echo "== 3. 等待就绪 =="
for i in $(seq 1 30); do
  if curl -sf -o /dev/null http://localhost:12001/; then echo "WebUI 就绪"; break; fi
  sleep 2
done

echo "== 4. 验证注册开关 =="
docker exec meerkat-webui python3 - <<'PYEOF'
import sqlite3, json
c = sqlite3.connect("/app/backend/data/webui.db")
d = json.loads(c.execute("SELECT data FROM config WHERE id=1").fetchone()[0])
ui = d.get("ui", {})
print("enable_signup:", ui.get("enable_signup"))
print("default_user_role:", ui.get("default_user_role"))
PYEOF
echo "完成"
