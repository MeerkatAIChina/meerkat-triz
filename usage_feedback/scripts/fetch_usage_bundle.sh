#!/usr/bin/env bash
# usage_feedback 采集器：从 DGX Spark 拉取 Open WebUI 对话+评分 与 pi 会话轨迹
# 在本机（workspace 所在 Mac）运行：bash usage_feedback/scripts/fetch_usage_bundle.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$ROOT/state/last_harvest.json"
DATE="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/raw/$DATE"
mkdir -p "$OUT" "$ROOT/state"

SINCE_EPOCH=0
if [[ -f "$STATE" ]]; then
  SINCE_EPOCH=$(python3 -c "import json; print(json.load(open('$STATE')).get('last_epoch', 0))")
fi
echo "[harvest] watermark: $SINCE_EPOCH, output: $OUT"

# ---------- 1. Open WebUI: chats + feedback ----------
ssh spark-855a 'docker exec meerkat-webui python3 -c "
import sqlite3, json, sys
db = sqlite3.connect(\"/app/backend/data/webui.db\")
db.row_factory = sqlite3.Row
chats = [dict(r) for r in db.execute(\"SELECT id, user_id, title, chat, created_at, updated_at FROM chat\")]
fbs = []
try:
    fbs = [dict(r) for r in db.execute(\"SELECT * FROM feedback\")]
except Exception as e:
    print(\"feedback table:\", e, file=sys.stderr)
print(len(chats), \"chats,\", len(fbs), \"feedback\", file=sys.stderr)
print(json.dumps({\"chats\": chats, \"feedback\": fbs}, ensure_ascii=False, default=str))
"' > "$OUT/owui_dump.json" 2>"$OUT/owui_dump.stderr" || true
head -c 200 "$OUT/owui_dump.stderr" | grep -v "^$" || true
python3 -c "import json; d=json.load(open('$OUT/owui_dump.json')); print('[harvest] open-webui:', len(d['chats']), 'chats,', len(d['feedback']), 'feedback')"

# ---------- 2. pi sessions（增量：mtime 新于水位线） ----------
# pi sessions 在 ~/.pi/agent/sessions/ 下按项目分目录
BUNDLE="pi_sessions_$DATE.tar.gz"
ssh spark-855a "cd ~/.pi/agent/sessions 2>/dev/null && find . -name '*.jsonl' -newermt \"@\${1:-0}\" 2>/dev/null | tar -czf /tmp/$BUNDLE -T - 2>/dev/null; ls -la /tmp/$BUNDLE 2>/dev/null || echo empty" _ "$SINCE_EPOCH" | tail -1
if ssh spark-855a "test -f /tmp/$BUNDLE"; then
  scp -q "spark-855a:/tmp/$BUNDLE" "$OUT/$BUNDLE"
  ssh spark-855a "rm -f /tmp/$BUNDLE"
  mkdir -p "$OUT/pi_sessions" && tar -xzf "$OUT/$BUNDLE" -C "$OUT/pi_sessions" 2>/dev/null || true
  N=$(find "$OUT/pi_sessions" -name '*.jsonl' | wc -l | tr -d ' ')
  echo "[harvest] pi sessions: $N files -> $OUT/pi_sessions/"
else
  echo "[harvest] no new pi sessions"
fi

# ---------- 3. 更新水位线 ----------
NOW=$(date +%s)
python3 -c "
import json, time
json.dump({'last_epoch': $NOW, 'last_run': '$DATE'}, open('$STATE', 'w'), indent=2)
"
echo "[harvest] done. watermark -> $NOW"
echo "OUT_DIR=$OUT"
