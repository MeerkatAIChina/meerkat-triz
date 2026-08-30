#!/usr/bin/env python3
"""
清理 OpenWebUI 数据库里的坏 function_call（arguments 是非法/截断的 JSON）。

背景: 模型调用工具时若 arguments 超长被 max_tokens 截断, 会留下 arguments 未闭合的
function_call 存进 chat_message 表。之后每次新请求, OpenWebUI 都会把这条坏 tool_call
发给 vLLM, vLLM 解析历史时 json.loads 失败 -> 会话被锁死 (每次请求都报
"Unterminated string")。

本脚本扫描 chat_message 表的 output(JSON 字符串), 删除坏 function_call 及其对应的
function_call_output, 并把 done=0 的 error 消息标记为 done=1, 解除锁死。

运行方式 (在 OpenWebUI 容器内):
  docker exec meerkat-webui python3 /tmp/fix_bad_toolcalls.py

可传 chat_id 只清理指定会话, 不传则全库扫描:
  docker exec meerkat-webui python3 /tmp/fix_bad_toolcalls.py <chat_id>
"""
import json
import sqlite3
import sys


def clean_chat_message_table(db, chat_id=None):
    """清理 chat_message 表 (OpenWebUI 新版真实数据源, output 是 JSON 字符串)。"""
    cur = db.cursor()
    if chat_id:
        cur.execute("SELECT id, chat_id, output FROM chat_message WHERE chat_id=?", (chat_id,))
    else:
        cur.execute("SELECT id, chat_id, output FROM chat_message")

    fixed_msgs = 0
    fixed_calls = 0
    for mid, cid, output in cur.fetchall():
        if not output or output == "null":
            continue
        try:
            out = json.loads(output)
        except Exception:
            continue
        if not isinstance(out, list):
            continue
        bad_ids = set()
        for x in out:
            if isinstance(x, dict) and x.get("type") == "function_call":
                args = x.get("arguments") or ""
                if isinstance(args, str) and args.strip():
                    try:
                        json.loads(args)
                    except Exception:
                        bad_ids.add(x.get("call_id") or x.get("id"))
        if bad_ids:
            new_out = [
                x for x in out
                if not (
                    isinstance(x, dict)
                    and (x.get("call_id") or x.get("id")) in bad_ids
                    and x.get("type") in ("function_call", "function_call_output")
                )
            ]
            db.execute(
                "UPDATE chat_message SET output=? WHERE id=?",
                (json.dumps(new_out, ensure_ascii=False), mid),
            )
            fixed_msgs += 1
            fixed_calls += len(bad_ids)

    # 解除 done=0 的 error 消息锁 (fork/继续 409 的根因)
    if chat_id:
        cur.execute("SELECT id FROM chat_message WHERE chat_id=? AND done=0", (chat_id,))
    else:
        cur.execute("SELECT id FROM chat_message WHERE done=0")
    err_rows = cur.fetchall()
    for (mid,) in err_rows:
        db.execute("UPDATE chat_message SET done=1, error=NULL WHERE id=?", (mid,))
    return fixed_msgs, fixed_calls, len(err_rows)


def clean_chat_table(db, chat_id=None):
    """同步清理 chat 表的 history.messages (旧副本, 前端可能读)。"""
    cur = db.cursor()
    if chat_id:
        cur.execute("SELECT id, chat FROM chat WHERE id=?", (chat_id,))
    else:
        cur.execute("SELECT id, chat FROM chat")
    fixed = 0
    for cid, chat_raw in cur.fetchall():
        try:
            chat = json.loads(chat_raw) if chat_raw else {}
        except Exception:
            continue
        msgs = chat.get("history", {}).get("messages", {})
        changed = False
        for mid, m in msgs.items():
            if m.get("role") == "assistant" and m.get("done") is False:
                m["done"] = True
                m["error"] = None
                changed = True
            out = m.get("output")
            if isinstance(out, list):
                bad_ids = set()
                for x in out:
                    if isinstance(x, dict) and x.get("type") == "function_call":
                        args = x.get("arguments") or ""
                        if isinstance(args, str) and args.strip():
                            try:
                                json.loads(args)
                            except Exception:
                                bad_ids.add(x.get("call_id") or x.get("id"))
                if bad_ids:
                    m["output"] = [
                        x for x in out
                        if not (
                            isinstance(x, dict)
                            and (x.get("call_id") or x.get("id")) in bad_ids
                            and x.get("type") in ("function_call", "function_call_output")
                        )
                    ]
                    changed = True
        if changed:
            db.execute("UPDATE chat SET chat=? WHERE id=?", (json.dumps(chat, ensure_ascii=False), cid))
            fixed += 1
    return fixed


def main():
    chat_id = sys.argv[1] if len(sys.argv) > 1 else None
    db = sqlite3.connect("/app/backend/data/webui.db")

    msgs, calls, errs = clean_chat_message_table(db, chat_id)
    chat_fixed = clean_chat_table(db, chat_id)
    db.commit()
    db.close()

    scope = f"会话 {chat_id[:12]}" if chat_id else "全库"
    print(f"[fix_bad_toolcalls] {scope}: 清理 {msgs} 条消息/{calls} 个坏调用, 解除 {errs} 个锁死消息, chat表同步 {chat_fixed} 条")


if __name__ == "__main__":
    main()
