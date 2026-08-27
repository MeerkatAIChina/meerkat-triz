import sqlite3, json, time, uuid

USER_ID = '2336e02f-77df-407d-857a-8b6c0154fc84'
MODEL_ID = 'Meerkat-TRIZ-v1-Qwen3.6-35B-A3B'

c = sqlite3.connect('/app/backend/data/webui.db')
now = int(time.time())

skills = json.load(open('/tmp/skills_manifest.json', encoding='utf-8'))

inserted = 0
for s in skills:
    sid = s['id']
    # 1. skill 表
    c.execute('INSERT OR REPLACE INTO skill (id, user_id, name, description, content, meta, is_active, updated_at, created_at) VALUES (?,?,?,?,?,?,?,?,?)',
              (sid, USER_ID, s['name'], s['description'], s['content'], json.dumps({'tags': []}), 1, now, now))
    # 2. access_grant
    exists = c.execute("SELECT id FROM access_grant WHERE resource_type='skill' AND resource_id=?", (sid,)).fetchone()
    if not exists:
        c.execute('INSERT INTO access_grant (id, resource_type, resource_id, principal_type, principal_id, permission, created_at) VALUES (?,?,?,?,?,?,?)',
                  (str(uuid.uuid4()), 'skill', sid, 'user', '*', 'read', now))
    inserted += 1

# 3. 更新 model.meta.skillIds（追加新 19 个，保留已有 9 个）
r = c.execute("SELECT meta FROM model WHERE id=?", (MODEL_ID,)).fetchone()
m = json.loads(r[0]) if r and r[0] else {}
existing = set(m.get('skillIds', []))
for s in skills:
    existing.add(s['id'])
m['skillIds'] = sorted(existing)
c.execute("UPDATE model SET meta=? WHERE id=?", (json.dumps(m, ensure_ascii=False), MODEL_ID))

c.commit()
print(f'导入完成: {inserted} 个 skill')
print(f'model.skillIds 共 {len(m["skillIds"])} 个:')
for sid in m['skillIds']:
    print(f'  {sid}')
