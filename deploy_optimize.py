import sqlite3, json

MODEL_ID = 'Meerkat-TRIZ-v1-Qwen3.6-35B-A3B'
c = sqlite3.connect('/app/backend/data/webui.db')

data = json.load(open('/tmp/skills_optimize.json', encoding='utf-8'))
desc_map = data['desc']
core_ids = data['core_ids']

# 1. 更新 28 个 skill 的 description
n = 0
for sid, desc in desc_map.items():
    c.execute('UPDATE skill SET description=? WHERE id=?', (desc, sid))
    n += 1
print(f'更新 description: {n} 个')

# 2. 分层：model.skillIds 只保留高频核心 10 个
r = c.execute("SELECT meta FROM model WHERE id=?", (MODEL_ID,)).fetchone()
m = json.loads(r[0]) if r and r[0] else {}
m['skillIds'] = core_ids
c.execute("UPDATE model SET meta=? WHERE id=?", (json.dumps(m, ensure_ascii=False), MODEL_ID))

c.commit()
print(f'高频核心 skillIds（{len(core_ids)} 个）:')
for sid in core_ids:
    name = c.execute('SELECT name FROM skill WHERE id=?', (sid,)).fetchone()[0]
    print(f'  {sid} -> {name}')
