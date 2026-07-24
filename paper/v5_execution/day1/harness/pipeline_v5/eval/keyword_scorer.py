#!/usr/bin/env python
"""
pipeline_v5 关键词轨: 子串匹配 + KEYWORD_MAP 别名表 (§6.2, §11.3)。

升级点 (相对 v4 纯子串):
  - 匹配 = 子串匹配(大小写不敏感) ∪ 别名表命中;
  - 别名表外置为可编辑文件 keyword_map_v5.json, 用 E3 漏判清单
    (paper/experiments/e3/e3_report.md, 19 条) 初始化;
  - 每条别名须 ≥2 个漏判案例或人工确认 (status=confirmed 才生效);
  - 每轮评测输出漏判审计队列 (rubric ≥0.5 而 kw <0.5 的题逐条提取);
  - 报告同时给"别名表更新前/后"双分数;
  - 漏判率 <5% 后冻结别名表; 历史数字不重写。
"""

import json

RUBRIC_PASS = 0.5
KW_PASS = 0.5
ALIAS_FREEZE_MISS_RATE = 0.05


def load_alias_map(path):
    """加载别名表。返回 {keyword: [alias, ...]}, 仅 status=confirmed 生效。"""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    amap = {}
    for kw, entry in raw.get("aliases", {}).items():
        if entry.get("status") == "confirmed":
            amap[kw] = list(entry.get("aliases", []))
    return amap


def keyword_score(response, keywords, alias_map=None):
    """单题关键词命中。

    alias_map: {keyword: [aliases]} 或 None (= 更新前口径, 纯子串)。
    返回 {"kw_hits", "kw_total", "kw_hit_rate", "hits", "misses",
          "alias_hits": {kw: 命中别名}}。
    """
    if not keywords:
        return {"kw_hits": 0, "kw_total": 0, "kw_hit_rate": None,
                "hits": [], "misses": [], "alias_hits": {}}
    low = response.lower()
    hits, misses, alias_hits = [], [], {}
    for k in keywords:
        if k.lower() in low:
            hits.append(k)
            continue
        matched = None
        if alias_map:
            for a in alias_map.get(k, []):
                if a.lower() in low:
                    matched = a
                    break
        if matched is not None:
            hits.append(k)
            alias_hits[k] = matched
        else:
            misses.append(k)
    return {"kw_hits": len(hits), "kw_total": len(keywords),
            "kw_hit_rate": len(hits) / len(keywords),
            "hits": hits, "misses": misses, "alias_hits": alias_hits}


def dual_score(records, alias_map=None):
    """别名表更新前/后双分数 (§6.2: 报告同时给双分数)。

    records: [{"id", "response", "keywords", ...}]
    返回 (pre_mean, post_mean, per_record_post)。
    """
    pre_rates, post_rates, per = [], [], {}
    for r in records:
        pre = keyword_score(r["response"], r.get("keywords", []), None)
        post = keyword_score(r["response"], r.get("keywords", []), alias_map)
        if pre["kw_hit_rate"] is not None:
            pre_rates.append(pre["kw_hit_rate"])
            post_rates.append(post["kw_hit_rate"])
        per[r["id"]] = {"pre": pre, "post": post}
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    return mean(pre_rates), mean(post_rates), per


def miss_audit(records, per_post, rubric_of, out_jsonl=None):
    """漏判审计队列: rubric ≥0.5 而 kw (别名表后) <0.5 的题逐条提取。

    rubric_of: callable(record) -> rubric 分 (0-1 归一) 或 None。
    返回 {"queue": [...], "miss_rate", "n", "freeze_recommended": bool}。
    queue 每条: id / subset / kw / rubric / 未命中期望词 / response 摘录,
    供人工确认后写入 keyword_map_v5.json。
    """
    queue, n_scored = [], 0
    for r in records:
        rub = rubric_of(r)
        kw = per_post[r["id"]]["post"]["kw_hit_rate"]
        if rub is None or kw is None:
            continue
        n_scored += 1
        if rub >= RUBRIC_PASS and kw < KW_PASS:
            queue.append({
                "id": r["id"], "subset": r.get("subset"),
                "kw_hit_rate": round(kw, 4), "rubric": rub,
                "missed_keywords": per_post[r["id"]]["post"]["misses"],
                "response_excerpt": r["response"][:600],
            })
    rate = len(queue) / n_scored if n_scored else None
    if out_jsonl and queue:
        with open(out_jsonl, "a", encoding="utf-8") as f:
            for q in queue:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
    return {"queue": queue, "n_queue": len(queue), "n_scored": n_scored,
            "miss_rate": rate,
            "freeze_recommended": (rate is not None
                                   and rate < ALIAS_FREEZE_MISS_RATE),
            "discipline": ("每条别名须 ≥2 个漏判案例或人工确认; 漏判率 "
                           f"<{ALIAS_FREEZE_MISS_RATE:.0%} 后冻结别名表; "
                           "历史数字不重写 (§6.2)")}
