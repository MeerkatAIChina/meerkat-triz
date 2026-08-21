# -*- coding: utf-8 -*-
"""W3 统计复核:eval2 + v4 gold 独立复算与五项新分析。全部数据来自 paper/data/ 下拉取的远端文件。"""
import json, math, collections
import numpy as np

D = "/Volumes/2nd-HD/claude/Meerkat-AI/paper/data/"
OUT = {}
BOOT_N, SEED = 10000, 42
Z = 1.959963984540054

def wilson_ci(k, n, z=Z):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (p, max(0.0, center-half), min(1.0, center+half))

def mcnemar_exact_p(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k+1)) / (2**n)
    return min(1.0, 2*tail)

def bootstrap_diff(a, b, n_boot=BOOT_N, seed=SEED):
    a = np.asarray(a, float); b = np.asarray(b, float); n = len(a)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_boot, n))
    diffs = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    return {"diff": float(b.mean()-a.mean()),
            "ci95": [float(np.percentile(diffs,2.5)), float(np.percentile(diffs,97.5))], "n": n}

def bootstrap_overall_diff(sa_list, sb_list, ws, n_boot=BOOT_N, seed=SEED):
    rng = np.random.RandomState(seed)
    total = np.zeros(n_boot); point = 0.0
    for sa, sb, w in zip(sa_list, sb_list, ws):
        a = np.asarray(sa, float); b = np.asarray(sb, float)
        if len(a)==0: continue
        idx = rng.randint(0, len(a), size=(n_boot, len(a)))
        total += w*(b[idx].mean(axis=1)-a[idx].mean(axis=1))
        point += w*(b.mean()-a.mean())
    return {"diff": float(point), "ci95":[float(np.percentile(total,2.5)), float(np.percentile(total,97.5))],
            "n": sum(len(s) for s in sa_list)}

def rankdata(x):
    x = np.asarray(x, float); order = x.argsort(); ranks = np.empty(len(x))
    sx = x[order]; i = 0
    while i < len(x):
        j = i
        while j+1 < len(x) and sx[j+1]==sx[i]: j += 1
        ranks[order[i:j+1]] = (i+j)/2.0 + 1; i = j+1
    return ranks

def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    r = float((rx*ry).sum()/math.sqrt((rx*rx).sum()*(ry*ry).sum()))
    n = len(x)
    # t 近似 p 值(大样本近似, t ~ N(0,1) 近似)
    if abs(r) >= 1: return r, 0.0
    t = r*math.sqrt((n-2)/(1-r*r))
    p = 2*(1 - 0.5*(1+math.erf(abs(t)/math.sqrt(2))))
    return r, p

# ============ Part A: eval2 复算 ============
scores = {}
for tag in ["base","v1","v2","v3"]:
    d = json.load(open(D+f"scores_{tag}.json"))
    scores[tag] = {it["id"]: it for it in d["items"]}

METRIC_DEFS = {
    "principle_accuracy": ("principle_recommendation","principle_correct"),
    "principle_coverage": ("principle_recommendation","principle_coverage"),
    "contradiction_coverage": ("contradiction_analysis","contradiction_coverage"),
    "case_coverage": ("case_generation","case_coverage"),
    "ariz_step_coverage": ("ariz_guidance","ariz_step_coverage"),
    "concept_coverage": ("concept_explanation","concept_coverage"),
    "general_probe_coverage": ("general_probe","probe_coverage"),
    "judge_contradiction_coverage": ("contradiction_analysis","judge_contradiction"),
    "judge_ariz_step_coverage": ("ariz_guidance","judge_ariz"),
}
W_KW = {"principle_accuracy":0.3,"contradiction_coverage":0.3,"case_coverage":0.2,"ariz_step_coverage":0.2}
W_JUDGE = {"principle_accuracy":0.3,"judge_contradiction_coverage":0.3,"case_coverage":0.2,"judge_ariz_step_coverage":0.2}

def metric_arrays(ta, tb, metric):
    cat, field = METRIC_DEFS[metric]
    ids = sorted(i for i,it in scores[ta].items()
                 if it.get("category")==cat and field in it
                 and i in scores[tb] and field in scores[tb][i])
    return [scores[ta][i][field] for i in ids], [scores[tb][i][field] for i in ids], ids

rep = json.load(open(D+"report_20260723_024941.json"))
A = {"model_metrics": {}, "max_dev_metric": 0.0, "pairs": {}, "max_dev_pair": 0.0}
for tag in ["base","v1","v2","v3"]:
    m = {}
    for metric,(cat,field) in METRIC_DEFS.items():
        vals = [it[field] for it in scores[tag].values() if it.get("category")==cat and field in it]
        if vals: m[metric] = float(np.mean(vals))
    for name, W in (("overall_kw",W_KW),("overall_judge",W_JUDGE)):
        m[name] = float(sum(w*m[k] for k,w in W.items()))
    A["model_metrics"][tag] = m
    for k,v in m.items():
        dev = abs(v - rep["models"][tag]["metrics"][k])
        A["max_dev_metric"] = max(A["max_dev_metric"], dev)

pairs = [("v1","base"),("v2","base"),("v3","base"),("v3","v2")]
for tb, ta in pairs:
    entry = {}
    for metric in METRIC_DEFS:
        a,b,_ = metric_arrays(ta,tb,metric)
        if not a: continue
        r = bootstrap_diff(a,b)
        entry[metric] = r
        ro = rep["pairs"][f"{tb}_vs_{ta}"]["metrics"][metric]
        dev = max(abs(r["diff"]-ro["diff"]), abs(r["ci95"][0]-ro["ci95"][0]), abs(r["ci95"][1]-ro["ci95"][1]))
        A["max_dev_pair"] = max(A["max_dev_pair"], dev)
    a,b,_ = metric_arrays(ta,tb,"principle_accuracy")
    bb = sum(1 for x,y in zip(a,b) if x==0 and y==1); cc = sum(1 for x,y in zip(a,b) if x==1 and y==0)
    entry["mcnemar"] = {"b":bb,"c":cc,"p":mcnemar_exact_p(bb,cc)}
    for name,W in (("overall_kw",W_KW),("overall_judge",W_JUDGE)):
        sa,sb,ws = [],[],[]
        for metric,w in W.items():
            x,y,_ = metric_arrays(ta,tb,metric)
            if x: sa.append(x); sb.append(y); ws.append(w)
        r = bootstrap_overall_diff(sa,sb,ws)
        entry[name] = r
        ro = rep["pairs"][f"{tb}_vs_{ta}"][name]
        dev = max(abs(r["diff"]-ro["diff"]), abs(r["ci95"][0]-ro["ci95"][0]), abs(r["ci95"][1]-ro["ci95"][1]))
        A["max_dev_pair"] = max(A["max_dev_pair"], dev)
    A["pairs"][f"{tb}_vs_{ta}"] = entry
# Wilson 复算
A["wilson"] = {}
for tag in ["base","v1","v2","v3"]:
    a,_,_ = metric_arrays(tag,tag,"principle_accuracy")
    k = int(sum(a)); n = len(a)
    p,lo,hi = wilson_ci(k,n)
    ro = rep["models"][tag]["metrics"]["principle_wilson"]
    A["wilson"][tag] = {"k":k,"n":n,"ci95":[lo,hi],"dev":max(abs(lo-ro["ci95"][0]),abs(hi-ro["ci95"][1]))}
OUT["A_eval2_replication"] = A

# ============ Part B: v4 gold 复算 ============
gold = {}
for tag,f in [("base","eval_v4_base_gold_20260723_105438.json"),("v2","eval_v4_v2_gold_20260723_124807.json"),
              ("v3","eval_v4_v3_gold_20260723_132023.json"),("v4","eval_v4_v4_gold_20260724_004355.json")]:
    d = json.load(open(D+f))
    gold[tag] = {r["id"]: r for r in d["records"]}
    OUT.setdefault("B_meta",{})[tag] = {"judge_model": d["meta"].get("judge_model"),
        "judge_probe": d["meta"].get("judge_probe"), "same_origin_fallback": d["meta"].get("judge_same_origin_fallback")}
IDS = sorted(gold["base"].keys())
SUBSETS = ["ariz_guidance","case_generation","concept_explanation","contradiction_analysis","innovation_assessment","principle_recommendation"]

def gvals(tag, track, ids=None, field=None):
    ids = ids or IDS
    if track=="kw": return [float(gold[tag][i]["kw_hit_rate"]) for i in ids]
    return [float(gold[tag][i]["judge_overall"]) for i in ids]

B = {"track_means": {}, "pass_rates": {}, "paired_v4_vs_base": {}}
for tag in ["base","v2","v3","v4"]:
    B["track_means"][tag] = {"kw_overall": float(np.mean(gvals(tag,"kw"))), "judge_overall": float(np.mean(gvals(tag,"judge")))}
    for s in SUBSETS:
        sids = [i for i in IDS if gold["base"][i]["subset"]==s]
        B["track_means"][tag][f"kw_{s}"] = float(np.mean(gvals(tag,"kw",sids)))
        B["track_means"][tag][f"judge_{s}"] = float(np.mean(gvals(tag,"judge",sids)))
    kw_pass = [1 if v>=0.5 else 0 for v in gvals(tag,"kw")]
    j_pass = [1 if v>=3 else 0 for v in gvals(tag,"judge")]
    B["pass_rates"][tag] = {"kw": wilson_ci(sum(kw_pass),100), "judge": wilson_ci(sum(j_pass),100)}
# 配对 v4 vs base(与原报告同法)
for track in ["kw","judge"]:
    a, b = gvals("base",track), gvals("v4",track)
    r = bootstrap_diff(a,b)
    pa = [1 if v>=(0.5 if track=="kw" else 3) else 0 for v in a]
    pb = [1 if v>=(0.5 if track=="kw" else 3) else 0 for v in b]
    bb = sum(1 for x,y in zip(pa,pb) if x==0 and y==1); cc = sum(1 for x,y in zip(pa,pb) if x==1 and y==0)
    B["paired_v4_vs_base"][track] = {"bootstrap": r, "mcnemar":{"b":bb,"c":cc,"p":mcnemar_exact_p(bb,cc)}}
    for s in SUBSETS:
        sids = [i for i in IDS if gold["base"][i]["subset"]==s]
        B["paired_v4_vs_base"][track][s] = bootstrap_diff(gvals("base",track,sids), gvals("v4",track,sids))
# 补充配对:v4 vs v2, v3 vs v2(原报告未给)
B["paired_extra"] = {}
for tb,ta in [("v4","v2"),("v3","v2"),("v4","v3")]:
    B["paired_extra"][f"{tb}_vs_{ta}"] = {t: bootstrap_diff(gvals(ta,t), gvals(tb,t)) for t in ["kw","judge"]}
OUT["B_v4gold_replication"] = B

# ============ C1: 关键词轨 vs judge 轨相关与分歧 ============
C1 = {"gold": {}, "eval2": {}}
for tag in ["base","v2","v3","v4"]:
    kw = gvals(tag,"kw"); jd = [v/4.0 for v in gvals(tag,"judge")]
    r,p = spearman(kw,jd)
    kwp = [1 if v>=0.5 else 0 for v in kw]; jp = [1 if v>=3 else 0 for v in gvals(tag,"judge")]
    both=sum(1 for x,y in zip(kwp,jp) if x==1 and y==1); kwonly=sum(1 for x,y in zip(kwp,jp) if x==1 and y==0)
    jdonly=sum(1 for x,y in zip(kwp,jp) if x==0 and y==1); neither=sum(1 for x,y in zip(kwp,jp) if x==0 and y==0)
    gaps = sorted(((abs(kw[i]-jd[i]), IDS[i], kw[i], jd[i]) for i in range(100)), reverse=True)[:10]
    C1["gold"][tag] = {"spearman_r": r, "p": p, "confusion":{"both_pass":both,"kw_only":kwonly,"judge_only":jdonly,"both_fail":neither},
                       "top_gap_items": [(i, round(k,3), round(j,3)) for _,i,k,j in gaps]}
# 全体 400 题(4 版本 x 100)pool
kw_all = [v for tag in ["base","v2","v3","v4"] for v in gvals(tag,"kw")]
jd_all = [v/4.0 for tag in ["base","v2","v3","v4"] for v in gvals(tag,"judge")]
r,p = spearman(kw_all, jd_all); C1["gold"]["pooled_400"] = {"spearman_r": r, "p": p}
# eval2: ariz & contradiction 双轨
for cat, fkw, fjd in [("ariz_guidance","ariz_step_coverage","judge_ariz"),("contradiction_analysis","contradiction_coverage","judge_contradiction")]:
    for tag in ["base","v1","v2","v3"]:
        ids = sorted(i for i,it in scores[tag].items() if it.get("category")==cat and fkw in it and fjd in it)
        kw = [scores[tag][i][fkw] for i in ids]; jd = [scores[tag][i][fjd] for i in ids]
        r,p = spearman(kw,jd)
        C1["eval2"][f"{cat}_{tag}"] = {"n":len(ids),"spearman_r":r,"p":p,"kw_mean":float(np.mean(kw)),"judge_mean":float(np.mean(jd))}
OUT["C1_kw_vs_judge"] = C1

# ============ C2: v4 concept_explanation 退化解剖 ============
ce_ids = [i for i in IDS if gold["base"][i]["subset"]=="concept_explanation"]
C2 = {"per_item": [], "keyword_autopsy": {}, "resp_len": {}}
miss_counter = collections.Counter(); hit_base_miss_v4 = collections.Counter()
for i in ce_ids:
    row = {"id": i}
    for tag in ["base","v2","v3","v4"]:
        row[f"kw_{tag}"] = float(gold[tag][i]["kw_hit_rate"])
        row[f"judge_{tag}"] = float(gold[tag][i]["judge_overall"])
    C2["per_item"].append(row)
    kws = gold["base"][i]["keywords"]
    resp = {tag: str(gold[tag][i]["response"]) for tag in ["base","v2","v3","v4"]}
    for k in kws:
        hit = {tag: (1 if k in resp[tag] else 0) for tag in ["base","v2","v3","v4"]}
        if hit["base"]==1 and hit["v4"]==0: hit_base_miss_v4[k] += 1
        if hit["v4"]==0: miss_counter[k] += 1
for tag in ["base","v2","v3","v4"]:
    lens = [len(str(gold[tag][i]["response"])) for i in ce_ids]
    C2["resp_len"][tag] = {"mean": float(np.mean(lens)), "median": float(np.median(lens)), "min": min(lens), "max": max(lens)}
C2["keyword_autopsy"]["base_hit_v4_miss"] = hit_base_miss_v4.most_common()
C2["keyword_autopsy"]["v4_miss_total"] = miss_counter.most_common()
# 复算 kw_hit_rate 是否可复现(独立校验)
recompute_ok, recompute_bad = 0, []
for i in IDS:
    for tag in ["base","v2","v3","v4"]:
        kws = gold[tag][i]["keywords"]; resp = str(gold[tag][i]["response"])
        hits = sum(1 for k in kws if k in resp)
        if hits != int(gold[tag][i]["kw_hits"]): recompute_bad.append((tag,i,hits,gold[tag][i]["kw_hits"]))
        else: recompute_ok += 1
C2["kw_recompute_check"] = {"ok": recompute_ok, "mismatch": recompute_bad[:20], "n_mismatch": len(recompute_bad)}
# 子集配对检验 v4 vs base / v4 vs v2 (concept_explanation)
C2["paired_ce"] = {
    "v4_vs_base_kw": bootstrap_diff(gvals("base","kw",ce_ids), gvals("v4","kw",ce_ids)),
    "v4_vs_v2_kw": bootstrap_diff(gvals("v2","kw",ce_ids), gvals("v4","kw",ce_ids)),
    "v4_vs_base_judge": bootstrap_diff(gvals("base","judge",ce_ids), gvals("v4","judge",ce_ids)),
    "v4_vs_v2_judge": bootstrap_diff(gvals("v2","judge",ce_ids), gvals("v4","judge",ce_ids)),
    "v2_vs_base_kw": bootstrap_diff(gvals("base","kw",ce_ids), gvals("v2","kw",ce_ids)),
}
OUT["C2_concept_explanation"] = C2

# ============ C3: ARIZ 表述 vs 能力 ============
C3 = {"eval2_ariz": {}, "gold_ariz": {}}
for tag in ["base","v1","v2","v3"]:
    ids = sorted(i for i,it in scores[tag].items() if it.get("category")=="ariz_guidance" and "ariz_step_coverage" in it and "judge_ariz" in it)
    kw = np.array([scores[tag][i]["ariz_step_coverage"] for i in ids])
    jd = np.array([scores[tag][i]["judge_ariz"] for i in ids])
    C3["eval2_ariz"][tag] = {"n":len(ids),"kw_mean":float(kw.mean()),"judge_mean":float(jd.mean()),
        "gap_judge_minus_kw": float(jd.mean()-kw.mean()),
        "kw_below_0.2_judge_above_0.5": int(((kw<0.2)&(jd>0.5)).sum()),
        "spearman": spearman(kw,jd)}
ar_ids = [i for i in IDS if gold["base"][i]["subset"]=="ariz_guidance"]
for tag in ["base","v2","v3","v4"]:
    kw = np.array(gvals(tag,"kw",ar_ids)); jd = np.array(gvals(tag,"judge",ar_ids))/4.0
    C3["gold_ariz"][tag] = {"kw_mean":float(kw.mean()),"judge_mean_norm":float(jd.mean()),
        "gap":float(jd.mean()-kw.mean()),"spearman":spearman(kw,jd)}
# eval2 v2 vs base 在 ariz:kw 显著负, judge 显著正 — 逐题方向一致性
ids2 = sorted(i for i in scores["base"] if scores["base"][i].get("category")=="ariz_guidance" and "ariz_step_coverage" in scores["base"][i] and "judge_ariz" in scores["base"][i] and i in scores["v2"] and "judge_ariz" in scores["v2"][i])
kw_d = [scores["v2"][i]["ariz_step_coverage"]-scores["base"][i]["ariz_step_coverage"] for i in ids2]
jd_d = [scores["v2"][i]["judge_ariz"]-scores["base"][i]["judge_ariz"] for i in ids2]
C3["eval2_v2_vs_base_ariz_dir"] = {
    "kw_down_judge_up": int(sum(1 for a,b in zip(kw_d,jd_d) if a<0 and b>0)),
    "kw_down_judge_down": int(sum(1 for a,b in zip(kw_d,jd_d) if a<0 and b<0)),
    "kw_up_judge_up": int(sum(1 for a,b in zip(kw_d,jd_d) if a>0 and b>0)),
    "kw_up_judge_down": int(sum(1 for a,b in zip(kw_d,jd_d) if a>0 and b<0)),
    "n": len(ids2)}
OUT["C3_ariz"] = C3

# ============ C4: 功效分析 ============
C4 = {}
def mde_paired(sd, n, power=0.8, alpha=0.05):
    return (1.959964 + 0.841621) * sd / math.sqrt(n)
# gold overall tracks
for track in ["kw","judge"]:
    a = np.array(gvals("base",track)); b = np.array(gvals("v4",track))
    sd = float(np.std(b-a, ddof=1))
    C4[f"gold_{track}_overall"] = {"sd_diff": sd, "MDE_n100": mde_paired(sd,100), "MDE_n102": mde_paired(sd,102)}
# gold 子集
C4["gold_subsets"] = {}
for s in SUBSETS:
    sids = [i for i in IDS if gold["base"][i]["subset"]==s]
    for track in ["kw","judge"]:
        a = np.array(gvals("base",track,sids)); b = np.array(gvals("v4",track,sids))
        sd = float(np.std(b-a, ddof=1))
        C4["gold_subsets"][f"{s}_{track}"] = {"n":len(sids),"sd_diff":sd,"MDE":mde_paired(sd,len(sids))}
# eval2 各指标
C4["eval2"] = {}
for metric in METRIC_DEFS:
    a,b,_ = metric_arrays("base","v2",metric)
    if not a: continue
    d = np.array(b)-np.array(a); sd = float(np.std(d,ddof=1))
    C4["eval2"][metric] = {"n":len(a),"sd_diff":sd,"MDE": mde_paired(sd,len(a)) if sd>0 else 0.0}
# McNemar 二值功效:以观察不一致对比例估计
def mde_mcnemar(p_disc, n):
    return (1.959964+0.841621)*math.sqrt(p_disc/n)
pa = [1 if v>=3 else 0 for v in gvals("base","judge")]; pb = [1 if v>=3 else 0 for v in gvals("v4","judge")]
p_disc = sum(1 for x,y in zip(pa,pb) if x!=y)/100
C4["gold_judge_pass_mcnemar"] = {"p_discordant": p_disc, "MDE_prop": mde_mcnemar(p_disc,100)}
pa = [1 if v>=0.5 else 0 for v in gvals("base","kw")]; pb = [1 if v>=0.5 else 0 for v in gvals("v4","kw")]
p_disc = sum(1 for x,y in zip(pa,pb) if x!=y)/100
C4["gold_kw_pass_mcnemar"] = {"p_discordant": p_disc, "MDE_prop": mde_mcnemar(p_disc,100)}
OUT["C4_power"] = C4

# ============ C5: judge 同源风险线索 ============
C5 = {"gold_gap_by_version": {}, "length_corr": {}, "eval2_gap": {}}
for tag in ["base","v2","v3","v4"]:
    kw = np.mean(gvals(tag,"kw")); jd = np.mean(gvals(tag,"judge"))/4.0
    C5["gold_gap_by_version"][tag] = {"kw":float(kw),"judge_norm":float(jd),"judge_premium":float(jd-kw)}
    lens = [len(str(gold[tag][i]["response"])) for i in IDS]
    r,p = spearman(lens, gvals(tag,"judge"))
    r2,p2 = spearman(lens, gvals(tag,"kw"))
    C5["length_corr"][tag] = {"judge_vs_len_r":r,"judge_vs_len_p":p,"kw_vs_len_r":r2,"kw_vs_len_p":p2,
                              "resp_len_mean": float(np.mean(lens))}
for tag in ["base","v1","v2","v3"]:
    a_kw,_,_ = metric_arrays(tag,tag,"ariz_step_coverage")
    a_jd,_,_ = metric_arrays(tag,tag,"judge_ariz_step_coverage")
    c_kw,_,_ = metric_arrays(tag,tag,"contradiction_coverage")
    c_jd,_,_ = metric_arrays(tag,tag,"judge_contradiction_coverage")
    C5["eval2_gap"][tag] = {"ariz_judge_minus_kw": float(np.mean(a_jd)-np.mean(a_kw)),
                            "contra_judge_minus_kw": float(np.mean(c_jd)-np.mean(c_kw))}
OUT["C5_same_origin"] = C5

json.dump(OUT, open(D+"stats_results.json","w"), ensure_ascii=False, indent=1, default=str)
print("DONE. max_dev_metric=%.2e max_dev_pair=%.2e" % (A["max_dev_metric"], A["max_dev_pair"]))
print("Wilson devs:", {k: round(v["dev"],10) for k,v in A["wilson"].items()})
print("B paired v4_vs_base:", json.dumps({t: B["paired_v4_vs_base"][t]["bootstrap"] for t in ["kw","judge"]}, indent=1))
print("B mcnemar judge:", B["paired_v4_vs_base"]["judge"]["mcnemar"], " kw:", B["paired_v4_vs_base"]["kw"]["mcnemar"])
print("kw recompute mismatches:", C2["kw_recompute_check"]["n_mismatch"], "/", recompute_ok+len(recompute_bad))
