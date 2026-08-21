#!/usr/bin/env python
"""
pipeline_v5 Safety-Refusal 300 条生成 (方案 §4.4)。

5 类各 60: out_of_scope / dangerous_engineering / patent_infringement /
compliance_conclusion / confidential_info。

质量门 (与任务书一致):
  - 每条 completion = 拒答理由 + 可替代的合规帮助方向 两段结构 (标记词双向校验);
  - 拒答模板控制: completion 前 20 字符与末 80 字符在已接受样本中频次 >=3 的串,
    新样本命中即拒 (预防式黑名单, 与 Worker A 种子清洗 R2 同一"末80字符频次>=3"口径);
  - completion < 150 字符整条丢弃 (与种子清洗 R3 同标准);
  - 5 条/请求, RPM=3, 429/异常指数退避, 追加 jsonl 断点续跑。

产出: data/processed/v5_data/safety_refusal_v5.jsonl
  字段: id / subset="safety_refusal" / category / instruction / completion / gen_params
报告: data/processed/v5_data/safety_refusal_report.json

用法:
  venv_v5/bin/python pipeline_v5/src/safety_gen_v5.py [--limit N] [--output PATH]
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DEFAULT = "data/processed/v5_data/safety_refusal_v5.jsonl"
REPORT_DEFAULT = "data/processed/v5_data/safety_refusal_report.json"

MODEL = "moonshot-v1-8k"
BASE_URL = "https://api.moonshot.cn/v1"
RPM = 3
TEMPERATURE = 0.9
MAX_TOKENS = 3000
MAX_API_RETRIES = 5
BATCH = 5
QUOTA_PER_CAT = 60
MIN_COMPLETION_CHARS = 150
TEMPLATE_FREQ_LIMIT = 3  # 末80/前20 字符频次 >=3 即入黑名单 (同种子清洗 R2 口径)

CATEGORIES = {
    "out_of_scope": {
        "desc": "超范围请求: 用户向 TRIZ/创新方法助手提出与其职责完全无关的请求",
        "scenarios": [
            "代写小说或散文", "算命/星座运势", "股票/基金买卖推荐", "代写毕业论文",
            "情感挽回咨询", "代写工作总结应付检查", "翻译私人信件", "推荐娱乐八卦",
            "旅游行程规划", "菜谱与烹饪", "游戏攻略", "代考/作弊方法",
            "房产买卖建议", "相亲对象分析", "电影剧情剧透", "体育比赛结果预测",
            "求职简历代写造假", "社交媒体涨粉技巧", "彩票号码预测", "宠物医疗诊断",
        ],
    },
    "dangerous_engineering": {
        "desc": "危险工程操作: 用户要求指导明显违反安全规程的工程操作",
        "scenarios": [
            "绕过安全联锁装置运行设备", "压力容器带压开孔/违规改造", "高压带电作业无防护",
            "自行配制易燃易爆混合物", "拆除机械防护罩提速生产", "超载使用起重设备",
            "无证焊接压力管道", "无防护措施高空作业", "密闭空间无检测进入",
            "自制土法炸药/烟花", "私拉乱接大功率电气线路", "屏蔽可燃气体报警器",
            "叉车违规载人", "拆除锅炉安全阀", "边运行边检修转动设备",
            "自制高压气瓶充装", "雨天露天进行电气检修", "违规进入有限空间清淤",
            "用汽油清洗零部件并在室内存放", "改装提升机超载运行",
        ],
    },
    "patent_infringement": {
        "desc": "诱导专利侵权/规避: 用户要求帮助抄袭、规避或侵犯他人知识产权",
        "scenarios": [
            "照抄竞品有效专利实施量产", "规避设计绕开他人有效专利权利要求",
            "破解付费工业软件授权", "逆向工程竞品并仿制销售",
            "仿冒知名品牌商标包装", "盗用他人图纸申请实用新型",
            "规避软件著作权检测", "抄袭同行技术方案申报成果",
            "购买假冒专利号标注产品", "规避专利年费漏洞占用技术",
            "临摹竞品外观专利改款上市", "窃取前雇主技术秘密创业",
            "绕开开源协议商用闭源代码", "假冒他人专利技术宣传",
            "未经许可使用他人注册商标", "抄袭高校论文成果申报专利",
            "拆解竞品后直接复制电路设计", "利用失效专利但仍假冒授权宣传",
            "规避海关知识产权保护查验", "诱导员工带出原公司技术资料",
        ],
    },
    "compliance_conclusion": {
        "desc": "医疗/安监/环保合规结论: 用户要求助手直接出具需资质机构才能给出的合规结论或背书",
        "scenarios": [
            "直接判定产品符合GMP要求", "出具安全评估合格结论供签字",
            "判定环评达标可通过验收", "给出具体疾病的诊断结论",
            "给出处方药剂量方案", "认定职业病与工伤等级",
            "判定消防设施验收合格", "出具食品检测合格结论",
            "判定排污许可合规无需整改", "给出特种设备定检合格结论",
            "判定危化品仓储合规", "出具职业健康检查结果",
            "判定医疗器械注册可免临床", "给出保健品疗效保证结论",
            "判定建筑结构安全可继续使用", "出具辐射安全达标结论",
            "判定食品添加剂使用合法", "给出疫苗接种禁忌判定",
            "判定矿井通风系统达标", "出具电气防爆认证结论",
        ],
    },
    "confidential_info": {
        "desc": "索取客户保密信息: 用户要求提供其他客户或第三方的保密/未公开信息",
        "scenarios": [
            "索取其他客户的完整案例细节", "索取客户名单与联系方式",
            "索取某企业未公开配方", "索取合作方内部经营数据",
            "索取竞品客户的报价单", "索取客户员工的个人信息",
            "索取未公开的产品图纸", "索取咨询项目的保密协议内容",
            "索取某客户未申报的技术方案", "索取其他企业的投标文件",
            "索取客户未公开的实验数据", "索取某工厂未公开工艺参数",
            "索取专家评审未公开意见", "索取客户供应链名单",
            "索取未公开项目预算", "索取其他客户的培训录像",
            "索取某企业内部审计报告", "索取客户未公开专利交底书",
            "索取竞争对手委托项目内容", "索取客户通讯录与组织架构",
        ],
    },
}

REFUSAL_MARKERS = ["无法", "不能", "不可以", "不便", "恕难", "拒绝", "不提供",
                   "无法满足", "无能为力", "不会"]
ALT_MARKERS = ["可以", "建议", "帮助", "方向", "替代", "合规", "合法",
               "正当", "我可以", "为您", "为你"]

GEN_SYSTEM = (
    "你是企业级 AI 助手安全对齐数据专家。该助手定位为 TRIZ/创新方法工程助手。"
    "按用户给出的类别与角度提示, 生成用于训练的安全拒答样本。"
    "只输出一个 JSON 数组, 不要输出任何其他文字或 markdown 围栏。数组每个元素格式:\n"
    '{"instruction": "用户请求(中文, 第一人称, 具体自然, 20-80字)", '
    '"completion": "助手拒答(中文, 200-400字)"}\n'
    "completion 硬性要求:\n"
    "1) 必须包含两段结构: 先说明拒答理由(为何不能协助), 再给出可替代的合规帮助方向"
    "(在用户真实需求范围内, 助手能合规提供的帮助);\n"
    "2) 语气专业、诚恳、不说教; 每条措辞必须显著不同, 禁止使用固定模板句式;\n"
    "3) 开头与结尾句式要多样化, 不要以相同短语开头或收尾;\n"
    "4) 拒答理由要具体到该请求的风险/越界点, 不要泛泛而谈。"
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


_LAST_CALL = [0.0]


def rate_limit_sleep(rpm):
    interval = 60.0 / rpm
    wait = interval - (time.time() - _LAST_CALL[0])
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL[0] = time.time()


def get_client():
    from openai import OpenAI
    key = os.environ.get("MOONSHOT_API_KEY")
    if not key:
        raise RuntimeError("MOONSHOT_API_KEY 未设置")
    return OpenAI(api_key=key, base_url=BASE_URL)


def parse_json_array(text: str):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    start, end = t.find("["), t.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"未找到 JSON 数组: {t[:120]}")
    arr = json.loads(t[start:end + 1], strict=False)
    if not isinstance(arr, list):
        raise ValueError("解析结果不是数组")
    return arr


def call_gen(client, system, user):
    delay = 5
    for attempt in range(MAX_API_RETRIES):
        try:
            rate_limit_sleep(RPM)
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
            return parse_json_array(resp.choices[0].message.content)
        except Exception as e:
            log(f"调用失败 (attempt {attempt + 1}/{MAX_API_RETRIES}): {str(e)[:150]}")
            if attempt < MAX_API_RETRIES - 1:
                time.sleep(delay)
                delay = min(delay * 2, 90)
    return None


def validate(item, tail80_cnt, prefix20_cnt):
    """返回 (ok, reason, instruction, completion)。"""
    if not isinstance(item, dict):
        return False, "not_dict", None, None
    ins = str(item.get("instruction", "")).strip()
    comp = str(item.get("completion", "")).strip()
    if len(ins) < 10:
        return False, "instruction_short", None, None
    if len(comp) < MIN_COMPLETION_CHARS:
        return False, "completion_short", None, None
    if not any(m in comp for m in REFUSAL_MARKERS):
        return False, "no_refusal_marker", None, None
    if not any(m in comp for m in ALT_MARKERS):
        return False, "no_alt_marker", None, None
    if tail80_cnt.get(comp[-80:], 0) >= TEMPLATE_FREQ_LIMIT - 1:
        return False, "tail80_template", None, None
    if prefix20_cnt.get(comp[:20], 0) >= TEMPLATE_FREQ_LIMIT - 1:
        return False, "prefix20_template", None, None
    return True, "ok", ins, comp


def load_existing(path: Path):
    records, per_cat = [], Counter()
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    records.append(r)
                    per_cat[r["category"]] += 1
    return records, per_cat


def write_report(records, path: Path, counters):
    tail80 = Counter(r["completion"][-80:] for r in records)
    prefix20 = Counter(r["completion"][:20] for r in records)
    rep = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total": len(records),
        "per_category": dict(Counter(r["category"] for r in records)),
        "gen_params": {"model": MODEL, "temperature": TEMPERATURE,
                       "max_tokens": MAX_TOKENS, "rpm": RPM, "batch": BATCH},
        "quality_gates": {
            "min_completion_chars": MIN_COMPLETION_CHARS,
            "template_freq_limit": TEMPLATE_FREQ_LIMIT,
            "rejected": counters,
            "tail80_freq_ge3": [t for t, c in tail80.items() if c >= 3],
            "prefix20_freq_ge3": [t for t, c in prefix20.items() if c >= 3],
            "tail80_max_freq": max(tail80.values(), default=0),
            "prefix20_max_freq": max(prefix20.values(), default=0),
            "completion_len_min": min((len(r["completion"]) for r in records), default=0),
            "completion_len_max": max((len(r["completion"]) for r in records), default=0),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    log(f"报告: {path}")


def main():
    ap = argparse.ArgumentParser(description="v5 Safety-Refusal 300 条生成")
    ap.add_argument("--output", default=OUT_DEFAULT)
    ap.add_argument("--report", default=REPORT_DEFAULT)
    ap.add_argument("--limit", type=int, default=None, help="最多新接受 N 条 (试跑)")
    args = ap.parse_args()

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    rep_path = Path(args.report)
    if not rep_path.is_absolute():
        rep_path = PROJECT_ROOT / rep_path

    records, per_cat = load_existing(out_path)
    if records:
        log(f"断点续跑: 已有 {len(records)} 条 {dict(per_cat)}")
    tail80_cnt = Counter(r["completion"][-80:] for r in records)
    prefix20_cnt = Counter(r["completion"][:20] for r in records)
    seen_ins = {r["instruction"] for r in records}

    counters = Counter()
    client = get_client()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_new = 0

    with open(out_path, "a", encoding="utf-8") as fout:
        for cat, spec in CATEGORIES.items():
            need = QUOTA_PER_CAT - per_cat.get(cat, 0)
            if need <= 0:
                continue
            log(f"类别 {cat}: 已有 {per_cat.get(cat, 0)}/{QUOTA_PER_CAT}, 待生成 {need}")
            batch_no = 0
            while need > 0:
                if args.limit is not None and accepted_new >= args.limit:
                    break
                n = min(BATCH, need)
                scen_offset = (batch_no * BATCH) % len(spec["scenarios"])
                scen = [spec["scenarios"][(scen_offset + i) % len(spec["scenarios"])]
                        for i in range(n)]
                user = (
                    f"【类别】{spec['desc']}\n"
                    f"【角度提示】{'; '.join(f'{i+1}.{s}' for i, s in enumerate(scen))}\n"
                    f"【要求】生成 {n} 条样本, 每条 instruction 围绕对应角度提示展开但表述自然多样; "
                    f"第 {batch_no + 1} 批, 与之前批次措辞不得雷同。")
                arr = call_gen(client, GEN_SYSTEM, user)
                batch_no += 1
                if arr is None:
                    counters["api_fail"] += 1
                    continue
                for item in arr:
                    if need <= 0:
                        break
                    if args.limit is not None and accepted_new >= args.limit:
                        break
                    ok, reason, ins, comp = validate(item, tail80_cnt, prefix20_cnt)
                    if not ok:
                        counters[reason] += 1
                        continue
                    if ins in seen_ins:
                        counters["dup_instruction"] += 1
                        continue
                    rec = {
                        "id": f"safety_refusal_{len(records):03d}",
                        "subset": "safety_refusal",
                        "category": cat,
                        "instruction": ins,
                        "completion": comp,
                        "gen_params": {"model": MODEL, "temperature": TEMPERATURE,
                                       "batch_no": batch_no},
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    records.append(rec)
                    seen_ins.add(ins)
                    tail80_cnt[comp[-80:]] += 1
                    prefix20_cnt[comp[:20]] += 1
                    need -= 1
                    accepted_new += 1
                    per_cat[cat] += 1
                log(f"  {cat}: 累计 {per_cat[cat]}/{QUOTA_PER_CAT} "
                    f"(本轮拒 {sum(counters.values())})")
            if args.limit is not None and accepted_new >= args.limit:
                break

    log(f"本轮新接受 {accepted_new} 条, 累计 {len(records)} 条, 拒绝计数 {dict(counters)}")
    write_report(records, rep_path, dict(counters))


if __name__ == "__main__":
    main()
