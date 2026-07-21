#!/usr/bin/env python3
"""Clean sample_data.json: fix typo, dedupe, and break rigid templates."""

import argparse
import json
import re
from collections import OrderedDict

INPUT_PATH = "data/sample_data.json"
OUTPUT_PATH = "data/sample_data.json"


def fix_typo(samples):
    """Replace duplicated '原理原理' with '原理' in all string fields."""
    for s in samples:
        for key, value in s.items():
            if isinstance(value, str):
                s[key] = value.replace("原理原理", "原理")
    return samples


def deduplicate(samples):
    """Remove exact duplicates based on instruction+input+output."""
    seen = set()
    unique = []
    for s in samples:
        sig = (s.get("instruction", ""), s.get("input", ""), s.get("output", ""))
        if sig not in seen:
            seen.add(sig)
            unique.append(s)
    return unique


def detect_instruction_conflicts(samples):
    """Find instructions appearing >=2 times with differing outputs.

    Returns a list of (instruction, [(index, output), ...]) conflict groups,
    in first-seen order.
    """
    groups = OrderedDict()
    for idx, s in enumerate(samples):
        groups.setdefault(s.get("instruction", ""), []).append(
            (idx, s.get("output", ""))
        )
    conflicts = []
    for instr, items in groups.items():
        if len(items) >= 2 and len({out for _, out in items}) >= 2:
            conflicts.append((instr, items))
    return conflicts


def resolve_instruction_conflicts(samples):
    """For each conflict group keep only the sample with the longest output.

    Ties keep the first occurrence. Returns (new_samples, removed_count).
    """
    drop = set()
    for _instr, items in detect_instruction_conflicts(samples):
        keep_idx = max(items, key=lambda t: len(t[1]))[0]
        for idx, _out in items:
            if idx != keep_idx:
                drop.add(idx)
    kept = [s for i, s in enumerate(samples) if i not in drop]
    return kept, len(drop)


def report_instruction_conflicts(conflicts, subset):
    """Print conflict groups for one subset (report only, no deletion)."""
    for instr, items in conflicts:
        print(
            f"  [CONFLICT] {subset}: instruction x{len(items)} "
            f"with differing outputs"
        )
        print(f"    instruction: {instr[:80]}")
        for idx, out in items:
            print(f"      - #{idx} (output {len(out)} chars): {out[:60]!r}")


def vary_concept_explanation(samples):
    """Vary rigid templates in concept_explanation subset."""
    # Alternative closings for detailed entries
    detailed_closings = [
        "使用场景：当现有结构限制了系统性能提升时，可运用{principle}重新构思设计方案。",
        "应用场景：在需要突破结构限制、提升系统某方面性能时，可参考{principle}寻找设计灵感。",
        "适用情况：当系统某部分受限于当前结构时，借助{principle}重新组织设计往往能获得突破。",
        "何时使用：当需要化解结构约束、实现性能跃升时，{principle}是值得优先考虑的方向。",
    ]

    # Alternative openings for brief entries
    brief_openings = [
        "{principle}（{english}）的核心要点是：{core}。",
        "{principle}（{english}）强调：{core}。",
        "{principle}（{english}）可以理解为：{core}。",
        "{principle}（{english}）的内涵是：{core}。",
    ]

    detailed_count = 0
    brief_count = 0
    varied = []

    for s in samples:
        instr = s.get("instruction", "")
        output = s.get("output", "")

        # Detailed entries: replace rigid closing sentence
        if instr.startswith("请详细解释TRIZ的发明原理"):
            # Extract principle name and English from instruction like:
            # "请详细解释TRIZ的发明原理1——分割原理（Segmentation）"
            m = re.search(r"发明原理\d+——(.+?)（(.+?)）", instr)
            if m:
                principle = m.group(1)
                closing = detailed_closings[detailed_count % len(detailed_closings)]
                output = re.sub(
                    r"使用场景：当系统需要在某方面实现突破但受限于现有结构时，可以考虑应用.+?重新思考设计方案。",
                    closing.format(principle=principle),
                    output,
                )
                detailed_count += 1

        # Brief entries: replace rigid opening
        elif instr.startswith("简要介绍TRIZ的"):
            m = re.search(r"简要介绍TRIZ的(.+?)（(.+?)）", instr)
            if m:
                principle = m.group(1)
                english = m.group(2)
                # Extract core and examples from existing output
                # Format: "X（English）：核心。例如：例子。"
                core_match = re.search(r"(?:.+?)：(.+?)。例如：(.+?)。", output)
                if core_match:
                    core = core_match.group(1)
                    examples = core_match.group(2)
                    opening = brief_openings[brief_count % len(brief_openings)]
                    output = opening.format(principle=principle, english=english, core=core)
                    output += f"常见例子包括：{examples}。"
                brief_count += 1

        s["output"] = output
        varied.append(s)

    return varied


def add_engineering_parameter_examples(samples):
    """Add concrete examples to the 39 engineering parameter explanations."""
    # Map parameter names to concrete examples
    parameter_examples = {
        "运动物体重量": "例如：无人机减重可延长续航，但可能降低结构强度。",
        "静止物体重量": "例如：建筑地基增重可提升稳定性，但会增加材料成本。",
        "运动物体长度": "例如：伸缩式天线在不同场景下调整长度。",
        "静止物体长度": "例如：桥梁跨度影响承重能力与建造成本。",
        "运动物体面积": "例如：飞机机翼面积影响升力与阻力。",
        "静止物体面积": "例如：散热器面积影响散热效率与占用空间。",
        "运动物体体积": "例如：行李箱容积与便携性之间的权衡。",
        "静止物体体积": "例如：数据中心机柜体积影响部署密度。",
        "速度": "例如：高速列车缩短通勤时间，但能耗和噪音增加。",
        "力": "例如：冲压机输出力影响加工精度与能耗。",
        "应力或压力": "例如：高压容器壁厚与安全裕度的设计。",
        "形状": "例如：流线型车身降低空气阻力。",
        "物体结构的稳定性": "例如：塔式起重机抗倾覆设计。",
        "强度": "例如：碳纤维材料在轻量化同时保持高强度。",
        "运动物体的耐久性": "例如：发动机活塞的磨损寿命。",
        "静止物体的耐久性": "例如：建筑外墙抗风化能力。",
        "温度": "例如：锂电池工作温度影响寿命与安全。",
        "照度": "例如：手术室照明需兼顾亮度与能耗。",
        "运动物体消耗能量": "例如：电动汽车行驶能耗。",
        "静止物体消耗能量": "例如：待机电器功耗。",
        "功率": "例如：电机功率决定设备出力上限。",
        "能量损失": "例如：输电线路电阻导致的热损耗。",
        "物质损失": "例如：化工生产中的原料泄漏。",
        "信息损失": "例如：无线信号传输中的数据丢包。",
        "时间损失": "例如：生产线换型导致的停机时间。",
        "物质的量": "例如：药液剂量精度影响疗效。",
        "可靠性": "例如：航空电子系统冗余设计提高可靠性。",
        "测量精度": "例如：光刻机对准精度决定芯片良率。",
        "制造精度": "例如：精密轴承加工公差控制。",
        "作用于物体的有害因素": "例如：腐蚀、辐射等外部损害。",
        "有害的副作用": "例如：药物治疗的副作用。",
        "可制造性": "例如：复杂曲面零件是否便于批量加工。",
        "使用的便利性": "例如：老年人产品的人机交互设计。",
        "可维修性": "例如：模块化设计便于快速更换故障部件。",
        "适应性/通用性": "例如：可调扳手适配多种规格螺栓。",
        "装置的复杂性": "例如：多功能一体机集成度与故障率权衡。",
        "控制的复杂性": "例如：自动驾驶多传感器融合决策。",
        "自动化程度": "例如：无人工厂与柔性生产的平衡。",
        "生产率": "例如：单位时间内装配线产出数量。",
    }

    varied = []
    for s in samples:
        instr = s.get("instruction", "")
        output = s.get("output", "")
        m = re.search(r"解释TRIZ 39个工程参数中的'(.+?)'", instr)
        if m:
            param = m.group(1)
            example = parameter_examples.get(param, "")
            if example and example not in output:
                output = output.rstrip() + "\n\n" + example
        s["output"] = output
        varied.append(s)
    return varied


def vary_domain_templates(samples, subset):
    """Vary outputs for domain-templated entries in case_generation, ariz_guidance, innovation_assessment."""
    if subset == "case_generation":
        return vary_case_generation(samples)
    elif subset == "ariz_guidance":
        return vary_ariz_guidance(samples)
    elif subset == "innovation_assessment":
        return vary_innovation_assessment(samples)
    return samples


def vary_case_generation(samples):
    """Make domain-specific case generation outputs less generic."""
    domain_specific = {
        "智能手机领域": {
            "problem": "如何使手机更轻薄但功能更强大",
            "principles": "分割（模块化内部结构）、嵌套（堆叠主板与电池）、多用性（一颗传感器兼顾多种功能）、复合材料（轻量化高强度机身）",
            "note": "可在保持握持舒适度的前提下，通过系统级封装提升集成度。",
        },
        "电动汽车领域": {
            "problem": "如何提高续航同时缩短充电时间",
            "principles": "分割（换电/电池包模块化）、动态化（智能充电策略）、反馈（BMS实时优化）、预先作用（预约热管理）",
            "note": "需要平衡能量密度、热安全与补能体验。",
        },
        "智能家居领域": {
            "problem": "如何让家居设备既节能又舒适",
            "principles": "反馈（环境传感器联动）、动态化（自适应运行）、自服务（利用自然能源）、等势（减少能量转换环节）",
            "note": "以人为本的场景联动比单一设备节能更重要。",
        },
        "医疗器械领域": {
            "problem": "如何设计无痛注射器",
            "principles": "分割（微针阵列）、柔性壳体和薄膜（超薄针头）、抛弃与再生（一次性安全耗材）、预先反作用（表皮麻醉预处理）",
            "note": "需兼顾注射有效性与患者心理接受度。",
        },
        "食品保鲜领域": {
            "problem": "如何延长食品保鲜期同时保持口感",
            "principles": "抽取（去除氧气/乙烯）、预先防护（气调包装）、局部质量（差异化保鲜环境）、自服务（天然抑菌成分）",
            "note": "不同食材的呼吸特性决定保鲜方案。",
        },
        "建筑隔音领域": {
            "problem": "如何提升隔音效果同时减少墙体厚度",
            "principles": "复合材料（阻尼隔音板）、多孔材料（吸音层）、不对称（错层阻断声桥）、曲面化（扩散体造型）",
            "note": "隔音设计应针对空气声与撞击声分别处理。",
        },
        "无人机领域": {
            "problem": "如何提高无人机续航能力",
            "principles": "重量补偿（升力优化翼型）、能量转换（太阳能薄膜）、预先作用（任务前路径优化）、反馈（实时风速与功耗闭环）",
            "note": "续航提升需在气动、结构与任务规划上协同优化。",
        },
        "水净化领域": {
            "problem": "如何低成本高效净化水质",
            "principles": "抽取（过滤/吸附杂质）、自服务（利用重力与自然沉降）、多孔材料（滤芯）、变害为利（回收浓缩物）",
            "note": "应根据水源污染物类型组合净化工艺。",
        },
        "可穿戴设备领域": {
            "problem": "如何让可穿戴设备更舒适且功能丰富",
            "principles": "柔性壳体和薄膜（亲肤材料）、分割（模块化传感器）、动态化（自适应松紧/显示）、局部质量（不同区域差异化设计）",
            "note": "舒适性优先，再叠加健康监测功能。",
        },
        "物流运输领域": {
            "problem": "如何提高物流运输效率并降低成本",
            "principles": "预先作用（预分拣与路径规划）、合并（共同配送）、有效作用连续性（减少中转等待）、反馈（实时追踪与调度）",
            "note": "数字化调度与装载优化同样关键。",
        },
    }

    varied = []
    for s in samples:
        instr = s.get("instruction", "")
        output = s.get("output", "")
        matched = False
        for domain, content in domain_specific.items():
            if domain in instr:
                output = (
                    f"{domain}TRIZ创新方案：\n"
                    f"问题：{content['problem']}\n"
                    f"核心原理组合：{content['principles']}\n"
                    f"补充说明：{content['note']}"
                )
                matched = True
                break
        s["output"] = output
        varied.append(s)
    return varied


def vary_ariz_guidance(samples):
    """Make ARIZ domain guidance outputs less generic."""
    domain_specific = {
        "制造业领域": {
            "context": "制造业",
            "problem": "产线效率瓶颈或工艺缺陷",
            "focus": "识别关键工艺参数冲突，如精度与速度、成本与质量",
            "deliverable": "形成可落地的工艺改进清单与验证方案。",
        },
        "医疗领域": {
            "context": "医疗",
            "problem": "器械安全性与治疗效果的矛盾",
            "focus": "从患者生理约束出发，定义物理矛盾与理想最终解",
            "deliverable": "输出兼顾疗效、安全与法规要求的概念方案。",
        },
        "建筑领域": {
            "context": "建筑",
            "problem": "空间、结构与能耗之间的冲突",
            "focus": "将结构功能与环境性能参数化，寻找矛盾矩阵推荐原理",
            "deliverable": "形成可持续建筑设计策略与关键构造节点。",
        },
        "能源领域": {
            "context": "能源",
            "problem": "能量转换效率与系统复杂度",
            "focus": "识别能量损失环节，将多目标冲突转化为技术矛盾",
            "deliverable": "给出能效提升路径与风险评估。",
        },
        "交通领域": {
            "context": "交通",
            "problem": "运力、安全与能耗的平衡",
            "focus": "以用户出行链为场景，定位关键矛盾参数",
            "deliverable": "形成系统集成优化方案与分阶段实施计划。",
        },
        "通信领域": {
            "context": "通信",
            "problem": "带宽、延迟与可靠性的三角约束",
            "focus": "将网络资源调度问题抽象为参数矛盾",
            "deliverable": "输出协议或架构层面的创新方向。",
        },
        "农业领域": {
            "context": "农业",
            "problem": "产量、成本与环境影响",
            "focus": "从作物生长周期与资源约束定义问题模型",
            "deliverable": "形成精准农业或绿色种植改进方案。",
        },
        "环保领域": {
            "context": "环保",
            "problem": "污染物去除效率与处理成本",
            "focus": "将变害为利思维融入ARIZ，寻找副产品再利用机会",
            "deliverable": "给出清洁生产或末端治理的技术路线。",
        },
    }

    varied = []
    for s in samples:
        instr = s.get("instruction", "")
        output = s.get("output", "")
        for domain, content in domain_specific.items():
            if domain in instr:
                output = (
                    f"{content['context']}领域的ARIZ应用：\n"
                    f"1) 明确领域核心问题：{content['problem']}\n"
                    f"2) 构建问题模型：{content['focus']}\n"
                    f"3) 按ARIZ 9步法系统化分析，识别技术与物理矛盾\n"
                    f"4) 利用矛盾矩阵与发明原理生成候选方案\n"
                    f"5) {content['deliverable']}"
                )
                break
        s["output"] = output
        varied.append(s)
    return varied


def vary_innovation_assessment(samples):
    """Make patent/innovation assessment outputs less generic."""
    tech_specific = {
        "智能传感器网络": {
            "core": "多节点协同感知与边缘决策算法",
            "peripheral": "低功耗通信协议、自校准机制、数据融合方法",
            "defensive": "节点部署拓扑、异常检测模型、安全认证方案",
            "note": "重点关注物联网标准必要专利布局。",
        },
        "自适应算法": {
            "core": "在线学习与参数自适应核心算法",
            "peripheral": "特征工程优化、模型压缩、迁移学习方案",
            "defensive": "训练数据增强、损失函数设计、解释性方法",
            "note": "需关注与上游基础模型专利的交叉许可。",
        },
        "新能源材料": {
            "core": "高能量密度/高稳定性材料配方与制备工艺",
            "peripheral": "回收再生方法、复合改性方案、表征技术",
            "defensive": "替代材料路线、失效分析模型、安全结构",
            "note": "专利应覆盖材料、工艺与应用场景。",
        },
        "生物医学设备": {
            "core": "诊疗一体化设备结构与控制算法",
            "peripheral": "一次性耗材、人机交互界面、数据管理平台",
            "defensive": "监管合规设计、生物相容性材料、安全机制",
            "note": "医疗器械专利需与注册路径同步规划。",
        },
        "环保技术": {
            "core": "高效低耗污染物去除核心工艺",
            "peripheral": "副产物资源化、设备模块化、智能监控系统",
            "defensive": "替代处理路线、排放标准适配方案",
            "note": "可考虑绿色技术许可与碳减排方法学。",
        },
        "机器人系统": {
            "core": "运动规划与多模态感知融合算法",
            "peripheral": "执行器结构、末端夹具、协作安全机制",
            "defensive": "不同构型机器人、云边协同控制",
            "note": "重点布局应用场景专利，避免仅保护通用算法。",
        },
        "虚拟现实应用": {
            "core": "低延迟渲染与空间定位算法",
            "peripheral": "交互外设、内容创作工具、云端分发",
            "defensive": "显示光学方案、眩晕缓解技术、数字资产保护",
            "note": "关注硬件、软件与内容三层专利组合。",
        },
        "无人机技术": {
            "core": "飞控、导航与任务规划一体化系统",
            "peripheral": "载荷接口、快拆结构、电池热管理",
            "defensive": "避障算法、通信链路、低空管控适配",
            "note": "空域法规与出口管制也是布局考量。",
        },
        "物联网平台": {
            "core": "设备接入、规则引擎与数据治理平台",
            "peripheral": "边缘网关、OTA升级、数字孪生",
            "defensive": "多协议兼容、数据安全、灾备方案",
            "note": "平台专利应突出行业know-how与数据闭环。",
        },
        "人工智能芯片": {
            "core": "神经网络加速器架构与数据流调度",
            "peripheral": "编译器优化、低比特量化、片上存储设计",
            "defensive": "不同精度/场景适配、散热封装、安全启动",
            "note": "需关注国际半导体专利格局与FTO风险。",
        },
    }

    varied = []
    for s in samples:
        instr = s.get("instruction", "")
        output = s.get("output", "")
        for tech, content in tech_specific.items():
            if tech in instr:
                output = (
                    f"{tech}专利布局策略：\n"
                    f"1) 核心专利：保护{content['core']}\n"
                    f"2) 外围专利：覆盖{content['peripheral']}\n"
                    f"3) 防御性专利：布局{content['defensive']}\n"
                    f"4) {content['note']}"
                )
                break
        s["output"] = output
        varied.append(s)
    return varied


def main():
    parser = argparse.ArgumentParser(
        description="Clean sample_data.json: fix typo, dedupe, break rigid templates, "
        "and detect instruction-level conflicts (same instruction, different outputs)."
    )
    parser.add_argument(
        "--resolve",
        action="store_true",
        help="Resolve instruction conflicts by keeping the longest-output version "
        "(default: report only, no deletion).",
    )
    args = parser.parse_args()

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = OrderedDict()
    counts = {}
    conflict_counts = {}
    resolved_removed = {}

    for subset in data:
        samples = data[subset]
        before = len(samples)

        samples = fix_typo(samples)
        samples = deduplicate(samples)

        # Instruction-level conflict detection: same instruction, differing outputs
        conflicts = detect_instruction_conflicts(samples)
        conflict_counts[subset] = len(conflicts)
        if conflicts:
            report_instruction_conflicts(conflicts, subset)
            if args.resolve:
                samples, removed = resolve_instruction_conflicts(samples)
                resolved_removed[subset] = removed

        if subset == "concept_explanation":
            samples = vary_concept_explanation(samples)
            samples = add_engineering_parameter_examples(samples)
        elif subset in ("case_generation", "ariz_guidance", "innovation_assessment"):
            samples = vary_domain_templates(samples, subset)

        after = len(samples)
        counts[subset] = (before, after)
        cleaned[subset] = samples

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print("Cleaning complete.")
    total_before = sum(b for b, _ in counts.values())
    total_after = sum(a for _, a in counts.values())
    total_conflicts = sum(conflict_counts.values())
    print(f"Total: {total_before} → {total_after} samples")
    print(f"Instruction conflicts: {total_conflicts} groups")
    if args.resolve:
        print(f"Resolved by removing {sum(resolved_removed.values())} samples "
              f"(kept longest output per group)")
    for subset, (b, a) in counts.items():
        line = f"  {subset}: {b} → {a}, conflicts: {conflict_counts[subset]}"
        if subset in resolved_removed:
            line += f" (resolved: -{resolved_removed[subset]})"
        print(line)


if __name__ == "__main__":
    main()
