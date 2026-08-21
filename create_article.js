import fs from "node:fs";
import path from "node:path";
import {
  AlignmentType,
  Document,
  Footer,
  Header,
  HeadingLevel,
  ImportedXmlComponent,
  Packer,
  PageNumber,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableRow,
  TextRun,
  WidthType,
  convertInchesToTwip,
} from "docx";

const outputPath = process.argv[2];
if (!outputPath) throw new Error("Usage: node create.js /absolute/path/output.docx");

const outputDir = path.dirname(outputPath);
fs.mkdirSync(outputDir, { recursive: true });

const T = String.raw;

const palette = {
  dark: "1A365D",
  primary: "2C5282",
  light: "718096",
  border: "E2E8F0",
  fill: "EDF2F7",
  accent: "C53030",
};

const font = {
  ascii: "Calibri",
  hAnsi: "Calibri",
  cs: "Calibri",
  eastAsia: "Microsoft YaHei",
};

const run = (text, options = {}) => new TextRun({ text, font, size: 24, ...options });
const para = (children, options = {}) => new Paragraph({
  spacing: { after: 160, line: 340 },
  ...options,
  children: Array.isArray(children) ? children : [children],
});

const bodyPara = (text, options = {}) => para(run(text), {
  indent: { firstLine: convertInchesToTwip(0.33) },
  ...options,
});

const heading = (text, level = 1) => para(run(text, {
  bold: true,
  size: level === 1 ? 32 : level === 2 ? 28 : 26,
  color: palette.dark,
}), {
  heading: level === 1 ? HeadingLevel.HEADING_1 : level === 2 ? HeadingLevel.HEADING_2 : HeadingLevel.HEADING_3,
  spacing: { before: 320, after: 160 },
});

const cell = (text, options = {}) => new TableCell({
  children: [para(run(text, { size: 22 }), { spacing: { after: 60, line: 280 } })],
  margins: { top: 100, bottom: 100, left: 120, right: 120 },
  ...options,
});

const xmlEscape = (value) => String(value)
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;");

const toc = (entries) => {
  const cached = entries.map(({ title: entryTitle, level, page }) => {
    const indent = Math.max(0, level - 1) * 360;
    return `<w:p><w:pPr><w:pStyle w:val="TOC${level}"/>
      <w:tabs><w:tab w:val="right" w:leader="dot" w:pos="9000"/></w:tabs>
      <w:ind w:left="${indent}"/></w:pPr>
      <w:r><w:t>${xmlEscape(entryTitle)}</w:t></w:r><w:r><w:tab/></w:r><w:r><w:t>${page}</w:t></w:r></w:p>`;
  }).join("");

  return ImportedXmlComponent.fromXmlString(`<w:sdt xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
    <w:sdtPr><w:alias w:val="目录"/></w:sdtPr>
    <w:sdtContent>
      <w:p><w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/>
        <w:instrText xml:space="preserve"> TOC \\o &quot;1-3&quot; \\h \\z \\u </w:instrText>
        <w:fldChar w:fldCharType="separate"/></w:r></w:p>
      ${cached}
      <w:p><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>
    </w:sdtContent>
  </w:sdt>`).root[0];
};

const sections = [
  {
    title: T`一、为什么现在是企业部署 TRIZ 大模型的最佳时机`,
    level: 1,
    page: 2,
    paragraphs: [
      T`TRIZ（发明问题解决理论）从来不是「玄学」，而是一套经过 70 余年验证的系统化创新方法论。从苏联军工体系到三星、宝洁、波音，全球 500 强中超过 400 家已将 TRIZ 纳入研发流程。`,
      T`但传统 TRIZ 咨询面临一个结构性瓶颈：培养一名合格的 TRIZ 分析师需要 2-3 年，而企业的创新需求是 7×24 小时不间断的。`,
      T`大语言模型的成熟，正在改变这一游戏规则。核心逻辑不是「用 AI 替代 TRIZ 专家」，而是「让 TRIZ 能力成为每位工程师触手可及的基础设施」。`,
      T`猫鼬 AI（Meerkat AI）正是在这一趋势下，推出了面向企业级场景的 Meerkat-TRIZ 系列产品——不是实验室玩具，而是可部署、可评估、可迭代的工程化产品。`,
    ],
  },
  {
    title: T`二、Meerkat-TRIZ v1：双版本并行，按需选择`,
    level: 1,
    page: 3,
    paragraphs: [
      T`2026 年 8 月，猫鼬 AI 正式发布 Meerkat-TRIZ v1 的第二个基座版本——基于 Qwen3.8-27B 稠密架构的微调适配器。`,
      T`需要特别强调的是：这不是「替代」，而是「增加选项」。`,
      T`两个版本将长期并行维护，企业可根据自身的算力预算、延迟要求、部署环境，灵活选择最适合的方案。`,
    ],
  },
  {
    title: T`2.1 版本对比与选型指南`,
    level: 2,
    page: 3,
    paragraphs: [
      T`以下从六个企业最关心的维度，给出两个版本的直接对比：`,
    ],
    table: () => {
      const widths = [2200, 3200, 3200];
      const headerCell = (text) => cell(text, {
        shading: { type: ShadingType.CLEAR, fill: palette.fill },
        width: { size: widths[0], type: WidthType.DXA },
      });
      const dataCell = (text) => cell(text, { width: { size: widths[0], type: WidthType.DXA } });

      return new Table({
        width: { size: 100, type: WidthType.PERCENTAGE },
        columnWidths: widths,
        rows: [
          new TableRow({ children: [
            headerCell("对比维度"),
            headerCell("v1-Qwen3.6-35B-A3B"),
            headerCell("v1-Qwen3.8-27B（新增）"),
          ]}),
          new TableRow({ children: [
            dataCell("架构类型"),
            dataCell("MoE（混合专家）"),
            dataCell("Dense（稠密架构）"),
          ]}),
          new TableRow({ children: [
            dataCell("推理成本"),
            dataCell("中等（受专家路由影响）"),
            dataCell("更低（稠密架构更可控）"),
          ]}),
          new TableRow({ children: [
            dataCell("部署门槛"),
            dataCell("需理解 MoE 显存模式"),
            dataCell("标准 Transformer，开箱即用"),
          ]}),
          new TableRow({ children: [
            dataCell("边缘部署"),
            dataCell("较复杂"),
            dataCell("更友好（量化生态成熟）"),
          ]}),
          new TableRow({ children: [
            dataCell("工具链支持"),
            dataCell("标准"),
            dataCell("更丰富（Qwen3.5 生态）"),
          ]}),
          new TableRow({ children: [
            dataCell("推荐场景"),
            dataCell("已有 MoE 基建的团队"),
            dataCell("新建项目、边缘部署、成本敏感场景"),
          ]}),
        ],
      });
    },
  },
  {
    title: T`2.2 选型建议：三类典型场景`,
    level: 2,
    page: 4,
    paragraphs: [
      T`【场景一：研发知识库问答系统】`,
      T`如果您的目标是构建一个面向数千名工程师的 TRIZ 知识问答系统，推荐选择 Qwen3.8-27B 版本。稠密架构的推理成本更可预测，便于做容量规划和预算控制。`,
      T`【场景二：已有大模型中台的集团企业】`,
      T`如果您的大模型中台已经部署了 Qwen3.6-35B-A3B 或其他 MoE 模型，两个版本均可接入。建议通过内部 A/B 测试，用实际业务数据验证哪个版本在您的任务上表现更优。`,
      T`【场景三：边缘设备或私有化轻量部署】`,
      T`对于需要在工厂现场、研发中心本地服务器或边缘设备上部署的场景，Qwen3.8-27B 版本是更务实的选择。其量化方案（AWQ、GPTQ、GGUF）的社区支持更为成熟。`,
    ],
  },
  {
    title: T`三、技术透明，是建立信任的前提`,
    level: 1,
    page: 5,
    paragraphs: [
      T`在企业级 AI 应用中，「黑盒」是最大的信任杀手。猫鼬 AI 在 Meerkat-TRIZ 项目中坚持「全链路透明」原则——这不仅是对技术负责，更是对客户的商业承诺。`,
    ],
  },
  {
    title: T`3.1 训练过程完全可追溯`,
    level: 2,
    page: 5,
    paragraphs: [
      T`从基座选择到最终模型，每一步都有明确的决策记录：`,
      T`• 预检阶段（P1-P3）：架构兼容性确认、数据契约验证、生成冒烟测试——三项全过才启动训练；`,
      T`• 锚点评测：300 道 TRIZ 专业题目全量评测，裸基座 Judge armA 均分 2.93，invalid 率 0%，三扇决策门全部通过；`,
      T`• 训练监控：Cosine Decay 学习率调度、EarlyStopping（patience=3, threshold=0.002）、自动续跑机制；`,
      T`• 最佳 Checkpoint：eval_loss 1.505834 @ step 1300，总训练耗时 9.73 小时。`,
      T`所有配置、日志、评测报告均已同步公开至 Hugging Face 模型页面，客户可自行审计。`,
    ],
  },
  {
    title: T`3.2 双轨评估：让效果「可量化」`,
    level: 2,
    page: 6,
    paragraphs: [
      T`Meerkat-TRIZ 的核心差异化之一，是建立了严格的双轨评估体系：`,
      T`• Keyword 轨道：验证输出是否覆盖 TRIZ 核心概念（发明原理、矛盾参数、标准解等）；`,
      T`• Judge 轨道：由独立大模型评委进行 1-5 分制专业评判。`,
      T`两条轨道并行，任何一条不通过即触发重生成。这种「冗余验证」机制，确保了输出质量的可量化、可复现——这正是企业采购 AI 产品时最关心的「确定性」。`,
    ],
  },
  {
    title: T`四、猫鼬 AI 能为您做什么`,
    level: 1,
    page: 7,
    paragraphs: [
      T`Meerkat-TRIZ 不是终点，而是猫鼬 AI 与企业客户合作的起点。我们提供三层服务能力：`,
    ],
  },
  {
    title: T`4.1 产品层：即插即用的领域模型`,
    level: 2,
    page: 7,
    paragraphs: [
      T`两个基座版本的 LoRA 适配器均已上架 Hugging Face（Meerkat-AI 组织），支持标准 Peft 加载方式。无论是云端部署、私有化部署还是边缘部署，均可快速集成。`,
    ],
  },
  {
    title: T`4.2 咨询层：方法论落地与知识工程`,
    level: 2,
    page: 7,
    paragraphs: [
      T`猫鼬 AI 的核心团队拥有深厚的 TRIZ 方法论背景，可为企业提供：`,
      T`• 研发流程创新瓶颈诊断；`,
      T`• 企业内部 TRIZ 知识库构建（专利、技术方案、失败案例的结构化）；`,
      T`• 行业专属评测标准共建——与企业一起定义「什么是好的 TRIZ 回答」。`,
    ],
  },
  {
    title: T`4.3 技术层：领域大模型迁移框架`,
    level: 2,
    page: 8,
    paragraphs: [
      T`基于 Meerkat-TRIZ 的工程实践，猫鼬 AI 已沉淀出一套可复用的「领域大模型迁移框架」：`,
      T`• 基座评估矩阵（架构兼容性 × 推理成本 × 生态成熟度 × 许可证合规）；`,
      T`• 预检流水线（数据契约 → Tokenizer 对齐 → 生成冒烟测试）；`,
      T`• 训练监控与版本管理规范。`,
      T`当您的企业需要为 FMEA、DFSS、六西格玛等其他方法论构建专用大模型时，这套框架可直接复用，大幅缩短从 0 到 1 的周期。`,
    ],
  },
  {
    title: T`五、下一步：我们期待的合作`,
    level: 1,
    page: 9,
    paragraphs: [
      T`Meerkat-TRIZ v1 的基座跃迁，验证了猫鼬 AI 在「方法论产品化」上的工程能力。但我们深知，真正的价值创造发生在技术与业务场景的交汇点。`,
      T`我们正在寻找以下合作伙伴：`,
      T`【工业企业】拥有丰富研发数据和 TRIZ 应用经验，希望将内部知识沉淀为可复用的 AI 能力。`,
      T`【咨询机构】在创新管理、研发体系咨询领域有深厚积累，希望借助大模型技术提升服务效率与交付标准。`,
      T`【技术团队】正在探索领域大模型的落地路径，需要经过验证的工程框架和最佳实践参考。`,
      T`我们相信：只有当客户能够追溯每一个环节，才能真正建立信任。猫鼬 AI 不做「魔法」，做工程。`,
    ],
  },
  {
    title: T`六、获取方式`,
    level: 1,
    page: 10,
    paragraphs: [
      T`🌐 官网：https://meerkatai.cn/`,
      T`🤗 Hugging Face 模型主页：https://huggingface.co/Meerkat-AI`,
      T`💻 GitHub：https://github.com/MeerkatAIChina`,
      T`📮 商务合作：请通过官网或公众号后台留言，我们将在 24 小时内回复。`,
      T``,
      T`—— 猫鼬 AI 产品团队`,
      T`2026 年 8 月`,
    ],
  },
];

const children = [
  // Cover title
  para(run(T`猫鼬 AI 技术发布`, { size: 22, color: palette.light }), {
    alignment: AlignmentType.CENTER,
    spacing: { after: 120 },
  }),
  para(run(T`Meerkat-TRIZ v1 双版本并行发布`, { bold: true, size: 40, color: palette.dark }), {
    alignment: AlignmentType.CENTER,
    spacing: { after: 80 },
  }),
  para(run(T`Qwen3.6-35B-A3B × Qwen3.8-27B｜按需选型｜透明工程`, { size: 24, color: palette.primary }), {
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
  }),
  para(run(T`—— 让企业级 TRIZ 创新方法论真正「可部署、可评估、可迭代」`, { italics: true, size: 22, color: palette.light }), {
    alignment: AlignmentType.CENTER,
    spacing: { after: 400 },
  }),
  // TOC heading
  heading("目录", 1),
  para(run(T`右键目录，选择"更新域"刷新页码。`, { italics: true, color: palette.light, size: 20 })),
  toc(sections.map(({ title: entryTitle, level, page }) => ({ title: entryTitle, level, page }))),
];

for (const section of sections) {
  children.push(heading(section.title, section.level));
  for (const paragraph of section.paragraphs) {
    children.push(bodyPara(paragraph));
  }
  if (section.table) {
    children.push(para(run(""), { spacing: { after: 120 } }));
    children.push(section.table());
    children.push(para(run(""), { spacing: { after: 200 } }));
  }
}

const doc = new Document({
  features: { updateFields: true },
  sections: [{
    properties: {
      page: {
        margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
      },
    },
    headers: {
      default: new Header({
        children: [para(run(T`猫鼬 AI · Meerkat-TRIZ v1 发布稿`, { size: 20, color: palette.light }), {
          alignment: AlignmentType.CENTER,
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [para(new TextRun({ children: [PageNumber.CURRENT], size: 20 }), {
          alignment: AlignmentType.CENTER,
        })],
      }),
    },
    children,
  }],
});

const buffer = await Packer.toBuffer(doc);
fs.writeFileSync(outputPath, buffer);
