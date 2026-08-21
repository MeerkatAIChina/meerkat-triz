import fs from "node:fs";
import path from "node:path";
import {
  AlignmentType,
  BorderStyle,
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
  VerticalAlign,
  WidthType,
  convertInchesToTwip,
} from "docx";

const outputPath = process.argv[2];
if (!outputPath) {
  throw new Error("Usage: node create.js /absolute/path/output.docx");
}

const MD_PATH = "/Volumes/2nd-HD/claude/Meerkat-AI/paper/assessment_report.md";
const md = fs.readFileSync(MD_PATH, "utf8");
const lines = md.split(/\r?\n/);

const palette = {
  dark: "263238",
  primary: "37474F",
  light: "78909C",
  border: "C9D4D9",
  fill: "EEF3F6",
  code: "8A4B2A",
};

const font = {
  ascii: "Times New Roman",
  hAnsi: "Times New Roman",
  cs: "Times New Roman",
  eastAsia: "SimSun",
};
const codeFont = {
  ascii: "Consolas",
  hAnsi: "Consolas",
  cs: "Consolas",
  eastAsia: "SimSun",
};

const run = (text, options = {}) =>
  new TextRun({ text, font, size: 21, ...options });

const para = (children, options = {}) =>
  new Paragraph({
    spacing: { after: 120, line: 300 },
    ...options,
    children: Array.isArray(children) ? children : [children],
  });

// ---------- inline markdown -> runs (bold / code / italic), text kept verbatim ----------
function inlineRuns(text, options = {}) {
  const runs = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*\n]+\*)/g;
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) runs.push(run(text.slice(last, m.index), options));
    const tok = m[0];
    if (tok.startsWith("**")) {
      runs.push(run(tok.slice(2, -2), { ...options, bold: true }));
    } else if (tok.startsWith("`")) {
      runs.push(
        new TextRun({
          text: tok.slice(1, -1),
          font: codeFont,
          size: 19,
          color: palette.code,
        }),
      );
    } else {
      runs.push(run(tok.slice(1, -1), { ...options, italics: true }));
    }
    last = m.index + tok.length;
  }
  if (last < text.length) runs.push(run(text.slice(last), options));
  if (runs.length === 0) runs.push(run("", options));
  return runs;
}

// ---------- parse markdown into blocks ----------
const blocks = [];
let i = 0;
while (i < lines.length) {
  const line = lines[i];
  if (line.trim() === "") {
    i += 1;
    continue;
  }
  if (line.startsWith("# ")) {
    blocks.push({ kind: "title", text: line.slice(2) });
    i += 1;
  } else if (line.startsWith("## ")) {
    blocks.push({ kind: "h1", text: line.slice(3) });
    i += 1;
  } else if (line.startsWith("### ")) {
    blocks.push({ kind: "h2", text: line.slice(4) });
    i += 1;
  } else if (line.startsWith("> ")) {
    const buf = [];
    while (i < lines.length && lines[i].startsWith("> ")) {
      buf.push(lines[i].slice(2));
      i += 1;
    }
    blocks.push({ kind: "quote", lines: buf });
  } else if (/^\s*---\s*$/.test(line)) {
    blocks.push({ kind: "hr" });
    i += 1;
  } else if (line.startsWith("|")) {
    const rows = [];
    while (i < lines.length && lines[i].startsWith("|")) {
      const raw = lines[i];
      if (!/^\|[\s:|-]+\|\s*$/.test(raw)) {
        const cells = raw
          .slice(1, raw.endsWith("|") ? -1 : undefined)
          .split("|")
          .map((c) => c.trim());
        rows.push(cells);
      }
      i += 1;
    }
    blocks.push({ kind: "table", rows });
  } else {
    const m = /^(\s*)([-*]|\d+\.)\s+(.*)$/.exec(line);
    if (m) {
      const level = Math.floor(m[1].length / 2);
      blocks.push({
        kind: "list",
        level,
        marker: m[2],
        text: m[3],
      });
    } else {
      blocks.push({ kind: "p", text: line });
    }
    i += 1;
  }
}

// ---------- estimate TOC page numbers ----------
const tocEntries = [];
let page = 3;
let weight = 0;
for (const b of blocks) {
  if (b.kind === "h1" || b.kind === "h2") {
    tocEntries.push({ title: b.text, level: b.kind === "h1" ? 1 : 2, page });
    weight += b.kind === "h1" ? 1.2 : 0.8;
  } else if (b.kind === "table") {
    weight += b.rows.length * 0.7;
  } else if (b.kind === "quote") {
    weight += b.lines.length * 0.5;
  } else if (b.kind !== "hr") {
    const t = b.text || (b.lines ? b.lines.join("") : "");
    weight += Math.max(0.5, t.length / 220);
  }
  if (weight > 14) {
    page += 1;
    weight = 0;
  }
}

const xmlEscape = (value) =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

const toc = (entries) => {
  const cached = entries
    .map(({ title: t, level, page: pg }) => {
      const indent = Math.max(0, level - 1) * 360;
      return `<w:p>
        <w:pPr>
          <w:pStyle w:val="TOC${level}"/>
          <w:tabs><w:tab w:val="right" w:leader="dot" w:pos="9000"/></w:tabs>
          <w:ind w:left="${indent}"/>
        </w:pPr>
        <w:r><w:t>${xmlEscape(t)}</w:t></w:r>
        <w:r><w:tab/></w:r>
        <w:r><w:t>${xmlEscape(pg)}</w:t></w:r>
      </w:p>`;
    })
    .join("");

  return ImportedXmlComponent.fromXmlString(`<w:sdt xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
    <w:sdtPr><w:alias w:val="目录"/></w:sdtPr>
    <w:sdtContent>
      <w:p>
        <w:r>
          <w:fldChar w:fldCharType="begin" w:dirty="true"/>
          <w:instrText xml:space="preserve"> TOC \\o &quot;1-3&quot; \\h \\z \\u </w:instrText>
          <w:fldChar w:fldCharType="separate"/>
        </w:r>
      </w:p>
      ${cached}
      <w:p><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>
    </w:sdtContent>
  </w:sdt>`).root[0];
};

// ---------- render blocks ----------
const children = [];
const docTitle =
  (blocks.find((b) => b.kind === "title") || {}).text || "科研评估报告";

// column widths (DXA) proportional to content length, CJK counted double
const displayLen = (s) => {
  let n = 0;
  for (const ch of s) n += ch.charCodeAt(0) > 0xff ? 2 : 1;
  return n;
};

function makeTable(rows) {
  const nCols = Math.max(...rows.map((r) => r.length));
  const colWeight = Array.from({ length: nCols }, (_, c) =>
    Math.max(6, ...rows.map((r) => displayLen(r[c] || ""))),
  );
  const totalW = 9360;
  const sum = colWeight.reduce((a, b) => a + b, 0);
  const widths = colWeight.map((w) =>
    Math.max(1100, Math.round((w / sum) * totalW)),
  );
  const borders = {
    top: { style: BorderStyle.SINGLE, size: 4, color: palette.border },
    bottom: { style: BorderStyle.SINGLE, size: 4, color: palette.border },
    left: { style: BorderStyle.SINGLE, size: 4, color: palette.border },
    right: { style: BorderStyle.SINGLE, size: 4, color: palette.border },
    insideHorizontal: { style: BorderStyle.SINGLE, size: 4, color: palette.border },
    insideVertical: { style: BorderStyle.SINGLE, size: 4, color: palette.border },
  };
  const mkCell = (text, c, isHeader) =>
    new TableCell({
      verticalAlign: VerticalAlign.TOP,
      width: { size: widths[c], type: WidthType.DXA },
      margins: { top: 80, bottom: 80, left: 100, right: 100 },
      shading: isHeader
        ? { type: ShadingType.CLEAR, fill: palette.fill }
        : undefined,
      children: [
        para(inlineRuns(text, { size: 18, bold: isHeader || undefined }), {
          spacing: { after: 0, line: 260 },
        }),
      ],
    });
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    columnWidths: widths,
    borders,
    rows: rows.map(
      (r, ri) =>
        new TableRow({
          tableHeader: ri === 0,
          children: Array.from({ length: nCols }, (_, c) =>
            mkCell(r[c] || "", c, ri === 0),
          ),
        }),
    ),
  });
}

let titleDone = false;
let tocInserted = false;
for (const b of blocks) {
  if (b.kind === "title") {
    children.push(
      para(run(b.text, { bold: true, size: 34, color: palette.dark }), {
        heading: HeadingLevel.TITLE,
        alignment: AlignmentType.CENTER,
        spacing: { before: 240, after: 240, line: 320 },
      }),
    );
    titleDone = true;
  } else if (b.kind === "quote") {
    for (const q of b.lines) {
      children.push(
        para(inlineRuns(q, { size: 18, color: palette.light }), {
          alignment: AlignmentType.CENTER,
          spacing: { after: 60, line: 260 },
        }),
      );
    }
    if (titleDone && !tocInserted) {
      children.push(
        new Paragraph({
          pageBreakBefore: true,
          spacing: { after: 160 },
          children: [run("目录", { bold: true, size: 30, color: palette.dark })],
        }),
      );
      children.push(
        para(
          run("在 Word/WPS 中右键目录选择“更新域”可刷新页码。", {
            italics: true,
            size: 18,
            color: palette.light,
          }),
        ),
      );
      children.push(toc(tocEntries));
      tocInserted = true;
    }
  } else if (b.kind === "h1") {
    children.push(
      para(inlineRuns(b.text, { bold: true, size: 28, color: palette.dark }), {
        heading: HeadingLevel.HEADING_1,
        spacing: { before: 320, after: 160 },
      }),
    );
  } else if (b.kind === "h2") {
    children.push(
      para(inlineRuns(b.text, { bold: true, size: 24, color: palette.primary }), {
        heading: HeadingLevel.HEADING_2,
        spacing: { before: 240, after: 120 },
      }),
    );
  } else if (b.kind === "hr") {
    children.push(
      new Paragraph({
        spacing: { before: 60, after: 120 },
        border: {
          bottom: { style: BorderStyle.SINGLE, size: 6, color: palette.border, space: 1 },
        },
        children: [],
      }),
    );
  } else if (b.kind === "table") {
    children.push(makeTable(b.rows));
    children.push(para(run("", { size: 8 }), { spacing: { after: 60 } }));
  } else if (b.kind === "list") {
    const prefix = /^\d/.test(b.marker) ? `${b.marker} ` : "• ";
    children.push(
      para([run(prefix, { bold: /^\d/.test(b.marker) }), ...inlineRuns(b.text)], {
        indent: {
          left: convertInchesToTwip(0.3 + b.level * 0.3),
          hanging: convertInchesToTwip(0.22),
        },
      }),
    );
  } else if (b.kind === "p") {
    children.push(
      para(inlineRuns(b.text), {
        indent: { firstLine: convertInchesToTwip(0.33) },
      }),
    );
  }
}

const doc = new Document({
  features: { updateFields: true },
  sections: [
    {
      properties: {
        page: {
          margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
        },
      },
      headers: {
        default: new Header({
          children: [
            para(run(docTitle, { size: 18, color: palette.primary }), {
              alignment: AlignmentType.CENTER,
              spacing: { after: 0 },
            }),
          ],
        }),
      },
      footers: {
        default: new Footer({
          children: [
            para(
              new TextRun({
                font,
                size: 18,
                children: [PageNumber.CURRENT],
              }),
              { alignment: AlignmentType.CENTER, spacing: { after: 0 } },
            ),
          ],
        }),
      },
      children,
    },
  ],
});

const buffer = await Packer.toBuffer(doc);
fs.writeFileSync(outputPath, buffer);
