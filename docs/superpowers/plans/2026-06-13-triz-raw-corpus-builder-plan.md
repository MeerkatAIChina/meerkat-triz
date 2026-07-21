# TRIZ-raw Corpus Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **路径已迁移（2026-07-20 标注）：** 本计划中所有 `ref/mongoose_ai_dgx/` 前缀指向的旧布局已于 2026-07-12 迁移至仓库根（`config.py`、`utils/`、`notebooks/`、`tests/` 现均在仓库根目录，与 DGX Spark 部署路径 `/home/meerkat/mongoose_ai` 一致）。阅读/执行本计划时，请将 `ref/mongoose_ai_dgx/` 一律视为仓库根；正文保留原样以维持历史记录。

**Goal:** Implement a corpus builder that extracts text from `TRIZ-raw/` PDF/DOCX/PPTX/DOC files, semantically chunks it, and writes a quality-gated `triz_corpus.jsonl` ready for continued pre-training on DGX Spark.

**Architecture:** A dispatcher routes files to type-specific extractors, all returning a common `ExtractedDocument` structure. A `SemanticChunker` splits and merges segments by headings/paragraphs up to a target token count. A `CorpusWriter` emits JSONL with metadata plus stats/failure reports. A Jupyter notebook orchestrates the pipeline on DGX Spark.

**Tech Stack:** Python 3.10+, `pymupdf`, `python-docx`, `python-pptx`, `pytesseract` (optional OCR), `pytest`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `ref/mongoose_ai_dgx/config.py` | Add `CORPUS_CONFIG` with paths, chunk sizes, quality gates |
| `ref/mongoose_ai_dgx/utils/corpus_builder.py` | Extractors, chunker, writer, and orchestration |
| `ref/mongoose_ai_dgx/notebooks/02c_build_corpus_from_raw.ipynb` | End-to-end DGX Spark notebook |
| `ref/mongoose_ai_dgx/tests/test_corpus_builder.py` | Unit tests for extractors, chunker, writer |
| `ref/mongoose_ai_dgx/tests/fixtures/corpus/` | Minimal PDF/DOCX/PPTX fixtures for tests |
| `ref/mongoose_ai_dgx/requirements.txt` | Add `pymupdf`, `python-docx`, `python-pptx`, `pytesseract` |

---

## Task 1: Add `CORPUS_CONFIG` to `config.py`

**Files:**
- Modify: `ref/mongoose_ai_dgx/config.py`

- [ ] **Step 1: Insert `CORPUS_CONFIG` after `SYNTHETIC_CONFIG`**

Add the following block after `SYNTHETIC_CONFIG` closes:

```python
# ==================== 语料库构建配置 (TRIZ-raw 原始材料) ====================
CORPUS_CONFIG = {
    # 原始TRIZ材料目录 (DGX Spark上应复制到 BASE_DIR/TRIZ-raw/)
    "raw_dir": str(BASE_DIR / "TRIZ-raw"),
    # 输出目录
    "output_dir": str(DATA_DIR / "processed" / "corpus"),
    # 文件名
    "output_filename": "triz_corpus.jsonl",
    "stats_filename": "triz_corpus_stats.json",
    "failed_files_filename": "failed_files.json",
    # 分块配置
    "chunk": {
        "target_tokens": 2048,
        "max_tokens": 4096,
        # 中文字符token估算: 1 token ~ 1字符 (保守估计)
        "chars_per_token": 1.0,
    },
    # 质量关卡
    "quality_gates": {
        "min_chars": 50,
        "deduplicate": True,
        "language_filter": False,
    },
    # 支持提取的文件扩展名
    "supported_extensions": [".pdf", ".docx", ".pptx", ".doc"],
    # 默认跳过的文件扩展名 (图片/视频/压缩包/表格)
    "skip_extensions": [
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".jfif",
        ".mov", ".mp4", ".avi", ".mkv",
        ".zip", ".rar", ".7z",
        ".xlsx", ".xls", ".csv",
    ],
    # OCR配置 (用于扫描版PDF)
    "ocr": {
        "enabled": True,
        "min_text_chars": 20,
    },
}
```

- [ ] **Step 2: Verify `config.py` imports cleanly**

Run:

```bash
cd /Volumes/2nd-HD/claude/Meerkat-AI/ref/mongoose_ai_dgx
python -c "import config; print(config.CORPUS_CONFIG['raw_dir'])"
```

Expected output: `/home/meerkat/mongoose_ai/TRIZ-raw`

- [ ] **Step 3: Commit**

```bash
git add ref/mongoose_ai_dgx/config.py
git commit -m "config: add CORPUS_CONFIG for TRIZ-raw corpus builder"
```

---

## Task 2: Create `ExtractedDocument` dataclasses and `Extractor` protocol

**Files:**
- Create: `ref/mongoose_ai_dgx/utils/corpus_builder.py`

- [ ] **Step 1: Write the data structures and protocol**

Create `ref/mongoose_ai_dgx/utils/corpus_builder.py` with:

```python
"""
TRIZ-raw 语料库构建器
将PDF/DOCX/PPTX/DOC原始材料提取、分块、metadata化，输出JSONL语料。
"""

import json
import logging
import os
import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Iterator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Page:
    """单个页面/幻灯片/章节提取结果"""
    text: str
    page_num: int = 1
    heading: Optional[str] = None


@dataclass
class ExtractedDocument:
    """单个文件提取结果"""
    source_path: str
    title: Optional[str] = None
    category: Optional[str] = None
    file_type: Optional[str] = None
    pages: List[Page] = field(default_factory=list)


class Extractor(ABC):
    """文件提取器基类"""

    @abstractmethod
    def extract(self, path: Path) -> ExtractedDocument:
        """从文件中提取文本，返回ExtractedDocument"""
        pass
```

- [ ] **Step 2: Verify file syntax**

Run:

```bash
cd /Volumes/2nd-HD/claude/Meerkat-AI/ref/mongoose_ai_dgx
python -c "from utils.corpus_builder import ExtractedDocument, Page, Extractor; print('ok')"
```

Expected output: `ok`

- [ ] **Step 3: Commit**

```bash
git add ref/mongoose_ai_dgx/utils/corpus_builder.py
git commit -m "feat(corpus): add ExtractedDocument dataclasses and Extractor base class"
```

---

## Task 3: Implement `PDFExtractor`

**Files:**
- Modify: `ref/mongoose_ai_dgx/utils/corpus_builder.py`

- [ ] **Step 1: Add `PDFExtractor` class**

Append to `corpus_builder.py`:

```python
class PDFExtractor(Extractor):
    """PDF提取器：优先提取可选中文字，可选OCR回退"""

    def __init__(self, ocr_enabled: bool = True, min_text_chars: int = 20):
        self.ocr_enabled = ocr_enabled
        self.min_text_chars = min_text_chars

    def extract(self, path: Path) -> ExtractedDocument:
        import fitz  # pymupdf

        doc = ExtractedDocument(
            source_path=str(path),
            file_type="pdf",
        )

        try:
            pdf = fitz.open(str(path))
        except Exception as e:
            logger.error(f"无法打开PDF {path}: {e}")
            return doc

        for page_idx in range(len(pdf)):
            page = pdf.load_page(page_idx)
            text = page.get_text().strip()

            # OCR回退：页面文字太少时尝试OCR
            if self.ocr_enabled and len(text) < self.min_text_chars:
                try:
                    from PIL import Image
                    import pytesseract
                    pix = page.get_pixmap(dpi=200)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = pytesseract.image_to_string(img, lang="chi_sim+eng").strip()
                except Exception as e:
                    logger.warning(f"OCR失败 {path} 第{page_idx + 1}页: {e}")

            if text:
                doc.pages.append(Page(text=text, page_num=page_idx + 1))

        pdf.close()
        return doc
```

- [ ] **Step 2: Create a minimal PDF fixture for testing**

Run:

```bash
mkdir -p /Volumes/2nd-HD/claude/Meerkat-AI/ref/mongoose_ai_dgx/tests/fixtures/corpus
cd /Volumes/2nd-HD/claude/Meerkat-AI/ref/mongoose_ai_dgx/tests/fixtures/corpus
python - <<'PY'
import fitz
doc = fitz.open()
page = doc.new_page()
page.insert_text((72, 72), "分割原理：将物体分成独立部分。\n典型应用：智能手机模块化设计。")
doc.save("sample.pdf")
doc.close()
PY
```

Expected: `sample.pdf` created in the fixtures directory.

- [ ] **Step 3: Write a test for `PDFExtractor`**

Create `ref/mongoose_ai_dgx/tests/test_corpus_builder.py`:

```python
"""
Corpus builder tests
Covers TRIZ-raw text extraction, semantic chunking, and corpus writing.
"""

import sys
sys.path.insert(0, "ref/mongoose_ai_dgx")

import importlib.util
spec = importlib.util.spec_from_file_location(
    "corpus_builder", "ref/mongoose_ai_dgx/utils/corpus_builder.py"
)
corpus_builder_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corpus_builder_mod)

PDFExtractor = corpus_builder_mod.PDFExtractor
ExtractedDocument = corpus_builder_mod.ExtractedDocument
Page = corpus_builder_mod.Page

FIXTURES_DIR = Path("ref/mongoose_ai_dgx/tests/fixtures/corpus")


def test_pdf_extractor():
    extractor = PDFExtractor(ocr_enabled=False)
    doc = extractor.extract(FIXTURES_DIR / "sample.pdf")
    assert doc.file_type == "pdf"
    assert len(doc.pages) == 1
    assert "分割原理" in doc.pages[0].text
```

Add at the top of the test file:

```python
from pathlib import Path
```

- [ ] **Step 4: Run the test**

```bash
cd /Volumes/2nd-HD/claude/Meerkat-AI
pytest ref/mongoose_ai_dgx/tests/test_corpus_builder.py::test_pdf_extractor -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ref/mongoose_ai_dgx/utils/corpus_builder.py \
        ref/mongoose_ai_dgx/tests/fixtures/corpus/sample.pdf \
        ref/mongoose_ai_dgx/tests/test_corpus_builder.py
git commit -m "feat(corpus): add PDFExtractor with OCR fallback and test"
```

---

## Task 4: Implement `DocxExtractor` and `PptxExtractor`

**Files:**
- Modify: `ref/mongoose_ai_dgx/utils/corpus_builder.py`
- Create: fixture DOCX and PPTX files

- [ ] **Step 1: Add `DocxExtractor` and `PptxExtractor` classes**

Append to `corpus_builder.py`:

```python
class DocxExtractor(Extractor):
    """DOCX提取器：按段落和标题提取"""

    def extract(self, path: Path) -> ExtractedDocument:
        from docx import Document

        doc = ExtractedDocument(source_path=str(path), file_type="docx")

        try:
            document = Document(str(path))
        except Exception as e:
            logger.error(f"无法打开DOCX {path}: {e}")
            return doc

        doc.title = document.core_properties.title or None
        current_heading = None

        for para in document.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            # 识别标题样式
            if para.style and para.style.name and para.style.name.startswith("Heading"):
                current_heading = text
                continue

            doc.pages.append(Page(text=text, page_num=1, heading=current_heading))

        return doc


class PptxExtractor(Extractor):
    """PPTX提取器：每页幻灯片作为一个Page"""

    def extract(self, path: Path) -> ExtractedDocument:
        from pptx import Presentation

        doc = ExtractedDocument(source_path=str(path), file_type="pptx")

        try:
            prs = Presentation(str(path))
        except Exception as e:
            logger.error(f"无法打开PPTX {path}: {e}")
            return doc

        for slide_idx, slide in enumerate(prs.slides, start=1):
            texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    texts.append(shape.text.strip())
            text = "\n".join(texts)
            if text:
                doc.pages.append(Page(text=text, page_num=slide_idx))

        return doc
```

- [ ] **Step 2: Create fixture DOCX and PPTX files**

Run:

```bash
cd /Volumes/2nd-HD/claude/Meerkat-AI/ref/mongoose_ai_dgx/tests/fixtures/corpus
python - <<'PY'
from docx import Document
doc = Document()
doc.add_heading("局部质量原理", level=1)
doc.add_paragraph("将物体的不同部分赋予不同功能。")
doc.add_paragraph("典型案例：瑞士军刀。")
doc.save("sample.docx")

from pptx import Presentation
prs = Presentation()
slide = prs.slides.add_slide(prs.slide_layouts[6])
textbox = slide.shapes.add_textbox(0, 0, 400, 200)
textbox.text_frame.text = "抽取原理\n从物体中抽取有害部分。"
prs.save("sample.pptx")
PY
```

Expected: `sample.docx` and `sample.pptx` created.

- [ ] **Step 3: Add tests**

Append to `tests/test_corpus_builder.py`:

```python
DocxExtractor = corpus_builder_mod.DocxExtractor
PptxExtractor = corpus_builder_mod.PptxExtractor


def test_docx_extractor():
    extractor = DocxExtractor()
    doc = extractor.extract(FIXTURES_DIR / "sample.docx")
    assert doc.file_type == "docx"
    assert len(doc.pages) >= 2
    assert any("瑞士军刀" in p.text for p in doc.pages)


def test_pptx_extractor():
    extractor = PptxExtractor()
    doc = extractor.extract(FIXTURES_DIR / "sample.pptx")
    assert doc.file_type == "pptx"
    assert len(doc.pages) == 1
    assert "抽取原理" in doc.pages[0].text
```

- [ ] **Step 4: Run tests**

```bash
cd /Volumes/2nd-HD/claude/Meerkat-AI
pytest ref/mongoose_ai_dgx/tests/test_corpus_builder.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ref/mongoose_ai_dgx/utils/corpus_builder.py \
        ref/mongoose_ai_dgx/tests/fixtures/corpus/sample.docx \
        ref/mongoose_ai_dgx/tests/fixtures/corpus/sample.pptx \
        ref/mongoose_ai_dgx/tests/test_corpus_builder.py
git commit -m "feat(corpus): add DocxExtractor and PptxExtractor with tests"
```

---

## Task 5: Implement `SemanticChunker`

**Files:**
- Modify: `ref/mongoose_ai_dgx/utils/corpus_builder.py`
- Modify: `ref/mongoose_ai_dgx/tests/test_corpus_builder.py`

- [ ] **Step 1: Add `SemanticChunker` class**

Append to `corpus_builder.py`:

```python
@dataclass
class Chunk:
    """分块后的语料记录"""
    text: str
    source_path: str
    category: Optional[str]
    file_type: Optional[str]
    page_num: Optional[int]
    heading: Optional[str]
    chunk_index: int
    token_count: int


class SemanticChunker:
    """语义分块器：按标题/段落边界合并短段落到目标token数"""

    def __init__(
        self,
        target_tokens: int = 2048,
        max_tokens: int = 4096,
        chars_per_token: float = 1.0,
        min_chars: int = 50,
    ):
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.chars_per_token = chars_per_token
        self.min_chars = min_chars

    def _estimate_tokens(self, text: str) -> int:
        return max(1, int(len(text) / self.chars_per_token))

    def _split_into_segments(self, text: str) -> List[str]:
        """按空行分割成语义段"""
        segments = []
        current = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                if current:
                    segments.append("\n".join(current))
                    current = []
            else:
                current.append(stripped)
        if current:
            segments.append("\n".join(current))
        return [s for s in segments if len(s) >= self.min_chars]

    def chunk_document(self, doc: ExtractedDocument, chunk_index_start: int = 0) -> Tuple[List[Chunk], int]:
        """对一个ExtractedDocument进行分块，返回Chunk列表和下一个chunk_index"""
        chunks = []
        chunk_index = chunk_index_start
        current_texts: List[str] = []
        current_tokens = 0
        current_heading: Optional[str] = None
        current_page: Optional[int] = None

        def flush():
            nonlocal chunks, chunk_index, current_texts, current_tokens, current_heading, current_page
            if not current_texts:
                return
            text = "\n\n".join(current_texts)
            if len(text) >= self.min_chars:
                chunks.append(Chunk(
                    text=text,
                    source_path=doc.source_path,
                    category=doc.category,
                    file_type=doc.file_type,
                    page_num=current_page,
                    heading=current_heading,
                    chunk_index=chunk_index,
                    token_count=self._estimate_tokens(text),
                ))
                chunk_index += 1
            current_texts = []
            current_tokens = 0

        for page in doc.pages:
            segments = self._split_into_segments(page.text)
            if not segments:
                continue

            # 记录当前页的起始heading
            if page.heading:
                current_heading = page.heading
            if page.page_num:
                current_page = page.page_num

            for seg in segments:
                seg_tokens = self._estimate_tokens(seg)

                # 单个段超过max_tokens则强行截断
                if seg_tokens > self.max_tokens:
                    flush()
                    chunks.append(Chunk(
                        text=seg[: self.max_tokens * self.chars_per_token],
                        source_path=doc.source_path,
                        category=doc.category,
                        file_type=doc.file_type,
                        page_num=current_page,
                        heading=current_heading,
                        chunk_index=chunk_index,
                        token_count=self.max_tokens,
                    ))
                    chunk_index += 1
                    continue

                # 合并到当前chunk会超过目标token数，先flush
                if current_tokens + seg_tokens > self.target_tokens and current_texts:
                    flush()

                current_texts.append(seg)
                current_tokens += seg_tokens

                # 如果当前heading来自这个segment的上一段标题，更新heading
                if page.heading and not current_heading:
                    current_heading = page.heading

        flush()
        return chunks, chunk_index
```

- [ ] **Step 2: Add chunker tests**

Append to `tests/test_corpus_builder.py`:

```python
SemanticChunker = corpus_builder_mod.SemanticChunker
Chunk = corpus_builder_mod.Chunk


def test_semantic_chunker_basic():
    doc = ExtractedDocument(
        source_path="TRIZ-raw/sample.txt",
        category="test",
        file_type="txt",
        pages=[
            Page(text="第一章\n\n分割原理：将物体分成独立部分。\n\n典型应用：智能手机。\n\n第二章\n\n抽取原理：从物体中抽取有害部分。"),
        ],
    )
    chunker = SemanticChunker(target_tokens=100, chars_per_token=1.0)
    chunks, _ = chunker.chunk_document(doc)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert any("分割原理" in c.text for c in chunks)


def test_semantic_chunker_respects_min_chars():
    doc = ExtractedDocument(
        source_path="TRIZ-raw/short.txt",
        category="test",
        file_type="txt",
        pages=[Page(text="A")],
    )
    chunker = SemanticChunker(min_chars=5)
    chunks, _ = chunker.chunk_document(doc)
    assert len(chunks) == 0
```

- [ ] **Step 3: Run tests**

```bash
cd /Volumes/2nd-HD/claude/Meerkat-AI
pytest ref/mongoose_ai_dgx/tests/test_corpus_builder.py -v
```

Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add ref/mongoose_ai_dgx/utils/corpus_builder.py \
        ref/mongoose_ai_dgx/tests/test_corpus_builder.py
git commit -m "feat(corpus): add SemanticChunker with tests"
```

---

## Task 6: Implement `CorpusWriter` and quality gates

**Files:**
- Modify: `ref/mongoose_ai_dgx/utils/corpus_builder.py`
- Modify: `ref/mongoose_ai_dgx/tests/test_corpus_builder.py`

- [ ] **Step 1: Add `CorpusWriter` and quality gate functions**

Append to `corpus_builder.py`:

```python
class CorpusWriter:
    """语料库写入器：输出JSONL、统计信息和失败文件列表"""

    def __init__(
        self,
        output_dir: str,
        output_filename: str = "triz_corpus.jsonl",
        stats_filename: str = "triz_corpus_stats.json",
        failed_files_filename: str = "failed_files.json",
        deduplicate: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / output_filename
        self.stats_path = self.output_dir / stats_filename
        self.failed_files_path = self.output_dir / failed_files_filename
        self.deduplicate = deduplicate

    def _deduplicate(self, chunks: List[Chunk]) -> List[Chunk]:
        if not self.deduplicate:
            return chunks
        seen: Set[str] = set()
        unique = []
        for c in chunks:
            key = hashlib.md5(c.text.encode("utf-8")).hexdigest()
            if key not in seen:
                seen.add(key)
                unique.append(c)
        removed = len(chunks) - len(unique)
        if removed > 0:
            logger.info(f"去重: 移除 {removed} 个重复chunk")
        return unique

    def write(
        self,
        chunks: List[Chunk],
        failed_files: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        chunks = self._deduplicate(chunks)

        # 写JSONL
        with open(self.output_path, "w", encoding="utf-8") as f:
            for idx, chunk in enumerate(chunks):
                record = {
                    "id": f"triz-raw-{idx:08d}",
                    "text": chunk.text,
                    "metadata": {
                        "source_path": chunk.source_path,
                        "category": chunk.category,
                        "file_type": chunk.file_type,
                        "page_num": chunk.page_num,
                        "heading": chunk.heading,
                        "chunk_index": chunk.chunk_index,
                        "token_count": chunk.token_count,
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 统计信息
        category_counts: Dict[Optional[str], int] = {}
        file_type_counts: Dict[Optional[str], int] = {}
        token_counts = [c.token_count for c in chunks]

        for c in chunks:
            category_counts[c.category] = category_counts.get(c.category, 0) + 1
            file_type_counts[c.file_type] = file_type_counts.get(c.file_type, 0) + 1

        stats = {
            "total_records": len(chunks),
            "total_tokens": sum(token_counts),
            "avg_tokens": round(sum(token_counts) / len(token_counts), 2) if token_counts else 0,
            "max_tokens": max(token_counts) if token_counts else 0,
            "min_tokens": min(token_counts) if token_counts else 0,
            "category_distribution": category_counts,
            "file_type_distribution": file_type_counts,
        }

        with open(self.stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        # 失败文件
        if failed_files:
            with open(self.failed_files_path, "w", encoding="utf-8") as f:
                json.dump(failed_files, f, ensure_ascii=False, indent=2)

        logger.info(f"语料库写入完成: {len(chunks)} 条记录 -> {self.output_path}")
        return stats
```

- [ ] **Step 2: Add writer tests**

Append to `tests/test_corpus_builder.py`:

```python
import tempfile
CorpusWriter = corpus_builder_mod.CorpusWriter


def test_corpus_writer(tmp_path):
    chunks = [
        Chunk(
            text="分割原理：将物体分成独立部分。",
            source_path="TRIZ-raw/sample.txt",
            category="test",
            file_type="txt",
            page_num=1,
            heading="第一章",
            chunk_index=0,
            token_count=15,
        ),
        Chunk(
            text="分割原理：将物体分成独立部分。",  # duplicate
            source_path="TRIZ-raw/sample.txt",
            category="test",
            file_type="txt",
            page_num=1,
            heading="第一章",
            chunk_index=1,
            token_count=15,
        ),
    ]
    writer = CorpusWriter(output_dir=str(tmp_path), deduplicate=True)
    stats = writer.write(chunks)
    assert stats["total_records"] == 1

    output_file = tmp_path / "triz_corpus.jsonl"
    assert output_file.exists()
    records = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["id"] == "triz-raw-00000000"
    assert records[0]["metadata"]["category"] == "test"
```

Add `import json` and `import tempfile` at the top of the test file if not already present.

- [ ] **Step 3: Run tests**

```bash
cd /Volumes/2nd-HD/claude/Meerkat-AI
pytest ref/mongoose_ai_dgx/tests/test_corpus_builder.py -v
```

Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add ref/mongoose_ai_dgx/utils/corpus_builder.py \
        ref/mongoose_ai_dgx/tests/test_corpus_builder.py
git commit -m "feat(corpus): add CorpusWriter with deduplication and stats"
```

---

## Task 7: Implement the orchestration function `build_corpus`

**Files:**
- Modify: `ref/mongoose_ai_dgx/utils/corpus_builder.py`
- Modify: `ref/mongoose_ai_dgx/tests/test_corpus_builder.py`

- [ ] **Step 1: Add `build_corpus` function**

Append to `corpus_builder.py`:

```python
def build_corpus(
    raw_dir: str,
    output_dir: str,
    output_filename: str = "triz_corpus.jsonl",
    stats_filename: str = "triz_corpus_stats.json",
    failed_files_filename: str = "failed_files.json",
    supported_extensions: Optional[List[str]] = None,
    skip_extensions: Optional[List[str]] = None,
    chunk_target_tokens: int = 2048,
    chunk_max_tokens: int = 4096,
    chars_per_token: float = 1.0,
    min_chars: int = 50,
    deduplicate: bool = True,
    ocr_enabled: bool = True,
    ocr_min_text_chars: int = 20,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    构建TRIZ语料库主入口

    Args:
        raw_dir: TRIZ-raw目录路径
        output_dir: 输出目录
        output_filename: 输出JSONL文件名
        stats_filename: 统计文件名
        failed_files_filename: 失败文件列表名
        supported_extensions: 支持提取的扩展名
        skip_extensions: 默认跳过的扩展名
        chunk_target_tokens: 目标chunk token数
        chunk_max_tokens: 最大chunk token数
        chars_per_token: 字符到token的估算比例
        min_chars: 最小chunk字符数
        deduplicate: 是否去重
        ocr_enabled: 是否启用OCR
        ocr_min_text_chars: OCR触发阈值
        resume: 是否跳过已处理的文件

    Returns:
        统计信息字典
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    supported_extensions = supported_extensions or [".pdf", ".docx", ".pptx", ".doc"]
    skip_extensions = set(skip_extensions or [".jpg", ".jpeg", ".png", ".gif", ".bmp",
                                                ".webp", ".jfif", ".mov", ".mp4", ".avi",
                                                ".mkv", ".zip", ".rar", ".7z", ".xlsx",
                                                ".xls", ".csv"])

    # 已处理文件集合 (用于断点续跑)
    processed_sources: Set[str] = set()
    output_path = output_dir / output_filename
    if resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    processed_sources.add(record["metadata"]["source_path"])
                except Exception:
                    pass
        logger.info(f"断点续跑: 已跳过 {len(processed_sources)} 个已处理文件")

    # 提取器映射
    extractors = {
        ".pdf": PDFExtractor(ocr_enabled=ocr_enabled, min_text_chars=ocr_min_text_chars),
        ".docx": DocxExtractor(),
        ".pptx": PptxExtractor(),
        ".doc": DocxExtractor(),  # .doc用python-docx尝试，失败则跳过
    }

    chunker = SemanticChunker(
        target_tokens=chunk_target_tokens,
        max_tokens=chunk_max_tokens,
        chars_per_token=chars_per_token,
        min_chars=min_chars,
    )
    writer = CorpusWriter(
        output_dir=str(output_dir),
        output_filename=output_filename,
        stats_filename=stats_filename,
        failed_files_filename=failed_files_filename,
        deduplicate=deduplicate,
    )

    all_chunks: List[Chunk] = []
    failed_files: List[Dict[str, str]] = []
    chunk_index_counter = 0

    if not raw_dir.exists():
        raise FileNotFoundError(f"原始目录不存在: {raw_dir}")

    # 收集所有文件
    files = []
    for ext in supported_extensions:
        files.extend(raw_dir.rglob(f"*{ext}"))

    # 过滤跳过扩展名
    files = [f for f in files if f.suffix.lower() not in skip_extensions]

    logger.info(f"发现 {len(files)} 个待处理文件")

    for file_path in files:
        rel_path = str(file_path.relative_to(raw_dir))
        source_path = f"TRIZ-raw/{rel_path}"

        if source_path in processed_sources:
            logger.info(f"跳过已处理: {source_path}")
            continue

        ext = file_path.suffix.lower()
        extractor = extractors.get(ext)
        if extractor is None:
            logger.warning(f"无可用提取器: {file_path}")
            continue

        try:
            doc = extractor.extract(file_path)
            doc.source_path = source_path
            doc.category = file_path.parent.relative_to(raw_dir).parts[0] if file_path.parent != raw_dir else "root"

            chunks, chunk_index_counter = chunker.chunk_document(doc, chunk_index_start=chunk_index_counter)
            all_chunks.extend(chunks)
            logger.info(f"处理完成: {source_path} -> {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"处理失败 {source_path}: {e}")
            failed_files.append({"path": source_path, "error": str(e)})

    stats = writer.write(all_chunks, failed_files=failed_files)
    return stats
```

- [ ] **Step 2: Add integration test**

Append to `tests/test_corpus_builder.py`:

```python
build_corpus = corpus_builder_mod.build_corpus


def test_build_corpus(tmp_path):
    import shutil
    # 构造临时TRIZ-raw目录
    raw_dir = tmp_path / "TRIZ-raw" / "test_category"
    raw_dir.mkdir(parents=True)
    shutil.copy(FIXTURES_DIR / "sample.pdf", raw_dir / "sample.pdf")

    output_dir = tmp_path / "corpus"
    stats = build_corpus(
        raw_dir=str(tmp_path / "TRIZ-raw"),
        output_dir=str(output_dir),
        resume=False,
    )

    assert stats["total_records"] >= 1
    assert (output_dir / "triz_corpus.jsonl").exists()
    assert (output_dir / "triz_corpus_stats.json").exists()
```

- [ ] **Step 3: Run tests**

```bash
cd /Volumes/2nd-HD/claude/Meerkat-AI
pytest ref/mongoose_ai_dgx/tests/test_corpus_builder.py -v
```

Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add ref/mongoose_ai_dgx/utils/corpus_builder.py \
        ref/mongoose_ai_dgx/tests/test_corpus_builder.py
git commit -m "feat(corpus): add build_corpus orchestration with resume support"
```

---

## Task 8: Update `requirements.txt`

**Files:**
- Modify: `ref/mongoose_ai_dgx/requirements.txt`

- [ ] **Step 1: Add corpus builder dependencies**

Append to `ref/mongoose_ai_dgx/requirements.txt`:

```text
# TRIZ-raw corpus builder
pymupdf>=1.23.0
python-docx>=0.8.11
python-pptx>=0.6.21
pytesseract>=0.3.10
Pillow>=10.0.0
```

- [ ] **Step 2: Commit**

```bash
git add ref/mongoose_ai_dgx/requirements.txt
git commit -m "chore(requirements): add corpus builder dependencies"
```

---

## Task 9: Create `02c_build_corpus_from_raw.ipynb`

**Files:**
- Create: `ref/mongoose_ai_dgx/notebooks/02c_build_corpus_from_raw.ipynb`

- [ ] **Step 1: Create the notebook with 6 cells**

Use the following Jupyter notebook JSON (save as `ref/mongoose_ai_dgx/notebooks/02c_build_corpus_from_raw.ipynb`):

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 02c - 从TRIZ-raw构建预训练语料库\n",
    "\n",
    "将 `TRIZ-raw/` 中的 PDF/DOCX/PPTX/DOC 材料提取为结构化 JSONL 语料。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import sys\n",
    "sys.path.append('/home/meerkat/mongoose_ai')\n",
    "\n",
    "from pathlib import Path\n",
    "from config import CORPUS_CONFIG\n",
    "from utils.corpus_builder import build_corpus\n",
    "\n",
    "print('CORPUS_CONFIG:', CORPUS_CONFIG['raw_dir'])"
   ],
   "outputs": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 检查 TRIZ-raw 目录\n",
    "raw_dir = Path(CORPUS_CONFIG['raw_dir'])\n",
    "print(f'Raw dir exists: {raw_dir.exists()}')\n",
    "print(f'Files found: {len(list(raw_dir.rglob(\"*\")))}')\n",
    "\n",
    "from collections import Counter\n",
    "exts = Counter([p.suffix.lower() for p in raw_dir.rglob(\"*\") if p.is_file()])\n",
    "print('Extension counts:')\n",
    "for ext, count in exts.most_common():\n",
    "    print(f'  {ext}: {count}')"
   ],
   "outputs": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 构建语料库\n",
    "stats = build_corpus(\n",
    "    raw_dir=CORPUS_CONFIG['raw_dir'],\n",
    "    output_dir=CORPUS_CONFIG['output_dir'],\n",
    "    output_filename=CORPUS_CONFIG['output_filename'],\n",
    "    stats_filename=CORPUS_CONFIG['stats_filename'],\n",
    "    failed_files_filename=CORPUS_CONFIG['failed_files_filename'],\n",
    "    chunk_target_tokens=CORPUS_CONFIG['chunk']['target_tokens'],\n",
    "    chunk_max_tokens=CORPUS_CONFIG['chunk']['max_tokens'],\n",
    "    chars_per_token=CORPUS_CONFIG['chunk']['chars_per_token'],\n",
    "    min_chars=CORPUS_CONFIG['quality_gates']['min_chars'],\n",
    "    deduplicate=CORPUS_CONFIG['quality_gates']['deduplicate'],\n",
    "    ocr_enabled=CORPUS_CONFIG['ocr']['enabled'],\n",
    "    ocr_min_text_chars=CORPUS_CONFIG['ocr']['min_text_chars'],\n",
    "    resume=True,\n",
    ")\n",
    "\n",
    "print('Build complete.')\n",
    "print(stats)"
   ],
   "outputs": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 查看统计和样本\n",
    "import json\n",
    "output_path = Path(CORPUS_CONFIG['output_dir']) / CORPUS_CONFIG['output_filename']\n",
    "with open(output_path, 'r', encoding='utf-8') as f:\n",
    "    records = [json.loads(line) for line in f][:5]\n",
    "\n",
    "for r in records:\n",
    "    print(f\"ID: {r['id']}\")\n",
    "    print(f\"Category: {r['metadata']['category']}\")\n",
    "    print(f\"Tokens: {r['metadata']['token_count']}\")\n",
    "    print(f\"Text preview: {r['text'][:200]}...\")\n",
    "    print('-' * 40)"
   ],
   "outputs": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 检查失败文件\n",
    "failed_path = Path(CORPUS_CONFIG['output_dir']) / CORPUS_CONFIG['failed_files_filename']\n",
    "if failed_path.exists():\n",
    "    with open(failed_path, 'r', encoding='utf-8') as f:\n",
    "        failed = json.load(f)\n",
    "    print(f'Failed files: {len(failed)}')\n",
    "    for item in failed[:10]:\n",
    "        print(item)\n",
    "else:\n",
    "    print('No failed files.')"
   ],
   "outputs": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
```

- [ ] **Step 2: Validate notebook JSON**

Run:

```bash
python -c "import json; json.load(open('ref/mongoose_ai_dgx/notebooks/02c_build_corpus_from_raw.ipynb')); print('notebook OK')"
```

Expected: `notebook OK`

- [ ] **Step 3: Commit**

```bash
git add ref/mongoose_ai_dgx/notebooks/02c_build_corpus_from_raw.ipynb
git commit -m "feat(notebook): add 02c_build_corpus_from_raw.ipynb"
```

---

## Task 10: DGX Spark Deployment

**Files:** None (deployment steps)

- [ ] **Step 1: Push code to GitHub**

```bash
git push origin main
```

- [ ] **Step 2: On DGX Spark, pull the updated code**

```bash
ssh meerkat@192.168.5.246  # or appropriate user
 cd /home/meerkat/mongoose_ai
 git pull origin main
```

- [ ] **Step 3: Copy TRIZ-raw to DGX Spark**

From local machine:

```bash
rsync -avz /Volumes/2nd-HD/Downloads/TRIZ-raw/ meerkat@192.168.5.246:/home/meerkat/mongoose_ai/TRIZ-raw/
```

- [ ] **Step 4: Install dependencies on DGX**

```bash
cd /home/meerkat/mongoose_ai
source venv_v5/bin/activate
pip install -r ref/mongoose_ai_dgx/requirements.txt
sudo apt-get install -y tesseract-ocr tesseract-ocr-chi-sim
```

- [ ] **Step 5: Run notebook on DGX**

Open JupyterLab at `http://192.168.5.246:8888`, navigate to `notebooks/02c_build_corpus_from_raw.ipynb`, and run all cells.

- [ ] **Step 6: Verify outputs**

Check that the following files exist:

```bash
ls -lh /home/meerkat/mongoose_ai/data/processed/corpus/
# Expected: triz_corpus.jsonl, triz_corpus_stats.json, failed_files.json
```

---

## Self-Review

**Spec coverage:**
- ✅ Output format: structured `.jsonl` with metadata — Task 6
- ✅ Default raw path `BASE_DIR / "TRIZ-raw"` — Task 1
- ✅ Semantic chunking — Task 5
- ✅ PDF/DOCX/PPTX/DOC support — Tasks 3, 4
- ✅ OCR fallback — Task 3
- ✅ Quality gates — Tasks 5, 6
- ✅ DGX notebook — Task 9
- ✅ Tests — Tasks 3–7
- ✅ DGX deployment steps — Task 10

**Placeholder scan:**
- No "TBD", "TODO", or vague requirements.
- Each step includes actual code or exact commands.

**Type consistency:**
- `ExtractedDocument`, `Page`, `Chunk` dataclasses used consistently.
- `build_corpus` parameter names match `CORPUS_CONFIG` keys.

**Scope check:**
- Plan is focused on building the corpus builder and producing `triz_corpus.jsonl` on DGX. It does not include running continued pre-training.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-13-triz-raw-corpus-builder-plan.md`.**

Two execution options:

1. **Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

Which approach do you want?
