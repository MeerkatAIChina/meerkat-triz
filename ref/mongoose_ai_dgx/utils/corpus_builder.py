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
