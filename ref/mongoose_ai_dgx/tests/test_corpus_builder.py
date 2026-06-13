"""
Corpus builder tests
Covers TRIZ-raw text extraction, semantic chunking, and corpus writing.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

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
    assert "Segmentation Principle" in doc.pages[0].text


DocxExtractor = corpus_builder_mod.DocxExtractor
PptxExtractor = corpus_builder_mod.PptxExtractor


def test_docx_extractor():
    extractor = DocxExtractor()
    doc = extractor.extract(FIXTURES_DIR / "sample.docx")
    assert doc.file_type == "docx"
    assert len(doc.pages) >= 2
    assert any("Swiss Army" in p.text for p in doc.pages)


def test_pptx_extractor():
    extractor = PptxExtractor()
    doc = extractor.extract(FIXTURES_DIR / "sample.pptx")
    assert doc.file_type == "pptx"
    assert len(doc.pages) == 1
    assert "Taking Out Principle" in doc.pages[0].text
