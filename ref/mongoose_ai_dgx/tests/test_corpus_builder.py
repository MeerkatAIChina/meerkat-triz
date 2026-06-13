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


SemanticChunker = corpus_builder_mod.SemanticChunker
Chunk = corpus_builder_mod.Chunk


def test_semantic_chunker_basic():
    doc = ExtractedDocument(
        source_path="TRIZ-raw/sample.txt",
        category="test",
        file_type="txt",
        pages=[
            Page(text="Chapter 1\n\nSegmentation Principle: Divide an object into independent parts.\n\nTypical application: smartphone.\n\nChapter 2\n\nTaking Out Principle: Extract the harmful part from an object."),
        ],
    )
    chunker = SemanticChunker(target_tokens=100, chars_per_token=1.0)
    chunks, _ = chunker.chunk_document(doc)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert any("Segmentation Principle" in c.text for c in chunks)


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


CorpusWriter = corpus_builder_mod.CorpusWriter


def test_corpus_writer(tmp_path):
    chunks = [
        Chunk(
            text="Segmentation Principle: Divide an object into independent parts.",
            source_path="TRIZ-raw/sample.txt",
            category="test",
            file_type="txt",
            page_num=1,
            heading="Chapter 1",
            chunk_index=0,
            token_count=15,
        ),
        Chunk(
            text="Segmentation Principle: Divide an object into independent parts.",  # duplicate
            source_path="TRIZ-raw/sample.txt",
            category="test",
            file_type="txt",
            page_num=1,
            heading="Chapter 1",
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
