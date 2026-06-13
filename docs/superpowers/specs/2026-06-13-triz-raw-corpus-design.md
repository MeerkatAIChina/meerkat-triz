# TRIZ-raw Corpus Builder Design

**Date:** 2026-06-13  
**Status:** Approved  
**Approach:** Structured extraction + semantic chunking (Approach B)

## Problem

`TRIZ-raw/` contains ~300 raw TRIZ project files (PDFs, PPTX slides, DOCX/DOC documents, images, videos) accumulated from courses, conferences, case studies, and training materials. These assets are not yet in a format usable for model training.

## Goal

Convert the raw TRIZ materials into a single, structured, quality-gated `.jsonl` corpus suitable for causal language model continued pre-training. Each record preserves source metadata so the corpus can later be converted into supervised fine-tuning (SFT) seed data if needed.

## Decisions

| Decision | Rationale |
|----------|-----------|
| Output format: structured `.jsonl` with metadata | Enables both pre-training and future SFT conversion |
| Default raw path: `BASE_DIR / "TRIZ-raw"` | Works on DGX Spark; path is configurable |
| Chunking: semantic merge-up-to-target | Preserves TRIZ concepts and document structure better than fixed windows |
| Target chunk size: ~2048 tokens, hard cap 4096 | Matches `DATA_CONFIG["chatml"]["max_length"]`; chunks are merged until adding the next segment would exceed 2048 tokens, then a new chunk starts |
| File types supported: PDF, DOCX, PPTX, DOC | Covers the bulk of useful text; images/video skipped |
| OCR fallback for scanned PDFs | Avoids silently losing content from image-based PDFs |

## Architecture

```text
TRIZ-raw/
├── 40个发明原理/
├── Proceedings/
├── 2019年创新方法培训班课件/
└── ...
        │
        ▼
Dispatcher (by file extension)
        │
        ├── PDFExtractor
        ├── DocxExtractor
        ├── PptxExtractor
        └── DocExtractor (legacy, best-effort)
        │
        ▼
ExtractedDocument
{ title, source_path, category, pages: [{text, page_num, heading}] }
        │
        ▼
SemanticChunker
- Split on headings / page breaks
- Merge short adjacent segments up to target token count
- Drop empty / near-empty chunks
        │
        ▼
CorpusWriter
- Enrich metadata
- Write triz_corpus.jsonl
- Emit triz_corpus_stats.json
- Emit failed_files.json
```

A Jupyter notebook `notebooks/02c_build_corpus_from_raw.ipynb` orchestrates the pipeline end-to-end on DGX Spark.

## Components

### `utils/corpus_builder.py`

| Component | Responsibility |
|-----------|----------------|
| `Extractor` protocol | Common interface: `extract(path) -> ExtractedDocument` |
| `PDFExtractor` | Extract text + bounding boxes via `pymupdf`; OCR fallback via `pytesseract` |
| `DocxExtractor` | Extract paragraphs and headings via `python-docx`; map heading styles |
| `PptxExtractor` | Extract slide text via `python-pptx`; one slide ≈ one page |
| `DocExtractor` | Legacy `.doc` support via `antiword`/`textract`; skip if unavailable |
| `SemanticChunker` | Split and merge segments into target-size chunks |
| `CorpusWriter` | Write JSONL + statistics + failure report |

### `notebooks/02c_build_corpus_from_raw.ipynb`

Six-cell notebook:
1. Imports and config
2. Validate `TRIZ-raw/` path and list files
3. Extract all documents
4. Semantic chunking
5. Quality gates
6. Save, inspect stats, and sample records

## Data Flow

1. **Discover files** — recursively walk `TRIZ-raw/`, filter by extension.
2. **Dispatch extractor** — map `.pdf`/`.docx`/`.pptx`/`.doc` to the matching extractor.
3. **Normalize text** — collapse whitespace, preserve CJK characters, strip page numbers/footers heuristically.
4. **Chunk semantically** — split on headings and paragraph breaks; merge small segments until near target token count.
5. **Enrich metadata** — attach source path, category (parent folder), file type, page/slide number, heading, chunk index, token count.
6. **Apply quality gates** — drop short/empty/duplicate/garbled chunks.
7. **Write outputs** — `triz_corpus.jsonl`, `triz_corpus_stats.json`, `failed_files.json`.

## Output Schema

```json
{
  "id": "triz-raw-000042",
  "text": "局部质量原理（Local Quality）...",
  "metadata": {
    "source_path": "TRIZ-raw/40个发明原理/40个发明原理详解(带详细案例).docx",
    "category": "40个发明原理",
    "file_type": "docx",
    "page_num": 12,
    "heading": "原理3：局部质量",
    "chunk_index": 3,
    "token_count": 1876
  }
}
```

## Error Handling & Quality Gates

### Per-file error handling
- Extractors catch exceptions, log the file path, and return an empty `ExtractedDocument`.
- The pipeline continues; failures are collected in `failed_files.json`.

### Quality gates (configurable in `CORPUS_CONFIG`)
- **Min chunk length:** Drop chunks < 50 characters.
- **Max chunk length:** Hard cap at 4096 tokens.
- **Deduplication:** Drop exact-duplicate text chunks within the same category.
- **Language filter:** Optional regex to drop chunks with excessive non-CJK/non-ASCII symbols.
- **File-type skip:** Skip `.jpg`, `.png`, `.jfif`, `.mov`, `.mp4`, `.webp`, `.zip`, `.rar`, `.xlsx`, `.xls` by default.

### OCR fallback
- If a PDF page yields < 20 characters, attempt OCR once.
- If OCR fails, mark the page empty and continue.

### Resumability
- Track processed `source_path` values in the output JSONL.
- On restart, skip files already represented in the corpus.

## Testing

`tests/test_corpus_builder.py`:
- Fixture files: one `.pdf`, one `.docx`, one `.pptx` in `tests/fixtures/`.
- Assert chunk count, metadata schema, and empty/garbage file handling.
- Unit-test `SemanticChunker` with synthetic paragraph lists.

## Integration

- Add `CORPUS_CONFIG` to `config.py` with raw path, output path, chunk defaults, and quality thresholds.
- Add the notebook as `notebooks/02c_build_corpus_from_raw.ipynb` in the existing notebook sequence (after `02b_synthetic_generation.ipynb`).
- Output `triz_corpus.jsonl` can feed a causal LM pre-training script or be converted to SFT format via `convert_to_chatml()` later.

## Dependencies

Add to `requirements.txt`:

```text
pymupdf>=1.23.0
python-docx>=0.8.11
python-pptx>=0.6.21
pytesseract>=0.3.10
```

System dependency (noted in notebook):

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim
```

## Open Questions / Next Steps

1. Confirm `TRIZ-raw/` is copied to `/home/meerkat/mongoose_ai/TRIZ-raw/` on DGX Spark.
2. Decide whether to run OCR on all PDFs or only on text-empty pages.
3. After first corpus build, review `triz_corpus_stats.json` and adjust chunk size / quality gates if needed.
4. Decide if the corpus will be used for continued pre-training before or after the planned QLoRA SFT run.
