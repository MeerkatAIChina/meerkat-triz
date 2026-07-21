import sys
sys.path.append('/home/meerkat/mongoose_ai')

from config import CORPUS_CONFIG
from utils.corpus_builder import build_corpus

stats = build_corpus(
    raw_dir=CORPUS_CONFIG['raw_dir'],
    output_dir=CORPUS_CONFIG['output_dir'],
    output_filename=CORPUS_CONFIG['output_filename'],
    stats_filename=CORPUS_CONFIG['stats_filename'],
    failed_files_filename=CORPUS_CONFIG['failed_files_filename'],
    chunk_target_tokens=CORPUS_CONFIG['chunk']['target_tokens'],
    chunk_max_tokens=CORPUS_CONFIG['chunk']['max_tokens'],
    chars_per_token=CORPUS_CONFIG['chunk']['chars_per_token'],
    min_chars=CORPUS_CONFIG['quality_gates']['min_chars'],
    deduplicate=CORPUS_CONFIG['quality_gates']['deduplicate'],
    ocr_enabled=CORPUS_CONFIG['ocr']['enabled'],
    ocr_min_text_chars=CORPUS_CONFIG['ocr']['min_text_chars'],
    resume=False,
)
print('BUILD_STATS:', stats)
