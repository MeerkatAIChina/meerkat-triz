---
phase: 03-evaluation-and-hardening
plan: 01
status: complete
completed: 2026-05-29
---

## Summary

Created the unified `format_messages()` utility in `data_utils.py` and added the `sacrebleu` dependency to `requirements.txt`. This utility replaces all hardcoded ChatML strings across the codebase by centralizing prompt formatting via `tokenizer.apply_chat_template()`.

## What Was Built

- **`format_messages()`** in `data_utils.py` — a reusable function that formats chat messages using the tokenizer's native chat template. Supports inference mode (`add_generation_prompt=True`) and training mode (`add_generation_prompt=False`). Pulls the default TRIZ expert system message from `DATA_CONFIG['chatml']['system_message']`.
- **`sacrebleu==2.4.3`** in `requirements.txt` — research-standard BLEU implementation with Chinese tokenization support.

## Commits

- `0fec50f` — feat(03-01): add format_messages() utility and sacrebleu dependency

## Key Files

| File | Change |
|------|--------|
| `ref/mongoose_ai_dgx/utils/data_utils.py` | Added `format_messages()` function after `_build_chatml_hardcoded()` |
| `ref/mongoose_ai_dgx/requirements.txt` | Added `sacrebleu==2.4.3` after `rouge-score==0.1.2` |

## Verification

- ✓ `grep "def format_messages"` — matches
- ✓ `grep "from config import DATA_CONFIG"` — matches
- ✓ `grep "add_generation_prompt"` — 4 matches (>= 2 required)
- ✓ `grep "apply_chat_template"` — 8 matches (>= 2 required)
- ✓ `grep "sacrebleu==2.4.3"` — matches

## Self-Check: PASSED
