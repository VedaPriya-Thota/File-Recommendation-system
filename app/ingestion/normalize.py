"""
Deterministic text normalization -- the single function whose output feeds BM25
(via files.extracted_text) and chunking/embeddings. No rewriting, no
summarisation, no LLM: only making equal text compare equal.
"""
import re
import unicodedata

_LINE_BREAKS = re.compile("[\u2028\u2029\x0b\x0c\x85]")
_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u00ad]")
# C0/C1 control characters except \t and \n (NUL in particular cannot be stored in Postgres TEXT)
_CONTROL = re.compile("[\x00-\x08\x0e-\x1f\x7f-\x9f]")
_HSPACE = re.compile(r"[^\S\n]+")
_AROUND_NL = re.compile(r" ?\n ?")
_BLANKS = re.compile(r"\n{3,}")


def normalize_text(text) -> str:
    """NFKC (ligatures/full-width forms/NBSP fold to plain characters), unified
    line endings, control + zero-width characters removed, runs of spaces/tabs
    collapsed, at most one blank line between paragraphs, trimmed. Idempotent.
    Whitespace-only input -> ""."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _LINE_BREAKS.sub("\n", text)
    text = _ZERO_WIDTH.sub("", text)
    text = _CONTROL.sub("", text)
    text = _HSPACE.sub(" ", text)
    text = _AROUND_NL.sub("\n", text)
    text = _BLANKS.sub("\n\n", text)
    return text.strip()
