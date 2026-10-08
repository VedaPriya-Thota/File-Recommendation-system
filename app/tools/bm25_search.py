"""
bm25_search -- genuine BM25 lexical retrieval (architecture_final.md §6, §10;
Phase 2A).

Library: `bm25s` (real BM25, not ILIKE/ts_rank/pg_trgm dressed up as BM25).

Indexing decision (Phase 2A brief, Step 5):
    Each file is indexed as a single weighted text field --
    filename (x3) + tags (x2) + summary + extracted_text -- concatenated
    before tokenization. Repetition is the weighting mechanism: repeated
    terms get a higher term frequency and therefore a higher BM25
    contribution, without needing a multi-field BM25 variant. Filename is
    weighted highest because in this corpus a filename match is almost
    always the strongest lexical signal. `extracted_text` is included for
    forward-compatibility (currently unpopulated in seed data, per the
    Phase 1.1 report) -- once ingestion populates it, bm25_search benefits
    with no code change. No chunking (that is future work, §17).

Access control (architecture_final.md §11, non-negotiable):
    The corpus is fetched with `owner_id = ANY(%s)` applied in SQL --
    identical predicate to metadata_search -- BEFORE any tokenization or
    indexing happens. No unscoped row is ever part of the BM25 corpus, so
    there is no code path where all users' files are indexed and then
    filtered afterward. `file_type`/`date_range` are ordinary (non-access-
    control) constraints and are applied via bm25s's `weight_mask` *within*
    this already-scoped index -- they narrow results, they never widen the
    corpus beyond user_scope.

Index lifecycle (Step 6):
    A small in-process cache keyed by the caller's `owner_ids`. Every call
    still does one cheap scoped SQL fetch (correctness/access-control must
    never depend on a stale cache), but the expensive part -- tokenizing
    and building the BM25 index -- is skipped and reused whenever the
    fetched rows are identical (by count + file_id set + max(updated_at))
    to what's cached for that scope. This avoids rebuilding the index on
    every single query while keeping the corpus provably fresh and scoped.
"""
import re
from typing import List, Optional

import bm25s
import numpy as np

from app.db.connection import get_cursor
from app.scope import UserScope
from app.tools.registry import Tool, ToolResult

TOOL_NAME = "bm25_search"
DEFAULT_TOP_K = 20
_FILENAME_WEIGHT = 3
_TAGS_WEIGHT = 2

# owner_ids tuple -> (fingerprint, BM25 retriever | None, rows used to build it)
_INDEX_CACHE = {}


def _fetch_scoped_rows(owner_ids) -> List[dict]:
    sql = """
        SELECT file_id, filename, file_type, file_path, owner_id,
               created_at, updated_at, tags, summary, extracted_text
        FROM files
        WHERE owner_id = ANY(%(owner_ids)s)
        ORDER BY file_id;
    """
    with get_cursor() as cur:
        cur.execute(sql, {"owner_ids": list(owner_ids)})
        return [dict(row) for row in cur.fetchall()]


def _fingerprint(rows: List[dict]):
    if not rows:
        return (0, None, ())
    return (
        len(rows),
        max(r["updated_at"] for r in rows),
        tuple(sorted(r["file_id"] for r in rows)),
        # Phase 10: ingested content can change without touching the scope's max(updated_at)
        # (e.g. an older file edited, or a restored mtime); include a content signal so the
        # cached index is never served stale. Same single BM25 implementation.
        hash(tuple((r["file_id"], len(r.get("extracted_text") or "")) for r in rows)),
    )


_FILENAME_SPLIT_RE = re.compile(r"[_\-.]+")


def _normalize_filename(filename: str) -> str:
    """bm25s's default tokenizer treats `_`/`-` as word characters, so a raw
    "AI_Architecture_Overview.pptx" would tokenize as one opaque blob
    ("ai_architecture_overview") that a query for "architecture" alone
    could never match. Splitting on the filename's own separators first
    makes filename terms matchable the same way tags/summary words are."""
    return " ".join(t for t in _FILENAME_SPLIT_RE.split(filename) if t)


def _doc_text(row: dict) -> str:
    filename = _normalize_filename(row.get("filename") or "")
    tags = " ".join(row.get("tags") or [])
    summary = row.get("summary") or ""
    extracted = row.get("extracted_text") or ""
    parts = [filename] * _FILENAME_WEIGHT + [tags] * _TAGS_WEIGHT + [summary, extracted]
    return " ".join(p for p in parts if p)


def _get_or_build_index(owner_ids):
    """Returns (retriever | None, rows) for this scope. `retriever` is None
    (and `rows` empty) when the scope has no accessible files at all."""
    rows = _fetch_scoped_rows(owner_ids)
    key = tuple(sorted(owner_ids))
    fp = _fingerprint(rows)

    cached = _INDEX_CACHE.get(key)
    if cached is not None and cached[0] == fp:
        return cached[1], cached[2]

    if not rows:
        _INDEX_CACHE[key] = (fp, None, [])
        return None, []

    texts = [_doc_text(r) for r in rows]
    corpus_tokens = bm25s.tokenize(texts, stopwords="en", show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    _INDEX_CACHE[key] = (fp, retriever, rows)
    return retriever, rows


def _snippet(row: dict) -> str:
    text = row.get("extracted_text") or row.get("summary") or ""
    return text[:200]


def _build_mask(rows: List[dict], args: dict) -> Optional[np.ndarray]:
    file_type = args.get("file_type")
    date_from = args.get("date_from")
    date_to = args.get("date_to")
    if not (file_type or date_from or date_to):
        return None

    mask = np.ones(len(rows), dtype="float32")
    for i, row in enumerate(rows):
        if file_type and row["file_type"] != file_type:
            mask[i] = 0.0
            continue
        if date_from and row["updated_at"] < date_from:
            mask[i] = 0.0
            continue
        if date_to and row["updated_at"] > date_to:
            mask[i] = 0.0
            continue
    return mask


def _unique_terms(tokenized) -> set:
    """Distinct query terms (strings) after the same stop-word removal bm25s applies."""
    vocab = {i: w for w, i in tokenized.vocab.items()}
    return {vocab[i] for i in tokenized.ids[0]}


def _term_coverage(query_terms: set, row: dict) -> Optional[float]:
    """Fraction of the query's distinct content terms present in this document's searchable text
    (same weighted document text, same tokenizer/stop-words as retrieval). None if the query has no terms."""
    if not query_terms:
        return None
    doc = bm25s.tokenize([_doc_text(row)], stopwords="en", show_progress=False)
    ids = set(doc.ids[0])
    present = {w for w, i in doc.vocab.items() if i in ids}
    return round(len(query_terms & present) / len(query_terms), 4)


def bm25_search(args: dict, user_scope: UserScope) -> ToolResult:
    query = (args.get("query") or "").strip()
    if not query:
        return ToolResult(tool=TOOL_NAME, candidates=[])

    retriever, rows = _get_or_build_index(user_scope.owner_ids)
    if retriever is None or not rows:
        return ToolResult(tool=TOOL_NAME, candidates=[])

    mask = _build_mask(rows, args)
    query_tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
    query_terms = _unique_terms(query_tokens)
    top_k = min(int(args.get("top_k") or DEFAULT_TOP_K), len(rows))

    doc_indices, scores = retriever.retrieve(
        query_tokens, k=top_k, show_progress=False, weight_mask=mask
    )
    doc_indices, scores = doc_indices[0], scores[0]
    max_score = float(scores.max()) if len(scores) else 0.0

    candidates = []
    for idx, raw_score in zip(doc_indices, scores):
        raw_score = float(raw_score)
        # bm25s pads top-k with zero-score docs when fewer than k docs
        # genuinely match; a zero score is not lexical evidence.
        if raw_score <= 0.0:
            continue
        row = rows[int(idx)]
        # Per-request min-max-style normalization against the top hit
        # (architecture_final.md §10's normalization principle, applied
        # here per-source since final cross-source fusion is deferred).
        normalized = round(raw_score / max_score, 4) if max_score > 0 else 0.0
        candidates.append(
            {
                # Phase 12: ABSOLUTE lexical strength. `score` is divided by the query's own best hit, so the
                # top hit is always 1.0 whether it matched every query term or a single function word.
                # `coverage` = fraction of the query's distinct content terms this document contains.
                "coverage": _term_coverage(query_terms, row),
                "file_id": row["file_id"],
                "metadata": {
                    "filename": row["filename"],
                    "file_type": row["file_type"],
                    "file_path": row["file_path"],
                    "owner_id": row["owner_id"],
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat(),
                    "tags": row["tags"],
                    "summary": row["summary"],
                    "snippet": _snippet(row),
                },
                "score": normalized,
                "source": TOOL_NAME,
            }
        )

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return ToolResult(tool=TOOL_NAME, candidates=candidates)


def make_tool() -> Tool:
    return Tool(
        name=TOOL_NAME,
        description=(
            "Genuine BM25 lexical retrieval (bm25s) over a weighted "
            "filename/tags/summary/extracted_text document per file, "
            "restricted to the caller's user_scope at corpus-fetch time. "
            "Best for free-text/conceptual queries with a lexical anchor; "
            "supports optional file_type/date_from/date_to constraints."
        ),
        fn=bm25_search,
    )
