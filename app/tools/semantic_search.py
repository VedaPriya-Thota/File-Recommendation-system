"""
semantic_search -- genuine semantic retrieval via local embeddings + Qdrant
(architecture_final.md §6, §10; Phase 2B).

Embeddings: see app/embeddings.py for the full provider story (production
target: sentence-transformers/BAAI/bge-small-en-v1.5, 384-dim; this sandbox
falls back to a real local model -- spaCy en_core_web_md, 300-dim -- since
Hugging Face is unreachable here). Similarity metric: cosine, always.

Chunking decision:
    `files.extracted_text` is split into fixed-size, non-overlapping,
    whitespace-word chunks (SEMANTIC_CHUNK_WORDS words each, default 120).
    No sentence-aware/semantic chunking, no overlap -- deterministic and
    simple, appropriate for the demo/small-team corpus scale
    (architecture_final.md §19). A file with empty/NULL extracted_text
    produces zero chunks and is never embedded or indexed (explicit, not
    an error -- most seed files have no extracted_text yet per the
    Phase 1.1 report, so this is the common case today).

Qdrant point payload (at least the required fields, plus a couple more
needed for filtering/candidate-building):
    file_id, chunk_id, owner_id, filename, file_type, content_text,
    updated_at_ts (unix seconds, for date_from/date_to filtering).
Point id = file_id * 100000 + chunk_id (deterministic -- re-running
rebuild_index() upserts/overwrites rather than duplicating points).

Access control (architecture_final.md §11, non-negotiable):
    owner_id is applied as a Qdrant payload `Filter` (MatchAny over the
    caller's user_scope.owner_ids) passed directly into `query_points`.
    Qdrant applies this filter as part of the ANN search itself -- there
    is no code path that searches all owners' vectors and discards
    unauthorized hits afterward. `file_type`/date constraints are ordinary
    (non-access-control) filters ANDed into the same Filter object.
    Belt-and-suspenders: candidate file metadata is re-fetched from
    Postgres with the SAME `owner_id = ANY(...)` predicate used by
    metadata_search/bm25_search, so a candidate can only reach the
    candidate pool if it clears scope at two independent layers.

File aggregation:
    Qdrant is queried at chunk granularity (SEMANTIC_CHUNK_QUERY_LIMIT
    chunks, to have enough chunks on hand to cover top_k distinct files);
    results are grouped by file_id, each file's score = its single best
    (maximum) matching chunk's score, and that chunk's content_text is
    kept as the candidate's evidence snippet. One file -> one candidate,
    per the brief. No RRF / cross-source fusion (deferred).

Index lifecycle:
    `rebuild_index(force=False)` is the explicit build/rebuild entrypoint
    (safe to call any time; upserts are idempotent). `semantic_search()`
    calls a lazy `_ensure_index_ready()` once per process that triggers a
    build only if the collection is empty, so a normal query never pays
    an unnecessary rebuild.
"""
import logging
from typing import Dict, List, Optional

from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchAny, MatchValue, PointStruct, Range

from app.config import SEMANTIC_CHUNK_QUERY_LIMIT, SEMANTIC_CHUNK_WORDS, QDRANT_COLLECTION
from app.db.connection import get_cursor
from app.embeddings import embed_texts
from app.scope import UserScope
from app.tools.registry import Tool, ToolResult
from app.vectorstore import ensure_collection, get_client

logger = logging.getLogger("file_discovery_agent.semantic_search")

TOOL_NAME = "semantic_search"
DEFAULT_TOP_K = 10

_build_attempted = False


def _chunk_text(text: str) -> List[str]:
    words = (text or "").split()
    if not words:
        return []
    return [
        " ".join(words[i : i + SEMANTIC_CHUNK_WORDS])
        for i in range(0, len(words), SEMANTIC_CHUNK_WORDS)
    ]


def _fetch_all_files_for_indexing() -> List[dict]:
    """Unlike bm25_search's per-query scoped fetch, indexing is a one-time
    global build across all owners -- owner_id travels into each point's
    payload and access control is enforced at QUERY time (see module
    docstring), not by restricting what gets indexed."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT file_id, filename, file_type, owner_id, updated_at, extracted_text
            FROM files
            ORDER BY file_id;
            """
        )
        return [dict(row) for row in cur.fetchall()]


MAX_CHUNKS_PER_FILE = 100000  # point ids are file_id * 100000 + chunk_id; more chunks would collide with the next file


def _points_for_row(row: dict) -> List[PointStruct]:
    """The ONE place a file's chunks become Qdrant points (used by both the
    global rebuild and per-file ingestion). Empty text -> [] (never embedded)."""
    chunks = _chunk_text(row.get("extracted_text"))
    if not chunks:
        return []
    if len(chunks) >= MAX_CHUNKS_PER_FILE:
        raise ValueError("too many chunks for deterministic point ids")
    vectors = embed_texts(chunks)
    updated_at_ts = row["updated_at"].timestamp()
    return [
        PointStruct(
            id=row["file_id"] * 100000 + chunk_id,
            vector=vector,
            payload={
                "file_id": row["file_id"],
                "chunk_id": chunk_id,
                "owner_id": row["owner_id"],
                "filename": row["filename"],
                "file_type": row["file_type"],
                "content_text": chunk_text,
                "updated_at_ts": updated_at_ts,
            },
        )
        for chunk_id, (chunk_text, vector) in enumerate(zip(chunks, vectors))
    ]


def delete_file_points(file_id: int) -> None:
    """Remove EVERY chunk point of one file (any chunk count), so a deleted or
    shrunken file leaves no stale searchable evidence behind."""
    ensure_collection()
    get_client().delete(
        QDRANT_COLLECTION,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="file_id", match=MatchValue(value=file_id))])
        ),
    )


def index_file(file_id: int) -> int:
    """Incremental (re)index of ONE file from its committed `files` row: old
    points are deleted FIRST (a failed re-embed must never leave stale content
    searchable), then the current chunks are upserted. Returns the chunk count.
    Same chunking, embeddings, ids and payload as rebuild_index()."""
    delete_file_points(file_id)
    with get_cursor() as cur:
        cur.execute(
            "SELECT file_id, filename, file_type, owner_id, updated_at, extracted_text FROM files WHERE file_id = %s;",
            (file_id,),
        )
        row = cur.fetchone()
    if row is None:
        return 0
    points = _points_for_row(dict(row))
    if points:
        get_client().upsert(QDRANT_COLLECTION, points=points)
    return len(points)


def rebuild_index(force: bool = False) -> dict:
    """Explicit build/rebuild entrypoint. Returns {"files_indexed",
    "chunks_indexed"}. Idempotent: point ids are deterministic
    (file_id*100000 + chunk_id), so re-running without `force` just
    upserts. `force=True` drops and recreates the collection first (e.g.
    after the embedding provider/dimension changed)."""
    global _build_attempted
    client = get_client()
    if force and client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)
    ensure_collection()

    rows = _fetch_all_files_for_indexing()
    points = []
    files_indexed = 0
    for row in rows:
        row_points = _points_for_row(row)
        if not row_points:
            continue  # empty extracted_text -> no chunks, not indexed
        files_indexed += 1
        points.extend(row_points)
    if points:
        client.upsert(QDRANT_COLLECTION, points=points)
    _build_attempted = True
    logger.info("semantic index rebuilt: %d files, %d chunks", files_indexed, len(points))
    return {"files_indexed": files_indexed, "chunks_indexed": len(points)}


def _ensure_index_ready() -> None:
    global _build_attempted
    ensure_collection()
    if _build_attempted:
        return
    client = get_client()
    count = client.count(QDRANT_COLLECTION, exact=True).count
    if count == 0:
        rebuild_index(force=False)
    _build_attempted = True


def _build_filter(owner_ids, args: dict) -> Filter:
    must = [FieldCondition(key="owner_id", match=MatchAny(any=list(owner_ids)))]
    file_type = args.get("file_type")
    if file_type:
        must.append(FieldCondition(key="file_type", match=MatchValue(value=file_type)))
    date_from = args.get("date_from")
    date_to = args.get("date_to")
    if date_from or date_to:
        must.append(
            FieldCondition(
                key="updated_at_ts",
                range=Range(
                    gte=date_from.timestamp() if date_from else None,
                    lte=date_to.timestamp() if date_to else None,
                ),
            )
        )
    return Filter(must=must)


def _fetch_scoped_metadata(file_ids: List[int], owner_ids) -> Dict[int, dict]:
    if not file_ids:
        return {}
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT file_id, filename, file_type, file_path, owner_id,
                   created_at, updated_at, tags, summary
            FROM files
            WHERE file_id = ANY(%(file_ids)s) AND owner_id = ANY(%(owner_ids)s);
            """,
            {"file_ids": file_ids, "owner_ids": list(owner_ids)},
        )
        return {row["file_id"]: dict(row) for row in cur.fetchall()}


def semantic_search(args: dict, user_scope: UserScope) -> ToolResult:
    query = (args.get("query") or "").strip()
    if not query:
        return ToolResult(tool=TOOL_NAME, candidates=[])

    _ensure_index_ready()
    client = get_client()

    [query_vector] = embed_texts([query])
    qfilter = _build_filter(user_scope.owner_ids, args)
    top_k = int(args.get("top_k") or DEFAULT_TOP_K)
    chunk_limit = max(SEMANTIC_CHUNK_QUERY_LIMIT, top_k)

    hits = client.query_points(
        QDRANT_COLLECTION, query=query_vector, query_filter=qfilter, limit=chunk_limit
    ).points

    # --- chunk -> file aggregation: file_score = max matching chunk score
    best_per_file: Dict[int, dict] = {}
    for hit in hits:
        payload = hit.payload
        fid = payload["file_id"]
        if fid not in best_per_file or hit.score > best_per_file[fid]["score"]:
            best_per_file[fid] = {"score": hit.score, "chunk_text": payload["content_text"]}

    if not best_per_file:
        return ToolResult(tool=TOOL_NAME, candidates=[])

    # Belt-and-suspenders re-scope at the Postgres layer (see module docstring).
    metadata_by_id = _fetch_scoped_metadata(list(best_per_file.keys()), user_scope.owner_ids)

    max_score = max(v["score"] for v in best_per_file.values())
    candidates = []
    for fid, evidence in best_per_file.items():
        row = metadata_by_id.get(fid)
        if row is None:
            continue  # Qdrant point existed but Postgres scope check failed -- excluded
        normalized = round(evidence["score"] / max_score, 4) if max_score > 0 else 0.0
        candidates.append(
            {
                "file_id": fid,
                "metadata": {
                    "filename": row["filename"],
                    "file_type": row["file_type"],
                    "file_path": row["file_path"],
                    "owner_id": row["owner_id"],
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat(),
                    "tags": row["tags"],
                    "summary": row["summary"],
                    "matched_chunk": evidence["chunk_text"],
                },
                "score": normalized,
                "source": TOOL_NAME,
            }
        )

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return ToolResult(tool=TOOL_NAME, candidates=candidates[:top_k])


def make_tool() -> Tool:
    return Tool(
        name=TOOL_NAME,
        description=(
            "Semantic retrieval via local embeddings + Qdrant, chunk-level "
            "search aggregated to one candidate per file (max matching "
            "chunk score). Best for conceptual queries worded differently "
            "than the stored text. Supports optional file_type/date_from/"
            "date_to constraints."
        ),
        fn=semantic_search,
    )
