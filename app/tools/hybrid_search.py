"""
hybrid_search -- BM25 + semantic retrieval fused with Reciprocal Rank
Fusion (RRF) (architecture_final.md §6, §10; Phase 2C).

RRF formula (standard, RANK-based -- never combines raw scores across
sources, which live on different, incomparable scales/distributions):

    RRF(d) = sum_i 1 / (RRF_K + rank_i(d))

RRF_K = 60 (app/config.py), the conventional constant from Cormack et al.
2009 -- not tuned here. Ranks are 1-based, taken directly from each
sub-tool's OWN already-sorted candidate order (both bm25_search and
semantic_search already return their candidates sorted by descending
score, so rank 1 = candidates[0], rank 2 = candidates[1], etc). A file
that appears in only one source's list gets only that source's term in
the sum -- the missing source is omitted, never treated as rank=0 or
score=0 (that would silently become a raw-score blend, which is exactly
what RRF is meant to avoid). A file in both lists gets both terms added.

RRF scores are explicitly NOT normalized to [0,1] and are NOT combined
with, or compared against, bm25_search's or semantic_search's own scores
anywhere -- they are a distinct scale (architecture_final.md §10's final
weighted formula, α·S_retrieval + β·S_recency + γ·S_affinity, is a
separate, still-deferred, later-phase step; RRF here is retrieval-time
fusion, not that).

No reimplementation: this module calls
`app.tools.bm25_search.bm25_search()` and
`app.tools.semantic_search.semantic_search()` directly and fuses their
results. Both sub-tools already enforce user_scope (bm25 via a scoped SQL
fetch, semantic via a Qdrant payload filter + a belt-and-suspenders scoped
Postgres re-fetch) -- hybrid_search performs no additional scope checks of
its own and adds no new access path; it inherits access control entirely
from the two tools it calls.
"""
from typing import Dict, List, Optional

from app.config import HYBRID_SUBQUERY_TOP_K, RRF_K
from app.scope import UserScope
from app.tools.bm25_search import bm25_search
from app.tools.registry import Tool, ToolResult
from app.tools.semantic_search import semantic_search

TOOL_NAME = "hybrid_search"
DEFAULT_TOP_K = 10

_COMMON_METADATA_FIELDS = (
    "filename",
    "file_type",
    "file_path",
    "owner_id",
    "created_at",
    "updated_at",
    "tags",
    "summary",
)


def _sub_args(args: dict) -> dict:
    """Args forwarded, unchanged, to BOTH bm25_search and semantic_search --
    same query text and same optional file_type/date_from/date_to
    constraints for both, so the two rankings being fused are answering
    the same question. `query` is passed through as-is (the caller's raw
    text); this module doesn't reinterpret or strip it -- that's each
    sub-tool's own concern (bm25_search tokenizes it, semantic_search
    embeds it), consistent with "do not modify the existing BM25 or
    semantic scoring implementations"."""
    sub = {"query": args.get("query"), "top_k": HYBRID_SUBQUERY_TOP_K}
    for key in ("file_type", "date_from", "date_to"):
        if args.get(key) is not None:
            sub[key] = args[key]
    return sub


def hybrid_search(args: dict, user_scope: UserScope) -> ToolResult:
    query = (args.get("query") or "").strip()
    if not query:
        return ToolResult(tool=TOOL_NAME, candidates=[])

    sub_args = _sub_args(args)
    bm25_result = bm25_search(sub_args, user_scope)
    semantic_result = semantic_search(sub_args, user_scope)

    # rank_i(d): 1-based position within each source's own ranked list.
    bm25_by_id = {c["file_id"]: c for c in bm25_result.candidates}
    semantic_by_id = {c["file_id"]: c for c in semantic_result.candidates}
    bm25_ranks = {fid: i + 1 for i, fid in enumerate(bm25_by_id)}
    semantic_ranks = {fid: i + 1 for i, fid in enumerate(semantic_by_id)}

    # sorted(): candidates tied on RRF score keep this iteration order through the stable sort below, and a set of
    # integers iterates in an order that depends on the ids' absolute values -- so the order of tied candidates
    # varied with the database's serial ids. Ascending file_id is the system-wide tie-break (as in final_rank).
    all_file_ids = sorted(set(bm25_by_id) | set(semantic_by_id))

    candidates = []
    for fid in all_file_ids:
        rrf_score = 0.0
        sources: List[str] = []
        evidence: Dict[str, object] = {}

        if fid in bm25_by_id:
            rank = bm25_ranks[fid]
            rrf_score += 1.0 / (RRF_K + rank)
            sources.append("bm25_search")
            evidence["bm25_rank"] = rank
            evidence["bm25_score"] = bm25_by_id[fid]["score"]
            evidence["bm25_snippet"] = bm25_by_id[fid]["metadata"].get("snippet")

        if fid in semantic_by_id:
            rank = semantic_ranks[fid]
            rrf_score += 1.0 / (RRF_K + rank)
            sources.append("semantic_search")
            evidence["semantic_rank"] = rank
            evidence["semantic_score"] = semantic_by_id[fid]["score"]
            evidence["semantic_matched_chunk"] = semantic_by_id[fid]["metadata"].get("matched_chunk")

        source_row = semantic_by_id.get(fid) or bm25_by_id.get(fid)
        metadata = {k: source_row["metadata"].get(k) for k in _COMMON_METADATA_FIELDS}
        metadata["retrieval_sources"] = sources
        metadata.update(evidence)

        candidates.append(
            {
                "file_id": fid,
                "metadata": metadata,
                "score": rrf_score,  # not normalized -- see module docstring
                "source": TOOL_NAME,
            }
        )

    candidates.sort(key=lambda c: c["score"], reverse=True)
    top_k = int(args.get("top_k") or DEFAULT_TOP_K)
    return ToolResult(tool=TOOL_NAME, candidates=candidates[:top_k])


def make_tool() -> Tool:
    return Tool(
        name=TOOL_NAME,
        description=(
            "Hybrid retrieval: fuses bm25_search and semantic_search "
            "results via Reciprocal Rank Fusion (RRF, k=60) -- rank-based, "
            "not a raw-score blend. Best when a query benefits from "
            "combining lexical and semantic signals, e.g. after each has "
            "independently returned a flat/tied candidate set. Supports "
            "optional file_type/date_from/date_to constraints."
        ),
        fn=hybrid_search,
    )
