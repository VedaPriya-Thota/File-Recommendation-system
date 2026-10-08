"""
rerank_candidates -- deterministic cross-encoder reranking of an existing
candidate set (architecture_final.md §6 `rerank_candidates`; Phase 3).

This tool NEVER performs a new corpus search and NEVER touches Postgres or
Qdrant. It receives the candidates already produced by prior (authorized)
retrieval actions -- via `args["candidates"]`, built from the current
`candidate_pool` by `plan_next_action`/`candidates_from_pool` below -- and
returns them in cross-encoder order. No candidate outside the supplied set
can ever appear in the result: the tool has no code path that looks up,
queries, or fetches anything beyond what it was handed
(architecture_final.md §6/§11 -- reranking must never expand the candidate
set or scope; it inherits access control entirely from the tools that
produced the candidates it's given, the same inheritance pattern
hybrid_search uses for bm25_search/semantic_search).

Candidate evidence (prompt §4): each candidate's "strongest available
evidence" is picked from its accumulated metadata, in this priority order:
    1. semantic matched chunk (`matched_chunk` -- semantic_search's own key
       -- or `semantic_matched_chunk` -- hybrid_search's fused-metadata key
       for the same evidence)
    2. bm25 snippet (`snippet` -- bm25_search's own key -- or
       `bm25_snippet` -- hybrid_search's fused-metadata key)
    3. summary
    4. tags + filename (last-resort metadata representation; never
       fabricated content)

Score handling (prompt §5, architecture_final.md §10's normalization
principle): the raw cross-encoder score is min-max normalized to [0, 1]
per-request (across just the supplied candidate set, not globally
calibrated) -- the same "per-source normalization" convention
bm25_search/semantic_search already use, and the normalization
architecture_final.md §10 specifies for rerank scores. The *normalized*
score is what becomes this candidate's `rerank_candidates` entry in
candidate_pool's per-source `scores` dict; the raw score is kept alongside
in metadata (`rerank_raw_score`) for observability. Neither raw nor
normalized score is combined with (summed into) bm25/semantic/RRF scores
anywhere -- they remain separate per-source entries, exactly like every
other tool's score, and neither is claimed to be a calibrated probability.
"""
from typing import Dict, List, Optional

from app.reranker import score_pairs
from app.scope import UserScope
from app.tools.registry import Tool, ToolResult

TOOL_NAME = "rerank_candidates"


def _evidence_text(metadata: dict) -> str:
    for key in ("matched_chunk", "semantic_matched_chunk"):
        val = metadata.get(key)
        if val:
            return val
    for key in ("snippet", "bm25_snippet"):
        val = metadata.get(key)
        if val:
            return val
    summary = metadata.get("summary")
    if summary:
        return summary
    tags = metadata.get("tags") or []
    filename = metadata.get("filename") or ""
    return " ".join(list(tags) + [filename]).strip()


def candidates_from_pool(candidate_pool: Dict[int, dict]) -> List[dict]:
    """Builds the `args["candidates"]` shape expected by tools that operate
    on an already-retrieved candidate set rather than performing a fresh
    search -- rerank_candidates (Phase 3) and, reusing this same helper,
    get_user_activity (Phase 4A; see app/tools/get_user_activity.py). Kept
    here (not in app/candidate_pool.py) deliberately: candidate_pool.py
    imports app.tools.registry, and app.tools.registry's package import
    (app/tools/__init__.py) imports every tool module including this one --
    defining this helper in candidate_pool.py and importing it back from a
    tool module would be a circular import. file_id + metadata only, so
    callers never need to know candidate_pool's internal accumulation
    shape (evidence_sources/scores/iteration bookkeeping)."""
    return [
        {"file_id": file_id, "metadata": entry["metadata"]}
        for file_id, entry in candidate_pool.items()
    ]


def rerank_candidates(args: dict, user_scope: UserScope) -> ToolResult:
    query = (args.get("query") or "").strip()
    candidates = args.get("candidates") or []

    if not query or not candidates:
        return ToolResult(tool=TOOL_NAME, candidates=[])

    texts = [_evidence_text(c.get("metadata") or {}) for c in candidates]
    raw_scores = score_pairs(query, texts)

    lo, hi = min(raw_scores), max(raw_scores)
    spread = hi - lo

    reranked = []
    for c, raw_score in zip(candidates, raw_scores):
        normalized = round((raw_score - lo) / spread, 4) if spread > 0 else 1.0
        metadata = dict(c.get("metadata") or {})
        metadata["rerank_raw_score"] = raw_score
        reranked.append(
            {
                "file_id": c["file_id"],
                "metadata": metadata,
                "score": normalized,
                "source": TOOL_NAME,
            }
        )

    reranked.sort(key=lambda c: c["score"], reverse=True)

    top_k = args.get("top_k")
    if top_k is not None:
        reranked = reranked[: int(top_k)]

    return ToolResult(tool=TOOL_NAME, candidates=reranked)


def make_tool() -> Tool:
    return Tool(
        name=TOOL_NAME,
        description=(
            "Deterministic cross-encoder reranking of an already-retrieved "
            "candidate set. Never performs a new corpus search and cannot "
            "introduce a file outside the supplied candidates -- it only "
            "reorders evidence the caller already has. Best when "
            "reflection reports multiple plausible/close candidates after "
            "retrieval (a flat/tied pool), not as a first-choice action. "
            "Requires `query` and `candidates` (file_id + metadata pairs, "
            "e.g. from the current candidate_pool); supports optional "
            "`top_k`."
        ),
        fn=rerank_candidates,
    )
