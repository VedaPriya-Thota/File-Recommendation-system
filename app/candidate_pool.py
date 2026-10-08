"""
candidate_pool accumulation -- architecture_final.md §4, §6.

Merges a tool's evidence into the running candidate_pool rather than
replacing it, so evidence from multiple actions/iterations on the same
file_id accumulates instead of duplicating. Only scores/metadata
references are kept -- no raw tool payloads.
"""
from typing import Dict

from app.final_ranking import PERSONALIZATION_SOURCE
from app.tools.registry import ToolResult


def update_candidate_pool(
    candidate_pool: Dict[int, dict], tool_result: ToolResult, iteration: int
) -> Dict[int, dict]:
    pool = dict(candidate_pool)  # shallow copy -- functional update

    for c in tool_result.candidates:
        file_id = c["file_id"]
        existing = pool.get(file_id)
        if existing is None:
            # Phase 8 structural guard (architecture_final.md §8 anti-circularity):
            # personalization is evidence-only. It may add its score to a candidate
            # that independent retrieval already put in the pool, but can NEVER
            # create a pool entry -- regardless of what the tool returned. (The
            # real get_user_activity tool already behaves this way; this makes the
            # guarantee hold at the merge, not just by the tool's good behaviour.)
            if c["source"] == PERSONALIZATION_SOURCE:
                continue
            pool[file_id] = {
                "evidence_sources": [c["source"]],
                "scores": {c["source"]: c["score"]},
                "coverage": ({c["source"]: c["coverage"]} if c.get("coverage") is not None else {}),
                "metadata": c["metadata"],
                "first_seen_iteration": iteration,
                "last_updated_iteration": iteration,
            }
        else:
            sources = list(existing["evidence_sources"])
            if c["source"] not in sources:
                sources.append(c["source"])
            scores = dict(existing["scores"])
            scores[c["source"]] = c["score"]
            coverage = dict(existing.get("coverage") or {})  # Phase 12: per-source absolute lexical strength
            if c.get("coverage") is not None:
                coverage[c["source"]] = c["coverage"]
            pool[file_id] = {
                "evidence_sources": sources,
                "scores": scores,
                "coverage": coverage,
                "metadata": c["metadata"],  # refreshed metadata is fine (same file)
                "first_seen_iteration": existing["first_seen_iteration"],
                "last_updated_iteration": iteration,
            }

    return pool
