"""
LLM-assisted result explanation -- Phase 5D (architecture_final.md §8
touchpoint 4 "Explanation" and §9 `generate_explanation`: "final synthesis
only, never influences ranking").

This is a PRESENTATION layer. It runs as the last graph stage
(`finish_stage -> compare_stage -> explain_stage -> END`), after
`final_rank` and Phase 5C have fully determined the result, and it returns
text only. It never retrieves, searches, runs a tool, reorders, rescores,
or touches candidates, the pool or `user_scope` -- the graph node that calls
it writes only `candidate_explanation` and `explanations`, never
`final_ranking`.

Invocation condition (`should_explain`)
---------------------------------------
    final_ranking is non-empty  AND  no explanation has been generated yet.
A clarification / no-result / error-only run has no ranked candidate, so no
LLM call is made and no explanation is fabricated. (The clarify path never
reaches this stage at all.) Only the single top-ranked result -- the one the
system actually returned first -- is explained.

Result binding
--------------
The LLM is never shown, and cannot return, a file ID (the schema has no such
field). The explained `file_id` is taken from `final_ranking[0]` by this
module and stored alongside the text in state, so the model has no way to
choose, change or introduce which file is being explained.

Grounding
---------
The prompt instructs the model to use only the supplied evidence and to say
so when evidence is insufficient. Because free text can't be proven
grounded, the deterministic checks are: (1) the model is shown only bounded,
whitelisted evidence -- no owner_id, file_path, user_scope, user_id, raw
records or unrelated candidates; and (2) an explanation that quotes any of
those withheld values is rejected as `invalid_output` (it can only have come
from outside the supplied evidence). Personalization is passed only when
it's a real signal (not cold-start, positive affinity).

Phase 5C interaction
--------------------
If comparison was actually APPLIED, the evidence says so (`selection_basis`)
and carries the 5C rationale as supporting evidence only; the model is told
the result is not simply the highest-scoring candidate. No comparison is
performed here.

Fallback
--------
Never raises. Unavailable provider, provider error, non-string / malformed /
schema-invalid / leaking output all return `None` plus metadata
(`unavailable` | `provider_error` | `invalid_output`); the run's ranking is
unaffected. Success is `generated`. No chain-of-thought is requested or
stored.
"""
import json
from typing import List, Optional, Tuple

from app.llm.candidate_comparer import MAX_TAGS, _truncate
from app.llm.provider import LLMProvider, LLMUnavailableError
from app.llm.schema import (
    LLMResponseInvalidError,
    ResultExplanation,
    parse_result_explanation_response,
)

MAX_QUERY_CHARS = 500
PERSONALIZATION_SOURCE = "get_user_activity"

# Whitelist of personalization signals ever forwarded, and only when the
# signal is meaningful (see `_meaningful_personalization`).
_PERSONALIZATION_KEYS = (
    "affinity_score",
    "preferred_file_type_match",
    "frequent_topic_score",
    "recency_score",
    "open_count",
    "view_count",
    "edit_count",
)

SYSTEM_PROMPT = (
    "You write a short user-facing explanation of why a file-search result was "
    "returned, for a result that a deterministic system has ALREADY selected. "
    "You do not search, rank, choose or change anything, and you have no access "
    "to files, databases or tools. Use ONLY the evidence in the message: do not "
    "invent file contents, dates, topics, ownership, past user behavior, or "
    "retrieval methods that are not listed. Mention a retrieval source, "
    "reranking or personalization only if the evidence shows it was used. If "
    "selection_basis says the result was chosen by a comparison of near-tied "
    "candidates, do not claim it has the highest score. If the evidence is "
    "insufficient to explain the match, say so plainly and use a low "
    "confidence. Describe evidence, not your reasoning steps. Respond with "
    "EXACTLY one JSON object and nothing else: no markdown fences, no prose "
    "outside the object. Allowed keys only: summary (1-2 sentences, at most 400 "
    "characters), evidence_points (a list of up to 5 short statements, each at "
    "most 200 characters, each stating a concrete fact from the evidence), and "
    "confidence (a number from 0 to 1 for how well the evidence supports the "
    "explanation)."
)


def should_explain(final_ranking: List[dict], existing_explanation: Optional[dict] = None) -> bool:
    if not final_ranking:
        return False
    return not (existing_explanation and existing_explanation.get("explanation_status") == "generated")


def _meaningful_personalization(entry: dict) -> dict:
    metadata = entry.get("metadata") or {}
    p = entry.get("personalization") or metadata.get("personalization") or {}
    try:
        affinity = float(p.get("affinity_score") or 0.0)
    except (TypeError, ValueError):
        return {}
    if p.get("cold_start") or affinity <= 0.0:
        return {}
    return {k: p[k] for k in _PERSONALIZATION_KEYS if k in p}


def build_explanation_evidence(
    raw_query: str, goal: dict, entry: dict, comparison: Optional[dict] = None
) -> dict:
    """The exact, bounded evidence the LLM sees for the ONE explained
    result. No file_id, owner_id, file_path, user_scope, timestamps or other
    candidates."""
    metadata = entry.get("metadata") or {}
    scores = entry.get("scores") or {}
    filters = goal.get("filters") or {}
    sources = [s for s in (entry.get("evidence_sources") or []) if s != PERSONALIZATION_SOURCE]

    result = {
        "filename": metadata.get("filename"),
        "file_type": metadata.get("file_type"),
        "tags": list(metadata.get("tags") or [])[:MAX_TAGS],
        "summary": _truncate(metadata.get("summary")),
        "bm25_snippet": _truncate(metadata.get("snippet") or metadata.get("bm25_snippet")),
        "semantic_matched_chunk": _truncate(
            metadata.get("matched_chunk") or metadata.get("semantic_matched_chunk")
        ),
        "retrieval_sources": sources,
        "reranked": "rerank_candidates" in scores,
        "reranker_score": round(float(scores["rerank_candidates"]), 4) if "rerank_candidates" in scores else None,
        "retrieval_score": entry.get("retrieval_score"),
    }

    comparison = comparison or {}
    applied = bool(comparison.get("comparison_applied"))
    evidence = {
        "user_query": (raw_query or "")[:MAX_QUERY_CHARS],
        "goal": {
            "file_type": filters.get("file_type"),
            "filename_candidate": filters.get("filename_candidate"),
            "date_range": filters.get("date_range"),
            "topic_tokens": goal.get("topic_tokens"),
            "expanded_terms": goal.get("expanded_terms") or [],
        },
        "result": result,
        "selection_basis": "llm_comparison_of_near_tied_candidates" if applied else "deterministic_ranking",
    }
    personalization = _meaningful_personalization(entry)
    if personalization:
        evidence["personalization"] = personalization
    if applied:
        evidence["comparison"] = {"rationale": _truncate(comparison.get("comparison_rationale"))}
    return evidence


def build_user_prompt(evidence: dict) -> str:
    return "Explain why this result was returned.\n" + json.dumps(
        evidence, default=str, ensure_ascii=False, indent=1
    )


def _withheld_values(entry: dict, user_id: Optional[str], user_scope: Optional[List[str]]) -> List[str]:
    metadata = entry.get("metadata") or {}
    vals = [metadata.get("owner_id"), metadata.get("file_path"), user_id, *(user_scope or [])]
    return [str(v).lower() for v in vals if v and len(str(v)) >= 3]


def explain_result_with_llm(
    raw_query: str,
    goal: dict,
    entry: dict,
    comparison: Optional[dict],
    provider: LLMProvider,
    user_id: Optional[str] = None,
    user_scope: Optional[List[str]] = None,
) -> Tuple[Optional[ResultExplanation], dict]:
    """Returns (validated explanation or None, decision metadata). Never
    raises. The metadata binds the explanation to `entry["file_id"]`."""
    meta = {
        "explanation_invoked": True,
        "explanation_provider": getattr(provider, "name", "unknown"),
        "explanation_status": None,
        "explained_file_id": entry["file_id"],
    }
    evidence = build_explanation_evidence(raw_query, goal, entry, comparison)

    try:
        raw_response = provider.complete(SYSTEM_PROMPT, build_user_prompt(evidence))
    except LLMUnavailableError as exc:
        meta["explanation_status"] = "unavailable"
        meta["explanation_error"] = str(exc)[:200]
        return None, meta
    except Exception as exc:  # noqa: BLE001 -- any other provider failure is still a safe-fallback case
        meta["explanation_status"] = "provider_error"
        meta["explanation_error"] = str(exc)[:200]
        return None, meta

    if not isinstance(raw_response, str):
        meta["explanation_status"] = "invalid_output"
        meta["explanation_error"] = f"provider returned {type(raw_response).__name__}, expected str"
        return None, meta

    try:
        parsed = parse_result_explanation_response(raw_response)
    except LLMResponseInvalidError as exc:
        meta["explanation_status"] = "invalid_output"
        meta["explanation_error"] = str(exc)[:200]
        return None, meta

    text = " ".join([parsed.summary, *parsed.evidence_points]).lower()
    if any(v in text for v in _withheld_values(entry, user_id, user_scope)):
        meta["explanation_status"] = "invalid_output"
        meta["explanation_error"] = "explanation references information that was not supplied as evidence"
        return None, meta

    meta["explanation_status"] = "generated"
    meta["explanation_confidence"] = parsed.confidence
    meta["explanation_summary"] = parsed.summary
    meta["explanation_evidence_points"] = parsed.evidence_points
    return parsed, meta
