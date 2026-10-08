"""
LLM-assisted candidate comparison -- Phase 5C (architecture_final.md §6
`compare_candidates`: "resolves a specific, narrow ambiguity (e.g. which of
these 2-3 tied candidates) using structured evidence handed to it; it is a
tie-breaker, not a general ranking authority -- it never re-ranks a full
candidate pool or substitutes for `final_rank`").

Invoked from exactly one place: app/graph/build.py's node_compare_candidates,
which runs AFTER the deterministic `final_rank` (app/final_ranking.py) and
before END. Nothing upstream of it -- planning, retrieval, reranking,
personalization, reflection, scope -- is touched or aware of it.

Invocation condition (`select_comparison_set`)
----------------------------------------------
Given the deterministic final ranking, take the top COMPARISON_TOP_N
candidates and keep those whose `final_score` is within DOMINANT_SCORE_GAP of
the top candidate's (the same "no clear winner" threshold reflect() uses:
`gap < DOMINANT_SCORE_GAP`). The LLM is consulted only if at least TWO
candidates survive; otherwise (single candidate, or a clearly dominant top
candidate) there is nothing ambiguous to resolve and the LLM is never
called. The surviving set is also exactly what the LLM is shown -- it never
sees, and can never select, a candidate that isn't genuinely close.

Guarantees
----------
  - The LLM returns ONLY an app.llm.schema.CandidateComparison: one
    `selected_file_id`, a confidence, and a short rationale. It cannot
    express SQL, a tool call, a new candidate, a score or a scope change.
  - `selected_file_id` must be one of the supplied candidate IDs. Anything
    else (an invented ID, an out-of-set real ID) is rejected as
    `unknown_candidate` and the deterministic order stands.
  - The candidate set is immutable: `reorder_with_selection` only moves the
    chosen candidate to the front; it never adds, removes or copies-with-
    changes an entry, and no score is recomputed or altered. The original
    deterministic order is preserved in the decision metadata.
  - Any failure -- provider unavailable/erroring, non-string or non-JSON or
    schema-invalid output, unknown candidate, confidence below
    COMPARISON_MIN_CONFIDENCE -- returns `None` (no preference) plus metadata
    explaining why, and the deterministic ranking is left unchanged.
  - No chain-of-thought is requested, stored or logged: only the validated
    selection, confidence and a <=200 char rationale.
"""
import json
from typing import List, Optional, Tuple

from app.config import COMPARISON_MIN_CONFIDENCE, COMPARISON_TOP_N, DOMINANT_SCORE_GAP
from app.llm.provider import LLMProvider, LLMUnavailableError
from app.llm.schema import (
    CandidateComparison,
    LLMResponseInvalidError,
    parse_candidate_comparison_response,
)

MAX_EVIDENCE_TEXT_CHARS = 400
MAX_TAGS = 10

# The only personalization keys ever forwarded (a fixed whitelist, so
# nothing else that might live in metadata["personalization"] leaks through).
_PERSONALIZATION_KEYS = (
    "preferred_file_type_match",
    "frequent_topic_score",
    "open_count",
    "view_count",
    "edit_count",
    "recency_score",
    "frequency_score",
    "affinity_score",
    "cold_start",
)

SYSTEM_PROMPT = (
    "You choose which of a small set of already-retrieved candidate files best "
    "matches a user's file-search request, when a deterministic system found "
    "them nearly tied. You are not a file search engine and you have no access "
    "to any files, database, or tools: you may only pick from the candidates "
    "listed in the message, using the evidence given. Never invent a file_id. "
    "Retrieval, reranker and personalization scores are provided as context; "
    "you do not change them. Respond with EXACTLY one JSON object and nothing "
    "else: no markdown fences, no prose outside the object. Allowed keys only: "
    "selected_file_id (an integer that is exactly one of the listed file_id "
    "values), confidence (a number from 0 to 1), and rationale (a single "
    "sentence, at most 200 characters, stating the gist only -- not your "
    "reasoning steps)."
)


def select_comparison_set(
    final_ranking: List[dict],
    top_n: int = COMPARISON_TOP_N,
    gap: float = DOMINANT_SCORE_GAP,
) -> List[dict]:
    """Returns the bounded set of near-tied candidates worth asking the LLM
    about, in their existing deterministic order -- or [] if there is no
    genuine ambiguity (see module docstring). Never mutates its input."""
    if len(final_ranking) < 2 or top_n < 2:
        return []
    top_score = final_ranking[0]["final_score"]
    close = [c for c in final_ranking[:top_n] if (top_score - c["final_score"]) < gap]
    return close if len(close) >= 2 else []


def _truncate(text) -> Optional[str]:
    if not text:
        return None
    text = str(text).strip()
    return text[:MAX_EVIDENCE_TEXT_CHARS] or None


def candidate_evidence(entry: dict) -> dict:
    """The exact, bounded evidence the LLM sees for ONE ranked candidate.
    Deliberately excludes owner_id, file_path, timestamps and anything else
    not needed to judge relevance."""
    metadata = entry.get("metadata") or {}
    personalization = entry.get("personalization") or metadata.get("personalization") or {}
    return {
        "file_id": entry["file_id"],
        "filename": metadata.get("filename"),
        "file_type": metadata.get("file_type"),
        "tags": list(metadata.get("tags") or [])[:MAX_TAGS],
        "summary": _truncate(metadata.get("summary")),
        "bm25_snippet": _truncate(metadata.get("snippet") or metadata.get("bm25_snippet")),
        "semantic_matched_chunk": _truncate(
            metadata.get("matched_chunk") or metadata.get("semantic_matched_chunk")
        ),
        "evidence_sources": list(entry.get("evidence_sources") or []),
        "source_scores": {k: round(float(v), 4) for k, v in (entry.get("scores") or {}).items()},
        "retrieval_score": entry.get("retrieval_score"),
        "personalization_score": entry.get("personalization_score"),
        "final_score": entry.get("final_score"),
        "personalization": {k: personalization[k] for k in _PERSONALIZATION_KEYS if k in personalization},
    }


def _goal_context(goal: dict) -> dict:
    filters = goal.get("filters") or {}
    return {
        "file_type": filters.get("file_type"),
        "filename_candidate": filters.get("filename_candidate"),
        "date_range": filters.get("date_range"),
        "topic_tokens": goal.get("topic_tokens"),
        "expanded_terms": goal.get("expanded_terms") or [],
        "ambiguity_score": goal.get("ambiguity_score"),
    }


def build_user_prompt(raw_query: str, goal: dict, comparison_set: List[dict]) -> str:
    payload = {
        "user_query": raw_query,
        "goal": _goal_context(goal),
        "candidates": [candidate_evidence(c) for c in comparison_set],
    }
    return (
        "Pick the single best-matching candidate for the user's request.\n"
        + json.dumps(payload, default=str, ensure_ascii=False, indent=1)
    )


def compare_candidates_with_llm(
    raw_query: str, goal: dict, comparison_set: List[dict], provider: LLMProvider
) -> Tuple[Optional[int], dict]:
    """Returns (selected_file_id or None, decision metadata). Never raises:
    every failure mode returns `(None, metadata explaining why)`, meaning
    "no preference -- keep the deterministic order"."""
    supplied_ids = [c["file_id"] for c in comparison_set]
    meta = {
        "comparison_invoked": True,
        "comparison_provider": getattr(provider, "name", "unknown"),
        "comparison_status": None,
        "compared_file_ids": supplied_ids,
        "selected_file_id": None,
    }

    try:
        raw_response = provider.complete(SYSTEM_PROMPT, build_user_prompt(raw_query, goal, comparison_set))
    except LLMUnavailableError as exc:
        meta["comparison_status"] = "unavailable"
        meta["comparison_error"] = str(exc)[:200]
        return None, meta
    except Exception as exc:  # noqa: BLE001 -- any other provider failure is still a safe-fallback case
        meta["comparison_status"] = "provider_error"
        meta["comparison_error"] = str(exc)[:200]
        return None, meta

    if not isinstance(raw_response, str):
        meta["comparison_status"] = "invalid_output"
        meta["comparison_error"] = f"provider returned {type(raw_response).__name__}, expected str"
        return None, meta

    try:
        parsed: CandidateComparison = parse_candidate_comparison_response(raw_response)
    except LLMResponseInvalidError as exc:
        meta["comparison_status"] = "invalid_output"
        meta["comparison_error"] = str(exc)[:200]
        return None, meta

    if parsed.selected_file_id not in supplied_ids:
        # Generated / out-of-set ID: rejected outright, never looked up.
        meta["comparison_status"] = "unknown_candidate"
        meta["comparison_error"] = "selected_file_id is not one of the supplied candidates"
        return None, meta

    meta["comparison_confidence"] = parsed.confidence
    meta["comparison_rationale"] = parsed.rationale
    if parsed.confidence < COMPARISON_MIN_CONFIDENCE:
        meta["comparison_status"] = "low_confidence"
        return None, meta

    meta["comparison_status"] = "selected"
    meta["selected_file_id"] = parsed.selected_file_id
    return parsed.selected_file_id, meta


def reorder_with_selection(final_ranking: List[dict], selected_file_id: int) -> List[dict]:
    """Moves `selected_file_id` to the front; every other entry keeps its
    existing relative order. Returns a NEW list holding the SAME entry
    objects (no score or field is touched). If the ID isn't in the ranking
    the input order is returned unchanged."""
    selected = [c for c in final_ranking if c["file_id"] == selected_file_id]
    if not selected:
        return list(final_ranking)
    rest = [c for c in final_ranking if c["file_id"] != selected_file_id]
    return selected[:1] + rest
