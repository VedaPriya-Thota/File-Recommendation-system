"""
LLM-assisted goal augmentation -- Phase 5A (architecture_final.md §9:
"Intent extraction fallback, when the deterministic heuristic is
inconclusive").

Invoked from exactly one place: app/graph/build.py's node_understand_goal,
and ONLY when the deterministic app.goal.understand_goal heuristic has
already flagged the query as genuinely ambiguous
(`ambiguity_score >= AMBIGUITY_CLARIFY_THRESHOLD`) -- prompt requirement 7:
never call the LLM for every simple query. The deterministic goal is always
computed first and is what gets augmented, never replaced.

Guarantees:
  - The LLM returns ONLY an app.llm.schema.LLMGoalInterpretation --
    validated, whitelisted-field JSON. It cannot express SQL, code, or a
    tool call (requirement 3): the schema has no field for any of those.
  - Merging (`_merge_llm_interpretation`) is additive and deterministic:
    an LLM-suggested filter is used only to fill a gap the deterministic
    heuristic left empty. It can never overwrite a deterministically
    detected filter, and it never touches `user_scope`, retrieval
    algorithms, reranking, personalization, or final ranking -- none of
    those are represented in the schema, so there's nothing to enforce a
    boundary around at merge time (requirements 4/5/10).
  - Any failure -- provider unavailable, provider error/timeout, invalid
    or non-JSON output, schema validation failure -- is caught here and
    degrades to returning the ORIGINAL goal dict completely unchanged,
    plus decision metadata explaining why (requirement 6). The graph node
    itself needs no try/except.
  - No chain-of-thought is requested, stored, or logged -- only the
    validated structured fields plus an optional short (<=200 char)
    rationale (requirement 9).
"""
from typing import Tuple

from app.llm.provider import LLMProvider, LLMUnavailableError
from app.llm.schema import LLMGoalInterpretation, LLMResponseInvalidError, parse_llm_goal_response

SYSTEM_PROMPT = (
    "You interpret a user's natural-language file-search request into a small, "
    "strictly structured JSON object. You are not a file search engine and you "
    "have no access to any files, database, or tools -- you only propose "
    "interpretation hints for a separate deterministic system to validate and "
    "apply; that system, not you, decides what actually happens. Respond with "
    "EXACTLY one JSON object and nothing else: no markdown fences, no prose, "
    "no explanation outside the object. Allowed keys only: "
    "file_type (one of pptx/xlsx/docx/pdf/png, or null), "
    "topic_tokens (a list of up to 8 short lowercase plain-language keywords), "
    "filename_candidate (a literal filename fragment if one is clearly implied, "
    "or null), retrieval_preference (one of metadata/lexical/semantic/hybrid, "
    "or null), confidence (a number from 0 to 1), and rationale (a single "
    "sentence, at most 200 characters, stating the gist only -- not your "
    "reasoning steps). Do not invent a file_type or filename_candidate you "
    "aren't reasonably confident about; leave it null instead."
)


def _build_user_prompt(raw_query: str, deterministic_goal: dict) -> str:
    filters = deterministic_goal.get("filters") or {}
    return (
        f"User query: {raw_query!r}\n"
        "The deterministic system already detected: "
        f"file_type={filters.get('file_type')!r}, "
        f"filename_candidate={filters.get('filename_candidate')!r}, "
        f"topic_tokens={deterministic_goal.get('topic_tokens')!r}, "
        f"ambiguity_score={deterministic_goal.get('ambiguity_score')!r} (0=clear, 1=very ambiguous).\n"
        "Only suggest a field if you are reasonably confident and it adds "
        "information the deterministic system doesn't already have."
    )


def augment_goal_with_llm(raw_query: str, goal_dict: dict, provider: LLMProvider) -> Tuple[dict, dict]:
    """Returns (possibly-augmented goal dict, decision metadata dict) for
    observability (architecture_final.md §12: "concise decision metadata",
    not chain-of-thought). Never raises: every failure mode degrades to
    `(goal_dict unchanged, metadata explaining why)`."""
    meta = {
        "llm_invoked": True,
        "llm_provider": getattr(provider, "name", "unknown"),
        "llm_status": None,
    }

    try:
        raw_response = provider.complete(SYSTEM_PROMPT, _build_user_prompt(raw_query, goal_dict))
    except LLMUnavailableError as exc:
        meta["llm_status"] = "unavailable"
        meta["llm_error"] = str(exc)[:200]
        return goal_dict, meta
    except Exception as exc:  # noqa: BLE001 -- any other provider failure is still a safe-fallback case
        meta["llm_status"] = "provider_error"
        meta["llm_error"] = str(exc)[:200]
        return goal_dict, meta

    try:
        parsed = parse_llm_goal_response(raw_response)
    except LLMResponseInvalidError as exc:
        meta["llm_status"] = "invalid_output"
        meta["llm_error"] = str(exc)[:200]
        return goal_dict, meta

    merged = _merge_llm_interpretation(goal_dict, parsed)
    meta["llm_status"] = "applied"
    meta["llm_confidence"] = parsed.confidence
    meta["llm_rationale"] = parsed.rationale
    meta["llm_retrieval_preference"] = parsed.retrieval_preference
    return merged, meta


def _merge_llm_interpretation(goal_dict: dict, parsed: LLMGoalInterpretation) -> dict:
    """Deterministic, additive-only merge (requirement 5: "deterministic
    code must validate and enforce"). Every already-populated
    deterministic field wins; the LLM can only fill a genuine gap. `intent`
    and `raw_query` are never touched -- they aren't represented in the
    schema at all, so there's nothing for the LLM to override."""
    filters = dict(goal_dict.get("filters") or {})

    if not filters.get("file_type") and parsed.file_type:
        filters["file_type"] = parsed.file_type

    if not filters.get("filename_candidate") and parsed.filename_candidate:
        filters["filename_candidate"] = parsed.filename_candidate

    topic_tokens = list(goal_dict.get("topic_tokens") or [])
    for tok in parsed.topic_tokens:
        if tok not in topic_tokens:
            topic_tokens.append(tok)

    concept_text = " ".join(topic_tokens) if topic_tokens else goal_dict.get("concept_text")

    merged = dict(goal_dict)
    merged["filters"] = filters
    merged["topic_tokens"] = topic_tokens
    merged["concept_text"] = concept_text
    # Advisory only -- not consumed by app/planning.py in Phase 5A. Carried
    # through purely for observability / a future phase to act on.
    merged["llm_retrieval_preference"] = parsed.retrieval_preference
    return merged
