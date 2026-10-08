"""
LLM-assisted query expansion -- Phase 5B (architecture_final.md §9:
"query_expansion, to reformulate a query that returned weak lexical/
semantic evidence"; §6's tool table).

Invoked from exactly one place: app/graph/build.py's node_expand_query,
positioned BEFORE plan_next_action ever runs (requirement 9), and ONLY
when the deterministic app.goal.understand_goal heuristic already flagged
the query as genuinely ambiguous/underspecified
(`ambiguity_score >= AMBIGUITY_CLARIFY_THRESHOLD` -- the same signal
Phase 5A's goal interpretation gates on; requirement 6: never call the LLM
for every simple query).

Deviation from the architecture doc's original placement, intentional and
explicit per this phase's brief: §7's adaptation table originally proposed
query_expansion reactively, after `bm25_search` returns weak/zero scores
(mid-loop, observation-driven). This phase instead runs it proactively,
once, during goal understanding -- before any retrieval tool has executed
-- so that metadata_search/bm25_search/semantic_search/hybrid_search can
all benefit from the richer term set from their very first attempt, not
only after an observed failure. The reactive, evidence-triggered form
from §7 remains unimplemented and is not what this module does.

Guarantees:
  - The LLM returns ONLY an app.llm.schema.QueryExpansion -- a short,
    validated, whitelisted-field list of search phrases. It cannot select
    or invoke a tool, express SQL/code, or reference a file/user ID
    (requirement 4/8): the schema has no field for any of those.
  - The original `raw_query` and deterministic `topic_tokens`/
    `concept_text` are NEVER modified. Expanded terms are stored under a
    new, separate `goal["expanded_terms"]` key (requirement 5); it is
    app/planning.py's job (deterministic, unchanged decision structure)
    to decide how/whether to fold them into a given tool's search text.
  - Any failure -- provider unavailable, provider error/timeout, invalid
    or non-JSON output, schema validation failure -- is caught here and
    degrades to an empty expansion (`goal["expanded_terms"] = []`),
    i.e. exactly the original deterministic query, plus decision metadata
    explaining why (requirement 7).
  - No chain-of-thought is requested, stored, or logged -- only the
    validated terms plus an optional short (<=200 char) rationale.
"""
from typing import List, Tuple

from app.llm.provider import LLMProvider, LLMUnavailableError
from app.llm.schema import LLMResponseInvalidError, QueryExpansion, parse_query_expansion_response

SYSTEM_PROMPT = (
    "You propose additional search terms for a natural-language file-search "
    "request that a deterministic system found ambiguous or underspecified. "
    "You are not a file search engine and you have no access to any files, "
    "database, or tools -- you only propose extra vocabulary; a separate "
    "deterministic system decides whether and how to use it. Respond with "
    "EXACTLY one JSON object and nothing else: no markdown fences, no prose, "
    "no explanation outside the object. Allowed keys only: expanded_terms "
    "(a list of up to 6 short lowercase search phrases -- synonyms, related "
    "terms, or likely alternate phrasings of the user's request; never a "
    "file name, ID, or number), confidence (a number from 0 to 1), and "
    "rationale (a single sentence, at most 200 characters, stating the gist "
    "only -- not your reasoning steps). Only propose a term if it plausibly "
    "broadens or clarifies the search; do not repeat words already in the "
    "query."
)


def _build_user_prompt(raw_query: str, goal_dict: dict) -> str:
    return (
        f"User query: {raw_query!r}\n"
        f"Deterministic system already extracted topic tokens: "
        f"{goal_dict.get('topic_tokens')!r}\n"
        f"ambiguity_score={goal_dict.get('ambiguity_score')!r} (0=clear, 1=very ambiguous).\n"
        "Suggest a small number of additional search terms/phrases that "
        "would help find the file the user means, if any genuinely would."
    )


def expand_query_with_llm(raw_query: str, goal_dict: dict, provider: LLMProvider) -> Tuple[List[str], dict]:
    """Returns (expanded_terms, decision metadata dict). Never raises:
    every failure mode degrades to `([], metadata explaining why)`."""
    meta = {
        "expansion_invoked": True,
        "expansion_provider": getattr(provider, "name", "unknown"),
        "expansion_status": None,
    }

    try:
        raw_response = provider.complete(SYSTEM_PROMPT, _build_user_prompt(raw_query, goal_dict))
    except LLMUnavailableError as exc:
        meta["expansion_status"] = "unavailable"
        meta["expansion_error"] = str(exc)[:200]
        return [], meta
    except Exception as exc:  # noqa: BLE001 -- any other provider failure is still a safe-fallback case
        meta["expansion_status"] = "provider_error"
        meta["expansion_error"] = str(exc)[:200]
        return [], meta

    try:
        parsed: QueryExpansion = parse_query_expansion_response(raw_response)
    except LLMResponseInvalidError as exc:
        meta["expansion_status"] = "invalid_output"
        meta["expansion_error"] = str(exc)[:200]
        return [], meta

    meta["expansion_status"] = "applied"
    meta["expansion_confidence"] = parsed.confidence
    meta["expansion_rationale"] = parsed.rationale
    return parsed.expanded_terms, meta
