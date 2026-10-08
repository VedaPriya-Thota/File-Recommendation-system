"""
Deterministic Tier-1 reflection -- architecture_final.md §7.

Tier 2 (LLM-assisted, for genuinely tied/conflicting evidence) is
explicitly deferred and NOT implemented here.

reflect() looks only at:
  - what the just-executed action returned (`last_tool_result`)
  - the accumulated `candidate_pool`
  - iteration/max_iterations

...and returns a tentative status (`finish` / `replan` / `clarify`) plus an
`observation_hint` ("zero_results" | "too_many" | "error" | None) that the
graph passes to `plan_next_action` when replanning. If replanning turns out
to be impossible (plan_next_action has no untried strategy left), the graph
resolves the tentative "replan" into a final "finish" (explicit nothing-
found) or "clarify" -- see `app/graph/build.py`.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.config import DOMINANT_SCORE_GAP, TOO_MANY_CANDIDATES_THRESHOLD
from app.final_ranking import is_weak_lexical, retrieval_evidence_score
from app.planning import args_signature
from app.pool_resolution import (
    BASIS_NEAR_TIE_UNRESOLVED,
    BASIS_PERSONALIZATION_RESOLVED,
    BASIS_RETRIEVAL_DOMINANCE,
    BASIS_WEAK_LEXICAL,
    personalization_tiebreak,
)
from app.state import Action

# Phase 6 correctness fix: dominance/flatness here MUST be judged on
# retrieval/reranker evidence alone (S_retrieval, architecture_final.md
# §10), the exact same definition app/final_ranking.py uses -- never on the
# raw per-source `scores` dict as a whole. That dict also holds
# get_user_activity's personalization/affinity score once that tool has
# run (app/candidate_pool.py merges every source into one shared dict by
# design). The previous code here did `max(entry["scores"].values())`,
# which therefore let a candidate's affinity score alone make it look
# "dominant" to reflect() even when its actual retrieval evidence was weak
# or tied with another candidate's -- silently letting personalization
# decide evidence-sufficiency, which is exactly what §8's anti-circularity
# guarantee (and this same codebase's own final_ranking.py bug-fix
# docstring) says personalization must never do. It could not change
# *which* candidate the system finally returns (final_rank() is the only
# place that produces final_ranking's order, and is unaffected by
# reflect()), but it could make reflect() report an inflated `confidence`
# and mark `degraded_mode=False` for a result whose real retrieval
# evidence was in fact flat/ambiguous -- a real observability/correctness
# bug per architecture_final.md §12. See tests/test_reflect.py's
# Phase 6 regression tests.


@dataclass
class ReflectionResult:
    status: str  # "finish" | "replan" | "clarify" (tentative for "replan")
    reason: str
    confidence: float
    observation_hint: Optional[str]  # "zero_results" | "too_many" | "error" | None
    failed_strategies: List[str]
    errors: List[dict]
    low_confidence: bool = False
    resolution_basis: Optional[str] = None  # Phase 9: why a "finish" was decided (BASIS_*)


def reflect(
    last_action: Action,
    last_tool_result: dict,
    candidate_pool: Dict[int, dict],
    failed_strategies: List[str],
    errors: List[dict],
    iteration_count: int,
    max_iterations: int,
) -> ReflectionResult:
    failed_strategies = list(failed_strategies)
    errors = list(errors)
    last_sig = args_signature(last_action["tool"], last_action["args"])
    at_iteration_cap = iteration_count >= max_iterations

    # --- Tool error -------------------------------------------------
    if last_tool_result.get("error"):
        failed_strategies.append(last_sig)
        errors.append(
            {
                "tool": last_action["tool"],
                "error": last_tool_result["error"],
                "fallback_taken": "replan" if not at_iteration_cap else "forced_finish",
            }
        )
        if at_iteration_cap:
            return ReflectionResult(
                status="finish",
                reason="Tool error and max_iterations reached",
                confidence=0.0,
                observation_hint="error",
                failed_strategies=failed_strategies,
                errors=errors,
                low_confidence=True,
            )
        return ReflectionResult(
            status="replan",
            reason=f"Tool error on {last_action['tool']}: {last_tool_result['error']}",
            confidence=0.0,
            observation_hint="error",
            failed_strategies=failed_strategies,
            errors=errors,
        )

    this_action_candidates = last_tool_result.get("candidates", [])

    # --- Zero candidates from the action just executed ---------------
    if len(this_action_candidates) == 0:
        failed_strategies.append(last_sig)
        if at_iteration_cap:
            return ReflectionResult(
                status="finish",
                reason="Max iterations reached; last strategy found nothing",
                confidence=0.0,
                observation_hint="zero_results",
                failed_strategies=failed_strategies,
                errors=errors,
                low_confidence=True,
            )
        return ReflectionResult(
            status="replan",
            reason="Zero candidates from the last strategy; will relax filters",
            confidence=0.0,
            observation_hint="zero_results",
            failed_strategies=failed_strategies,
            errors=errors,
        )

    # --- Evaluate the accumulated pool for dominance / flatness ------
    # retrieval_evidence_score (not a raw max over the whole scores dict)
    # -- see the module-level note above.
    scored = sorted(
        candidate_pool.items(),
        key=lambda kv: retrieval_evidence_score(kv[1]),
        reverse=True,
    )
    top_score = retrieval_evidence_score(scored[0][1]) if scored else 0.0
    second_score = retrieval_evidence_score(scored[1][1]) if len(scored) > 1 else 0.0
    gap = top_score - second_score
    n_candidates = len(scored)

    # --- Phase 12: weak lexical evidence is not dominance --------------------------------------
    # bm25_search max-normalises, so its best hit is always 1.0, and a lone candidate is "dominant"
    # at any strength. If the leading candidate's ONLY retrieval evidence is a lexical match covering
    # a minority of the query's terms (final_ranking.is_weak_lexical), that is not enough to stop on:
    # seek independent evidence first (architecture §7: "bm25_search -> 0/weak scores -> fall through
    # to semantic_search"). The planner answers the "weak_evidence" hint; if it has nothing left the
    # graph finishes with this candidate flagged low-confidence instead of discarding it.
    if scored and is_weak_lexical(scored[0][1]):
        # Confidence is how much of the query the lexical evidence explains -- never more than that, even if a
        # later stage (e.g. min-max reranking, which hands its own best candidate 1.0) inflated the raw score.
        weak_confidence = round(min(top_score, (scored[0][1].get("coverage") or {}).get("bm25_search", top_score)), 4)
        if at_iteration_cap:
            return ReflectionResult(
                status="finish",
                reason="Max iterations reached; only weak lexical evidence found",
                confidence=weak_confidence,
                observation_hint="weak_evidence",
                failed_strategies=failed_strategies,
                errors=errors,
                low_confidence=True,
                resolution_basis=BASIS_WEAK_LEXICAL,
            )
        return ReflectionResult(
            status="replan",
            reason=(
                "Leading candidate rests only on a lexical match covering a minority of the query's terms "
                f"(evidence {round(top_score, 4)}); seeking independent evidence"
            ),
            confidence=weak_confidence,
            observation_hint="weak_evidence",
            failed_strategies=failed_strategies,
            errors=errors,
        )
    is_dominant = n_candidates == 1 or (n_candidates > 1 and gap >= DOMINANT_SCORE_GAP)
    is_too_many_or_flat = n_candidates > TOO_MANY_CANDIDATES_THRESHOLD or (
        n_candidates > 1 and gap < DOMINANT_SCORE_GAP
    )

    if is_dominant:
        return ReflectionResult(
            status="finish",
            reason="One dominant candidate with constraints satisfied",
            confidence=round(top_score, 4),
            observation_hint=None,
            failed_strategies=failed_strategies,
            errors=errors,
            resolution_basis=BASIS_RETRIEVAL_DOMINANCE,
        )

    # --- Phase 8: personalization may resolve a tie retrieval left open --
    # Retrieval evidence is flat here (not dominant). If get_user_activity has
    # run and the final-ranking leader is separated from every other
    # retrieval-contender by a meaningful AFFINITY margin, the tie is resolved
    # -- via a distinct, labelled path. This is NOT retrieval dominance
    # (is_dominant above is untouched, still retrieval-only), and the reported
    # confidence stays derived from retrieval evidence, never from affinity.
    # See app/pool_resolution.py (Rule A).
    if n_candidates > 1:
        tiebreak = personalization_tiebreak(candidate_pool)
        if tiebreak["resolved"]:
            return ReflectionResult(
                status="finish",
                reason=tiebreak["reason"],
                confidence=round(top_score * 0.5, 4),
                observation_hint=None,
                failed_strategies=failed_strategies,
                errors=errors,
                resolution_basis=BASIS_PERSONALIZATION_RESOLVED,
            )

    if is_too_many_or_flat:
        if at_iteration_cap:
            return ReflectionResult(
                status="finish",
                reason="Max iterations reached with flat/too-many candidates",
                confidence=round(top_score * 0.5, 4),
                observation_hint="too_many",
                failed_strategies=failed_strategies,
                errors=errors,
                low_confidence=True,
                resolution_basis=BASIS_NEAR_TIE_UNRESOLVED,
            )
        return ReflectionResult(
            status="replan",
            reason="Too many / flat-score candidates; attempting to narrow",
            confidence=round(top_score * 0.5, 4),
            observation_hint="too_many",
            failed_strategies=failed_strategies,
            errors=errors,
        )

    # --- Fallback: small, non-flat set but below the dominance gap ---
    # (e.g. exactly 2 candidates with a middling gap). Treat like
    # too-many/flat: attempt to narrow, else finish with the best guess.
    if at_iteration_cap:
        return ReflectionResult(
            status="finish",
            reason="Max iterations reached; returning best-available ranking",
            confidence=round(top_score * 0.6, 4),
            observation_hint="too_many",
            failed_strategies=failed_strategies,
            errors=errors,
            low_confidence=True,
            resolution_basis=BASIS_NEAR_TIE_UNRESOLVED,
        )
    return ReflectionResult(
        status="replan",
        reason="Ambiguous candidate set (no clear winner); attempting to narrow",
        confidence=round(top_score * 0.6, 4),
        observation_hint="too_many",
        failed_strategies=failed_strategies,
        errors=errors,
    )
