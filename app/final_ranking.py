"""
Final deterministic ranking policy -- architecture_final.md §8 (touchpoint
3, "Final ranking") and §10 ("Final scoring formula"); Phase 4B.

Bug this fixes: `finish_stage` (app/graph/build.py) used to compute
`final_score = max(entry["scores"].values())`. Because Phase 4A's
get_user_activity tool adds its affinity score into that SAME shared
per-source `scores` dict as every retrieval/rerank tool
(`scores["get_user_activity"]`, via the generic app/candidate_pool.py
merge -- by design, so it's preserved alongside other evidence, see that
tool's docstring), `max()` over the whole dict could let personalization's
affinity score silently WIN and become the final score whenever it
happened to be numerically larger than genuine retrieval/reranker
evidence. That directly violates architecture_final.md §8's
anti-circularity guarantee ("[personalization] can never introduce a
candidate with zero retrieval evidence... it never re-ranks a full
candidate pool or substitutes for final_rank") and Phase 4A's own
personalization.py docstring ("never a substitute for that evidence").

This module is the single, explicit, testable place that separates the
two kinds of evidence and combines them the way architecture_final.md §10
specifies, instead of an unweighted max over an undifferentiated bag of
scores.

Retrieval / reranking evidence (S_retrieval)
---------------------------------------------
The max of every per-source score in `entry["scores"]` EXCEPT
`get_user_activity`'s -- i.e. exactly what the old `max(entry["scores"]
.values())` computed, minus the one source that was never supposed to be
pooled in there. If `rerank_candidates` ran for a candidate, its
normalized score is already the largest of the retrieval-stage scores by
construction (architecture_final.md §10: the reranked score REPLACES,
never sums with, the RRF/lexical/semantic score it superseded for that
candidate) -- taking max() over the remaining sources reproduces that
"replaces" semantics for free, no special-casing needed here.

Personalization evidence (S_affinity)
---------------------------------------------
`get_user_activity`'s composite affinity score for this candidate, read
from wherever it actually lives: `scores["get_user_activity"]` in the real
graph flow (candidate_pool.py's generic merge), falling back to a
top-level `entry["personalization"]["affinity_score"]` or
`entry["metadata"]["personalization"]["affinity_score"]` for callers that
used app/personalization.py's `attach_personalization_signals` directly
instead of running the tool through the graph. Missing everywhere ->
0.0 (neutral) -- this is what makes cold start / "get_user_activity never
ran for this candidate" behave identically: no affinity evidence, no
influence on the final score.

Final score
---------------------------------------------
    FinalScore = ALPHA * S_retrieval + GAMMA * S_affinity

architecture_final.md §10's full formula is
`α·S_retrieval + β·S_recency + γ·S_affinity`. There is no independent
S_recency in this codebase yet to weight: Phase 4A's `affinity_score`
already folds a recency sub-signal into its single [0,1] composite (see
app/personalization.py's `recency_score` component of that composite), so
introducing a second, separate recency term here would double-count it /
invent a value with nothing behind it. Phase 4B therefore implements the
two-term form of the same formula (β=0 by construction, not a silent
omission) rather than fabricating a redundant weight just to fill in the
full formula shape. A future phase that separates S_recency out of the
affinity composite can extend ALPHA/BETA/GAMMA without changing this
module's shape.

ALPHA/GAMMA (app/config.py; architecture_final.md §19: "α/β/γ per
route-equivalent... configurable V1 defaults, tunable via evaluation, not
fixed architectural constants") default to a single retrieval-dominant
profile -- architecture_final.md §10's "HIGH-equivalent ~
retrieval-dominant with meaningful affinity weight" -- ALPHA=0.85,
GAMMA=0.15. Both component scores are already normalized to [0, 1] by
their own tools, so this stays a plain weighted sum, never a rescale.

Candidate-set / evidence preservation
---------------------------------------------
This module only ever reads `candidate_pool`; it never adds, removes, or
mutates a `file_id`, and every existing key on each entry (scores,
metadata, evidence_sources, iteration bookkeeping, ...) is carried forward
into the ranked output completely unchanged -- only the three new keys
(`retrieval_score`, `personalization_score`, `final_score`) are added.
"""
from typing import Dict, List

from app.config import FINAL_RANKING_ALPHA, FINAL_RANKING_GAMMA

# The one source name that means "personalization evidence", never
# "retrieval evidence" -- kept as a single constant so both score-extraction
# helpers below (and any future caller) agree on it.
PERSONALIZATION_SOURCE = "get_user_activity"


def retrieval_evidence_score(entry: dict) -> float:
    """Max of all per-source scores that are genuine retrieval/reranker
    evidence -- i.e. every source except get_user_activity's. Absent ->
    0.0 (a candidate with literally no retrieval evidence has none to
    report; it should not have been in the pool in the first place, but
    this stays defensive rather than raising).

    Public (Phase 6): this is the one place S_retrieval / "retrieval
    dominance" is computed. app/reflect.py's Tier-1 dominance/flatness
    check (architecture_final.md §7) imports this directly rather than
    re-deriving its own notion of "best score" -- see reflect.py's
    docstring for why a second, divergent definition there was a
    correctness bug, not just duplication."""
    scores = entry.get("scores") or {}
    retrieval_scores = [_source_evidence(entry, src, v) for src, v in scores.items() if src != PERSONALIZATION_SOURCE]
    return float(max(retrieval_scores)) if retrieval_scores else 0.0


# --- Phase 12: absolute strength of lexical evidence -----------------------------------------------------
# bm25_search divides every score by the query's best hit, so its top candidate is ALWAYS 1.0 -- whether it
# matched every query term or one function word. Each bm25 candidate therefore also carries `coverage`: the
# fraction of the query's distinct content terms it contains (carried per source in entry["coverage"]).
# A lexical match covering a MINORITY of the query (strictly less than half) is partial evidence: it
# contributes only `score * coverage`, and if it is a candidate's ONLY retrieval evidence the candidate is
# "weakly lexical" (see is_weak_lexical). "Half" is the majority boundary, not a tuned weight: on the Phase 11
# data every correct lexical top hit covered >= 0.5 of its query and the two spurious ones covered 0.25.
LEXICAL_SOURCE = "bm25_search"
LEXICAL_MAJORITY = 0.5
_INDEPENDENT_SOURCES = ("metadata_search", "semantic_search")  # evidence that does not come from the lexical match


def _lexical_coverage(entry: dict):
    return (entry.get("coverage") or {}).get(LEXICAL_SOURCE)


def _source_evidence(entry: dict, source: str, value: float) -> float:
    if source == LEXICAL_SOURCE:
        cov = _lexical_coverage(entry)
        if cov is not None and cov < LEXICAL_MAJORITY:
            return value * cov
    return value


def leading_is_weak_lexical(candidate_pool: dict) -> bool:
    """Is the pool's best-supported candidate (by retrieval evidence) weakly lexical? False for an empty pool."""
    if not candidate_pool:
        return False
    top = max(candidate_pool.values(), key=retrieval_evidence_score)
    return is_weak_lexical(top)


def weak_lexical_confidence(candidate_pool: dict) -> float:
    """Confidence for a finish whose best-supported candidate is weakly lexical: how much of the query the lexical
    evidence explains (its coverage), never more than the candidate's evidence score."""
    top = max(candidate_pool.values(), key=retrieval_evidence_score)
    cov = _lexical_coverage(top)
    ev = retrieval_evidence_score(top)
    return round(min(ev, cov) if cov is not None else ev, 4)


def is_weak_lexical(entry: dict) -> bool:
    """True iff the candidate's only retrieval evidence is a bm25 match covering a minority of the query's
    terms. Corroboration by metadata_search or semantic_search makes it NOT weak. Entries without coverage
    information (e.g. built by older callers/tests) are never weak."""
    scores = entry.get("scores") or {}
    if LEXICAL_SOURCE not in scores or any(s in scores for s in _INDEPENDENT_SOURCES):
        return False
    cov = _lexical_coverage(entry)
    return cov is not None and cov < LEXICAL_MAJORITY


# Back-compat internal alias -- this module's own code below predates the
# public rename and is left as-is to keep the diff minimal.
_retrieval_score = retrieval_evidence_score


def _personalization_score(entry: dict) -> float:
    """The get_user_activity affinity score, wherever it actually lives.
    Checked in priority order:
      1. scores["get_user_activity"] -- the real graph/finish-stage path
         (app/tools/get_user_activity.py -> app/candidate_pool.py merge).
      2. entry["personalization"]["affinity_score"] -- the direct
         app/personalization.py `attach_personalization_signals` path
         (used standalone / in tests, not wired into the graph).
      3. entry["metadata"]["personalization"]["affinity_score"] -- the
         signal breakdown get_user_activity always embeds in metadata for
         observability, as a last-resort fallback.
    Missing everywhere -> 0.0 (neutral / cold start / never investigated)."""
    scores = entry.get("scores") or {}
    if PERSONALIZATION_SOURCE in scores:
        return float(scores[PERSONALIZATION_SOURCE] or 0.0)

    personalization = entry.get("personalization")
    if personalization is None:
        personalization = (entry.get("metadata") or {}).get("personalization")
    return float((personalization or {}).get("affinity_score", 0.0) or 0.0)


def compute_component_scores(entry: dict) -> dict:
    """Pure, single-candidate computation -- does NOT mutate `entry`.
    Returns the three explicit, separated scores that make up the ranking
    policy for one candidate_pool entry."""
    retrieval_score = _retrieval_score(entry)
    personalization_score = _personalization_score(entry)
    final_score = (FINAL_RANKING_ALPHA * retrieval_score) + (FINAL_RANKING_GAMMA * personalization_score)
    return {
        "retrieval_score": round(retrieval_score, 6),
        "personalization_score": round(personalization_score, 6),
        "final_score": round(final_score, 6),
    }


def final_rank(candidate_pool: Dict[int, dict]) -> List[dict]:
    """The one final-ranking pass over an existing candidate_pool
    (architecture_final.md §8 touchpoint 3: "the deterministic weighted
    formula runs once, at finish, over the accumulated, investigated
    candidate_pool"). This is the actual policy wired into
    app/graph/build.py's finish_stage.

    - Never introduces or drops a file_id: output has exactly one entry
      per input file_id (§8's anti-circularity guarantee / requirement
      that personalization can only reorder candidates already present).
    - Every pre-existing key on each entry is preserved unchanged; only
      retrieval_score/personalization_score/final_score are added.
    - Deterministic ordering: sorted by final_score desc, ties broken by
      retrieval_score desc (real evidence wins ties over personalization),
      remaining ties broken by ascending file_id for full determinism.
    """
    ranked = []
    for file_id, entry in candidate_pool.items():
        component_scores = compute_component_scores(entry)
        ranked.append({"file_id": file_id, **entry, **component_scores})

    ranked.sort(key=lambda item: (-item["final_score"], -item["retrieval_score"], item["file_id"]))
    return ranked


def personalized_order(candidate_pool: Dict[int, dict]) -> List[int]:
    """File-id ordering only -- a thin wrapper over `final_rank` so there
    is exactly one ranking policy in the codebase (reconciles Phase 4A's
    standalone ordering helper with Phase 4B's actual finish-stage policy;
    see app/personalization.py, which re-exports this for backward
    compatibility)."""
    return [item["file_id"] for item in final_rank(candidate_pool)]
