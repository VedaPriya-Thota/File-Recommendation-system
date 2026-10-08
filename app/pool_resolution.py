"""
Deterministic pool-resolution rules (Phase 8) -- architecture_final.md §7,
§8 (touchpoint 2, "investigation/disambiguation") and §14.

Phase 7's evaluation showed two control-flow gaps that share one cause: the
agent could never *use* the evidence it had just gathered to stop.

  1. `retrieve -> rerank -> get_user_activity -> reflect` kept seeing a flat
     RETRIEVAL pool (correctly -- see app/reflect.py's Phase 6 note), so even
     when personalization had clearly separated the contenders the loop ran
     out of strategies and `clarify`'d.
  2. `clarify` always returned an empty ranking, discarding a pool that
     final_rank() could already order defensibly.

This module holds the two small, pure, deterministic rules that fix that. It
never changes final_rank(), the ALPHA/GAMMA weights or any threshold, never
adds/removes a candidate, and never reads anything but the existing pool.

Rule A -- `personalization_tiebreak` (used by reflect(); confident finish)
    Applies only when RETRIEVAL evidence is flat (>= 2 candidates within
    DOMINANT_SCORE_GAP of the top retrieval score) AND get_user_activity has
    actually run. The tie is "meaningfully resolved" iff the candidate that
    final_rank() puts first also leads every other retrieval-contender on
    S_affinity by at least DOMINANT_SCORE_GAP -- the same normalised-[0,1]
    "meaningful gap" the system already uses for retrieval dominance, applied
    to the *other* evidence term. Personalization is therefore never counted
    as retrieval evidence; it only breaks a tie retrieval left open.

Rule B -- `assess_exhausted_pool` (used by the graph when plan_next_action has
    no strategy left on a flat pool)
    Clarification is necessary only when RETRIEVAL evidence leaves the top
    candidates genuinely TIED (equal at the 6-decimal precision final_rank
    stores) AND personalization did not meaningfully resolve them (Rule A is
    checked by reflect() before the loop gets here). If retrieval evidence
    gives a strict leader, the evidence does support an ordering, so the agent
    finishes with the final_rank() shortlist flagged low-confidence
    (architecture §14: "best-available pool, flagged low-confidence") instead
    of discarding it.

    "Tied" is judged on retrieval evidence on purpose: "best-supported" is a
    claim about relevance, and a small affinity edge must not convert an
    ambiguous query into an answer (that would let personalization invent
    confidence, which Rule A forbids by requiring a meaningful margin). No new
    threshold is introduced: "tied" means "indistinguishable at stored
    precision", not a tuned cut-off.
"""
from typing import Dict, List

from app.config import DOMINANT_SCORE_GAP
from app.final_ranking import PERSONALIZATION_SOURCE, final_rank

# --- Decision basis (Phase 9) -------------------------------------------------
# WHY the loop decided to finish. Carried explicitly in stopping_decision["basis"]
# so downstream stages never have to infer it from score gaps (a
# personalization-resolved pool always has a final-score gap below
# DOMINANT_SCORE_GAP, which is indistinguishable from an ordinary near-tie).
#   retrieval_dominance      one candidate clearly dominates on RETRIEVAL evidence
#   personalization_resolved retrieval was flat; Rule A's affinity margin resolved it
#   near_tie_unresolved      finished (cap / exhaustion) with candidates still near-tied
# Any other finish (nothing found, tool error) leaves the basis unset.
BASIS_RETRIEVAL_DOMINANCE = "retrieval_dominance"
BASIS_PERSONALIZATION_RESOLVED = "personalization_resolved"
BASIS_NEAR_TIE_UNRESOLVED = "near_tie_unresolved"
BASIS_WEAK_LEXICAL = "weak_lexical_evidence"  # Phase 12: only a minority-coverage lexical match was ever found

# final_rank() rounds scores to 6 decimals; two scores closer than half a unit
# in the last place are the same stored value.
TIE_TOLERANCE = 5e-7


def has_personalization_evidence(candidate_pool: Dict[int, dict]) -> bool:
    """True once get_user_activity has contributed to at least one candidate."""
    return any(PERSONALIZATION_SOURCE in (e.get("scores") or {}) for e in candidate_pool.values())


def personalization_tiebreak(candidate_pool: Dict[int, dict]) -> dict:
    """Rule A. Pure; never mutates the pool. Returned keys: resolved (bool),
    leader_id, margin, contender_ids, reason."""
    out = {"resolved": False, "leader_id": None, "margin": 0.0, "contender_ids": [], "reason": ""}
    if len(candidate_pool) < 2:
        out["reason"] = "fewer than two candidates"
        return out
    if not has_personalization_evidence(candidate_pool):
        out["reason"] = "no personalization evidence has been gathered"
        return out
    ranked = final_rank(candidate_pool)
    top_retrieval = max(e["retrieval_score"] for e in ranked)
    if top_retrieval <= 0.0:
        out["reason"] = "no retrieval evidence to break a tie among"
        return out
    contenders = [e for e in ranked if top_retrieval - e["retrieval_score"] < DOMINANT_SCORE_GAP]
    out["contender_ids"] = [e["file_id"] for e in contenders]
    if len(contenders) < 2:
        out["reason"] = "retrieval evidence is not flat (handled by retrieval dominance)"
        return out
    leader = ranked[0]
    if leader["file_id"] not in out["contender_ids"]:
        out["reason"] = "final-ranking leader is not a retrieval contender"
        return out
    others = [e for e in contenders if e["file_id"] != leader["file_id"]]
    margin = leader["personalization_score"] - max(e["personalization_score"] for e in others)
    out.update(leader_id=leader["file_id"], margin=round(margin, 6))
    if margin >= DOMINANT_SCORE_GAP:
        out["resolved"] = True
        out["reason"] = (
            f"Retrieval evidence is flat across {len(contenders)} candidates; personalization "
            f"(get_user_activity) separates file {leader['file_id']} by an affinity margin of "
            f"{margin:.2f} (>= {DOMINANT_SCORE_GAP}). Not counted as retrieval dominance."
        )
    else:
        out["reason"] = (
            f"Personalization margin {margin:.2f} is below {DOMINANT_SCORE_GAP}; "
            "it does not meaningfully resolve the flat retrieval pool."
        )
    return out


def assess_exhausted_pool(candidate_pool: Dict[int, dict]) -> dict:
    """Rule B. Pure; never mutates the pool. Returned keys: decision
    ("finish" | "clarify"), reason, leader_id, runner_up_id, gap (retrieval-
    evidence gap between the two best-supported candidates), tied_ids."""
    ranked = final_rank(candidate_pool)
    out = {"decision": "clarify", "reason": "", "leader_id": None, "runner_up_id": None, "gap": None, "tied_ids": []}
    if not ranked:
        out["reason"] = "No candidates to rank."
        return out
    out["leader_id"] = ranked[0]["file_id"]  # the product's own ordering (final_rank) decides who leads
    if len(ranked) == 1:
        out.update(decision="finish", reason="Single candidate remains after all evidence sources.")
        return out
    by_retrieval = sorted(ranked, key=lambda e: (-e["retrieval_score"], e["file_id"]))
    top_r = by_retrieval[0]["retrieval_score"]
    gap = round(top_r - by_retrieval[1]["retrieval_score"], 6)
    out.update(runner_up_id=by_retrieval[1]["file_id"], gap=gap)
    if gap < TIE_TOLERANCE:
        out["tied_ids"] = [e["file_id"] for e in by_retrieval if top_r - e["retrieval_score"] < TIE_TOLERANCE]
        out["reason"] = (
            f"Genuinely ambiguous: {len(out['tied_ids'])} candidates are tied on retrieval evidence and "
            "personalization does not meaningfully separate them; asking the user to disambiguate."
        )
        return out
    out.update(
        decision="finish",
        reason=(
            f"Evidence exhausted without a dominant candidate; file {out['leader_id']} is the best-supported "
            f"result (retrieval-evidence lead {gap:.3f}, below the dominance gap). Returning the ranked "
            "shortlist flagged low-confidence."
        ),
    )
    return out
