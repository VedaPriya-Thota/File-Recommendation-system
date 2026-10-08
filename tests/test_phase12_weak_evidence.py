"""
Phase 12 -- weak lexical evidence must not be treated as dominant.

Demonstrated by the Phase 11 real-content evaluation: for S3 ("obligations when renting an apartment") and S6
("generating electricity from sunlight at home") the semantic stage ALONE ranks the right file first, but the
agent never asked it, because bm25_search's single hit (matching 1 of 4 query terms: `when` / `sunlight`) was
max-normalised to 1.0 and a lone candidate counts as dominant at any strength.

The fix is evidence-driven, not a fixed pipeline: bm25 candidates report `coverage`; a lexical match covering a
MINORITY of the query's terms is partial evidence (scaled by its coverage); if it is a candidate's only
retrieval evidence the planner seeks independent (semantic) evidence once; if none exists the agent finishes
with it flagged low-confidence rather than clarifying or discarding it.
"""
import json

import pytest

from app import config
from app.candidate_pool import update_candidate_pool
from app.final_ranking import (LEXICAL_MAJORITY, final_rank, is_weak_lexical, leading_is_weak_lexical,
                               retrieval_evidence_score)
from app.graph.build import build_graph
from app.llm import candidate_comparer, goal_interpreter, query_expander, result_explainer
from app.llm.provider import LLMUnavailableError, NullProvider
from app.planning import plan_next_action
from app.pool_resolution import BASIS_WEAK_LEXICAL
from app.reflect import reflect
from app.run import _initial_state
from app.goal import understand_goal
from app.tools.registry import ToolResult


def meta(fid):
    return {"filename": f"f{fid}.txt", "file_type": "txt", "owner_id": "user_a"}


def entry(fid, scores, coverage=None):
    e = {"file_id": fid, "scores": dict(scores), "metadata": meta(fid), "evidence_sources": list(scores)}
    if coverage is not None:
        e["coverage"] = dict(coverage)
    return e


# ============================================================ evidence representation
def test_a_minority_lexical_match_contributes_only_score_times_coverage():
    weak = entry(1, {"bm25_search": 1.0}, {"bm25_search": 0.25})
    assert retrieval_evidence_score(weak) == pytest.approx(0.25)
    assert retrieval_evidence_score(entry(2, {"bm25_search": 0.8}, {"bm25_search": 0.25})) == pytest.approx(0.2)


@pytest.mark.parametrize("coverage,weak", [(0.25, True), (0.49, True), (0.5, False), (0.75, False), (1.0, False), (None, False)])
def test_the_boundary_is_strictly_less_than_half(coverage, weak):
    e = entry(1, {"bm25_search": 1.0}, {"bm25_search": coverage} if coverage is not None else None)
    assert is_weak_lexical(e) is weak
    assert retrieval_evidence_score(e) == pytest.approx(1.0 if not weak else coverage)
    assert LEXICAL_MAJORITY == 0.5


def test_corroboration_by_independent_evidence_means_not_weak():
    base = {"bm25_search": 1.0}
    cov = {"bm25_search": 0.25}
    assert is_weak_lexical(entry(1, base, cov))
    assert not is_weak_lexical(entry(1, {**base, "semantic_search": 0.4}, cov))
    assert not is_weak_lexical(entry(1, {**base, "metadata_search": 0.6}, cov))
    assert is_weak_lexical(entry(1, {**base, "hybrid_search": 0.03, "rerank_candidates": 1.0}, cov))  # derived/reordering evidence is not independent
    assert retrieval_evidence_score(entry(1, {**base, "semantic_search": 0.4}, cov)) == pytest.approx(0.4)


def test_entries_without_coverage_keep_exactly_the_previous_behaviour():
    legacy = {"scores": {"bm25_search": 1.0, "semantic_search": 0.3}, "metadata": {}}
    assert retrieval_evidence_score(legacy) == 1.0 and not is_weak_lexical(legacy)
    assert not leading_is_weak_lexical({}) and not leading_is_weak_lexical({1: legacy})


def test_final_score_formula_and_weights_are_unchanged():
    assert (config.FINAL_RANKING_ALPHA, config.FINAL_RANKING_GAMMA, config.DOMINANT_SCORE_GAP) == (0.85, 0.15, 0.25)
    r = final_rank({1: entry(1, {"semantic_search": 0.6}), 2: entry(2, {"bm25_search": 1.0}, {"bm25_search": 0.25})})
    assert [x["file_id"] for x in r] == [1, 2]                              # 0.6 beats a quarter-explained lexical hit
    assert r[0]["final_score"] == pytest.approx(0.85 * 0.6)


def test_pool_carries_coverage_per_source_and_it_survives_later_merges_that_replace_metadata():
    p = update_candidate_pool({}, ToolResult(tool="bm25_search", candidates=[
        {"file_id": 1, "score": 1.0, "source": "bm25_search", "coverage": 0.25, "metadata": meta(1)}]), 1)
    assert p[1]["coverage"] == {"bm25_search": 0.25}
    p = update_candidate_pool(p, ToolResult(tool="hybrid_search", candidates=[
        {"file_id": 1, "score": 0.03, "source": "hybrid_search", "metadata": {"filename": "other-metadata"}}]), 2)
    assert p[1]["metadata"] == {"filename": "other-metadata"}              # metadata WAS replaced...
    assert p[1]["coverage"] == {"bm25_search": 0.25}                         # ...the evidence strength was not lost
    assert p[1]["scores"] == {"bm25_search": 1.0, "hybrid_search": 0.03}


def test_coverage_that_arrives_on_a_later_merge_is_recorded():
    """A file first found by metadata (no coverage) and later surfaced by bm25 must gain its coverage."""
    p = update_candidate_pool({}, ToolResult(tool="metadata_search", candidates=[
        {"file_id": 1, "score": 0.4, "source": "metadata_search", "metadata": meta(1)}]), 1)
    assert p[1]["coverage"] == {}
    p = update_candidate_pool(p, ToolResult(tool="bm25_search", candidates=[
        {"file_id": 1, "score": 1.0, "source": "bm25_search", "coverage": 0.25, "metadata": meta(1)}]), 2)
    assert p[1]["coverage"] == {"bm25_search": 0.25} and set(p[1]["scores"]) == {"metadata_search", "bm25_search"}
    assert not is_weak_lexical(p[1])                                    # corroborated by metadata evidence -> not weak


def test_pool_entries_for_tools_without_coverage_get_an_empty_slot():
    p = update_candidate_pool({}, ToolResult(tool="metadata_search", candidates=[
        {"file_id": 1, "score": 0.9, "source": "metadata_search", "metadata": meta(1)}]), 1)
    assert p[1]["coverage"] == {}


# ============================================================ reflect
ACT = {"tool": "bm25_search", "args": {}, "reason": "x", "iteration": 2}
OBS = {"candidates": [{"file_id": 1}], "error": None}


def test_reflect_does_not_accept_a_lone_weak_lexical_candidate_as_dominant():
    pool = {1: entry(1, {"bm25_search": 1.0}, {"bm25_search": 0.25})}
    rr = reflect(ACT, OBS, pool, [], [], 2, 6)
    assert rr.status == "replan" and rr.observation_hint == "weak_evidence"
    assert rr.confidence == 0.25 and "minority" in rr.reason


def test_reflect_still_finishes_on_full_or_majority_lexical_evidence():
    for cov in (0.5, 1.0, None):
        pool = {1: entry(1, {"bm25_search": 1.0}, {"bm25_search": cov} if cov is not None else None)}
        rr = reflect(ACT, OBS, pool, [], [], 2, 6)
        assert rr.status == "finish" and rr.resolution_basis == "retrieval_dominance", cov


def test_reflect_at_the_iteration_cap_finishes_the_weak_candidate_flagged_low_confidence():
    pool = {1: entry(1, {"bm25_search": 1.0}, {"bm25_search": 0.25})}
    rr = reflect(ACT, OBS, pool, [], [], 6, 6)
    assert rr.status == "finish" and rr.low_confidence is True and rr.resolution_basis == BASIS_WEAK_LEXICAL


def test_weak_confidence_is_capped_at_coverage_even_if_a_later_stage_inflated_the_score():
    pool = {1: entry(1, {"bm25_search": 1.0, "rerank_candidates": 1.0}, {"bm25_search": 0.25})}   # min-max rerank gave 1.0
    rr = reflect({"tool": "rerank_candidates", "args": {}, "reason": "x", "iteration": 5}, OBS, pool, [], [], 5, 6)
    assert rr.observation_hint == "weak_evidence" and rr.confidence == 0.25


def test_a_corroborated_candidate_is_not_treated_as_weak():
    pool = {1: entry(1, {"bm25_search": 1.0, "semantic_search": 1.0}, {"bm25_search": 0.25}),
            2: entry(2, {"semantic_search": 0.5})}
    rr = reflect(ACT, OBS, pool, [], [], 3, 6)
    assert rr.status == "finish" and rr.resolution_basis == "retrieval_dominance"


# ============================================================ planner
def goal_for(q):
    g = understand_goal(q).to_dict()
    g["expanded_terms"] = []
    return g


def test_planner_answers_weak_evidence_with_semantic_search_on_the_raw_query_once():
    g = goal_for("obligations when renting an apartment")
    a = plan_next_action(g, [], [], 3, observation_hint="weak_evidence")
    assert a["tool"] == "semantic_search" and a["args"]["query"] == "obligations when renting an apartment"
    assert "weak" in a["reason"].lower()
    committed = [a]
    assert plan_next_action(g, committed, [], 4, observation_hint="weak_evidence") is None     # never proposed twice


def test_planner_has_nothing_to_offer_when_semantic_was_already_tried_or_failed():
    from app.planning import args_signature
    g = goal_for("obligations when renting an apartment")
    sem = {"tool": "semantic_search", "args": {"query": "x"}, "reason": "", "iteration": 3}
    assert plan_next_action(g, [sem], [], 4, observation_hint="weak_evidence") is None
    sig = args_signature("semantic_search", {"query": g["raw_query"]})
    assert plan_next_action(g, [], [sig], 4, observation_hint="weak_evidence") is None


def test_other_hints_are_unaffected_by_the_new_branch():
    g = goal_for("quarterly budget review")
    assert plan_next_action(g, [], [], 1)["tool"] == "metadata_search"                       # first action unchanged
    assert plan_next_action(g, [], [], 3, observation_hint="too_many") is None                # nothing to fuse/rerank without a pool


# ============================================================ the real graph (scripted tools)
class Registry:
    """metadata finds nothing; bm25/semantic return what the test scripts; everything else returns the pool unchanged."""

    def __init__(self, bm25=(), semantic=()):
        self.bm25, self.semantic, self.calls = list(bm25), list(semantic), []

    def execute(self, tool, args, user_scope):
        self.calls.append((tool, user_scope))
        cands = {"bm25_search": self.bm25, "semantic_search": self.semantic}.get(tool, [])
        return ToolResult(tool=tool, candidates=[dict(c, source=tool) for c in cands])


def bm25c(fid, score=1.0, coverage=None):
    c = {"file_id": fid, "score": score, "metadata": meta(fid)}
    if coverage is not None:
        c["coverage"] = coverage
    return c


def semc(fid, score):
    return {"file_id": fid, "score": score, "metadata": meta(fid)}


def run(reg, query="obligations when renting an apartment", provider=None, cap=6):
    g = build_graph(registry=reg, max_iterations=cap, llm_provider=provider or NullProvider())
    return g.invoke(_initial_state("user_a", query, cap), config={"recursion_limit": 100})


def tools(r):
    return [a["tool"] for a in r["committed_actions"]]


def ids(r):
    return [x["file_id"] for x in r["final_ranking"]]


def test_weak_lexical_hit_does_not_stop_the_agent_and_independent_evidence_wins():
    reg = Registry(bm25=[bm25c(1, 1.0, 0.25)], semantic=[semc(2, 1.0), semc(3, 0.4)])
    r = run(reg)
    assert tools(r) == ["metadata_search", "bm25_search", "semantic_search"]
    assert r["stopping_decision"]["status"] == "finish" and ids(r)[0] == 2           # the semantic leader, not the weak lexical hit
    assert r["stopping_decision"]["basis"] == "retrieval_dominance" and r["iteration_count"] == 3
    weak = r["candidate_pool"][1]
    assert weak["coverage"] == {"bm25_search": 0.25} and 1 in ids(r)                   # still in the pool/ranking, just not leading


def test_strong_lexical_evidence_still_stops_immediately_so_the_agent_is_not_a_fixed_pipeline():
    reg = Registry(bm25=[bm25c(1, 1.0, 1.0)], semantic=[semc(2, 1.0)])
    r = run(reg, query="quarterly budget review")
    assert tools(r) == ["metadata_search", "bm25_search"] and ids(r) == [1]            # semantic was NOT consulted
    assert r["stopping_decision"]["basis"] == "retrieval_dominance"
    assert "semantic_search" not in [t for t, _ in reg.calls]


def test_majority_coverage_exactly_half_is_still_accepted():
    r = run(Registry(bm25=[bm25c(1, 1.0, 0.5)], semantic=[semc(2, 1.0)]), query="how photosynthesis works")
    assert tools(r) == ["metadata_search", "bm25_search"] and ids(r) == [1]


def test_weak_candidate_with_no_independent_evidence_is_returned_flagged_not_discarded_or_clarified():
    reg = Registry(bm25=[bm25c(1, 1.0, 0.25)], semantic=[])
    r = run(reg)
    assert r["stopping_decision"]["status"] == "finish"                                # never "clarify"
    assert ids(r) == [1]                                                               # the candidate is kept
    assert r["stopping_decision"]["basis"] == BASIS_WEAK_LEXICAL and r["degraded_mode"] is True
    assert "weak lexical" in r["stopping_decision"]["reason"].lower()
    assert r["confidence"] == 0.25                                                     # honest: not 0.0 ("nothing found"), not 1.0
    assert tools(r).count("semantic_search") == 1


def test_semantic_is_consulted_at_most_once_and_iterations_stay_bounded():
    for cap in (2, 3, 4, 6):
        r = run(Registry(bm25=[bm25c(1, 1.0, 0.25)], semantic=[semc(2, 0.9), semc(3, 0.85)]), cap=cap)
        assert r["iteration_count"] <= cap and tools(r).count("semantic_search") <= 1


def test_scope_is_unchanged_and_candidate_set_closure_holds():
    reg = Registry(bm25=[bm25c(1, 1.0, 0.25)], semantic=[semc(2, 1.0)])
    r = run(reg)
    assert {s.owner_ids for _, s in reg.calls} == {("user_a",)}                       # the same immutable scope on every call
    assert set(r["candidate_pool"]) == {1, 2} and set(ids(r)) <= {1, 2}


def test_no_new_llm_call_site_and_outage_changes_nothing():
    STAGE = {goal_interpreter.SYSTEM_PROMPT: "goal", query_expander.SYSTEM_PROMPT: "expansion",
             candidate_comparer.SYSTEM_PROMPT: "comparison", result_explainer.SYSTEM_PROMPT: "explanation"}

    class Off:
        name = "off"

        def __init__(self):
            self.stages = []

        def complete(self, system_prompt, user_prompt):
            self.stages.append(STAGE.get(system_prompt, "UNKNOWN"))
            raise LLMUnavailableError("offline")

    off = Off()
    reg = lambda: Registry(bm25=[bm25c(1, 1.0, 0.25)], semantic=[semc(2, 1.0), semc(3, 0.4)])  # noqa: E731
    with_off, plain = run(reg(), provider=off), run(reg())
    assert "UNKNOWN" not in off.stages and set(off.stages) <= {"goal", "expansion", "comparison", "explanation"}
    assert ids(with_off) == ids(plain) and with_off["stopping_decision"] == plain["stopping_decision"]


def test_resolved_basis_vocabulary_is_strings_only_and_5c_gate_is_unaffected():
    from app import pool_resolution as pr
    assert pr.BASIS_WEAK_LEXICAL == "weak_lexical_evidence"
    numeric = {k for k, v in vars(pr).items() if not k.startswith("_") and isinstance(v, (int, float)) and not isinstance(v, bool) and k not in vars(config)}
    assert numeric == {"TIE_TOLERANCE"}
    # a weak-lexical finish is NOT a personalization-resolved pool, so 5C keeps its existing gate for it
    r = run(Registry(bm25=[bm25c(1, 1.0, 0.25)], semantic=[]))
    assert r["candidate_comparison"]["comparison_status"] != "skipped_personalization_resolved"


# ============================================================ hybrid determinism (found by the Phase 12 extended evaluation)
def test_hybrid_search_orders_tied_candidates_by_file_id_whatever_their_magnitude(monkeypatch):
    """Candidates tied on RRF score used to keep SET-iteration order, which for integers depends on their absolute
    values -- so the same corpus ingested with different serial ids produced a different tail order."""
    from app.tools import hybrid_search as hs

    # ids chosen so that a plain set iterates NON-ascending (64, 17, 3, 100 on CPython)
    assert list({100, 3, 17, 64}) != sorted({100, 3, 17, 64})

    def cand(fid, src):
        return {"file_id": fid, "score": 1.0, "source": src, "metadata": meta(fid)}

    bm25 = [cand(100, "bm25_search"), cand(3, "bm25_search")]            # ranks 1, 2
    sem = [cand(64, "semantic_search"), cand(17, "semantic_search")]      # ranks 1, 2  -> tied pairwise with the above
    monkeypatch.setattr(hs, "bm25_search", lambda a, s: ToolResult(tool="bm25_search", candidates=bm25))
    monkeypatch.setattr(hs, "semantic_search", lambda a, s: ToolResult(tool="semantic_search", candidates=sem))
    res = hs.hybrid_search({"query": "anything"}, None)
    assert [c["file_id"] for c in res.candidates] == [64, 100, 3, 17]    # ties: ascending file_id, scores descending


def test_hybrid_ordering_is_independent_of_id_offsets(monkeypatch):
    from app.tools import hybrid_search as hs

    def run_with_offset(off):
        mk = lambda fid, src: {"file_id": fid + off, "score": 1.0, "source": src, "metadata": meta(fid + off)}  # noqa: E731
        monkeypatch.setattr(hs, "bm25_search", lambda a, s: ToolResult(tool="bm25_search", candidates=[mk(i, "bm25_search") for i in (9, 4, 7)]))
        monkeypatch.setattr(hs, "semantic_search", lambda a, s: ToolResult(tool="semantic_search", candidates=[mk(i, "semantic_search") for i in (2, 8, 1)]))
        return [c["file_id"] - off for c in hs.hybrid_search({"query": "x"}, None).candidates]

    assert run_with_offset(0) == run_with_offset(1000) == run_with_offset(7919)
