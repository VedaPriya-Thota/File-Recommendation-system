"""
Phase 8 -- regression tests for the agent-loop fixes found by the Phase 7
benchmark:

  * personalization-aware termination (pool_resolution Rule A, reflect())
  * clarify vs finish at strategy exhaustion (pool_resolution Rule B, graph)
  * structural "personalization cannot introduce a candidate" guard

The loop is driven through the REAL LangGraph with a scripted registry, so the
tests exercise real control flow (reflect -> plan -> reflect -> ...) without
depending on corpus contents.
"""
import json

import pytest

from app.candidate_pool import update_candidate_pool
from app.config import DOMINANT_SCORE_GAP, FINAL_RANKING_ALPHA, FINAL_RANKING_GAMMA
from app.final_ranking import PERSONALIZATION_SOURCE, compute_component_scores, final_rank, retrieval_evidence_score
from app.graph.build import build_graph
from app.llm import candidate_comparer, goal_interpreter, query_expander, result_explainer
from app.llm.provider import LLMUnavailableError, NullProvider
from app.pool_resolution import (
    TIE_TOLERANCE,
    assess_exhausted_pool,
    has_personalization_evidence,
    personalization_tiebreak,
)
from app.reflect import reflect
from app.run import _initial_state
from app.tools.registry import ToolResult


# --------------------------------------------------------------------------
# scripted registry
# --------------------------------------------------------------------------
def _meta(file_id):
    return {"filename": f"f{file_id}.docx", "file_type": "docx", "owner_id": "user_a"}


class ScriptedRegistry:
    """Every non-personalization tool reports `retrieval` (file_id -> score);
    get_user_activity scores the candidates it is handed from `affinity` and may
    additionally try to smuggle in `smuggle` ids (a misbehaving tool)."""

    def __init__(self, retrieval, affinity=None, smuggle=()):
        self.retrieval, self.affinity, self.smuggle = retrieval, affinity or {}, smuggle
        self.calls = []

    def execute(self, tool, args, user_scope):
        self.calls.append((tool, user_scope))
        if tool == PERSONALIZATION_SOURCE:
            out = [{"file_id": c["file_id"], "score": self.affinity.get(c["file_id"], 0.0), "source": tool,
                    "metadata": _meta(c["file_id"])} for c in args["candidates"]]
            out += [{"file_id": fid, "score": 0.99, "source": tool, "metadata": _meta(fid)} for fid in self.smuggle]
            return ToolResult(tool=tool, candidates=out)
        return ToolResult(tool=tool, candidates=[
            {"file_id": fid, "score": sc, "source": tool, "metadata": _meta(fid)} for fid, sc in self.retrieval.items()
        ])


def run(registry, query="budget report", max_iterations=6, provider=None):
    graph = build_graph(registry=registry, max_iterations=max_iterations, llm_provider=provider or NullProvider())
    return graph.invoke(_initial_state("user_a", query, max_iterations), config={"recursion_limit": 100})


def tools(result):
    return [a["tool"] for a in result["committed_actions"]]


# --------------------------------------------------------------------------
# 1. flat retrieval pool + useful personalization -> finish
# --------------------------------------------------------------------------
def test_flat_pool_with_useful_personalization_finishes_via_tiebreak():
    reg = ScriptedRegistry({1: 0.60, 2: 0.60}, affinity={1: 0.10, 2: 0.90})
    result = run(reg)
    assert tools(result)[-1] == "get_user_activity"
    assert result["stopping_decision"]["status"] == "finish"
    assert "personalization" in result["stopping_decision"]["reason"].lower()
    assert [r["file_id"] for r in result["final_ranking"]] == [2, 1]  # personalized order
    assert result["degraded_mode"] is False  # a resolved tie is not a low-confidence finish
    assert result["confidence"] <= 0.60 * 0.5 + 1e-9  # retrieval-derived, never affinity-derived
    assert result["iteration_count"] <= 6


def test_tiebreak_leader_must_actually_lead_final_ranking():
    # Affinity favours file 1 by a wide margin, but file 2's retrieval edge keeps it
    # ahead in final_rank (0.85*0.80 + 0.15*0.0 > 0.85*0.60 + 0.15*0.9): the tie-break
    # must not claim to resolve in favour of a candidate final_rank doesn't put first.
    pool = {1: {"scores": {"bm25_search": 0.60, PERSONALIZATION_SOURCE: 0.90}, "metadata": _meta(1)},
            2: {"scores": {"bm25_search": 0.80, PERSONALIZATION_SOURCE: 0.00}, "metadata": _meta(2)}}
    tb = personalization_tiebreak(pool)
    assert final_rank(pool)[0]["file_id"] == 2
    assert tb["resolved"] is False and tb["margin"] < DOMINANT_SCORE_GAP


# --------------------------------------------------------------------------
# 2. flat retrieval pool + no useful personalization -> does not finish as "resolved"
# --------------------------------------------------------------------------
def test_flat_pool_with_no_useful_personalization_clarifies_when_tied():
    reg = ScriptedRegistry({1: 0.60, 2: 0.60}, affinity={1: 0.0, 2: 0.0})  # cold start: neutral
    result = run(reg)
    assert "get_user_activity" in tools(result)
    assert result["stopping_decision"]["status"] == "clarify"
    assert "tied" in result["stopping_decision"]["reason"].lower()
    assert result["final_ranking"] == []
    assert "f1.docx" in result["clarification_question"] and "f2.docx" in result["clarification_question"]


def test_weak_personalization_cannot_turn_an_ambiguous_query_into_an_answer():
    # Retrieval is exactly tied and affinity differs by only 0.10 (< DOMINANT_SCORE_GAP): a small affinity
    # edge must not manufacture confidence -- the genuinely ambiguous query still clarifies.
    reg = ScriptedRegistry({1: 0.60, 2: 0.60}, affinity={1: 0.50, 2: 0.40})
    result = run(reg)
    assert "get_user_activity" in tools(result)
    assert result["stopping_decision"]["status"] == "clarify"
    assert result["final_ranking"] == []


def test_strict_retrieval_lead_with_weak_personalization_finishes_low_confidence():
    # Retrieval has a (sub-dominance) strict leader; personalization is too weak to resolve anything on its
    # own, so this is NOT reported as a resolved tie -- it is the low-confidence "best-supported" finish.
    reg = ScriptedRegistry({1: 0.60, 2: 0.50}, affinity={1: 0.50, 2: 0.40})
    result = run(reg)
    assert "not counted as retrieval dominance" not in result["stopping_decision"]["reason"].lower()
    assert result["stopping_decision"]["status"] == "finish" and result["degraded_mode"] is True
    assert [r["file_id"] for r in result["final_ranking"]] == [1, 2]


# --------------------------------------------------------------------------
# 3. personalization cannot introduce a candidate
# --------------------------------------------------------------------------
def test_pool_merge_refuses_personalization_only_candidates():
    pool = update_candidate_pool({}, ToolResult(tool="bm25_search", candidates=[
        {"file_id": 1, "score": 0.5, "source": "bm25_search", "metadata": _meta(1)}]), 1)
    merged = update_candidate_pool(pool, ToolResult(tool=PERSONALIZATION_SOURCE, candidates=[
        {"file_id": 1, "score": 0.9, "source": PERSONALIZATION_SOURCE, "metadata": _meta(1)},
        {"file_id": 99, "score": 0.99, "source": PERSONALIZATION_SOURCE, "metadata": _meta(99)}]), 2)
    assert set(merged) == {1}  # 99 had no retrieval evidence -> never enters
    assert merged[1]["scores"][PERSONALIZATION_SOURCE] == 0.9  # existing candidate still gets its evidence
    # and from an empty pool it can create nothing at all
    assert update_candidate_pool({}, ToolResult(tool=PERSONALIZATION_SOURCE, candidates=[
        {"file_id": 7, "score": 1.0, "source": PERSONALIZATION_SOURCE, "metadata": _meta(7)}]), 1) == {}


def test_misbehaving_personalization_tool_cannot_expand_the_run():
    reg = ScriptedRegistry({1: 0.60, 2: 0.60}, affinity={1: 0.1, 2: 0.9}, smuggle=(99, 100))
    result = run(reg)
    assert set(result["candidate_pool"]) == {1, 2}
    assert {r["file_id"] for r in result["final_ranking"]} <= {1, 2}


def test_resolution_functions_never_add_or_mutate_candidates():
    pool = {1: {"scores": {"bm25_search": 0.6, PERSONALIZATION_SOURCE: 0.9}, "metadata": _meta(1)},
            2: {"scores": {"bm25_search": 0.6, PERSONALIZATION_SOURCE: 0.1}, "metadata": _meta(2)}}
    snapshot = json.dumps(pool, sort_keys=True)
    tb, verdict = personalization_tiebreak(pool), assess_exhausted_pool(pool)
    assert json.dumps(pool, sort_keys=True) == snapshot
    assert set(tb["contender_ids"]) <= set(pool) and verdict["leader_id"] in pool


# --------------------------------------------------------------------------
# 4. deterministic final ranking / formula unchanged
# --------------------------------------------------------------------------
def test_final_score_formula_is_unchanged():
    assert (FINAL_RANKING_ALPHA, FINAL_RANKING_GAMMA) == (0.85, 0.15)
    entry = {"scores": {"bm25_search": 0.7, "rerank_candidates": 0.4, PERSONALIZATION_SOURCE: 0.5}, "metadata": {}}
    comp = compute_component_scores(entry)
    assert comp["retrieval_score"] == 0.7  # max over non-personalization sources
    assert comp["final_score"] == pytest.approx(0.85 * 0.7 + 0.15 * 0.5, abs=1e-6)


def test_ranking_is_deterministic_including_tie_break_by_file_id():
    pool = {5: {"scores": {"bm25_search": 0.5}, "metadata": _meta(5)}, 3: {"scores": {"bm25_search": 0.5}, "metadata": _meta(3)}}
    assert [e["file_id"] for e in final_rank(pool)] == [3, 5]
    reg = lambda: ScriptedRegistry({1: 0.60, 2: 0.60}, affinity={1: 0.1, 2: 0.9})  # noqa: E731
    a, b = run(reg()), run(reg())
    assert [(r["file_id"], r["final_score"]) for r in a["final_ranking"]] == [(r["file_id"], r["final_score"]) for r in b["final_ranking"]]


# --------------------------------------------------------------------------
# 5. retrieval-only dominance is not corrupted
# --------------------------------------------------------------------------
def test_tiebreak_never_applies_when_retrieval_is_not_flat():
    pool = {1: {"scores": {"bm25_search": 0.95, PERSONALIZATION_SOURCE: 0.0}, "metadata": _meta(1)},
            2: {"scores": {"bm25_search": 0.30, PERSONALIZATION_SOURCE: 1.0}, "metadata": _meta(2)}}
    tb = personalization_tiebreak(pool)
    assert tb["resolved"] is False and "not flat" in tb["reason"]
    action = {"tool": "get_user_activity", "args": {}, "reason": "x", "iteration": 3}
    rr = reflect(action, {"candidates": [{"file_id": 1}], "error": None}, pool, [], [], 3, 6)
    assert rr.status == "finish" and rr.confidence == 0.95  # retrieval dominance, untouched
    assert retrieval_evidence_score(pool[2]) == 0.30  # affinity 1.0 is not retrieval evidence


def test_tiebreak_requires_actual_personalization_evidence():
    pool = {1: {"scores": {"bm25_search": 0.6}, "metadata": _meta(1)}, 2: {"scores": {"bm25_search": 0.6}, "metadata": _meta(2)}}
    assert has_personalization_evidence(pool) is False
    assert personalization_tiebreak(pool)["resolved"] is False


# --------------------------------------------------------------------------
# Clarification rule (Rule B)
# --------------------------------------------------------------------------
def test_useful_candidate_with_strict_lead_is_not_discarded():
    # leader 1.0 vs 0.875 -- below the dominance gap, but a real, evidence-based ordering exists
    reg = ScriptedRegistry({1: 1.0, 2: 0.875})
    result = run(reg)
    assert result["stopping_decision"]["status"] == "finish"
    assert [r["file_id"] for r in result["final_ranking"]] == [1, 2]
    assert result["degraded_mode"] is True  # flagged low-confidence, not presented as dominant
    assert "best-supported" in result["stopping_decision"]["reason"]
    exhausted = [t for t in result["trace"] if t["event"] == "plan_next_action_exhausted"][-1]
    assert exhausted["leader_id"] == 1 and exhausted["gap"] > 0  # explainable from the trace


def test_ambiguous_tied_candidates_still_clarify_and_name_only_the_tied_ones():
    result = run(ScriptedRegistry({1: 0.9, 2: 0.9}))
    assert result["stopping_decision"]["status"] == "clarify"
    assert result["final_ranking"] == [] and result["candidate_explanation"] == {}
    v = assess_exhausted_pool({1: {"scores": {"x": 0.9}, "metadata": _meta(1)}, 2: {"scores": {"x": 0.9}, "metadata": _meta(2)},
                               3: {"scores": {"x": 0.5}, "metadata": _meta(3)}})
    assert v["decision"] == "clarify" and v["tied_ids"] == [1, 2]


def test_exhausted_pool_rule_unit_cases():
    mk = lambda *scores: {i + 1: {"scores": {"x": s}, "metadata": _meta(i + 1)} for i, s in enumerate(scores)}  # noqa: E731
    assert assess_exhausted_pool({})["decision"] == "clarify"
    assert assess_exhausted_pool(mk(0.7))["decision"] == "finish"
    assert assess_exhausted_pool(mk(0.7, 0.7))["decision"] == "clarify"
    assert assess_exhausted_pool(mk(0.7, 0.7 - 2 * TIE_TOLERANCE))["decision"] == "finish"
    # tied on retrieval but differing slightly on affinity: personalization alone must not break the tie
    tied_r = {1: {"scores": {"x": 0.7, PERSONALIZATION_SOURCE: 0.30}, "metadata": _meta(1)},
              2: {"scores": {"x": 0.7, PERSONALIZATION_SOURCE: 0.20}, "metadata": _meta(2)}}
    v = assess_exhausted_pool(tied_r)
    assert v["decision"] == "clarify" and v["tied_ids"] == [1, 2]
    v = assess_exhausted_pool(mk(0.8, 0.6))
    assert v["decision"] == "finish" and v["leader_id"] == 1 and v["runner_up_id"] == 2 and v["gap"] == 0.2


def test_no_result_is_safe_and_never_fabricates_or_clarifies():
    result = run(ScriptedRegistry({}))
    assert result["stopping_decision"]["status"] == "finish"
    assert result["final_ranking"] == [] and result["candidate_pool"] == {}
    assert result["degraded_mode"] is True and result["confidence"] == 0.0


def test_iteration_cap_still_bounds_every_path():
    for reg in (ScriptedRegistry({1: 0.6, 2: 0.6}, affinity={1: 0.1, 2: 0.9}),
                ScriptedRegistry({1: 0.6, 2: 0.6}),
                ScriptedRegistry({1: 1.0, 2: 0.875}),
                ScriptedRegistry({})):
        for cap in (1, 2, 3):
            assert run(reg, max_iterations=cap)["iteration_count"] <= cap


# --------------------------------------------------------------------------
# Security: scope isolation and immutability
# --------------------------------------------------------------------------
def test_user_scope_is_immutable_and_passed_unchanged_to_every_tool():
    reg = ScriptedRegistry({1: 0.6, 2: 0.6}, affinity={1: 0.1, 2: 0.9})
    result = run(reg)
    assert result["user_scope"] == ["user_a"]
    scopes = {s.owner_ids for _, s in reg.calls}
    assert scopes == {("user_a",)}  # identical scope object contents on every call


def test_cross_user_isolation_end_to_end_on_the_real_corpus():
    from app.run import run_query
    for user, query, forbidden_owner in (
        ("user_b", "AI architecture slide deck", {"user_a", "user_x", "user_c"}),
        ("user_a", "AI architecture secret roadmap", {"user_x", "user_c"}),
        ("user_c", "salary information hr", {"user_x", "user_a"}),
    ):
        r = run_query(user, query, llm_provider=NullProvider())
        owners = {e["metadata"]["owner_id"] for e in r["final_ranking"]} | {
            e["metadata"]["owner_id"] for e in r["candidate_pool"].values()}
        assert not owners & forbidden_owner and owners <= {user}


# --------------------------------------------------------------------------
# LLM boundaries (5A-5D) and graceful degradation
# --------------------------------------------------------------------------
STAGE_BY_PROMPT = {goal_interpreter.SYSTEM_PROMPT: "goal", query_expander.SYSTEM_PROMPT: "expansion",
                   candidate_comparer.SYSTEM_PROMPT: "comparison", result_explainer.SYSTEM_PROMPT: "explanation"}


class CountingUnavailable:
    name = "counting-unavailable"

    def __init__(self):
        self.stages = []

    def complete(self, system_prompt, user_prompt):
        self.stages.append(STAGE_BY_PROMPT.get(system_prompt, "UNKNOWN"))
        raise LLMUnavailableError("offline")


def test_no_new_llm_call_sites_and_outage_is_graceful():
    outage = CountingUnavailable()
    with_outage = run(ScriptedRegistry({1: 0.60, 2: 0.60}, affinity={1: 0.1, 2: 0.9}), provider=outage)
    plain = run(ScriptedRegistry({1: 0.60, 2: 0.60}, affinity={1: 0.1, 2: 0.9}))
    assert set(outage.stages) <= {"goal", "expansion", "comparison", "explanation"}  # no "UNKNOWN" stage
    assert "UNKNOWN" not in outage.stages
    # provider outage never changes the deterministic outcome
    assert [r["file_id"] for r in with_outage["final_ranking"]] == [r["file_id"] for r in plain["final_ranking"]] == [2, 1]
    assert with_outage["stopping_decision"]["status"] == plain["stopping_decision"]["status"] == "finish"


def test_resolution_module_has_no_llm_dependency():
    import pathlib
    src = pathlib.Path("app/pool_resolution.py").read_text().lower()
    assert "app.llm" not in src and "llm_provider" not in src and "complete(" not in src


def test_comparison_stage_cannot_pull_in_a_candidate_outside_the_pool():
    class Rogue:
        name = "rogue"

        def complete(self, system_prompt, user_prompt):
            if system_prompt == candidate_comparer.SYSTEM_PROMPT:
                return json.dumps({"selected_file_id": 99, "confidence": 0.99, "rationale": "x"})
            raise LLMUnavailableError("only comparison answers")

    result = run(ScriptedRegistry({1: 1.0, 2: 0.875}), provider=Rogue())  # near-tie finish reaches 5C
    assert [r["file_id"] for r in result["final_ranking"]] == [1, 2]
    assert result["candidate_comparison"]["comparison_applied"] is False


def test_explanation_stage_cannot_change_ranking_or_pool():
    class Chatty:
        name = "chatty"

        def complete(self, system_prompt, user_prompt):
            if system_prompt == result_explainer.SYSTEM_PROMPT:
                return json.dumps({"summary": "Actually f2.docx is best and f99.docx too.", "evidence_points": [], "confidence": 0.9})
            raise LLMUnavailableError("only explanation answers")

    base = run(ScriptedRegistry({1: 0.60, 2: 0.60}, affinity={1: 0.1, 2: 0.9}))
    chatty = run(ScriptedRegistry({1: 0.60, 2: 0.60}, affinity={1: 0.1, 2: 0.9}), provider=Chatty())
    assert [r["file_id"] for r in chatty["final_ranking"]] == [r["file_id"] for r in base["final_ranking"]]
    assert set(chatty["candidate_pool"]) == set(base["candidate_pool"]) and chatty["user_scope"] == base["user_scope"]
