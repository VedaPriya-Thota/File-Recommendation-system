"""
Phase 9 -- Phase 5C must not re-adjudicate a decision the deterministic
personalization-resolution path (Phase 8, Rule A) already made.

Root cause being guarded: 5C's eligibility gate looked only at final-score
gaps, and a personalization-resolved pool ALWAYS has a final gap below
DOMINANT_SCORE_GAP (affinity is only 15% of the score), so it looked like an
ordinary near-tie. The fix carries the reason explicitly in
stopping_decision["basis"]; nothing is inferred from the gap.

The loop is driven through the real LangGraph with a scripted registry and a
recording provider that deliberately picks the OPPOSITE candidate in 5C.
"""
import json

import pytest

from app import config
from app import pool_resolution as pr
from app.config import DOMINANT_SCORE_GAP
from app.final_ranking import PERSONALIZATION_SOURCE
from app.graph.build import build_graph
from app.llm import candidate_comparer, goal_interpreter, query_expander, result_explainer
from app.llm.provider import LLMUnavailableError, NullProvider
from app.reflect import reflect
from app.run import _initial_state
from app.tools.registry import ToolResult

STAGE = {goal_interpreter.SYSTEM_PROMPT: "goal", query_expander.SYSTEM_PROMPT: "expansion",
         candidate_comparer.SYSTEM_PROMPT: "comparison", result_explainer.SYSTEM_PROMPT: "explanation"}


def _meta(fid):
    return {"filename": f"f{fid}.docx", "file_type": "docx", "owner_id": "user_a"}


class Registry:
    """Non-personalization tools report `retrieval`; get_user_activity scores supplied candidates from
    `affinity` and may also try to smuggle in extra ids."""

    def __init__(self, retrieval, affinity=None, smuggle=()):
        self.retrieval, self.affinity, self.smuggle = retrieval, affinity or {}, smuggle

    def execute(self, tool, args, user_scope):
        if tool == PERSONALIZATION_SOURCE:
            out = [{"file_id": c["file_id"], "score": self.affinity.get(c["file_id"], 0.0), "source": tool,
                    "metadata": _meta(c["file_id"])} for c in args["candidates"]]
            out += [{"file_id": f, "score": 0.99, "source": tool, "metadata": _meta(f)} for f in self.smuggle]
            return ToolResult(tool=tool, candidates=out)
        return ToolResult(tool=tool, candidates=[{"file_id": f, "score": s, "source": tool, "metadata": _meta(f)}
                                                 for f, s in self.retrieval.items()])


class Recorder:
    """Counts calls per stage. In 5C it selects the candidate that is NOT first (the opposite of the
    deterministic leader) with very high confidence; `mode` can make 5C misbehave instead."""
    name = "recorder"

    def __init__(self, mode="opposite"):
        self.mode, self.calls = mode, {s: 0 for s in ("goal", "expansion", "comparison", "explanation", "UNKNOWN")}

    def complete(self, system_prompt, user_prompt):
        stage = STAGE.get(system_prompt, "UNKNOWN")
        self.calls[stage] += 1
        if self.mode == "unavailable":
            raise LLMUnavailableError("offline")
        if stage == "comparison":
            if self.mode == "invalid":
                return "not json at all"
            cands = json.loads(user_prompt.split("\n", 1)[1])["candidates"]
            if self.mode == "unknown":
                return json.dumps({"selected_file_id": 999, "confidence": 0.99, "rationale": "x"})
            if self.mode == "low_confidence":
                return json.dumps({"selected_file_id": cands[-1]["file_id"], "confidence": 0.2, "rationale": "x"})
            return json.dumps({"selected_file_id": cands[-1]["file_id"], "confidence": 0.99, "rationale": "opposite"})
        if stage == "explanation":
            f = json.loads(user_prompt.split("\n", 1)[1])["result"]["filename"]
            return json.dumps({"summary": f"{f} is the selected result.", "evidence_points": [], "confidence": 0.8})
        raise LLMUnavailableError("only 5C/5D answer")


def run(registry, provider, query="budget report", max_iterations=6):
    g = build_graph(registry=registry, max_iterations=max_iterations, llm_provider=provider)
    return g.invoke(_initial_state("user_a", query, max_iterations), config={"recursion_limit": 100})


def ids(result):
    return [r["file_id"] for r in result["final_ranking"]]


RESOLVED = lambda **kw: Registry({1: 0.60, 2: 0.60}, affinity={1: 0.10, 2: 0.90}, **kw)   # noqa: E731  Rule A resolves -> [2, 1]
ORDINARY = lambda: Registry({1: 1.00, 2: 0.875})                                          # noqa: E731  Rule B near-tie  -> [1, 2]


# ---------------------------------------------------------------------------
# Test 1 + 5 -- personalization-resolved result skips 5C; the LLM cannot reverse it
# ---------------------------------------------------------------------------
def test_personalization_resolved_pool_is_not_readjudicated_even_by_an_opposing_llm():
    rec = Recorder("opposite")
    result = run(RESOLVED(), rec)
    assert result["stopping_decision"]["status"] == "finish"
    assert result["stopping_decision"]["basis"] == pr.BASIS_PERSONALIZATION_RESOLVED
    assert ids(result) == [2, 1]                      # the deterministic, personalized order
    assert rec.calls["comparison"] == 0               # provider never consulted by 5C
    cmp = result["candidate_comparison"]
    assert cmp["comparison_status"] == "skipped_personalization_resolved"
    assert cmp["comparison_applied"] is False and cmp["comparison_invoked"] is False
    assert cmp["near_tie_gate_open"] is True          # it WOULD have been eligible on score gap alone
    assert cmp["deterministic_order"] == [2, 1]
    assert [e["final_score"] for e in result["final_ranking"]] == sorted(
        [e["final_score"] for e in result["final_ranking"]], reverse=True)


def test_the_resolved_pool_really_does_look_like_an_ordinary_near_tie_on_score_gap_alone():
    """Guards the reason the explicit basis exists: gap-based inference cannot tell the two apart."""
    resolved, ordinary = run(RESOLVED(), NullProvider()), run(ORDINARY(), NullProvider())
    gap = lambda r: r["final_ranking"][0]["final_score"] - r["final_ranking"][1]["final_score"]  # noqa: E731
    assert gap(resolved) < DOMINANT_SCORE_GAP and gap(ordinary) < DOMINANT_SCORE_GAP  # both pass the 5C gate
    assert resolved["stopping_decision"]["basis"] != ordinary["stopping_decision"]["basis"]


# ---------------------------------------------------------------------------
# Test 7 -- 5D may still explain, and may not change anything
# ---------------------------------------------------------------------------
def test_explanation_still_runs_for_a_resolved_pool_and_changes_nothing():
    rec = Recorder("opposite")
    with_5d = run(RESOLVED(), rec)
    without = run(RESOLVED(), NullProvider())
    assert rec.calls["explanation"] == 1
    assert with_5d["candidate_explanation"]["explanation_status"] == "generated"
    assert with_5d["candidate_explanation"]["explained_file_id"] == 2
    assert ids(with_5d) == ids(without) == [2, 1]
    assert set(with_5d["candidate_pool"]) == set(without["candidate_pool"]) and with_5d["user_scope"] == without["user_scope"]


def test_5d_describes_a_skipped_comparison_as_deterministic_ranking():
    from app.llm.result_explainer import build_explanation_evidence
    skipped = {"comparison_status": "skipped_personalization_resolved", "comparison_applied": False}
    ev = build_explanation_evidence("q", {}, {"file_id": 2, "metadata": _meta(2), "final_score": 0.6}, skipped)
    assert ev["selection_basis"] == "deterministic_ranking"


# ---------------------------------------------------------------------------
# Test 2 -- ordinary near-ties still reach 5C, with its existing behaviour
# ---------------------------------------------------------------------------
def test_ordinary_near_tie_still_invokes_5c_and_it_may_apply():
    rec = Recorder("opposite")
    result = run(ORDINARY(), rec)
    assert result["stopping_decision"]["basis"] == pr.BASIS_NEAR_TIE_UNRESOLVED
    assert rec.calls["comparison"] == 1
    cmp = result["candidate_comparison"]
    assert cmp["comparison_status"] == "selected" and cmp["comparison_applied"] is True
    assert cmp["deterministic_order"] == [1, 2] and ids(result) == [2, 1]  # existing 5C behaviour: promoted within the set


def test_near_tie_finished_at_the_iteration_cap_is_also_5c_eligible():
    rec = Recorder("opposite")
    result = run(Registry({1: 0.60, 2: 0.60}, affinity={1: 0.0, 2: 0.0}), rec, max_iterations=3)  # flat at cap
    assert result["stopping_decision"]["basis"] == pr.BASIS_NEAR_TIE_UNRESOLVED
    assert result["iteration_count"] <= 3 and rec.calls["comparison"] == 1


@pytest.mark.parametrize("mode,status", [("low_confidence", "low_confidence"), ("invalid", "invalid_output"),
                                         ("unknown", "unknown_candidate"), ("unavailable", "unavailable")])
def test_5c_rejections_and_outage_leave_the_deterministic_order_untouched(mode, status):
    rec = Recorder(mode)
    result = run(ORDINARY(), rec)
    assert rec.calls["comparison"] == 1
    assert result["candidate_comparison"]["comparison_status"] == status
    assert result["candidate_comparison"]["comparison_applied"] is False
    assert ids(result) == [1, 2] and result["stopping_decision"]["status"] == "finish"


# ---------------------------------------------------------------------------
# Test 3 -- retrieval dominance is unaffected
# ---------------------------------------------------------------------------
def test_retrieval_dominance_keeps_its_own_basis_and_5c_stays_governed_by_its_existing_gate():
    rec = Recorder("opposite")
    result = run(Registry({1: 0.95, 2: 0.30}, affinity={1: 0.0, 2: 1.0}), rec)
    assert result["stopping_decision"]["basis"] == pr.BASIS_RETRIEVAL_DOMINANCE
    assert result["iteration_count"] == 1 and ids(result) == [1, 2]
    assert result["candidate_comparison"]["comparison_status"] == "not_needed"  # existing gate: clear winner
    assert rec.calls["comparison"] == 0


def test_affinity_never_masquerades_as_retrieval_dominance_in_reflect():
    pool = {1: {"scores": {"bm25_search": 0.95, PERSONALIZATION_SOURCE: 0.0}, "metadata": _meta(1)},
            2: {"scores": {"bm25_search": 0.30, PERSONALIZATION_SOURCE: 1.0}, "metadata": _meta(2)}}
    act = {"tool": "get_user_activity", "args": {}, "reason": "x", "iteration": 3}
    rr = reflect(act, {"candidates": [{"file_id": 1}], "error": None}, pool, [], [], 3, 6)
    assert rr.status == "finish" and rr.resolution_basis == pr.BASIS_RETRIEVAL_DOMINANCE and rr.confidence == 0.95
    flat = {1: {"scores": {"bm25_search": 0.60, PERSONALIZATION_SOURCE: 0.1}, "metadata": _meta(1)},
            2: {"scores": {"bm25_search": 0.60, PERSONALIZATION_SOURCE: 0.9}, "metadata": _meta(2)}}
    rr = reflect(act, {"candidates": [{"file_id": 1}], "error": None}, flat, [], [], 3, 6)
    assert rr.resolution_basis == pr.BASIS_PERSONALIZATION_RESOLVED and rr.confidence <= 0.30 + 1e-9  # still retrieval-derived


def test_basis_is_unset_where_no_ranking_basis_exists():
    nothing = run(Registry({}), NullProvider())
    assert nothing["stopping_decision"]["status"] == "finish" and "basis" not in nothing["stopping_decision"]
    tied = run(Registry({1: 0.60, 2: 0.60}), NullProvider())
    assert tied["stopping_decision"]["status"] == "clarify" and "basis" not in tied["stopping_decision"]


# ---------------------------------------------------------------------------
# Test 4 -- personalization cannot add candidates (Phase 8 guard preserved on the new path)
# ---------------------------------------------------------------------------
def test_resolved_pool_cannot_be_expanded_by_the_personalization_tool():
    rec = Recorder("opposite")
    result = run(RESOLVED(smuggle=(99, 100)), rec)
    assert result["stopping_decision"]["basis"] == pr.BASIS_PERSONALIZATION_RESOLVED
    assert set(result["candidate_pool"]) == {1, 2} and set(ids(result)) == {1, 2}
    assert rec.calls["comparison"] == 0


# ---------------------------------------------------------------------------
# Test 6 -- LLM outage stays safe on both paths
# ---------------------------------------------------------------------------
def test_outage_is_graceful_on_the_resolved_path_and_the_ordinary_path():
    off = Recorder("unavailable")
    resolved = run(RESOLVED(), off)
    assert ids(resolved) == [2, 1] and resolved["stopping_decision"]["status"] == "finish"
    assert off.calls["comparison"] == 0           # skipped, not even attempted
    assert resolved["candidate_explanation"]["explanation_status"] != "generated"  # 5D degrades, ranking intact
    off2 = Recorder("unavailable")
    ordinary = run(ORDINARY(), off2)
    assert ids(ordinary) == [1, 2] and off2.calls["comparison"] == 1
    assert ids(run(RESOLVED(), NullProvider())) == ids(resolved)  # identical to the no-LLM deterministic run


# ---------------------------------------------------------------------------
# Scope guards: no new thresholds / weights; determinism; no LLM in the resolution module
# ---------------------------------------------------------------------------
def test_no_threshold_or_weight_changed_and_no_new_numeric_constant_added():
    assert (config.FINAL_RANKING_ALPHA, config.FINAL_RANKING_GAMMA) == (0.85, 0.15)
    assert (config.DOMINANT_SCORE_GAP, config.COMPARISON_TOP_N, config.COMPARISON_MIN_CONFIDENCE) == (0.25, 3, 0.5)
    # numeric constants DEFINED by the module (imported config values excluded): only the Phase 8 tie tolerance
    numeric = {k for k, v in vars(pr).items() if not k.startswith("_") and isinstance(v, (int, float))
               and not isinstance(v, bool) and k not in vars(config)}
    assert numeric == {"TIE_TOLERANCE"}               # Phase 9 added vocabulary (strings) only
    assert {pr.BASIS_RETRIEVAL_DOMINANCE, pr.BASIS_PERSONALIZATION_RESOLVED, pr.BASIS_NEAR_TIE_UNRESOLVED} == {
        "retrieval_dominance", "personalization_resolved", "near_tie_unresolved"}


def test_resolved_path_is_deterministic_across_runs():
    a, b = run(RESOLVED(), Recorder()), run(RESOLVED(), Recorder())
    assert [(r["file_id"], r["final_score"]) for r in a["final_ranking"]] == [(r["file_id"], r["final_score"]) for r in b["final_ranking"]]


# ---------------------------------------------------------------------------
# Real-corpus integration: the Phase 7 H3 / E2 situations
# ---------------------------------------------------------------------------
def test_real_corpus_personalized_pool_is_not_flipped_by_an_opposing_5c():
    from app.run import run_query
    from evaluation.runner import personalization_fixture
    from evaluation.variants import VARIANTS_BY_NAME, build_variant_registry

    reg = build_variant_registry(VARIANTS_BY_NAME["C_agentic_personalized"])
    with personalization_fixture():
        rec = Recorder("opposite")
        r = run_query("user_a", "my slides", registry=reg, llm_provider=rec)
        assert r["status"] == "finish" and r["decision_basis"] == pr.BASIS_PERSONALIZATION_RESOLVED
        assert [e["file_id"] for e in r["final_ranking"]] == [7, 1]
        assert rec.calls["comparison"] == 0 and r["candidate_comparison"]["comparison_status"] == "skipped_personalization_resolved"
        assert all(e["metadata"]["owner_id"] == "user_a" for e in r["final_ranking"])
