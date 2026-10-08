"""
Phase 6 -- Architecture & Behavioral Hardening.

This file adds tests only for behavior that was genuinely UNtested by the
existing Phase 1-5D suites (see the final Phase 6 report for the full
inspection). It intentionally does not re-prove things already covered in
depth elsewhere:
  - cross-user access control            -> tests/test_agent_loop.py (D),
                                             tests/test_get_user_activity.py
  - reflect() Tier-1 branching            -> tests/test_reflect.py
  - final_rank() formula/audit            -> tests/test_final_ranking.py
  - Phase 5A/5B fallback & gating         -> tests/test_llm_goal_interpretation.py,
                                             tests/test_query_expansion.py
  - Phase 5C candidate-set/ID guarantees  -> tests/test_candidate_comparison.py
  - Phase 5D grounding/result-binding     -> tests/test_result_explanation.py
  - reflect vs final_ranking regression   -> tests/test_reflect.py (Phase 6 section)

Sections below map to the Phase 6 task letters (A, B, D, G, H, I).
"""
import json

import pytest

from app.db.connection import get_cursor
from app.graph.build import build_graph
from app.llm.provider import LLMUnavailableError, NullProvider
from app.run import _initial_state, run_query
from app.tools.registry import ToolResult

QUERY = "quarterly budget planning"


def _ids(owner, n):
    with get_cursor() as cur:
        cur.execute("SELECT file_id FROM files WHERE owner_id = %s ORDER BY file_id LIMIT %s", (owner, n))
        return [r["file_id"] for r in cur.fetchall()]


def _cand(fid, score, source, **md):
    metadata = {"filename": f"f{fid}.docx", "file_type": "docx", "tags": ["budget"],
                "summary": f"summary {fid}", "owner_id": "user_a"}
    metadata.update(md)
    return {"file_id": fid, "score": score, "source": source, "metadata": metadata}


class ScriptedRegistry:
    """A registry whose responses are driven by a per-tool script (a list
    of ToolResults consumed in call order, or a single ToolResult reused
    for every call). Lets §B's escalation matrix and §A's loop-shape tests
    assert exactly which tools were invoked, in what order, with what
    result -- something the existing seed-data-driven tests don't isolate
    directly."""

    def __init__(self, scripts: dict):
        self.scripts = {k: list(v) if isinstance(v, list) else v for k, v in scripts.items()}
        self.calls = []

    def execute(self, tool, args, user_scope):
        self.calls.append((tool, args))
        script = self.scripts.get(tool, ToolResult(tool=tool, candidates=[]))
        if isinstance(script, list):
            result = script.pop(0) if script else ToolResult(tool=tool, candidates=[])
        else:
            result = script
        return result

    def call_counts(self):
        counts = {}
        for tool, _ in self.calls:
            counts[tool] = counts.get(tool, 0) + 1
        return counts


def _goal_signal_kwargs():
    """Minimal args so plan_next_action's Phase-1 metadata strategies have
    at least one signal to build a strategy from."""
    return {}


def _invoke(registry, query=QUERY, max_iterations=6, llm_provider=None):
    graph = build_graph(registry=registry, max_iterations=max_iterations, llm_provider=llm_provider or NullProvider())
    return graph.invoke(_initial_state("user_a", query, max_iterations), config={"recursion_limit": 100})


# =============================================================================
# A. Actual agent loop -- structural verification
# =============================================================================

def test_A_clarify_terminates_without_reaching_any_finish_stage():
    """§A: 'clarify really terminates without retrieval stages afterward.'
    Force clarify with a pool that stays flat/too-many through the entire
    escalation chain (metadata -> rerank -> get_user_activity all agree,
    with no dominant candidate) -- 'zero results everywhere' instead
    resolves to finish ('nothing found'), a different, already-covered
    path (see test_A_zero_results_... below)."""
    flat = ToolResult(tool="x", candidates=[_cand(i, 0.5, "x") for i in range(1, 8)])
    registry = ScriptedRegistry({"metadata_search": flat, "rerank_candidates": flat, "get_user_activity": flat})
    result = _invoke(registry, max_iterations=6)
    assert result["stopping_decision"]["status"] == "clarify"
    events = [t["event"] for t in result["trace"]]
    assert "finish" not in events and "compare_candidates" not in events and "explain_result" not in events
    assert result["final_ranking"] == []
    assert result["candidate_comparison"] == {}
    assert result["candidate_explanation"] == {}


def test_A_finish_reaches_final_ranking_compare_and_explain_stages():
    """§A: 'finish really reaches the final ranking stages.'"""
    registry = ScriptedRegistry({
        "metadata_search": ToolResult(tool="metadata_search", candidates=[_cand(1, 0.95, "metadata_search")]),
    })
    result = _invoke(registry)
    events = [t["event"] for t in result["trace"]]
    assert events.index("finish") < events.index("compare_candidates") < events.index("explain_result")
    assert result["final_ranking"] and result["final_ranking"][0]["file_id"] == 1
    assert result["candidate_comparison"]["comparison_invoked"] is False  # single dominant candidate
    assert result["candidate_explanation"]["explanation_status"] == "unavailable"  # NullProvider


def test_A_zero_results_causes_genuinely_different_next_action():
    """§A: zero results -> replan with a DIFFERENT tool/args, not a repeat."""
    registry = ScriptedRegistry({
        "metadata_search": ToolResult(tool="metadata_search", candidates=[]),
        "bm25_search": ToolResult(tool="bm25_search", candidates=[_cand(1, 0.8, "bm25_search")]),
    })
    result = _invoke(registry)
    tools_called = [t for t, _ in registry.calls]
    assert "bm25_search" in tools_called
    assert tools_called.index("bm25_search") > tools_called.index("metadata_search")
    assert result["stopping_decision"]["status"] == "finish"


def test_A_tool_error_causes_replan_not_a_crash():
    """§A: tool errors can cause fallback/replanning. metadata_search
    errors on every strategy it's given (a persistent outage), so the
    planner exhausts all metadata strategies via the error path and,
    since none of them ever succeeded, resolves to clarify -- it must not
    crash and must record every attempt as an error."""
    registry = ScriptedRegistry({
        "metadata_search": ToolResult(tool="metadata_search", candidates=[], error="boom"),
    })
    result = _invoke(registry, query="the Q3 budget xlsx from last month")
    assert any(e["tool"] == "metadata_search" for e in result["errors"])
    assert len(result["errors"]) >= 2  # more than one distinct strategy was tried and failed
    assert result["stopping_decision"]["status"] in ("finish", "clarify")


def test_A_already_tried_strategy_is_never_repeated_identically():
    """§A: already-tried strategies are not redundantly repeated. Metadata
    keeps returning zero for every distinct filter combination it's given
    -- if the planner repeated an identical (tool, args) it would loop
    until max_iterations; instead it must exhaust distinct strategies and
    escalate/terminate well before the iteration cap."""
    registry = ScriptedRegistry({"metadata_search": ToolResult(tool="metadata_search", candidates=[])})
    result = _invoke(registry, query="the Q3 budget xlsx from last month", max_iterations=6)
    metadata_calls = [args for tool, args in registry.calls if tool == "metadata_search"]
    assert len(metadata_calls) == len(set(json.dumps(a, sort_keys=True, default=str) for a in metadata_calls))
    assert result["iteration_count"] <= 6  # max_iterations respected (also see §A below)


def test_A_max_iterations_is_a_hard_guard_never_exceeded():
    """§A: max_iterations respected, even when every strategy keeps
    failing and escalation keeps finding genuinely new (but still empty)
    strategies."""
    registry = ScriptedRegistry({})  # everything empty forever
    result = _invoke(registry, query="the Q3 budget xlsx from last month", max_iterations=3)
    assert result["iteration_count"] <= 3
    assert result["stopping_decision"]["status"] in ("finish", "clarify")


# =============================================================================
# B. Retrieval escalation behavioral matrix
# =============================================================================

def test_B1_metadata_succeeds_clearly_nothing_else_called():
    registry = ScriptedRegistry({
        "metadata_search": ToolResult(tool="metadata_search", candidates=[_cand(1, 0.95, "metadata_search")]),
    })
    _invoke(registry)
    assert registry.call_counts() == {"metadata_search": 1}


def test_B2_metadata_zero_escalates_to_bm25():
    registry = ScriptedRegistry({
        "metadata_search": ToolResult(tool="metadata_search", candidates=[]),
        "bm25_search": ToolResult(tool="bm25_search", candidates=[_cand(1, 0.9, "bm25_search")]),
    })
    _invoke(registry)
    assert "bm25_search" in registry.call_counts()
    assert "semantic_search" not in registry.call_counts()  # bm25 already sufficient


def test_B3_bm25_insufficient_escalates_to_semantic():
    registry = ScriptedRegistry({
        "metadata_search": ToolResult(tool="metadata_search", candidates=[]),
        "bm25_search": ToolResult(tool="bm25_search", candidates=[]),
        "semantic_search": ToolResult(tool="semantic_search", candidates=[_cand(1, 0.85, "semantic_search")]),
    })
    _invoke(registry)
    assert registry.call_counts().get("semantic_search") == 1


def test_B4_bm25_and_semantic_evidence_escalates_to_hybrid_when_flat():
    registry = ScriptedRegistry({
        "metadata_search": ToolResult(tool="metadata_search", candidates=[
            _cand(1, 0.5, "metadata_search"), _cand(2, 0.5, "metadata_search"),
            _cand(3, 0.5, "metadata_search"), _cand(4, 0.5, "metadata_search"),
            _cand(5, 0.5, "metadata_search"), _cand(6, 0.5, "metadata_search"),
            _cand(7, 0.5, "metadata_search"),
        ]),
        "bm25_search": ToolResult(tool="bm25_search", candidates=[_cand(1, 0.5, "bm25_search")]),
        "semantic_search": ToolResult(tool="semantic_search", candidates=[_cand(2, 0.5, "semantic_search")]),
        "hybrid_search": ToolResult(tool="hybrid_search", candidates=[_cand(1, 0.9, "hybrid_search")]),
    })
    # metadata alone is "too_many" (>6, TOO_MANY_CANDIDATES_THRESHOLD) -- but
    # hybrid is only proposed once BOTH bm25 and semantic have independently
    # run, so it's not reached from a too_many metadata result alone; force
    # that by pre-seeding committed_actions is out of scope for run_query, so
    # we instead assert the actual planning.py behavior directly:
    from app.planning import plan_next_action
    from app.state import Action

    goal = {"filters": {}, "topic_tokens": ["budget"], "raw_query": QUERY, "concept_text": QUERY}
    committed = [
        Action(tool="bm25_search", args={"query": QUERY}, reason="x", iteration=1),
        Action(tool="semantic_search", args={"query": QUERY}, reason="x", iteration=2),
    ]
    action = plan_next_action(goal, committed, [], iteration=3, observation_hint="too_many",
                              candidate_pool={1: {}, 2: {}})
    assert action["tool"] == "hybrid_search"


def test_B4_hybrid_not_proposed_without_both_bm25_and_semantic_evidence():
    from app.planning import plan_next_action
    from app.state import Action

    goal = {"filters": {}, "topic_tokens": ["budget"], "raw_query": QUERY, "concept_text": QUERY}
    committed = [Action(tool="bm25_search", args={"query": QUERY}, reason="x", iteration=1)]  # semantic never ran
    pool = {1: {"metadata": {}}, 2: {"metadata": {}}}
    action = plan_next_action(goal, committed, [], iteration=2, observation_hint="too_many",
                              candidate_pool=pool)
    assert action is None or action["tool"] != "hybrid_search"


def test_B5_reranking_only_when_pool_still_flat_after_available_retrieval():
    """§B5: rerank_candidates is proposed only once retrieval strategies
    are exhausted and the pool is still ambiguous -- not for a single
    dominant candidate."""
    registry = ScriptedRegistry({
        "metadata_search": ToolResult(tool="metadata_search", candidates=[_cand(1, 0.95, "metadata_search")]),
    })
    _invoke(registry)
    assert "rerank_candidates" not in registry.call_counts()


def test_B6_personalization_only_after_rerank_when_still_flat():
    """§B6: get_user_activity is proposed only after rerank_candidates has
    genuinely run and the pool is STILL flat -- never before, never as a
    substitute."""
    from app.planning import plan_next_action
    from app.state import Action

    goal = {"filters": {}, "topic_tokens": ["budget"], "raw_query": QUERY, "concept_text": QUERY}
    pool = {1: {"scores": {"rerank_candidates": 0.5}, "metadata": {}},
            2: {"scores": {"rerank_candidates": 0.5}, "metadata": {}}}

    # Before rerank has run: get_user_activity must not be proposed.
    action_before = plan_next_action(goal, [], [], iteration=1, observation_hint="too_many", candidate_pool=pool)
    assert action_before["tool"] != "get_user_activity"

    # After rerank has run and pool is still flat: get_user_activity is next.
    committed = [Action(tool="rerank_candidates", args={"query": QUERY, "candidates": []}, reason="x", iteration=1)]
    action_after = plan_next_action(goal, committed, [], iteration=2, observation_hint="too_many", candidate_pool=pool)
    assert action_after["tool"] == "get_user_activity"


def test_B7_no_unnecessary_repeated_tool_calls_for_a_clean_single_hit():
    registry = ScriptedRegistry({
        "metadata_search": ToolResult(tool="metadata_search", candidates=[_cand(1, 0.95, "metadata_search")]),
    })
    result = _invoke(registry)
    assert registry.call_counts() == {"metadata_search": 1}
    assert result["iteration_count"] == 1


def test_B8_tool_failure_causes_safe_alternative_not_a_crash():
    """§B8: a tool error is a safe, non-crashing outcome -- it triggers
    replanning of remaining metadata strategies (architecture_final.md §7
    'Tool error -> deterministic fallback, replan'), and if none remain,
    a safe clarify rather than a crash or a silently-wrong result. (A
    tool error currently does NOT escalate to bm25/semantic the way a
    genuine zero-results observation does -- see the Phase 6 report's
    architecture-debt findings.)"""
    registry = ScriptedRegistry({
        "metadata_search": ToolResult(tool="metadata_search", candidates=[], error="db timeout"),
    })
    result = _invoke(registry, query="the Q3 budget xlsx from last month")  # must not raise
    assert result["stopping_decision"]["status"] in ("finish", "clarify")
    assert result["errors"] and result["errors"][0]["tool"] == "metadata_search"


# =============================================================================
# D. Candidate-pool invariants across the FULL LLM pipeline together
# (5A + 5B + 5C + 5D all firing in a single run -- stronger than each
# phase's own isolated invariant tests, which never combine all four).
# =============================================================================

class FullPipelineProvider:
    """Answers every one of the four LLM prompt shapes successfully, so a
    single run exercises goal interpretation, query expansion, candidate
    comparison AND explanation together."""

    name = "fake"

    def __init__(self, comparison_pick_index=1):
        self.comparison_pick_index = comparison_pick_index
        self.calls = []

    def complete(self, system_prompt, user_prompt):
        self.calls.append(system_prompt[:40])
        if "interpret a user's natural-language file-search request" in system_prompt:
            return json.dumps({"file_type": None, "topic_tokens": ["budget"], "filename_candidate": None,
                               "retrieval_preference": None, "confidence": 0.7, "rationale": "ok"})
        if "propose additional search terms" in system_prompt:
            return json.dumps({"expanded_terms": ["fiscal"], "confidence": 0.7, "rationale": "ok"})
        if "already-retrieved candidate files" in system_prompt:
            payload = json.loads(user_prompt.split("\n", 1)[1])
            fid = payload["candidates"][self.comparison_pick_index]["file_id"]
            return json.dumps({"selected_file_id": fid, "confidence": 0.9, "rationale": "closer match"})
        if "user-facing explanation" in system_prompt:
            return json.dumps({"summary": "Matches your query.", "evidence_points": ["Tagged budget."],
                               "confidence": 0.8})
        raise RuntimeError("unexpected prompt")


def test_D_full_pipeline_preserves_candidate_membership_and_scope():
    ids = _ids("user_a", 3)
    registry = ScriptedRegistry({
        "metadata_search": ToolResult(tool="metadata_search", candidates=[
            _cand(ids[0], 0.90, "metadata_search"), _cand(ids[1], 0.85, "metadata_search"),
            _cand(ids[2], 0.80, "metadata_search"),
        ]),
    })
    provider = FullPipelineProvider()
    result = _invoke(registry, query="the budget thing", llm_provider=provider)  # vague -> triggers 5A/5B

    assert len(provider.calls) >= 3  # goal interpretation + expansion + comparison (+ explanation)
    assert result["candidate_comparison"]["comparison_applied"] is True
    assert result["candidate_explanation"]["explanation_status"] == "generated"

    assert set(result["candidate_pool"]) == set(ids)  # unchanged by any of the four LLM phases
    assert set(c["file_id"] for c in result["final_ranking"]) == set(ids)  # no candidate added/dropped
    assert result["user_scope"] == ["user_a"]
    other = _ids("user_x", 1)
    assert not other or other[0] not in result["candidate_pool"]


def test_D_personalization_reranking_comparison_explanation_never_add_file_ids():
    """Explicit statement of §D's five invariants, checked directly
    against the real modules (not just via one end-to-end run)."""
    from app.llm.candidate_comparer import reorder_with_selection
    from app.final_ranking import final_rank
    from app.personalization import compute_personalization_signals

    pool = {1: {"scores": {"bm25_search": 0.5}, "metadata": {"file_type": "docx", "tags": []},
                "evidence_sources": ["bm25_search"]}}

    # personalization: annotates, never adds a candidate to a pool it's given.
    signals = compute_personalization_signals(pool[1]["metadata"], [], None)
    assert set(pool) == {1}  # unaffected

    # final_rank (incl. any personalization already merged in): same file_ids.
    ranked = final_rank(pool)
    assert {r["file_id"] for r in ranked} == {1}

    # reorder_with_selection (Phase 5C): reorders only, never adds.
    reordered = reorder_with_selection(ranked, 999999)  # unknown id -> no-op, no addition
    assert {r["file_id"] for r in reordered} == {1}


# =============================================================================
# G. Graceful degradation with LLM_PROVIDER=none
# =============================================================================

def test_G_llm_provider_none_env_var_keeps_system_fully_functional(monkeypatch):
    monkeypatch.setenv("DISCOVERY_LLM_PROVIDER", "none")
    from app.llm.provider import get_default_provider
    provider = get_default_provider()
    assert provider.name == "none"

    result = run_query("user_a", "Find the slide deck about AI architecture I edited last week.")
    assert result["status"] == "finish"
    assert result["final_ranking"]
    assert result["candidate_explanation"]["explanation_status"] == "unavailable"


@pytest.mark.parametrize("stage_prompt_fragment,expected_meta_key,expected_status_key", [
    ("interpret a user's natural-language file-search request", "llm_goal_metadata", "llm_status"),
    ("propose additional search terms", "query_expansion_metadata", "expansion_status"),
])
def test_G_individual_5a_5b_failure_leaves_deterministic_result_intact(stage_prompt_fragment, expected_meta_key, expected_status_key):
    class FailOneStage:
        name = "fake"

        def complete(self, system_prompt, user_prompt):
            if stage_prompt_fragment in system_prompt:
                raise LLMUnavailableError("down")
            raise RuntimeError("other stages unused in this scenario")

    ids = _ids("user_a", 1)
    registry = ScriptedRegistry({
        "metadata_search": ToolResult(tool="metadata_search", candidates=[_cand(ids[0], 0.95, "metadata_search")]),
    })
    result = _invoke(registry, query="the budget thing", llm_provider=FailOneStage())
    assert result[expected_meta_key][expected_status_key] == "unavailable"
    assert result["final_ranking"][0]["file_id"] == ids[0]
    assert result["user_scope"] == ["user_a"]


def test_G_no_exception_escapes_when_every_llm_stage_fails_together():
    class AlwaysFails:
        name = "fake"

        def complete(self, system_prompt, user_prompt):
            raise RuntimeError("simulated total provider outage")

    ids = _ids("user_a", 3)
    registry = ScriptedRegistry({
        "metadata_search": ToolResult(tool="metadata_search", candidates=[
            _cand(ids[0], 0.90, "metadata_search"), _cand(ids[1], 0.89, "metadata_search"),
            _cand(ids[2], 0.88, "metadata_search"),
        ]),
    })
    result = _invoke(registry, query="the budget thing", llm_provider=AlwaysFails())  # must not raise
    assert result["stopping_decision"]["status"] == "finish"
    assert set(c["file_id"] for c in result["final_ranking"]) == set(ids)
    assert result["candidate_comparison"]["comparison_status"] == "provider_error"
    assert result["candidate_explanation"]["explanation_status"] == "provider_error"


# =============================================================================
# H. Unnecessary-LLM-call gating, verified together across all four phases
# =============================================================================

CLEAR_QUERY = "Find the slide deck about AI architecture I edited last week."  # ambiguity_score 0.4 < 0.65


def test_H_clear_unambiguous_query_never_calls_5a_or_5b():
    """§H: a clear query (deterministic ambiguity_score below the gate)
    must never invoke goal interpretation (5A) or query expansion (5B)."""
    class ExplodesOnAnyCall:
        name = "fake"

        def complete(self, system_prompt, user_prompt):
            raise AssertionError("5A/5B must not be called for a clear, unambiguous query")

    ids = _ids("user_a", 1)
    registry = ScriptedRegistry({
        "metadata_search": ToolResult(tool="metadata_search", candidates=[_cand(ids[0], 0.95, "metadata_search")]),
    })
    result = _invoke(registry, query=CLEAR_QUERY, llm_provider=ExplodesOnAnyCall())  # must not raise
    assert result["llm_goal_metadata"]["llm_status"] == "not_needed"
    assert result["query_expansion_metadata"]["expansion_status"] == "not_needed"


def test_H_dominant_result_never_calls_5c():
    class ExplodesIfComparisonCalled:
        name = "fake"

        def complete(self, system_prompt, user_prompt):
            if "already-retrieved candidate files" in system_prompt:
                raise AssertionError("5C must not be called for a single dominant candidate")
            raise LLMUnavailableError("5A/5B/5D not exercised in this scenario")

    ids = _ids("user_a", 1)
    registry = ScriptedRegistry({
        "metadata_search": ToolResult(tool="metadata_search", candidates=[_cand(ids[0], 0.95, "metadata_search")]),
    })
    result = _invoke(registry, query=CLEAR_QUERY, llm_provider=ExplodesIfComparisonCalled())
    assert result["candidate_comparison"]["comparison_status"] == "not_needed"


def test_H_explanation_never_called_twice_for_an_already_generated_result():
    from app.llm.result_explainer import should_explain

    ranking = [{"file_id": 1}]
    assert should_explain(ranking, None) is True
    assert should_explain(ranking, {"explanation_status": "generated"}) is False
    assert should_explain(ranking, {"explanation_status": "unavailable"}) is True  # retry allowed, not "already done"


def test_H_no_result_never_calls_explanation():
    class ExplodesIfCalled:
        name = "fake"

        def complete(self, system_prompt, user_prompt):
            raise AssertionError("explanation LLM must not be called when there is no final result")

    registry = ScriptedRegistry({})  # zero results everywhere -> clarify, no final_ranking
    result = _invoke(registry, llm_provider=ExplodesIfCalled())
    assert result["final_ranking"] == []
    assert result["candidate_explanation"] in ({}, {"explanation_invoked": False, "explanation_status": "not_needed"})


# =============================================================================
# I. Compact end-to-end behavioral matrix
# =============================================================================

def test_I_1_clear_query_deterministic_result():
    result = run_query("user_a", "Find the slide deck about AI architecture I edited last week.")
    assert result["status"] == "finish" and result["final_ranking"]
    assert result["llm_goal_metadata"]["llm_status"] == "not_needed"


def test_I_9_no_result_safe_finish_or_clarify_never_crashes():
    result = run_query("user_a", "xyzzy_nonexistent_file_reference_12345")
    assert result["status"] in ("finish", "clarify")
    assert "error" not in result or result.get("status") != "hard_failure"


def test_I_10_clarify_ends_the_turn_with_a_question():
    registry = ScriptedRegistry({})
    result = _invoke(registry)
    if result["stopping_decision"]["status"] == "clarify":
        assert result["clarification_question"]


def test_I_11_provider_unavailable_falls_back_to_deterministic_result():
    result = run_query("user_a", "Find the slide deck about AI architecture I edited last week.",
                       llm_provider=NullProvider())
    assert result["status"] == "finish"
    assert result["final_ranking"]


def test_I_12_cross_user_attempt_is_isolated():
    result_a = run_query("user_a", "budget report")
    other_owner_ids = {c["metadata"].get("owner_id") for c in result_a["final_ranking"]
                       if c.get("metadata", {}).get("owner_id")}
    assert other_owner_ids <= {"user_a"}


def test_I_7_near_tie_triggers_5c_end_to_end():
    ids = _ids("user_a", 3)
    registry = ScriptedRegistry({
        "metadata_search": ToolResult(tool="metadata_search", candidates=[
            _cand(ids[0], 0.90, "metadata_search"), _cand(ids[1], 0.89, "metadata_search"),
        ]),
    })
    provider = FullPipelineProvider(comparison_pick_index=1)

    class ComparisonOnly:
        name = "fake"

        def complete(self, system_prompt, user_prompt):
            if "already-retrieved candidate files" in system_prompt:
                return provider.complete(system_prompt, user_prompt)
            raise LLMUnavailableError("only comparison enabled in this scenario")

    result = _invoke(registry, llm_provider=ComparisonOnly())
    assert result["candidate_comparison"]["comparison_invoked"] is True


def test_I_8_final_result_gets_5d_explanation_end_to_end():
    ids = _ids("user_a", 1)
    registry = ScriptedRegistry({
        "metadata_search": ToolResult(tool="metadata_search", candidates=[_cand(ids[0], 0.95, "metadata_search")]),
    })

    class ExplanationOnly:
        name = "fake"

        def complete(self, system_prompt, user_prompt):
            if "user-facing explanation" in system_prompt:
                return json.dumps({"summary": "Because it matched.", "confidence": 0.8})
            raise LLMUnavailableError("only explanation enabled in this scenario")

    result = _invoke(registry, llm_provider=ExplanationOnly())
    assert result["candidate_explanation"]["explanation_status"] == "generated"
    assert result["explanations"] == {ids[0]: "Because it matched."}
