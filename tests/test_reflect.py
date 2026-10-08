from app.reflect import reflect

ACTION = {"tool": "metadata_search", "args": {"topic_tokens": ["budget"]}, "reason": "x", "iteration": 1}


def test_zero_candidates_triggers_replan():
    rr = reflect(
        last_action=ACTION,
        last_tool_result={"tool": "metadata_search", "candidates": [], "error": None},
        candidate_pool={},
        failed_strategies=[],
        errors=[],
        iteration_count=1,
        max_iterations=6,
    )
    assert rr.status == "replan"
    assert rr.observation_hint == "zero_results"
    assert len(rr.failed_strategies) == 1


def test_zero_candidates_at_max_iterations_forces_finish():
    rr = reflect(
        last_action=ACTION,
        last_tool_result={"tool": "metadata_search", "candidates": [], "error": None},
        candidate_pool={},
        failed_strategies=[],
        errors=[],
        iteration_count=6,
        max_iterations=6,
    )
    assert rr.status == "finish"
    assert rr.low_confidence is True


def test_one_dominant_candidate_finishes():
    pool = {
        1: {"scores": {"metadata_search": 0.95}, "metadata": {}, "evidence_sources": ["metadata_search"]}
    }
    rr = reflect(
        last_action=ACTION,
        last_tool_result={
            "tool": "metadata_search",
            "candidates": [{"file_id": 1, "score": 0.95}],
            "error": None,
        },
        candidate_pool=pool,
        failed_strategies=[],
        errors=[],
        iteration_count=1,
        max_iterations=6,
    )
    assert rr.status == "finish"
    assert rr.confidence == 0.95


def test_flat_scores_trigger_replan_to_narrow():
    pool = {
        1: {"scores": {"metadata_search": 0.5}, "metadata": {}, "evidence_sources": ["metadata_search"]},
        2: {"scores": {"metadata_search": 0.5}, "metadata": {}, "evidence_sources": ["metadata_search"]},
    }
    rr = reflect(
        last_action=ACTION,
        last_tool_result={
            "tool": "metadata_search",
            "candidates": [{"file_id": 1, "score": 0.5}, {"file_id": 2, "score": 0.5}],
            "error": None,
        },
        candidate_pool=pool,
        failed_strategies=[],
        errors=[],
        iteration_count=1,
        max_iterations=6,
    )
    assert rr.status == "replan"
    assert rr.observation_hint == "too_many"


def test_too_many_candidates_trigger_replan():
    pool = {i: {"scores": {"metadata_search": 0.5}, "metadata": {}, "evidence_sources": ["metadata_search"]} for i in range(10)}
    rr = reflect(
        last_action=ACTION,
        last_tool_result={
            "tool": "metadata_search",
            "candidates": [{"file_id": i, "score": 0.5} for i in range(10)],
            "error": None,
        },
        candidate_pool=pool,
        failed_strategies=[],
        errors=[],
        iteration_count=1,
        max_iterations=6,
    )
    assert rr.status == "replan"
    assert rr.observation_hint == "too_many"


def test_tool_error_triggers_replan_and_records_error():
    rr = reflect(
        last_action=ACTION,
        last_tool_result={"tool": "metadata_search", "candidates": [], "error": "boom"},
        candidate_pool={},
        failed_strategies=[],
        errors=[],
        iteration_count=1,
        max_iterations=6,
    )
    assert rr.status == "replan"
    assert rr.observation_hint == "error"
    assert rr.errors[0]["error"] == "boom"


def test_tool_error_at_max_iterations_forces_finish():
    rr = reflect(
        last_action=ACTION,
        last_tool_result={"tool": "metadata_search", "candidates": [], "error": "boom"},
        candidate_pool={},
        failed_strategies=[],
        errors=[],
        iteration_count=6,
        max_iterations=6,
    )
    assert rr.status == "finish"
    assert rr.low_confidence is True


def test_repeated_strategy_is_marked_failed():
    sig_before = []
    rr = reflect(
        last_action=ACTION,
        last_tool_result={"tool": "metadata_search", "candidates": [], "error": None},
        candidate_pool={},
        failed_strategies=sig_before,
        errors=[],
        iteration_count=1,
        max_iterations=6,
    )
    assert "metadata_search:" in rr.failed_strategies[0]
    # original list must not have been mutated in place
    assert sig_before == []


# =============================================================================
# Phase 6 (§C): reflect() dominance must be judged on RETRIEVAL evidence
# only (matching app/final_ranking.py's S_retrieval), never on the raw,
# undifferentiated `scores` dict -- which also holds get_user_activity's
# personalization/affinity score once that tool has run (see
# app/candidate_pool.py's single shared-dict merge). Regression tests for
# the bug fixed in app/reflect.py (previously: max(scores.values())).
# =============================================================================

def _pool_two_candidates_flat_retrieval_but_skewed_affinity():
    # Retrieval evidence (bm25_search) is close/flat: gap 0.05, well under
    # DOMINANT_SCORE_GAP (0.25) -- genuinely ambiguous by retrieval alone.
    # get_user_activity's affinity score is wildly skewed toward file 1,
    # well over DOMINANT_SCORE_GAP -- this must NOT count as retrieval
    # dominance.
    return {
        1: {"scores": {"bm25_search": 0.55, "get_user_activity": 0.95},
            "metadata": {"filename": "a.docx"}, "evidence_sources": ["bm25_search", "get_user_activity"]},
        2: {"scores": {"bm25_search": 0.60, "get_user_activity": 0.05},
            "metadata": {"filename": "b.docx"}, "evidence_sources": ["bm25_search", "get_user_activity"]},
    }


def _reflect_after_get_user_activity(pool, iteration_count=3, max_iterations=6):
    action = {"tool": "get_user_activity", "args": {"candidates": []}, "reason": "x", "iteration": iteration_count}
    result = {"tool": "get_user_activity", "candidates": [{"file_id": 1}, {"file_id": 2}], "error": None}
    return reflect(action, result, pool, [], [], iteration_count=iteration_count, max_iterations=max_iterations)


def test_personalization_score_alone_does_not_create_false_dominance():
    # Phase 8 (intentional behaviour change): with retrieval flat (0.55 vs 0.60)
    # and affinity skewed far beyond DOMINANT_SCORE_GAP, personalization MAY now
    # resolve the tie (architecture §7/§8) -- but never AS retrieval dominance.
    # What this Phase 6 regression test still guarantees: the decision is made on
    # the distinct personalization-tiebreak path, not the retrieval-dominance one.
    rr = _reflect_after_get_user_activity(_pool_two_candidates_flat_retrieval_but_skewed_affinity())
    assert rr.status == "finish"
    assert "dominant candidate" not in rr.reason  # not the retrieval-dominance reason
    assert "personalization" in rr.reason.lower()
    assert "not counted as retrieval dominance" in rr.reason.lower()


def test_flat_retrieval_with_small_affinity_margin_still_replans():
    # The original Phase 6 intent, on the case where it is still true: flat
    # retrieval + personalization that does NOT meaningfully separate the
    # candidates (margin 0.10 < DOMINANT_SCORE_GAP) must keep narrowing.
    pool = {
        1: {"scores": {"bm25_search": 0.55, "get_user_activity": 0.55},
            "metadata": {"filename": "a.docx"}, "evidence_sources": ["bm25_search", "get_user_activity"]},
        2: {"scores": {"bm25_search": 0.60, "get_user_activity": 0.45},
            "metadata": {"filename": "b.docx"}, "evidence_sources": ["bm25_search", "get_user_activity"]},
    }
    rr = _reflect_after_get_user_activity(pool)
    assert rr.status == "replan"
    assert rr.observation_hint == "too_many"


def test_personalization_score_does_not_inflate_reported_confidence():
    rr = _reflect_after_get_user_activity(_pool_two_candidates_flat_retrieval_but_skewed_affinity())
    # Confidence must be derived from the flat retrieval evidence (~0.60),
    # never from the skewed 0.95 affinity score -- including on Phase 8's
    # personalization-tiebreak finish path.
    assert rr.confidence < 0.60


def test_dominance_still_recognized_from_genuine_retrieval_evidence():
    # Sanity check the fix isn't overcorrected: real retrieval dominance
    # (not personalization) must still be recognized, even with
    # get_user_activity scores present in the same dict.
    pool = {
        1: {"scores": {"bm25_search": 0.95, "get_user_activity": 0.1},
            "metadata": {"filename": "a.docx"}, "evidence_sources": ["bm25_search", "get_user_activity"]},
        2: {"scores": {"bm25_search": 0.30, "get_user_activity": 0.9},
            "metadata": {"filename": "b.docx"}, "evidence_sources": ["bm25_search", "get_user_activity"]},
    }
    rr = _reflect_after_get_user_activity(pool)
    assert rr.status == "finish"
    assert rr.confidence == 0.95  # file 1's genuine retrieval score, not file 2's affinity score


def test_reflect_dominance_matches_final_ranking_retrieval_component():
    """The exact regression this fix targets: reflect()'s per-candidate
    'best score' must agree with app/final_ranking.py's S_retrieval for
    every candidate -- the two components are no longer allowed to diverge
    for what 'the evidence says' about a single candidate. (Their combined
    FinalScore still legitimately differs, since final_rank also weighs in
    S_affinity by design -- this test is about the shared evidence term,
    not the final weighted output.)"""
    from app.final_ranking import retrieval_evidence_score

    pool = _pool_two_candidates_flat_retrieval_but_skewed_affinity()
    for entry in pool.values():
        assert retrieval_evidence_score(entry) == max(
            v for src, v in entry["scores"].items() if src != "get_user_activity"
        )


def test_end_to_end_unresolved_after_full_escalation_clarifies_not_silently_finishes():
    """End-to-end (§C.3): when retrieval evidence is genuinely flat AND
    personalization does not discriminate either (identical retrieval and
    identical affinity), the escalation chain (rerank + get_user_activity)
    leaves the candidates indistinguishable, and the system must ask for
    clarification -- not silently 'finish' (architecture_final.md §7/§15: 'on
    unresolved ambiguity, the agent asks rather than guessing confidently').

    Phase 8 note: the earlier version of this test used SKEWED affinity and
    expected clarify. Phase 8 intentionally lets meaningful personalization
    resolve such a tie (see tests/test_phase8_termination.py), so the
    'unresolved' scenario is now the one where nothing separates the pool."""
    from app.graph.build import build_graph
    from app.llm.provider import NullProvider
    from app.run import _initial_state
    from app.tools.registry import ToolResult

    meta = lambda name: {"filename": name, "file_type": "docx", "owner_id": "user_a"}  # noqa: E731

    class TiedEverywhereRegistry:
        """Every tool reports the SAME two candidates with identical scores."""

        def execute(self, tool, args, user_scope):
            return ToolResult(tool=tool, candidates=[
                {"file_id": 1, "score": 0.60, "source": tool, "metadata": meta("a.docx")},
                {"file_id": 2, "score": 0.60, "source": tool, "metadata": meta("b.docx")},
            ])

    graph = build_graph(registry=TiedEverywhereRegistry(), max_iterations=6, llm_provider=NullProvider())
    result = graph.invoke(_initial_state("user_a", "budget report", 6), config={"recursion_limit": 100})
    assert "get_user_activity" in [a["tool"] for a in result["committed_actions"]]
    assert result["stopping_decision"]["status"] == "clarify"
    assert "tied" in result["stopping_decision"]["reason"].lower()
    assert result["final_ranking"] == []  # finish_stage/compare_stage/explain_stage never ran
    assert result["candidate_explanation"] == {}
