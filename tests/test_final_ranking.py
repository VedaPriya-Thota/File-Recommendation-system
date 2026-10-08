"""
Phase 4B tests for app/final_ranking.py (architecture_final.md §8
touchpoint 3 "Final ranking" + §10 "Final scoring formula") and its
wiring into app/graph/build.py's finish_stage.

Structure:
  1. Component-score separation (retrieval vs. personalization)
  2. Personalization cannot override strong retrieval evidence
  3. Personalization CAN affect ordering where the policy permits (ties /
     near-ties)
  4. Cold-start / no-profile / no-activity neutrality
  5. Evidence preservation -- nothing existing is dropped or mutated
  6. Candidate set cannot expand
  7. Deterministic ordering / tie behavior
  8. Single-candidate behavior
  9. finish_stage actually uses the new component (graph integration)
  10. personalized_order reconciliation (no divergent duplicate policy)
"""
import pytest

from app.config import FINAL_RANKING_ALPHA, FINAL_RANKING_GAMMA
from app.final_ranking import compute_component_scores, final_rank, personalized_order
from app.graph.build import build_graph
from app.personalization import personalized_order as personalization_module_order
from app.tools.registry import ToolResult


def _entry(scores, metadata=None, **extra):
    return {
        "scores": scores,
        "metadata": metadata or {},
        "evidence_sources": list(scores.keys()),
        "first_seen_iteration": 1,
        "last_updated_iteration": 1,
        **extra,
    }


# =============================================================================
# 1. Component-score separation
# =============================================================================

def test_1_retrieval_score_excludes_personalization_source():
    entry = _entry({"bm25_search": 0.4, "rerank_candidates": 0.75, "get_user_activity": 0.99})
    scores = compute_component_scores(entry)
    assert scores["retrieval_score"] == 0.75
    assert scores["personalization_score"] == 0.99


def test_1_personalization_score_falls_back_to_top_level_personalization_key():
    """The standalone attach_personalization_signals path stores affinity
    under entry['personalization'], not entry['scores']['get_user_activity']
    -- both must be readable."""
    entry = _entry({"bm25_search": 0.5}, personalization={"affinity_score": 0.42})
    scores = compute_component_scores(entry)
    assert scores["personalization_score"] == 0.42
    assert scores["retrieval_score"] == 0.5


def test_1_final_score_is_the_weighted_sum_not_a_max():
    entry = _entry({"bm25_search": 0.6, "get_user_activity": 0.6})
    scores = compute_component_scores(entry)
    expected = round(FINAL_RANKING_ALPHA * 0.6 + FINAL_RANKING_GAMMA * 0.6, 6)
    assert scores["final_score"] == expected
    # Critically, NOT max(0.6, 0.6) == 0.6 with no weighting distinction --
    # regression check: a differently-weighted case must actually differ
    # from an equal-weight max to prove real separation is happening.
    entry2 = _entry({"bm25_search": 0.9, "get_user_activity": 0.1})
    scores2 = compute_component_scores(entry2)
    assert scores2["final_score"] != max(0.9, 0.1)


# =============================================================================
# 2. Personalization cannot override strong retrieval evidence
# =============================================================================

def test_2_large_retrieval_gap_survives_extreme_affinity_difference():
    pool = {
        1: _entry({"hybrid_search": 0.95}, personalization={"affinity_score": 0.0}),
        2: _entry({"hybrid_search": 0.10}, personalization={"affinity_score": 1.0}),
    }
    ranked = final_rank(pool)
    assert [r["file_id"] for r in ranked] == [1, 2]
    assert ranked[0]["final_score"] > ranked[1]["final_score"]


def test_2_old_bug_regression_max_of_all_scores_would_have_flipped_this():
    """The exact scenario the old `max(entry['scores'].values())` bug broke:
    file 2's real retrieval evidence is much weaker than file 1's, but its
    personalization score is high enough that a plain max() over the
    unified scores dict would rank it first. The fixed policy must not."""
    pool = {
        1: {"scores": {"bm25_search": 0.9}, "metadata": {}, "evidence_sources": ["bm25_search"]},
        2: {
            "scores": {"bm25_search": 0.2, "get_user_activity": 0.97},
            "metadata": {},
            "evidence_sources": ["bm25_search", "get_user_activity"],
        },
    }
    # Demonstrate the old bug would have flipped the order:
    old_buggy_order = sorted(pool.items(), key=lambda kv: max(kv[1]["scores"].values()), reverse=True)
    assert [fid for fid, _ in old_buggy_order] == [2, 1]

    ranked = final_rank(pool)
    assert [r["file_id"] for r in ranked] == [1, 2], (
        "personalization must not be able to outrank stronger retrieval evidence"
    )


# =============================================================================
# 3. Personalization CAN affect ordering where the policy permits it
# =============================================================================

def test_3_affinity_breaks_a_tie_between_equal_retrieval_scores():
    pool = {
        1: _entry({"bm25_search": 0.7}, personalization={"affinity_score": 0.1}),
        2: _entry({"bm25_search": 0.7}, personalization={"affinity_score": 0.9}),
    }
    ranked = final_rank(pool)
    assert [r["file_id"] for r in ranked] == [2, 1]


def test_3_affinity_can_reorder_within_a_small_retrieval_margin():
    pool = {
        1: _entry({"bm25_search": 0.61}, personalization={"affinity_score": 0.0}),
        2: _entry({"bm25_search": 0.60}, personalization={"affinity_score": 1.0}),
    }
    ranked = final_rank(pool)
    # GAMMA * 1.0 outweighs a 0.01 retrieval gap under retrieval-dominant
    # weights, so personalization is allowed to matter here -- this is the
    # "meaningful affinity weight" half of the policy, not just a pure
    # max-wins tie-break.
    assert [r["file_id"] for r in ranked] == [2, 1]


# =============================================================================
# 4. Cold-start / no-profile / no-activity neutrality
# =============================================================================

def test_4_missing_personalization_evidence_is_neutral_zero():
    entry = _entry({"semantic_search": 0.55})
    scores = compute_component_scores(entry)
    assert scores["personalization_score"] == 0.0
    assert scores["final_score"] == round(FINAL_RANKING_ALPHA * 0.55, 6)


def test_4_cold_start_candidates_rank_purely_on_retrieval_evidence():
    pool = {
        1: _entry({"hybrid_search": 0.8}),  # get_user_activity never ran
        2: _entry({"hybrid_search": 0.3}),
    }
    ranked = final_rank(pool)
    assert [r["file_id"] for r in ranked] == [1, 2]
    assert all(r["personalization_score"] == 0.0 for r in ranked)


def test_4_explicit_cold_start_signal_affinity_zero_does_not_error():
    entry = _entry(
        {"bm25_search": 0.4, "get_user_activity": 0.0},
        metadata={"personalization": {"cold_start": True, "affinity_score": 0.0}},
    )
    scores = compute_component_scores(entry)
    assert scores["personalization_score"] == 0.0
    assert scores["retrieval_score"] == 0.4


# =============================================================================
# 5. Evidence preservation
# =============================================================================

def test_5_all_existing_entry_fields_survive_unchanged():
    pool = {
        3: _entry(
            {"bm25_search": 0.4, "semantic_search": 0.6, "rerank_candidates": 0.8},
            metadata={"filename": "Q3.xlsx", "summary": "budget"},
            first_seen_iteration=1,
            last_updated_iteration=4,
        )
    }
    ranked = final_rank(pool)
    entry = ranked[0]
    assert entry["scores"] == {"bm25_search": 0.4, "semantic_search": 0.6, "rerank_candidates": 0.8}
    assert entry["metadata"] == {"filename": "Q3.xlsx", "summary": "budget"}
    assert entry["evidence_sources"] == ["bm25_search", "semantic_search", "rerank_candidates"]
    assert entry["first_seen_iteration"] == 1
    assert entry["last_updated_iteration"] == 4


def test_5_input_candidate_pool_is_not_mutated():
    pool = {1: _entry({"bm25_search": 0.5})}
    import copy

    original = copy.deepcopy(pool)
    final_rank(pool)
    assert pool == original


# =============================================================================
# 6. Candidate set cannot expand
# =============================================================================

def test_6_output_has_exactly_one_entry_per_input_file_id():
    pool = {1: _entry({"bm25_search": 0.5}), 2: _entry({"bm25_search": 0.3}), 5: _entry({"bm25_search": 0.9})}
    ranked = final_rank(pool)
    assert {r["file_id"] for r in ranked} == {1, 2, 5}
    assert len(ranked) == 3


def test_6_empty_pool_ranks_to_empty_list():
    assert final_rank({}) == []


# =============================================================================
# 7. Deterministic ordering / tie behavior
# =============================================================================

def test_7_full_ties_break_on_ascending_file_id():
    pool = {
        7: _entry({"bm25_search": 0.5}),
        3: _entry({"bm25_search": 0.5}),
        9: _entry({"bm25_search": 0.5}),
    }
    ranked = final_rank(pool)
    assert [r["file_id"] for r in ranked] == [3, 7, 9]


def test_7_ranking_is_repeatable_across_calls():
    pool = {
        1: _entry({"bm25_search": 0.5}, personalization={"affinity_score": 0.3}),
        2: _entry({"bm25_search": 0.5}, personalization={"affinity_score": 0.3}),
        3: _entry({"hybrid_search": 0.9}),
    }
    first = final_rank(pool)
    second = final_rank(pool)
    assert [r["file_id"] for r in first] == [r["file_id"] for r in second]


# =============================================================================
# 8. Single-candidate behavior
# =============================================================================

def test_8_single_candidate_ranks_alone_with_correct_score():
    pool = {42: _entry({"bm25_search": 0.66}, personalization={"affinity_score": 0.5})}
    ranked = final_rank(pool)
    assert len(ranked) == 1
    assert ranked[0]["file_id"] == 42
    expected = round(FINAL_RANKING_ALPHA * 0.66 + FINAL_RANKING_GAMMA * 0.5, 6)
    assert ranked[0]["final_score"] == expected


# =============================================================================
# 9. finish_stage actually uses the new ranking component
# =============================================================================

def test_9_finish_stage_final_ranking_matches_final_rank_output():
    registry = _FakeRegistry(
        {
            "metadata_search": ToolResult(
                tool="metadata_search",
                candidates=[
                    {
                        "file_id": 1,
                        "score": 0.9,
                        "source": "metadata_search",
                        "metadata": {"filename": "a.txt", "owner_id": "user_a"},
                    }
                ],
            )
        }
    )
    graph = build_graph(registry=registry, max_iterations=3)
    from app.scope import UserScope

    import app.graph.build as build_module

    monkey_scope = UserScope(user_id="user_a", owner_ids=("user_a",))
    orig_resolve = build_module.resolve_scope_fn
    build_module.resolve_scope_fn = lambda user_id: monkey_scope
    try:
        result = graph.invoke(
            {
                "query_id": "q1",
                "user_id": "user_a",
                "raw_query": "a.txt",
                "committed_actions": [],
                "candidate_pool": {},
                "failed_strategies": [],
                "iteration_count": 0,
                "trace": [],
            }
        )
    finally:
        build_module.resolve_scope_fn = orig_resolve

    assert result["final_ranking"], "expected a non-empty final_ranking"
    expected = final_rank(result["candidate_pool"])
    assert result["final_ranking"] == expected
    # And explicitly NOT the old buggy behavior:
    for entry in result["final_ranking"]:
        assert "final_score" in entry
        assert entry["final_score"] != max(entry["scores"].values()) or len(entry["scores"]) == 1


class _FakeRegistry:
    def __init__(self, results_by_tool):
        self._results = results_by_tool

    def execute(self, tool, args, user_scope):
        return self._results.get(tool, ToolResult(tool=tool, candidates=[]))


# =============================================================================
# 10. personalized_order reconciliation -- one policy, not two
# =============================================================================

def test_10_personalization_module_reexports_final_ranking_personalized_order():
    pool = {
        1: _entry({"bm25_search": 0.9}, personalization={"affinity_score": 0.1}),
        2: _entry({"bm25_search": 0.5}, personalization={"affinity_score": 0.99}),
    }
    assert personalized_order(pool) == personalization_module_order(pool)
    assert personalized_order(pool) == [1, 2]


def test_10_personalized_order_matches_final_rank_file_id_order():
    pool = {
        1: _entry({"hybrid_search": 0.4}),
        2: _entry({"hybrid_search": 0.8}),
        3: _entry({"hybrid_search": 0.8}, personalization={"affinity_score": 0.9}),
    }
    assert personalized_order(pool) == [r["file_id"] for r in final_rank(pool)]
