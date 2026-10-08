"""
Phase 3 tests for rerank_candidates (architecture_final.md §6 `rerank_candidates`).

Two tiers, mirroring test_hybrid_search.py's structure:
  - DETERMINISTIC MOCKED-SCORER tests (1-9): app.reranker.score_pairs is
    monkeypatched with a fixed, hand-specified scoring function so ordering/
    normalization/access-control/evidence-selection can be checked exactly,
    independent of which real-or-fallback provider is active in this
    environment.
  - PLANNING/INTEGRATION tests (10-12): real plan_next_action, real
    candidate_pool, real registry -- proving the escalation and pool
    integration work end-to-end.
  - REAL-MODEL SMOKE TEST: only runs if a real network-reachable
    cross-encoder can actually be loaded here (see app/reranker.py's module
    docstring -- this sandbox blocks huggingface.co, same as
    app/embeddings.py's semantic-search provider, so it's expected to skip
    here).
"""
import pytest

from app.candidate_pool import update_candidate_pool
from app.reflect import reflect
from app.scope import UserScope, resolve_scope
from app.tools import build_registry
import app.tools.rerank_candidates as rerank_module
from app.tools.rerank_candidates import candidates_from_pool, rerank_candidates
from app.tools.registry import ToolResult

SCOPE = UserScope(user_id="user_a", owner_ids=("user_a",))


def _candidate(file_id, **metadata_overrides):
    metadata = {
        "filename": f"File_{file_id}.txt",
        "file_type": "txt",
        "owner_id": "user_a",
        "tags": [],
        "summary": "",
    }
    metadata.update(metadata_overrides)
    return {"file_id": file_id, "metadata": metadata}


@pytest.fixture
def fixed_scorer(monkeypatch):
    """Installs a deterministic scorer keyed by evidence text, patched onto
    the name as imported into rerank_candidates' own module namespace
    (`from app.reranker import score_pairs`), which is what
    rerank_candidates() actually calls -- same monkeypatch pattern
    test_hybrid_search.py uses for bm25_search/semantic_search."""

    def _install(text_to_score: dict):
        def _fake_score_pairs(query, texts):
            return [text_to_score.get(t, 0.0) for t in texts]

        monkeypatch.setattr(rerank_module, "score_pairs", _fake_score_pairs)

    return _install


# --- 1. Cross-encoder produces scores for supplied candidates --------------

def test_1_produces_a_score_for_every_supplied_candidate(fixed_scorer):
    candidates = [
        _candidate(1, summary="alpha document"),
        _candidate(2, summary="beta document"),
    ]
    fixed_scorer({"alpha document": 0.9, "beta document": 0.1})
    result = rerank_candidates({"query": "q", "candidates": candidates}, SCOPE)
    assert len(result.candidates) == 2
    assert {c["file_id"] for c in result.candidates} == {1, 2}
    for c in result.candidates:
        assert isinstance(c["score"], float)


# --- 2. Candidate order changes when reranker scores indicate a different order

def test_2_order_follows_reranker_score_not_input_order(fixed_scorer):
    candidates = [
        _candidate(1, summary="irrelevant document"),
        _candidate(2, summary="highly relevant document"),
    ]
    fixed_scorer({"irrelevant document": 0.1, "highly relevant document": 0.95})
    result = rerank_candidates({"query": "q", "candidates": candidates}, SCOPE)
    ranking = [c["file_id"] for c in result.candidates]
    assert ranking == [2, 1]  # input order was [1, 2]; reranked order flips it


# --- 3. Only supplied candidates are returned -------------------------------

def test_3_only_supplied_candidates_are_returned(fixed_scorer):
    candidates = [_candidate(1, summary="a"), _candidate(2, summary="b")]
    fixed_scorer({"a": 0.5, "b": 0.6})
    result = rerank_candidates({"query": "q", "candidates": candidates}, SCOPE)
    returned_ids = {c["file_id"] for c in result.candidates}
    assert returned_ids == {1, 2}
    assert returned_ids <= {c["file_id"] for c in candidates}


# --- 4. Original retrieval evidence is preserved ----------------------------

def test_4_original_metadata_and_evidence_preserved_alongside_rerank_score(fixed_scorer):
    candidates = [
        _candidate(
            1,
            summary="original summary",
            snippet="original bm25 snippet",
            matched_chunk="original semantic chunk",
        )
    ]
    fixed_scorer({"original semantic chunk": 0.7})
    result = rerank_candidates({"query": "q", "candidates": candidates}, SCOPE)
    meta = result.candidates[0]["metadata"]
    assert meta["summary"] == "original summary"
    assert meta["snippet"] == "original bm25 snippet"
    assert meta["matched_chunk"] == "original semantic chunk"
    assert meta["filename"] == "File_1.txt"
    # raw cross-encoder score kept for observability, distinct from the
    # normalized "score" used as the candidate's rerank_candidates source score
    assert meta["rerank_raw_score"] == 0.7


def test_4_merging_into_an_existing_candidate_pool_entry_keeps_prior_source_scores():
    """candidate_pool accumulation (app/candidate_pool.py) is untouched by
    Phase 3 -- rerank_candidates' ToolResult merges in exactly like any
    other tool's, adding a new source without erasing bm25/semantic/RRF
    evidence already on the entry."""
    pool = {
        7: {
            "evidence_sources": ["bm25_search", "semantic_search"],
            "scores": {"bm25_search": 0.4, "semantic_search": 0.6},
            "metadata": {"filename": "File_7.txt"},
            "first_seen_iteration": 1,
            "last_updated_iteration": 1,
        }
    }
    rerank_result = ToolResult(
        tool="rerank_candidates",
        candidates=[{"file_id": 7, "metadata": {"filename": "File_7.txt"}, "score": 0.88, "source": "rerank_candidates"}],
    )
    updated = update_candidate_pool(pool, rerank_result, iteration=3)
    entry = updated[7]
    assert entry["scores"]["bm25_search"] == 0.4
    assert entry["scores"]["semantic_search"] == 0.6
    assert entry["scores"]["rerank_candidates"] == 0.88
    assert set(entry["evidence_sources"]) == {"bm25_search", "semantic_search", "rerank_candidates"}


# --- 5. Reranker does not perform corpus retrieval --------------------------

def test_5_never_touches_postgres_or_qdrant(fixed_scorer, monkeypatch):
    def _explode(*a, **kw):
        raise AssertionError("rerank_candidates must never query Postgres/Qdrant")

    monkeypatch.setattr("app.db.connection.get_cursor", _explode)
    monkeypatch.setattr("app.vectorstore.get_client", _explode)

    candidates = [_candidate(1, summary="a")]
    fixed_scorer({"a": 0.5})
    # Would raise if rerank_candidates had any DB/Qdrant code path.
    result = rerank_candidates({"query": "q", "candidates": candidates}, SCOPE)
    assert len(result.candidates) == 1


# --- 6. Empty candidate list is handled safely ------------------------------

def test_6_empty_candidate_list_returns_empty_result(fixed_scorer):
    fixed_scorer({})
    result = rerank_candidates({"query": "q", "candidates": []}, SCOPE)
    assert result.candidates == []
    assert result.error is None


def test_6_empty_query_returns_empty_result(fixed_scorer):
    fixed_scorer({})
    result = rerank_candidates({"query": "", "candidates": [_candidate(1, summary="a")]}, SCOPE)
    assert result.candidates == []


# --- 7. Access-control boundary cannot be expanded --------------------------

def test_7_cannot_introduce_a_file_outside_the_supplied_candidate_set(fixed_scorer):
    """The reranker has no lookup path at all -- it can only ever emit
    file_ids it was handed. This is the access-control guarantee: whatever
    scope-checking the upstream retrieval tools already did is never
    bypassed or widened here."""
    candidates = [_candidate(101, summary="a"), _candidate(102, summary="b")]
    fixed_scorer({"a": 0.2, "b": 0.9})
    result = rerank_candidates({"query": "q", "candidates": candidates}, SCOPE)
    returned_ids = {c["file_id"] for c in result.candidates}
    supplied_ids = {c["file_id"] for c in candidates}
    assert returned_ids == supplied_ids
    assert 999 not in returned_ids  # an unsupplied id can never appear


def test_7_top_k_only_truncates_never_adds(fixed_scorer):
    candidates = [_candidate(i, summary=f"doc{i}") for i in range(5)]
    fixed_scorer({f"doc{i}": float(i) for i in range(5)})
    result = rerank_candidates({"query": "q", "candidates": candidates, "top_k": 2}, SCOPE)
    assert len(result.candidates) == 2
    assert {c["file_id"] for c in result.candidates} <= {0, 1, 2, 3, 4}


# --- 8. Tool registration works ---------------------------------------------

def test_8_rerank_candidates_is_registered_and_executable(fixed_scorer):
    fixed_scorer({"a": 0.5})
    registry = build_registry()
    assert "rerank_candidates" in registry.names()
    tool = registry.get("rerank_candidates")
    assert tool.name == "rerank_candidates"

    result = registry.execute(
        "rerank_candidates",
        {"query": "q", "candidates": [_candidate(1, summary="a")]},
        SCOPE,
    )
    assert isinstance(result, ToolResult)
    assert result.tool == "rerank_candidates"


# --- 9. Candidate-pool integration works ------------------------------------

def test_9_candidates_from_pool_builds_expected_shape():
    pool = {
        5: {
            "evidence_sources": ["bm25_search"],
            "scores": {"bm25_search": 0.3},
            "metadata": {"filename": "File_5.txt", "snippet": "snip"},
            "first_seen_iteration": 1,
            "last_updated_iteration": 1,
        }
    }
    built = candidates_from_pool(pool)
    assert built == [{"file_id": 5, "metadata": {"filename": "File_5.txt", "snippet": "snip"}}]


def test_9_rerank_result_updates_the_same_pool_object_shape_as_other_tools(fixed_scorer):
    pool = {
        1: {
            "evidence_sources": ["metadata_search"],
            "scores": {"metadata_search": 0.5},
            "metadata": {"filename": "File_1.txt", "summary": "a"},
            "first_seen_iteration": 1,
            "last_updated_iteration": 1,
        },
        2: {
            "evidence_sources": ["metadata_search"],
            "scores": {"metadata_search": 0.5},
            "metadata": {"filename": "File_2.txt", "summary": "b"},
            "first_seen_iteration": 1,
            "last_updated_iteration": 1,
        },
    }
    fixed_scorer({"a": 0.2, "b": 0.9})
    result = rerank_candidates({"query": "q", "candidates": candidates_from_pool(pool)}, SCOPE)
    updated = update_candidate_pool(pool, result, iteration=2)
    assert updated[1]["scores"]["rerank_candidates"] < updated[2]["scores"]["rerank_candidates"]
    assert updated[1]["scores"]["metadata_search"] == 0.5  # original evidence untouched


# =============================================================================
# PLANNING / FULL-LOOP INTEGRATION
# =============================================================================

def _flat_pool(n=3):
    return {
        i: {
            "evidence_sources": ["hybrid_search"],
            "scores": {"hybrid_search": 0.02},
            "metadata": {"filename": f"File_{i}.txt", "summary": f"doc {i}"},
            "first_seen_iteration": 1,
            "last_updated_iteration": 1,
        }
        for i in range(n)
    }


# --- 10. Agent can select reranking for an ambiguous/close candidate pool --

def test_10_plan_next_action_selects_rerank_after_hybrid_leaves_pool_flat():
    from app.goal import understand_goal
    from app.planning import plan_next_action
    from app.state import Action

    query = "quarterly budget planning review"
    goal = understand_goal(query).to_dict()
    committed = [
        Action(tool="metadata_search", args={"topic_tokens": goal["topic_tokens"]}, reason="x", iteration=1),
        Action(tool="bm25_search", args={"query": query}, reason="x", iteration=2),
        Action(tool="semantic_search", args={"query": query}, reason="x", iteration=3),
        Action(tool="hybrid_search", args={"query": query}, reason="x", iteration=4),
    ]
    action = plan_next_action(
        goal,
        committed_actions=committed,
        failed_strategies=[],
        iteration=5,
        observation_hint="too_many",
        candidate_pool=_flat_pool(),
    )
    assert action is not None
    assert action["tool"] == "rerank_candidates"
    assert action["args"]["query"]
    assert len(action["args"]["candidates"]) == 3


def test_10_plan_next_action_does_not_select_rerank_with_fewer_than_two_candidates():
    from app.goal import understand_goal
    from app.planning import plan_next_action
    from app.state import Action

    query = "quarterly budget planning review"
    goal = understand_goal(query).to_dict()
    committed = [
        Action(tool="metadata_search", args={"topic_tokens": goal["topic_tokens"]}, reason="x", iteration=1),
        Action(tool="bm25_search", args={"query": query}, reason="x", iteration=2),
        Action(tool="semantic_search", args={"query": query}, reason="x", iteration=3),
        Action(tool="hybrid_search", args={"query": query}, reason="x", iteration=4),
    ]
    action = plan_next_action(
        goal,
        committed_actions=committed,
        failed_strategies=[],
        iteration=5,
        observation_hint="too_many",
        candidate_pool=_flat_pool(n=1),
    )
    assert action is None


def test_10_plan_next_action_never_selects_rerank_twice_in_one_run():
    from app.goal import understand_goal
    from app.planning import plan_next_action
    from app.state import Action

    query = "quarterly budget planning review"
    goal = understand_goal(query).to_dict()
    committed = [
        Action(tool="hybrid_search", args={"query": query}, reason="x", iteration=1),
        Action(tool="rerank_candidates", args={"query": query, "candidates": []}, reason="x", iteration=2),
    ]
    action = plan_next_action(
        goal,
        committed_actions=committed,
        failed_strategies=[],
        iteration=3,
        observation_hint="too_many",
        candidate_pool=_flat_pool(),
    )
    # rerank_candidates itself is never proposed again this run -- but
    # Phase 4A adds get_user_activity as the next (and final) escalation
    # in this chain once rerank has already been tried, per
    # architecture_final.md §7's "Reranked, still tied -> get_user_activity".
    # See tests/test_get_user_activity.py for that escalation's own tests.
    assert action is not None
    assert action["tool"] != "rerank_candidates"


# --- 11. Reranking does not execute unnecessarily when one dominant candidate exists

def test_11_reflect_reports_finish_not_too_many_for_a_dominant_candidate():
    """When reflection sees a clearly dominant top candidate, status is
    'finish' with observation_hint=None -- plan_next_action is never even
    invoked with 'too_many', so rerank_candidates is never proposed."""
    pool = {
        1: {"scores": {"hybrid_search": 0.9}, "metadata": {}, "evidence_sources": ["hybrid_search"]},
        2: {"scores": {"hybrid_search": 0.1}, "metadata": {}, "evidence_sources": ["hybrid_search"]},
    }
    rr = reflect(
        last_action={"tool": "hybrid_search", "args": {"query": "x"}, "reason": "x", "iteration": 1},
        last_tool_result={"tool": "hybrid_search", "candidates": [{"file_id": 1, "score": 0.9}], "error": None},
        candidate_pool=pool,
        failed_strategies=[],
        errors=[],
        iteration_count=1,
        max_iterations=6,
    )
    assert rr.status == "finish"
    assert rr.observation_hint is None


# --- 12. Full run_query loop can reach rerank_candidates --------------------

def test_12_full_run_query_loop_terminates_cleanly_whether_or_not_rerank_is_reached():
    """Integration smoke test through the real graph: whatever path this
    corpus/query happens to take, the run must terminate cleanly, and if
    rerank_candidates was used, its evidence must be visible in the pool."""
    from app.run import run_query

    result = run_query("user_a", "quarterly budget planning review")
    tools_used = [a["tool"] for a in result["committed_actions"]]
    if "rerank_candidates" in tools_used:
        assert any(
            "rerank_candidates" in entry.get("evidence_sources", [])
            for entry in result["candidate_pool"].values()
        )
    assert result["status"] in ("finish", "clarify")


# =============================================================================
# REAL-MODEL SMOKE TEST (skipped automatically if the environment can't reach
# huggingface.co to download cross-encoder/ms-marco-MiniLM-L-6-v2 -- see
# app/reranker.py's module docstring, same sandbox constraint as
# app/embeddings.py's semantic-search provider).
# =============================================================================

def test_13_real_cross_encoder_smoke_test_if_available():
    from app.reranker import PREFERRED_MODEL_NAME, _CrossEncoderProvider

    try:
        provider = _CrossEncoderProvider(PREFERRED_MODEL_NAME)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"real cross-encoder unavailable in this environment: {exc}")

    scores = provider.score(
        "quarterly budget review",
        ["Quarterly budget planning review meeting notes.", "Unrelated recipe for banana bread."],
    )
    assert len(scores) == 2
    assert scores[0] > scores[1]
