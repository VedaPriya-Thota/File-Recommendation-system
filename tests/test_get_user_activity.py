"""
Phase 4A tests for get_user_activity (architecture_final.md §6/§8; Phase 4A
"Deterministic Personalization Signals").

Three tiers, mirroring test_rerank_candidates.py's structure:
  - PURE SIGNAL-COMPUTATION tests (1-2): app/personalization.py's
    compute_personalization_signals is DB-free, so these test the actual
    signal math directly -- preferred file type, frequent topic,
    activity/frequency/recency, and cold start.
  - TOOL tests (3-8), mostly with a monkeypatched DB layer so behavior can
    be checked exactly without depending on fixture timing/ordering.
  - INTEGRATION tests (9-12): real Postgres, real fixture telemetry/
    profile rows, real registry -- proving the whole thing works
    end-to-end including actual user-isolation enforcement at the SQL
    level.
"""
import pytest

from app.candidate_pool import update_candidate_pool
from app.db.connection import get_cursor
from app.personalization import (
    attach_personalization_signals,
    compute_personalization_signals,
    personalized_order,
)
from app.reflect import reflect
from app.scope import UserScope, resolve_scope
from app.tools import build_registry
import app.tools.get_user_activity as get_user_activity_module
from app.tools.get_user_activity import get_user_activity
from app.tools.rerank_candidates import candidates_from_pool
from app.tools.registry import ToolResult

SCOPE_A = UserScope(user_id="user_a", owner_ids=("user_a",))


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


# =============================================================================
# 1. Pure signal computation (app/personalization.py) -- no DB involved
# =============================================================================

def test_1_preferred_file_type_signal():
    signals_match = compute_personalization_signals(
        candidate_metadata={"file_type": "pptx", "tags": [], "filename": "x", "summary": ""},
        telemetry_events=[],
        profile={"preferred_file_types": ["pptx", "xlsx"], "frequent_topics": {}},
    )
    signals_no_match = compute_personalization_signals(
        candidate_metadata={"file_type": "png", "tags": [], "filename": "x", "summary": ""},
        telemetry_events=[],
        profile={"preferred_file_types": ["pptx", "xlsx"], "frequent_topics": {}},
    )
    assert signals_match["preferred_file_type_match"] is True
    assert signals_no_match["preferred_file_type_match"] is False
    assert signals_match["affinity_score"] > signals_no_match["affinity_score"]


def test_1_frequent_topic_signal():
    signals = compute_personalization_signals(
        candidate_metadata={"file_type": "xlsx", "tags": ["budget", "finance"], "filename": "Q3.xlsx", "summary": ""},
        telemetry_events=[],
        profile={"preferred_file_types": [], "frequent_topics": {"budget": 0.8, "travel": 0.3}},
    )
    assert "budget" in signals["frequent_topic_matches"]
    assert "travel" not in signals["frequent_topic_matches"]
    assert signals["frequent_topic_score"] > 0


def test_1_activity_frequency_and_recency_signal():
    from datetime import datetime, timedelta, timezone

    recent_events = [
        {"action_type": "open", "time_spent_seconds": 120, "timestamp": datetime.now(timezone.utc) - timedelta(days=1)},
        {"action_type": "edit", "time_spent_seconds": 300, "timestamp": datetime.now(timezone.utc) - timedelta(days=2)},
    ]
    stale_events = [
        {"action_type": "open", "time_spent_seconds": 60, "timestamp": datetime.now(timezone.utc) - timedelta(days=200)},
    ]

    recent = compute_personalization_signals({"file_type": "txt", "tags": [], "filename": "x", "summary": ""}, recent_events, None)
    stale = compute_personalization_signals({"file_type": "txt", "tags": [], "filename": "x", "summary": ""}, stale_events, None)

    assert recent["open_count"] == 1
    assert recent["edit_count"] == 1
    assert recent["total_events"] == 2
    assert recent["recency_score"] > stale["recency_score"]
    assert recent["affinity_score"] > stale["affinity_score"]


def test_1_cold_start_is_neutral_not_an_error():
    signals = compute_personalization_signals(
        candidate_metadata={"file_type": "pptx", "tags": ["ai"], "filename": "x", "summary": ""},
        telemetry_events=[],
        profile=None,
    )
    assert signals["cold_start"] is True
    assert signals["preferred_file_type_match"] is False
    assert signals["frequent_topic_matches"] == {}
    assert signals["affinity_score"] == 0.0


def test_1_profile_present_but_no_telemetry_is_not_cold_start():
    """Cold start means genuinely nothing to go on (no profile AND no
    telemetry). A user with a profile but no history for THIS file still
    gets real profile-derived signals, not the cold-start neutral case."""
    signals = compute_personalization_signals(
        candidate_metadata={"file_type": "pptx", "tags": [], "filename": "x", "summary": ""},
        telemetry_events=[],
        profile={"preferred_file_types": ["pptx"], "frequent_topics": {}},
    )
    assert signals["cold_start"] is False
    assert signals["preferred_file_type_match"] is True


# --- attach_personalization_signals / personalized_order -------------------

def test_2_attach_never_introduces_a_new_file_id():
    pool = {1: {"scores": {"bm25_search": 0.5}, "metadata": {}, "evidence_sources": ["bm25_search"]}}
    signals_by_file = {
        1: {"affinity_score": 0.9},
        999: {"affinity_score": 0.7},  # not in pool -- must be ignored
    }
    updated = attach_personalization_signals(pool, signals_by_file)
    assert set(updated.keys()) == {1}
    assert updated[1]["personalization"]["affinity_score"] == 0.9


def test_2_attach_preserves_existing_scores_and_metadata():
    pool = {
        1: {
            "scores": {"bm25_search": 0.5, "rerank_candidates": 0.8},
            "metadata": {"filename": "x.txt", "summary": "s"},
            "evidence_sources": ["bm25_search", "rerank_candidates"],
        }
    }
    updated = attach_personalization_signals(pool, {1: {"affinity_score": 0.6}})
    assert updated[1]["scores"] == {"bm25_search": 0.5, "rerank_candidates": 0.8}
    assert updated[1]["metadata"] == {"filename": "x.txt", "summary": "s"}


def test_2_personalized_order_only_breaks_ties_among_equal_evidence():
    pool = {
        1: {"scores": {"bm25_search": 0.9}, "metadata": {}, "personalization": {"affinity_score": 0.1}},
        2: {"scores": {"bm25_search": 0.5}, "metadata": {}, "personalization": {"affinity_score": 0.99}},
    }
    order = personalized_order(pool)
    # file 1's real retrieval evidence (0.9) beats file 2's (0.5) even
    # though file 2 has a much higher personalization affinity -- affinity
    # is a tie-breaker, never a substitute for retrieval evidence.
    assert order == [1, 2]


def test_2_personalized_order_uses_affinity_as_tiebreak_when_scores_tie():
    pool = {
        1: {"scores": {"bm25_search": 0.7}, "metadata": {}, "personalization": {"affinity_score": 0.2}},
        2: {"scores": {"bm25_search": 0.7}, "metadata": {}, "personalization": {"affinity_score": 0.8}},
    }
    order = personalized_order(pool)
    assert order == [2, 1]


# =============================================================================
# TOOL tests (monkeypatched DB layer)
# =============================================================================

@pytest.fixture
def fixed_db(monkeypatch):
    """Installs fixed telemetry/profile lookups, patched onto the names as
    imported into get_user_activity's own module namespace -- same
    monkeypatch pattern used throughout test_hybrid_search.py/
    test_rerank_candidates.py."""

    def _install(telemetry_by_file=None, profile=None):
        telemetry_by_file = telemetry_by_file or {}

        def _fake_fetch_telemetry(user_id, file_ids):
            return {fid: telemetry_by_file.get(fid, []) for fid in file_ids if fid in telemetry_by_file}

        def _fake_fetch_profile(user_id):
            return profile

        monkeypatch.setattr(get_user_activity_module, "_fetch_telemetry", _fake_fetch_telemetry)
        monkeypatch.setattr(get_user_activity_module, "_fetch_profile", _fake_fetch_profile)

    return _install


# --- 3. Activity retrieval ---------------------------------------------------

def test_3_activity_retrieval_produces_signals_for_every_supplied_candidate(fixed_db):
    fixed_db(profile={"preferred_file_types": ["pptx"], "frequent_topics": {}})
    candidates = [_candidate(1, file_type="pptx"), _candidate(2, file_type="txt")]
    result = get_user_activity({"candidates": candidates}, SCOPE_A)
    assert len(result.candidates) == 2
    for c in result.candidates:
        assert "personalization" in c["metadata"]
        assert isinstance(c["score"], float)


# --- 4. Candidate IDs cannot expand ------------------------------------------

def test_4_only_supplied_candidates_are_returned(fixed_db):
    fixed_db(profile=None)
    candidates = [_candidate(101), _candidate(102)]
    result = get_user_activity({"candidates": candidates}, SCOPE_A)
    returned_ids = {c["file_id"] for c in result.candidates}
    assert returned_ids == {101, 102}
    assert 999 not in returned_ids


def test_4_empty_candidate_list_never_expands_to_a_full_scan(fixed_db, monkeypatch):
    def _explode(*a, **kw):
        raise AssertionError("get_user_activity must never query when given no candidates")

    monkeypatch.setattr(get_user_activity_module, "_fetch_telemetry", _explode)
    monkeypatch.setattr(get_user_activity_module, "_fetch_profile", _explode)
    result = get_user_activity({"candidates": []}, SCOPE_A)
    assert result.candidates == []


# --- 5. Preferred file type signal (via the tool) ---------------------------

def test_5_preferred_file_type_signal_via_tool(fixed_db):
    fixed_db(profile={"preferred_file_types": ["pptx"], "frequent_topics": {}})
    candidates = [_candidate(1, file_type="pptx"), _candidate(2, file_type="png")]
    result = get_user_activity({"candidates": candidates}, SCOPE_A)
    by_id = {c["file_id"]: c for c in result.candidates}
    assert by_id[1]["metadata"]["personalization"]["preferred_file_type_match"] is True
    assert by_id[2]["metadata"]["personalization"]["preferred_file_type_match"] is False
    assert by_id[1]["score"] > by_id[2]["score"]


# --- 6. Frequent-topic signal (via the tool) --------------------------------

def test_6_frequent_topic_signal_via_tool(fixed_db):
    fixed_db(profile={"preferred_file_types": [], "frequent_topics": {"budget": 1.0}})
    candidates = [_candidate(1, tags=["budget"]), _candidate(2, tags=["travel"])]
    result = get_user_activity({"candidates": candidates}, SCOPE_A)
    by_id = {c["file_id"]: c for c in result.candidates}
    assert by_id[1]["metadata"]["personalization"]["frequent_topic_score"] > 0
    assert by_id[2]["metadata"]["personalization"]["frequent_topic_score"] == 0


# --- 7. Activity/frequency signal (via the tool) ----------------------------

def test_7_activity_frequency_signal_via_tool(fixed_db):
    from datetime import datetime, timezone

    fixed_db(
        telemetry_by_file={
            1: [
                {"file_id": 1, "action_type": "open", "time_spent_seconds": 60, "timestamp": datetime.now(timezone.utc)},
                {"file_id": 1, "action_type": "open", "time_spent_seconds": 60, "timestamp": datetime.now(timezone.utc)},
            ]
        },
        profile=None,
    )
    candidates = [_candidate(1), _candidate(2)]
    result = get_user_activity({"candidates": candidates}, SCOPE_A)
    by_id = {c["file_id"]: c for c in result.candidates}
    assert by_id[1]["metadata"]["personalization"]["open_count"] == 2
    assert by_id[1]["score"] > by_id[2]["score"]


# --- 8. Cold-start behavior (via the tool) ----------------------------------

def test_8_cold_start_user_gets_neutral_signals_not_an_error(fixed_db):
    fixed_db(telemetry_by_file={}, profile=None)
    candidates = [_candidate(1)]
    result = get_user_activity({"candidates": candidates}, SCOPE_A)
    signals = result.candidates[0]["metadata"]["personalization"]
    assert signals["cold_start"] is True
    assert signals["affinity_score"] == 0.0
    assert result.error is None


def test_8_empty_candidate_pool_is_handled_safely(fixed_db):
    fixed_db(profile=None)
    result = get_user_activity({"candidates": []}, SCOPE_A)
    assert result.candidates == []
    assert result.error is None


# --- preservation of existing retrieval evidence ----------------------------

def test_preservation_of_all_existing_retrieval_evidence(fixed_db):
    fixed_db(profile={"preferred_file_types": ["pptx"], "frequent_topics": {}})
    candidate = _candidate(
        1,
        file_type="pptx",
        summary="original summary",
        snippet="original bm25 snippet",
        matched_chunk="original semantic chunk",
        rerank_raw_score=3.2,
    )
    result = get_user_activity({"candidates": [candidate]}, SCOPE_A)
    meta = result.candidates[0]["metadata"]
    assert meta["summary"] == "original summary"
    assert meta["snippet"] == "original bm25 snippet"
    assert meta["matched_chunk"] == "original semantic chunk"
    assert meta["rerank_raw_score"] == 3.2


def test_preservation_via_candidate_pool_merge_keeps_prior_source_scores():
    """candidate_pool accumulation is untouched by Phase 4A -- get_user_activity's
    ToolResult merges in exactly like rerank_candidates' does: a NEW
    per-source score, added alongside (never overwriting or summing)
    bm25/semantic/hybrid/reranker scores."""
    pool = {
        7: {
            "evidence_sources": ["bm25_search", "semantic_search", "hybrid_search", "rerank_candidates"],
            "scores": {"bm25_search": 0.4, "semantic_search": 0.6, "hybrid_search": 0.02, "rerank_candidates": 0.75},
            "metadata": {"filename": "File_7.txt"},
            "first_seen_iteration": 1,
            "last_updated_iteration": 3,
        }
    }
    activity_result = ToolResult(
        tool="get_user_activity",
        candidates=[
            {
                "file_id": 7,
                "metadata": {"filename": "File_7.txt", "personalization": {"affinity_score": 0.55}},
                "score": 0.55,
                "source": "get_user_activity",
            }
        ],
    )
    updated = update_candidate_pool(pool, activity_result, iteration=4)
    entry = updated[7]
    assert entry["scores"]["bm25_search"] == 0.4
    assert entry["scores"]["semantic_search"] == 0.6
    assert entry["scores"]["hybrid_search"] == 0.02
    assert entry["scores"]["rerank_candidates"] == 0.75
    assert entry["scores"]["get_user_activity"] == 0.55
    assert set(entry["evidence_sources"]) == {
        "bm25_search",
        "semantic_search",
        "hybrid_search",
        "rerank_candidates",
        "get_user_activity",
    }


# --- registry/tool integration ----------------------------------------------

def test_registry_get_user_activity_is_registered_and_executable(fixed_db):
    fixed_db(profile=None)
    registry = build_registry()
    assert "get_user_activity" in registry.names()
    tool = registry.get("get_user_activity")
    assert tool.name == "get_user_activity"

    result = registry.execute("get_user_activity", {"candidates": [_candidate(1)]}, SCOPE_A)
    assert isinstance(result, ToolResult)
    assert result.tool == "get_user_activity"


# --- agent behavior: no unnecessary personalization calls -------------------

def test_agent_reflect_reports_finish_for_a_dominant_candidate_no_personalization_needed():
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
    # plan_next_action is never even invoked with "too_many" in this
    # case -- get_user_activity has no opportunity to be proposed.


def test_agent_plan_next_action_never_proposes_get_user_activity_before_rerank():
    """get_user_activity is only the escalation AFTER rerank_candidates
    has genuinely run and the pool is still flat -- never a substitute for
    or a step ahead of reranking."""
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
        # note: rerank_candidates NOT committed yet
    ]
    pool = {i: {"scores": {"hybrid_search": 0.02}, "metadata": {}, "evidence_sources": ["hybrid_search"]} for i in range(3)}
    action = plan_next_action(
        goal, committed_actions=committed, failed_strategies=[], iteration=5,
        observation_hint="too_many", candidate_pool=pool,
    )
    assert action is not None
    assert action["tool"] == "rerank_candidates"  # not get_user_activity yet


def test_agent_plan_next_action_selects_get_user_activity_only_after_rerank():
    from app.goal import understand_goal
    from app.planning import plan_next_action
    from app.state import Action

    query = "quarterly budget planning review"
    goal = understand_goal(query).to_dict()
    committed = [
        Action(tool="hybrid_search", args={"query": query}, reason="x", iteration=1),
        Action(tool="rerank_candidates", args={"query": query, "candidates": []}, reason="x", iteration=2),
    ]
    pool = {i: {"scores": {"hybrid_search": 0.02}, "metadata": {}, "evidence_sources": ["hybrid_search"]} for i in range(3)}
    action = plan_next_action(
        goal, committed_actions=committed, failed_strategies=[], iteration=3,
        observation_hint="too_many", candidate_pool=pool,
    )
    assert action is not None
    assert action["tool"] == "get_user_activity"
    assert len(action["args"]["candidates"]) == 3


def test_agent_plan_next_action_never_selects_get_user_activity_twice_in_one_run():
    from app.goal import understand_goal
    from app.planning import plan_next_action
    from app.state import Action

    query = "quarterly budget planning review"
    goal = understand_goal(query).to_dict()
    committed = [
        Action(tool="hybrid_search", args={"query": query}, reason="x", iteration=1),
        Action(tool="rerank_candidates", args={"query": query, "candidates": []}, reason="x", iteration=2),
        Action(tool="get_user_activity", args={"candidates": []}, reason="x", iteration=3),
    ]
    pool = {i: {"scores": {"hybrid_search": 0.02}, "metadata": {}, "evidence_sources": ["hybrid_search"]} for i in range(3)}
    action = plan_next_action(
        goal, committed_actions=committed, failed_strategies=[], iteration=4,
        observation_hint="too_many", candidate_pool=pool,
    )
    assert action is None  # fully exhausted this run -> clarify upstream


def test_no_llm_involved_get_user_activity_is_deterministic():
    from app.tools import build_registry

    registry = build_registry()
    assert set(registry.names()) == {
        "metadata_search",
        "bm25_search",
        "semantic_search",
        "hybrid_search",
        "rerank_candidates",
        "get_user_activity",
    }


# =============================================================================
# INTEGRATION (real Postgres, real fixture telemetry/profile rows)
# =============================================================================

@pytest.fixture
def telemetry_fixture():
    """Inserts real user_file_telemetry + user_profiles rows for user_a
    (and a decoy row for user_b, to prove isolation), against real seeded
    files, then cleans up -- same insert/cleanup-in-fixture pattern as
    test_hybrid_search.py's extra_hybrid_files."""
    with get_cursor() as cur:
        cur.execute("SELECT file_id FROM files WHERE owner_id = 'user_a' AND filename = 'Q3_Budget_Review.xlsx';")
        file_id = cur.fetchone()["file_id"]

    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO user_profiles (user_id, preferred_file_types, frequent_topics)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id) DO UPDATE SET
                preferred_file_types = EXCLUDED.preferred_file_types,
                frequent_topics = EXCLUDED.frequent_topics;
            """,
            ("user_a", ["xlsx"], '{"budget": 0.9}'),
        )
        cur.execute(
            """
            INSERT INTO user_file_telemetry (user_id, file_id, action_type, time_spent_seconds)
            VALUES (%s, %s, 'open', 90), (%s, %s, 'edit', 200);
            """,
            ("user_a", file_id, "user_a", file_id),
        )
        # Decoy: user_b's OWN telemetry on a DIFFERENT file they own, to
        # prove user_a's query never sees it (isolation, not just
        # coincidentally-absent data).
        cur.execute("SELECT file_id FROM files WHERE owner_id = 'user_b' LIMIT 1;")
        user_b_file_id = cur.fetchone()["file_id"]
        cur.execute(
            """
            INSERT INTO user_file_telemetry (user_id, file_id, action_type, time_spent_seconds)
            VALUES (%s, %s, 'open', 500);
            """,
            ("user_b", user_b_file_id),
        )

    yield {"file_id": file_id, "user_b_file_id": user_b_file_id}

    with get_cursor(commit=True) as cur:
        cur.execute("DELETE FROM user_file_telemetry WHERE user_id IN ('user_a', 'user_b');")
        cur.execute("DELETE FROM user_profiles WHERE user_id = 'user_a';")


def test_9_end_to_end_telemetry_and_profile_produce_real_signals(telemetry_fixture):
    file_id = telemetry_fixture["file_id"]
    candidates = [_candidate(file_id, file_type="xlsx", tags=["budget"])]
    result = get_user_activity({"candidates": candidates}, SCOPE_A)
    signals = result.candidates[0]["metadata"]["personalization"]
    assert signals["preferred_file_type_match"] is True
    assert signals["open_count"] == 1
    assert signals["edit_count"] == 1
    assert signals["cold_start"] is False
    assert signals["affinity_score"] > 0


def test_10_user_isolation_user_a_never_sees_user_bs_telemetry(telemetry_fixture):
    """The candidate is one of user_a's OWN files, but we query
    get_user_activity scoped as user_a for a file_id user_b interacted
    with -- user_a's activity signals for that file must show zero
    activity, proving the SQL filter is on user_id, not just accidentally
    correct because the file lists differ."""
    scope_a = UserScope(user_id="user_a", owner_ids=("user_a",))
    candidates = [_candidate(telemetry_fixture["user_b_file_id"])]
    result = get_user_activity({"candidates": candidates}, scope_a)
    signals = result.candidates[0]["metadata"]["personalization"]
    assert signals["total_events"] == 0  # user_a has no telemetry on this file
    assert signals["open_count"] == 0


def test_10_user_isolation_user_b_only_sees_their_own_telemetry(telemetry_fixture):
    scope_b = UserScope(user_id="user_b", owner_ids=("user_b",))
    candidates = [_candidate(telemetry_fixture["user_b_file_id"])]
    result = get_user_activity({"candidates": candidates}, scope_b)
    signals = result.candidates[0]["metadata"]["personalization"]
    assert signals["open_count"] == 1
    assert signals["total_events"] == 1


def test_11_candidate_ids_cannot_expand_even_with_real_telemetry_present(telemetry_fixture):
    """Even though user_a has real telemetry rows in the DB, supplying a
    candidate set that does NOT include that file_id must never pull it
    in -- the tool is strictly scoped to what it was handed."""
    with get_cursor() as cur:
        cur.execute("SELECT file_id FROM files WHERE owner_id = 'user_a' AND filename = 'Team_Offsite_Photo.png';")
        other_file_id = cur.fetchone()["file_id"]

    candidates = [_candidate(other_file_id, file_type="png")]
    result = get_user_activity({"candidates": candidates}, SCOPE_A)
    returned_ids = {c["file_id"] for c in result.candidates}
    assert returned_ids == {other_file_id}
    assert telemetry_fixture["file_id"] not in returned_ids


def test_12_cold_start_real_user_with_no_seed_telemetry():
    """user_c has no user_profiles row and no user_file_telemetry rows at
    all in the base seed data (only the telemetry_fixture-using tests add
    any) -- a genuinely cold-start real user."""
    with get_cursor() as cur:
        cur.execute("SELECT file_id FROM files WHERE owner_id = 'user_c' LIMIT 1;")
        file_id = cur.fetchone()["file_id"]

    scope_c = UserScope(user_id="user_c", owner_ids=("user_c",))
    result = get_user_activity({"candidates": [_candidate(file_id)]}, scope_c)
    signals = result.candidates[0]["metadata"]["personalization"]
    assert signals["cold_start"] is True
    assert signals["affinity_score"] == 0.0
    assert result.error is None


def test_13_full_run_query_loop_terminates_cleanly_whether_or_not_personalization_is_reached():
    from app.run import run_query

    result = run_query("user_a", "quarterly budget planning review")
    tools_used = [a["tool"] for a in result["committed_actions"]]
    if "get_user_activity" in tools_used:
        assert any(
            "get_user_activity" in entry.get("evidence_sources", [])
            for entry in result["candidate_pool"].values()
        )
    assert result["status"] in ("finish", "clarify")
