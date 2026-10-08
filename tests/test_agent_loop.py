"""
Integration tests for the Phase 1 LangGraph agent loop.

Maps directly to the prompt's required scenarios:
    Test A - specific query
    Test B - ambiguous query
    Test C - nonexistent query
    Test D - access control
    Test E - real replanning (observation-driven strategy change)
"""
import pytest

from app.run import run_query


# --- Test A: specific query ------------------------------------------

def test_A_specific_query_finds_the_deck_without_looping():
    result = run_query("user_a", "Find the slide deck about AI architecture I edited last week.")

    assert result["status"] == "finish"
    assert result["iteration_count"] == 1, "a well-specified query should resolve in one action"
    filenames = [c["metadata"]["filename"] for c in result["candidate_pool"].values()]
    assert "AI_Architecture_Overview.pptx" in filenames
    assert result["confidence"] > 0.9

    # no LLM tools exist in the Phase 1 registry at all
    for action in result["committed_actions"]:
        assert action["tool"] == "metadata_search"


# --- Test B: ambiguous query -------------------------------------------

def test_B_ambiguous_query_does_not_blindly_return_first_result():
    result = run_query("user_a", "The budget thing I was looking at.")

    # Must recognize ambiguity: either clarify, or finish explicitly
    # flagged low-confidence -- never a silent single guess.
    assert result["status"] in ("clarify", "finish")
    if result["status"] == "finish":
        assert result["degraded_mode"] is True
    else:
        assert result["clarification_question"]

    # Multiple genuinely tied candidates should have been observed, proving
    # the agent looked before concluding it couldn't pick one.
    assert len(result["candidate_pool"]) > 1

    # No infinite loop.
    assert result["iteration_count"] <= result["committed_actions"][0]["iteration"] + 3


def test_B_ambiguous_query_terminates_quickly():
    result = run_query("user_a", "The budget thing I was looking at.", max_iterations=6)
    assert result["iteration_count"] < 6


# --- Test C: nonexistent query ------------------------------------------

def test_C_nonexistent_query_no_infinite_retry():
    result = run_query(
        "user_a", "the zzznonexistentxyz122 pineapplesubmarine987 file", max_iterations=6
    )

    assert result["status"] == "finish"
    assert result["degraded_mode"] is True  # flagged low-confidence / nothing found
    assert result["candidate_pool"] == {}
    assert result["iteration_count"] < 6, "must not exhaust max_iterations via blind retrying"

    # failed_strategies must have tracked the attempted (tool, args)
    assert len(result["committed_actions"]) >= 1


def test_C_max_iterations_hard_guard_never_exceeded():
    result = run_query("user_a", "asdkjaslkdj qweoiqwoe", max_iterations=3)
    assert result["iteration_count"] <= 3


# --- Test D: access control ----------------------------------------------

def test_D_user_x_files_never_visible_to_user_a():
    result = run_query(
        "user_a", "confidential secret roadmap salaries HR AI architecture"
    )
    owners = {c["metadata"]["owner_id"] for c in result["candidate_pool"].values()}
    assert "user_x" not in owners
    assert owners <= {"user_a"}


def test_D_user_x_files_never_visible_to_user_b():
    result = run_query(
        "user_b", "confidential secret roadmap salaries HR AI architecture"
    )
    owners = {c["metadata"]["owner_id"] for c in result["candidate_pool"].values()}
    assert "user_x" not in owners
    assert owners <= {"user_b"}


def test_D_user_x_can_see_their_own_files():
    result = run_query("user_x", "confidential HR salaries")
    owners = {c["metadata"]["owner_id"] for c in result["candidate_pool"].values()}
    assert owners <= {"user_x"}
    filenames = [c["metadata"]["filename"] for c in result["candidate_pool"].values()]
    assert "Confidential_HR_Salaries.xlsx" in filenames


def test_D_user_c_only_sees_permitted_files():
    """user_c owns a topically-colliding AI-architecture file; a search for
    that exact topic from user_c must return ONLY user_c's own file, never
    user_a's or user_x's same-topic files.
    """
    result = run_query("user_c", "AI architecture notes")
    owners = {c["metadata"]["owner_id"] for c in result["candidate_pool"].values()}
    assert owners <= {"user_c"}
    filenames = [c["metadata"]["filename"] for c in result["candidate_pool"].values()]
    assert "AI_Architecture_Notes.docx" in filenames
    assert "AI_Architecture_Overview.pptx" not in filenames
    assert "AI_Architecture_Secret_Roadmap.pptx" not in filenames


def test_D_user_a_cannot_see_user_c_files_despite_topic_collision():
    result = run_query("user_a", "AI architecture notes")
    owners = {c["metadata"]["owner_id"] for c in result["candidate_pool"].values()}
    assert "user_c" not in owners
    assert owners <= {"user_a"}


def test_D_unknown_user_is_a_hard_failure_not_an_empty_result():
    result = run_query("totally_unknown_user_id", "anything")
    assert result["status"] == "hard_failure"
    assert "Unknown user_id" in result["error"]


# --- Test E: real observation-driven replanning --------------------------

def test_E_zero_results_causes_a_genuinely_different_next_action():
    """The initial action (file_type=xlsx, wrong guess) must return zero
    results, causing reflect -> replan, and the SECOND action must have
    different args (file_type dropped) that then succeeds. This proves the
    loop is observation-driven, not a fixed sequence.
    """
    result = run_query(
        "user_a", "Find the spreadsheet about AI architecture I edited last week."
    )

    assert len(result["committed_actions"]) == 2, "must have replanned exactly once"
    first, second = result["committed_actions"]

    assert first["args"].get("file_type") == "xlsx"
    assert "file_type" not in second["args"], "file_type must be dropped on replan"
    assert first["args"] != second["args"]

    assert result["status"] == "finish"
    filenames = [c["metadata"]["filename"] for c in result["candidate_pool"].values()]
    assert "AI_Architecture_Overview.pptx" in filenames

    # Confirm from the trace that this was genuinely observation-driven:
    # action 1 -> zero candidates -> reflect(replan) -> action 2 -> found it.
    events = [e["event"] for e in result["trace"]]
    assert events.count("execute_tool") == 2
    assert events.count("reflect") == 2
    reflect_events = [e for e in result["trace"] if e["event"] == "reflect"]
    assert reflect_events[0]["status"] == "replan"
    assert reflect_events[0]["observation_hint"] == "zero_results"
    assert reflect_events[1]["status"] == "finish"


def test_E_failed_strategy_is_never_repeated_identically():
    result = run_query(
        "user_a", "Find the spreadsheet about AI architecture I edited last week."
    )
    sigs = [
        (a["tool"], tuple(sorted(a["args"].keys())))
        for a in result["committed_actions"]
    ]
    assert len(sigs) == len(set(sigs)), "no two committed actions should have identical arg shape+values repeated"


# --- Test F: PostgreSQL failure is a hard failure -------------------------

def test_F_postgres_outage_during_scope_resolution_is_a_hard_failure():
    """Simulates a genuine Postgres outage (not a soft tool error) at the
    very first node (resolve_scope). architecture_final.md §7/§11/§14: this
    must propagate as an explicit hard failure -- never degraded around,
    never silently treated like a normal zero-result search.
    """
    import psycopg2
    from unittest.mock import patch

    with patch("app.db.connection.psycopg2.connect", side_effect=psycopg2.OperationalError(
        "simulated Postgres outage"
    )):
        result = run_query("user_a", "Find the slide deck about AI architecture I edited last week.")

    assert result["status"] == "hard_failure"
    assert "simulated Postgres outage" in result["error"]
    # No partial/degraded result should be fabricated.
    assert "final_ranking" not in result or not result.get("final_ranking")


def test_F_postgres_outage_during_metadata_search_is_also_a_hard_failure():
    """Same guarantee if Postgres dies mid-run (during the tool call rather
    than during scope resolution) -- the registry re-raises DatabaseError
    rather than converting it into a soft, replan-able ToolResult.error.
    """
    import psycopg2
    from unittest.mock import patch

    from app.scope import resolve_scope
    from app.tools import build_registry

    registry = build_registry()
    scope = resolve_scope("user_a")  # succeeds before the simulated outage

    with patch("app.db.connection.psycopg2.connect", side_effect=psycopg2.OperationalError(
        "simulated mid-run outage"
    )):
        with pytest.raises(Exception):
            # execute() re-raises DatabaseError for Postgres failures
            # instead of swallowing it into ToolResult.error.
            registry.execute("metadata_search", {"file_type": "pptx"}, scope)


# --- Cross-cutting: tool error handling (non-DB) --------------------------

def test_tool_error_is_handled_without_crashing_the_graph():
    """A non-DB tool exception must be caught as a soft ToolResult.error,
    trigger replanning, and the loop must still terminate cleanly (finish
    or clarify) rather than crash or loop forever.
    """
    from app.scope import UserScope
    from app.tools.registry import Tool, ToolRegistry, ToolResult

    def always_fails(args: dict, user_scope: UserScope) -> ToolResult:
        raise RuntimeError("simulated non-DB tool failure")

    broken_registry = ToolRegistry()
    broken_registry.register(
        Tool(name="metadata_search", description="broken for this test", fn=always_fails)
    )

    result = run_query(
        "user_a",
        "Find the slide deck about AI architecture I edited last week.",
        registry=broken_registry,
        max_iterations=6,
    )

    assert result["status"] in ("finish", "clarify")
    assert result["iteration_count"] <= 6
    assert len(result["errors"]) >= 1
    assert "simulated non-DB tool failure" in result["errors"][0]["error"]


# --- Cross-cutting: no LLM anywhere ---------------------------------------

def test_no_llm_tool_is_ever_registered_or_called():
    """Phase 4A adds get_user_activity (a deterministic, read-only
    telemetry/profile lookup -- not an LLM), but all six registered tools
    remain deterministic (architecture_final.md §3/§9 -- LLM roles are all
    deferred); no LLM tool is registered."""
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
