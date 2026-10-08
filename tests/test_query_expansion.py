"""
Phase 5B tests -- LLM-assisted query expansion (app/llm/query_expander.py,
QueryExpansion in app/llm/schema.py, node_expand_query in
app/graph/build.py, and its consumption in app/planning.py).

Structure:
  1. Valid structured expansion
  2. Bounded number/length of terms
  3. Invalid output rejection
  4. Provider failure fallback
  5. No provider call for a clear query
  6. Ambiguous/underspecified query invokes expansion
  7. Original query preserved / expanded terms stored separately
  8. No SQL/tool/file/user-scope fields accepted
  9. Expansion cannot introduce candidate IDs
 10. Planning consumes expanded terms deterministically
 11. End-to-end: expanded query reaches deterministic retrieval safely
"""
import json

import pytest
from pydantic import ValidationError

from app.db.connection import get_cursor
from app.goal import understand_goal
from app.graph.build import build_graph
from app.llm.provider import LLMUnavailableError
from app.llm.query_expander import expand_query_with_llm
from app.llm.schema import (
    MAX_EXPANDED_TERM_LENGTH,
    MAX_EXPANDED_TERMS,
    LLMResponseInvalidError,
    QueryExpansion,
    parse_query_expansion_response,
)
from app.planning import plan_next_action
from app.run import _initial_state, run_query

UNDERSPECIFIED_QUERY = "the fiscal thing"  # ambiguity 0.9, no metadata hit on its own
CLEAR_QUERY = "Find the slide deck about AI architecture I edited last week."


class RoutingProvider:
    """Answers the query-expansion prompt with `expansion_response`; any
    other prompt (i.e. Phase 5A goal interpretation) fails, so these tests
    isolate 5B. Records which prompts were seen."""

    name = "fake"

    def __init__(self, expansion_response=None, raises=None):
        self.expansion_response = expansion_response
        self.raises = raises
        self.expansion_calls = 0
        self.other_calls = 0

    def complete(self, system_prompt, user_prompt):
        if "additional search terms" in system_prompt:
            self.expansion_calls += 1
            if self.raises is not None:
                raise self.raises
            return self.expansion_response
        self.other_calls += 1
        raise RuntimeError("goal interpretation intentionally unavailable in this test")


class NeverCalledProvider:
    name = "never_called"

    def complete(self, system_prompt, user_prompt):
        raise AssertionError("LLM provider must not be invoked for this query")


def _payload(**overrides) -> str:
    payload = {"expanded_terms": ["budget", "quarterly"], "confidence": 0.7, "rationale": "fiscal ~ budget"}
    payload.update(overrides)
    return json.dumps(payload)


def _run_graph(user_id, query, provider):
    graph = build_graph(llm_provider=provider)
    return graph.invoke(_initial_state(user_id, query, 6), config={"recursion_limit": 100})


# 1. Valid structured expansion -------------------------------------------------

def test_1_valid_payload_parses_and_normalizes():
    parsed = parse_query_expansion_response(_payload(expanded_terms=["Budget ", "quarterly", "budget"]))
    assert isinstance(parsed, QueryExpansion)
    assert parsed.expanded_terms == ["budget", "quarterly"]  # lowercased, de-duplicated, order kept
    assert parsed.confidence == 0.7


def test_1_fenced_json_tolerated_and_empty_list_allowed():
    parsed = parse_query_expansion_response("```json\n" + _payload(expanded_terms=[]) + "\n```")
    assert parsed.expanded_terms == []


def test_1_expand_query_with_llm_returns_terms_and_metadata():
    goal = understand_goal(UNDERSPECIFIED_QUERY).to_dict()
    provider = RoutingProvider(expansion_response=_payload())
    terms, meta = expand_query_with_llm(UNDERSPECIFIED_QUERY, goal, provider)
    assert terms == ["budget", "quarterly"]
    assert meta["expansion_status"] == "applied"
    assert meta["expansion_invoked"] is True


# 2. Bounds ---------------------------------------------------------------------

def test_2_too_many_terms_rejected():
    terms = [f"term{i}" for i in range(MAX_EXPANDED_TERMS + 1)]
    with pytest.raises(LLMResponseInvalidError):
        parse_query_expansion_response(_payload(expanded_terms=terms))


def test_2_exactly_max_terms_accepted():
    terms = [f"term{i}" for i in range(MAX_EXPANDED_TERMS)]
    assert len(parse_query_expansion_response(_payload(expanded_terms=terms)).expanded_terms) == MAX_EXPANDED_TERMS


def test_2_overlong_term_rejected():
    with pytest.raises(LLMResponseInvalidError):
        parse_query_expansion_response(_payload(expanded_terms=["a" * (MAX_EXPANDED_TERM_LENGTH + 1)]))


def test_2_disallowed_characters_rejected():
    for bad in ["budget; DROP TABLE files;--", "SELECT * FROM files", "a/b", "$(rm -rf)", "x' OR '1'='1"]:
        with pytest.raises(LLMResponseInvalidError):
            parse_query_expansion_response(_payload(expanded_terms=[bad]))


# 3. Invalid output rejection ---------------------------------------------------

@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        json.dumps(["budget"]),
        json.dumps({"expanded_terms": ["budget"]}),  # confidence missing
        _payload(confidence=1.5),
        _payload(expanded_terms="budget"),  # not a list
        _payload(expanded_terms=[1, 2]),  # not strings
    ],
)
def test_3_invalid_outputs_rejected(raw):
    with pytest.raises(LLMResponseInvalidError):
        parse_query_expansion_response(raw)


def test_3_invalid_output_falls_back_to_empty_expansion():
    goal = understand_goal(UNDERSPECIFIED_QUERY).to_dict()
    terms, meta = expand_query_with_llm(UNDERSPECIFIED_QUERY, goal, RoutingProvider(expansion_response="{bad"))
    assert terms == []
    assert meta["expansion_status"] == "invalid_output"


# 4. Provider failure fallback --------------------------------------------------

def test_4_provider_unavailable_falls_back():
    goal = understand_goal(UNDERSPECIFIED_QUERY).to_dict()
    terms, meta = expand_query_with_llm(
        UNDERSPECIFIED_QUERY, goal, RoutingProvider(raises=LLMUnavailableError("no key"))
    )
    assert terms == []
    assert meta["expansion_status"] == "unavailable"


def test_4_unexpected_provider_error_falls_back():
    goal = understand_goal(UNDERSPECIFIED_QUERY).to_dict()
    terms, meta = expand_query_with_llm(UNDERSPECIFIED_QUERY, goal, RoutingProvider(raises=TimeoutError("t/o")))
    assert terms == []
    assert meta["expansion_status"] == "provider_error"


def test_4_end_to_end_default_null_provider_matches_no_expansion_baseline():
    result = run_query("user_a", UNDERSPECIFIED_QUERY)  # default provider -> NullProvider
    assert result["query_expansion_metadata"]["expansion_status"] == "unavailable"
    assert result["status"] == "finish"
    assert [a["tool"] for a in result["committed_actions"]] == ["metadata_search", "bm25_search", "semantic_search"]
    assert result["candidate_pool"] == {}


# 5. No provider call for a clear query -----------------------------------------

def test_5_clear_query_never_calls_provider_and_has_no_expansion():
    assert understand_goal(CLEAR_QUERY).to_dict()["ambiguity_score"] < 0.65
    state = _run_graph("user_a", CLEAR_QUERY, NeverCalledProvider())
    assert state["query_expansion_metadata"] == {"expansion_invoked": False, "expansion_status": "not_needed"}
    assert state["goal"]["expanded_terms"] == []


# 6. Ambiguous query invokes expansion ------------------------------------------

def test_6_ambiguous_query_invokes_expansion_exactly_once():
    provider = RoutingProvider(expansion_response=_payload())
    state = _run_graph("user_a", UNDERSPECIFIED_QUERY, provider)
    assert provider.expansion_calls == 1
    assert state["query_expansion_metadata"]["expansion_status"] == "applied"
    assert state["goal"]["expanded_terms"] == ["budget", "quarterly"]


# 7. Original query preserved / stored separately -------------------------------

def test_7_original_query_and_deterministic_goal_fields_unchanged():
    baseline = understand_goal(UNDERSPECIFIED_QUERY).to_dict()
    state = _run_graph("user_a", UNDERSPECIFIED_QUERY, RoutingProvider(expansion_response=_payload()))
    goal = state["goal"]
    assert state["raw_query"] == UNDERSPECIFIED_QUERY
    assert goal["raw_query"] == UNDERSPECIFIED_QUERY
    assert goal["topic_tokens"] == baseline["topic_tokens"]
    assert goal["concept_text"] == baseline["concept_text"]
    assert goal["ambiguity_score"] == baseline["ambiguity_score"]
    assert goal["expanded_terms"] == ["budget", "quarterly"]  # separate key


# 8. No SQL/tool/file/user-scope fields accepted --------------------------------

def test_8_schema_fields_are_exactly_the_whitelist():
    assert set(QueryExpansion.model_fields) == {"expanded_terms", "confidence", "rationale"}


@pytest.mark.parametrize(
    "field",
    ["sql", "tool", "tool_call", "action", "file_ids", "file_id", "user_id", "user_scope", "owner_id", "score", "scores"],
)
def test_8_forbidden_extra_fields_rejected(field):
    with pytest.raises(ValidationError):
        QueryExpansion.model_validate({"confidence": 0.5, "expanded_terms": [], field: "x"})
    with pytest.raises(LLMResponseInvalidError):
        parse_query_expansion_response(_payload(**{field: "x"}))


# 9. Expansion cannot introduce candidate IDs -----------------------------------

def test_9_numeric_id_like_terms_rejected():
    for bad in ["12345", "1 2 3", "42-7"]:
        with pytest.raises(LLMResponseInvalidError):
            parse_query_expansion_response(_payload(expanded_terms=[bad]))


def test_9_candidate_pool_only_contains_real_in_scope_files():
    with get_cursor() as cur:
        cur.execute("SELECT file_id FROM files WHERE owner_id = %s", ("user_a",))
        allowed = {r["file_id"] for r in cur.fetchall()}
    result = run_query(
        "user_a", UNDERSPECIFIED_QUERY, llm_provider=RoutingProvider(expansion_response=_payload())
    )
    assert result["candidate_pool"], "precondition: expansion should have found files"
    assert set(result["candidate_pool"]) <= allowed


def test_9_access_control_holds_with_expansion_targeting_other_users_data():
    provider = RoutingProvider(
        expansion_response=_payload(expanded_terms=["confidential", "salaries", "hr"])
    )
    result = run_query("user_a", UNDERSPECIFIED_QUERY, llm_provider=provider)
    owners = {c["metadata"]["owner_id"] for c in result["candidate_pool"].values()}
    assert "user_x" not in owners
    assert owners <= {"user_a"}


# 10. Planning consumes expanded terms deterministically ------------------------

def _goal_with_expansion(query="quarterly numbers", expanded=("budget",)):
    goal = understand_goal(query).to_dict()
    goal["expanded_terms"] = list(expanded)
    return goal


def test_10_metadata_search_topic_tokens_include_expanded_terms_after_originals():
    goal = _goal_with_expansion()
    action = plan_next_action(goal, [], [], iteration=1)
    assert action["tool"] == "metadata_search"
    assert action["args"]["topic_tokens"] == goal["topic_tokens"] + ["budget"]


def test_10_bm25_semantic_hybrid_queries_append_expanded_terms():
    from app.planning import _bm25_args, _hybrid_args, _semantic_args

    goal = _goal_with_expansion()
    assert _bm25_args(goal)["query"].endswith("budget")
    assert _semantic_args(goal)["query"] == goal["raw_query"] + " budget"
    assert _hybrid_args(goal)["query"] == goal["raw_query"] + " budget"
    # planning never mutates the goal itself
    assert goal["raw_query"] == "quarterly numbers"
    assert "budget" not in goal["topic_tokens"]


def test_10_terms_already_in_query_are_not_duplicated():
    from app.planning import _semantic_args

    goal = _goal_with_expansion(query="budget review", expanded=("budget",))
    assert _semantic_args(goal)["query"] == "budget review"


def test_10_goal_without_expanded_terms_key_behaves_exactly_as_before():
    goal = understand_goal("The budget thing I was looking at.").to_dict()
    assert "expanded_terms" not in goal
    action = plan_next_action(goal, [], [], iteration=1)
    assert action["args"] == {"topic_tokens": ["budget"]}


def test_10_expansion_alone_does_not_trigger_bm25_without_deterministic_tokens():
    from app.planning import _bm25_args

    goal = understand_goal("the thing").to_dict()
    goal["expanded_terms"] = ["budget"]
    assert not goal["topic_tokens"]
    assert _bm25_args(goal) is None


def test_10_planner_decision_structure_unchanged_by_expansion():
    goal = _goal_with_expansion()
    action = plan_next_action(goal, [], [], iteration=1)
    assert action["tool"] == "metadata_search"  # expansion never selects a tool


# 11. End-to-end ----------------------------------------------------------------

def test_11_expansion_recovers_an_underspecified_query_end_to_end():
    baseline = run_query("user_a", UNDERSPECIFIED_QUERY)
    assert baseline["candidate_pool"] == {}

    result = run_query(
        "user_a", UNDERSPECIFIED_QUERY, llm_provider=RoutingProvider(expansion_response=_payload())
    )
    assert result["query_expansion_metadata"]["expansion_status"] == "applied"
    first = result["committed_actions"][0]
    assert first["tool"] == "metadata_search"
    assert first["args"]["topic_tokens"] == ["fiscal", "budget", "quarterly"]
    filenames = {c["metadata"]["filename"] for c in result["candidate_pool"].values()}
    assert "Q3_Budget_Review.xlsx" in filenames
    assert result["final_ranking"], "final ranking (Phase 4B) still runs over the expanded pool"


def test_11_expand_query_trace_event_recorded_before_first_action():
    result = run_query(
        "user_a", UNDERSPECIFIED_QUERY, llm_provider=RoutingProvider(expansion_response=_payload())
    )
    events = [t["event"] for t in result["trace"]]
    assert events.index("understand_goal") < events.index("expand_query") < events.index("plan_next_action")
