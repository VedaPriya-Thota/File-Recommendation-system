"""
Phase 5A tests -- LLM provider abstraction (app/llm/provider.py),
structured goal-interpretation schema (app/llm/schema.py), the
deterministic-first augmentation step (app/llm/goal_interpreter.py), and
its wiring into app/graph/build.py's node_understand_goal.

Structure:
  1. Provider abstraction
  2. Structured output validation
  3. Ambiguous query -> LLM path
  4. Simple/unambiguous query -> deterministic path, no LLM call
  5. Invalid LLM output -> safe fallback
  6. Provider failure -> safe fallback
  7. No SQL/tool execution generated directly by the LLM
  8. user_scope cannot be changed by LLM output
"""
import json

import pytest
from pydantic import ValidationError

from app.goal import understand_goal
from app.llm.goal_interpreter import _merge_llm_interpretation, augment_goal_with_llm
from app.llm.provider import (
    AnthropicProvider,
    LLMProvider,
    LLMUnavailableError,
    NullProvider,
    get_default_provider,
)
from app.llm.schema import (
    LLMGoalInterpretation,
    LLMResponseInvalidError,
    parse_llm_goal_response,
)
from app.run import run_query

AMBIGUOUS_QUERY = "The budget thing I was looking at."
CLEAR_QUERY = "Find the slide deck about AI architecture I edited last week."


class FakeProvider:
    """Test double satisfying the LLMProvider protocol. Records every call
    so tests can assert whether/how it was invoked."""

    name = "fake"

    def __init__(self, response=None, raises=None):
        self.response = response
        self.raises = raises
        self.calls = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self.raises is not None:
            raise self.raises
        return self.response


class NeverCalledProvider:
    """A provider that fails the test immediately if the graph ever calls
    it -- used to prove the deterministic-only path truly never invokes
    the LLM."""

    name = "never_called"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise AssertionError("LLM provider should not have been invoked for this query")


def _valid_payload(**overrides) -> str:
    payload = {
        "file_type": "xlsx",
        "topic_tokens": ["forecast"],
        "filename_candidate": None,
        "retrieval_preference": None,
        "confidence": 0.8,
        "rationale": "likely the quarterly budget spreadsheet",
    }
    payload.update(overrides)
    return json.dumps(payload)


# =============================================================================
# 1. Provider abstraction
# =============================================================================

def test_1_null_provider_raises_without_any_io():
    provider = NullProvider()
    assert provider.name == "none"
    with pytest.raises(LLMUnavailableError):
        provider.complete("system", "user")


def test_1_get_default_provider_is_null_when_unset(monkeypatch):
    monkeypatch.delenv("DISCOVERY_LLM_PROVIDER", raising=False)
    provider = get_default_provider()
    assert isinstance(provider, NullProvider)


def test_1_get_default_provider_falls_back_to_null_for_unknown_name(monkeypatch):
    monkeypatch.setenv("DISCOVERY_LLM_PROVIDER", "some_unsupported_vendor")
    provider = get_default_provider()
    assert isinstance(provider, NullProvider)


def test_1_get_default_provider_falls_back_to_null_when_anthropic_key_missing(monkeypatch):
    monkeypatch.setenv("DISCOVERY_LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = get_default_provider()
    assert isinstance(provider, NullProvider)


def test_1_anthropic_provider_requires_api_key():
    with pytest.raises(LLMUnavailableError):
        AnthropicProvider(model="claude-haiku-4-5-20251001", api_key=None)


def test_1_fake_provider_satisfies_the_llm_provider_protocol():
    assert isinstance(FakeProvider(response="{}"), LLMProvider)


# =============================================================================
# 2. Structured output validation
# =============================================================================

def test_2_valid_payload_parses_to_the_schema():
    parsed = parse_llm_goal_response(_valid_payload())
    assert isinstance(parsed, LLMGoalInterpretation)
    assert parsed.file_type == "xlsx"
    assert parsed.topic_tokens == ["forecast"]
    assert parsed.confidence == 0.8


def test_2_fenced_json_block_is_tolerated():
    fenced = "```json\n" + _valid_payload() + "\n```"
    parsed = parse_llm_goal_response(fenced)
    assert parsed.file_type == "xlsx"


def test_2_malformed_json_is_rejected():
    with pytest.raises(LLMResponseInvalidError):
        parse_llm_goal_response("not json at all {{{")


def test_2_non_object_json_is_rejected():
    with pytest.raises(LLMResponseInvalidError):
        parse_llm_goal_response(json.dumps(["xlsx", "budget"]))


def test_2_unknown_file_type_is_rejected():
    with pytest.raises(LLMResponseInvalidError):
        parse_llm_goal_response(_valid_payload(file_type="exe"))


def test_2_confidence_out_of_range_is_rejected():
    with pytest.raises(LLMResponseInvalidError):
        parse_llm_goal_response(_valid_payload(confidence=1.5))


def test_2_missing_required_confidence_is_rejected():
    payload = json.loads(_valid_payload())
    del payload["confidence"]
    with pytest.raises(LLMResponseInvalidError):
        parse_llm_goal_response(json.dumps(payload))


def test_2_too_many_topic_tokens_is_rejected():
    payload = _valid_payload(topic_tokens=[f"tok{i}" for i in range(20)])
    with pytest.raises(LLMResponseInvalidError):
        parse_llm_goal_response(payload)


def test_2_topic_token_with_disallowed_characters_is_rejected():
    with pytest.raises(LLMResponseInvalidError):
        parse_llm_goal_response(_valid_payload(topic_tokens=["budget; DROP TABLE files;"]))


def test_2_unknown_retrieval_preference_is_rejected():
    with pytest.raises(LLMResponseInvalidError):
        parse_llm_goal_response(_valid_payload(retrieval_preference="direct_sql"))


# =============================================================================
# 3. Ambiguous query -> LLM path
# =============================================================================

def test_3_augment_goal_applies_a_valid_llm_suggestion_for_an_ambiguous_goal():
    goal = understand_goal(AMBIGUOUS_QUERY).to_dict()
    assert goal["ambiguity_score"] >= 0.65  # precondition: genuinely ambiguous
    assert goal["filters"]["file_type"] is None

    provider = FakeProvider(response=_valid_payload())
    merged, meta = augment_goal_with_llm(AMBIGUOUS_QUERY, goal, provider)

    assert provider.calls, "LLM should have been invoked for an ambiguous query"
    assert meta["llm_invoked"] is True
    assert meta["llm_status"] == "applied"
    assert merged["filters"]["file_type"] == "xlsx"
    assert "forecast" in merged["topic_tokens"]
    assert "budget" in merged["topic_tokens"], "deterministic tokens must be preserved, not replaced"


def test_3_end_to_end_llm_suggestion_narrows_the_first_planned_action():
    result = run_query("user_a", AMBIGUOUS_QUERY, llm_provider=FakeProvider(response=_valid_payload()))

    assert result["llm_goal_metadata"]["llm_invoked"] is True
    assert result["llm_goal_metadata"]["llm_status"] == "applied"
    assert result["committed_actions"][0]["args"]["file_type"] == "xlsx"
    # A real, meaningful effect: the LLM hint should let the run resolve to
    # a single dominant candidate in one iteration, instead of the
    # multi-candidate ambiguity the same query produces without it.
    assert result["status"] == "finish"
    assert result["iteration_count"] == 1
    filenames = [c["metadata"]["filename"] for c in result["candidate_pool"].values()]
    assert filenames == ["Q3_Budget_Review.xlsx"]


# =============================================================================
# 4. Simple/unambiguous query -> deterministic path, no LLM call
# =============================================================================

def test_4_clear_query_never_invokes_the_provider():
    goal = understand_goal(CLEAR_QUERY).to_dict()
    assert goal["ambiguity_score"] < 0.65  # precondition: not ambiguous

    provider = FakeProvider(response=_valid_payload())
    # augment_goal_with_llm is never even called by the graph for this
    # goal -- but if it WERE called, it would still just apply cleanly;
    # the real guarantee under test is the graph-level gate, exercised
    # end-to-end below.
    result = run_query("user_a", CLEAR_QUERY, llm_provider=NeverCalledProvider())

    assert result["llm_goal_metadata"] == {"llm_invoked": False, "llm_status": "not_needed"}
    assert result["status"] == "finish"


def test_4_run_query_default_provider_needs_no_network_for_clear_query():
    # No llm_provider passed at all -> get_default_provider() -> NullProvider.
    # This must succeed without any network access or API key.
    result = run_query("user_a", CLEAR_QUERY)
    assert result["status"] == "finish"
    assert result["llm_goal_metadata"]["llm_invoked"] is False


# =============================================================================
# 5. Invalid LLM output -> safe fallback
# =============================================================================

def test_5_invalid_json_falls_back_to_the_deterministic_goal_unchanged():
    goal = understand_goal(AMBIGUOUS_QUERY).to_dict()
    provider = FakeProvider(response="this is not valid json")

    merged, meta = augment_goal_with_llm(AMBIGUOUS_QUERY, goal, provider)

    assert merged == goal
    assert meta["llm_status"] == "invalid_output"


def test_5_schema_violation_falls_back_to_the_deterministic_goal_unchanged():
    goal = understand_goal(AMBIGUOUS_QUERY).to_dict()
    provider = FakeProvider(response=_valid_payload(file_type="not_a_real_type"))

    merged, meta = augment_goal_with_llm(AMBIGUOUS_QUERY, goal, provider)

    assert merged == goal
    assert meta["llm_status"] == "invalid_output"


def test_5_extra_unexpected_field_falls_back_safely():
    goal = understand_goal(AMBIGUOUS_QUERY).to_dict()
    provider = FakeProvider(response=_valid_payload(sql="DROP TABLE files;"))

    merged, meta = augment_goal_with_llm(AMBIGUOUS_QUERY, goal, provider)

    assert merged == goal
    assert meta["llm_status"] == "invalid_output"


def test_5_end_to_end_invalid_output_still_produces_a_normal_result():
    result = run_query("user_a", AMBIGUOUS_QUERY, llm_provider=FakeProvider(response="{garbage"))
    assert result["llm_goal_metadata"]["llm_status"] == "invalid_output"
    # Falls back to exactly the same deterministic behavior as no-LLM-at-all:
    assert result["status"] in ("clarify", "finish")
    assert len(result["candidate_pool"]) > 1


# =============================================================================
# 6. Provider failure -> safe fallback
# =============================================================================

def test_6_llm_unavailable_error_falls_back_safely():
    goal = understand_goal(AMBIGUOUS_QUERY).to_dict()
    provider = FakeProvider(raises=LLMUnavailableError("simulated: no credentials"))

    merged, meta = augment_goal_with_llm(AMBIGUOUS_QUERY, goal, provider)

    assert merged == goal
    assert meta["llm_status"] == "unavailable"


def test_6_unexpected_provider_exception_falls_back_safely():
    goal = understand_goal(AMBIGUOUS_QUERY).to_dict()
    provider = FakeProvider(raises=TimeoutError("simulated network timeout"))

    merged, meta = augment_goal_with_llm(AMBIGUOUS_QUERY, goal, provider)

    assert merged == goal
    assert meta["llm_status"] == "provider_error"


def test_6_default_null_provider_end_to_end_matches_pre_llm_baseline():
    """With no provider configured at all (the real-world default), an
    ambiguous query must behave EXACTLY as it did before Phase 5A existed
    -- LLM unavailability must be fully transparent to the rest of the
    loop."""
    result = run_query("user_a", AMBIGUOUS_QUERY, max_iterations=6)
    assert result["llm_goal_metadata"]["llm_status"] == "unavailable"
    assert result["status"] in ("clarify", "finish")
    if result["status"] == "finish":
        assert result["degraded_mode"] is True
    else:
        assert result["clarification_question"]
    assert len(result["candidate_pool"]) > 1


# =============================================================================
# 7. No SQL / tool execution generated directly by the LLM
# =============================================================================

def test_7_schema_has_no_field_capable_of_expressing_code_sql_or_a_tool_call():
    field_names = set(LLMGoalInterpretation.model_fields.keys())
    forbidden = {"sql", "query", "code", "tool", "tool_call", "tool_calls", "action", "command", "exec"}
    assert field_names.isdisjoint(forbidden)
    assert field_names == {
        "file_type",
        "topic_tokens",
        "filename_candidate",
        "retrieval_preference",
        "confidence",
        "rationale",
    }


def test_7_extra_field_is_rejected_outright_not_ignored():
    with pytest.raises(ValidationError):
        LLMGoalInterpretation.model_validate(
            {
                "confidence": 0.5,
                "sql": "SELECT * FROM files",
            }
        )


def test_7_sql_like_content_in_a_valid_field_cannot_reach_the_merged_goal():
    """Even attempting to smuggle SQL-shaped text through an allowed field
    (topic_tokens) fails validation -- it never reaches the merge step, so
    it can never reach planning.py's args-building, which is the only
    thing that ever constructs a SQL query (parameterized, in
    app/tools/metadata_search.py)."""
    goal = understand_goal(AMBIGUOUS_QUERY).to_dict()
    malicious = _valid_payload(topic_tokens=["'; DROP TABLE files; --"])
    provider = FakeProvider(response=malicious)

    merged, meta = augment_goal_with_llm(AMBIGUOUS_QUERY, goal, provider)

    assert meta["llm_status"] == "invalid_output"
    assert merged == goal


def test_7_merge_function_only_ever_touches_the_three_documented_goal_keys():
    goal = understand_goal(AMBIGUOUS_QUERY).to_dict()
    parsed = parse_llm_goal_response(_valid_payload())
    merged = _merge_llm_interpretation(goal, parsed)

    changed_keys = {k for k in merged if merged.get(k) != goal.get(k)}
    assert changed_keys <= {"filters", "topic_tokens", "concept_text", "llm_retrieval_preference"}
    # intent, raw_query, ambiguity_score are never touched by the merge.
    assert merged["intent"] == goal["intent"]
    assert merged["raw_query"] == goal["raw_query"]
    assert merged["ambiguity_score"] == goal["ambiguity_score"]


# =============================================================================
# 8. user_scope cannot be changed by LLM output
# =============================================================================

def test_8_schema_has_no_user_scope_or_owner_field():
    field_names = set(LLMGoalInterpretation.model_fields.keys())
    assert field_names.isdisjoint({"user_scope", "owner_id", "owner_ids", "user_id", "scope"})


def test_8_merge_never_introduces_a_scope_key():
    goal = understand_goal(AMBIGUOUS_QUERY).to_dict()
    parsed = parse_llm_goal_response(_valid_payload())
    merged = _merge_llm_interpretation(goal, parsed)
    assert "user_scope" not in merged
    assert "owner_id" not in merged
    assert "owner_ids" not in merged


def test_8_access_control_holds_even_when_llm_is_invoked_and_applied():
    """Reruns the architecture's core access-control guarantee (see
    test_agent_loop.py Test D) with an LLM in the loop -- user_a must
    never see user_x's same-topic confidential file, regardless of what
    the (fake, validated) LLM suggested."""
    provider = FakeProvider(
        response=_valid_payload(file_type=None, topic_tokens=["confidential", "roadmap"])
    )
    result = run_query(
        "user_a", "confidential secret roadmap salaries HR AI architecture", llm_provider=provider
    )
    assert result["llm_goal_metadata"]["llm_invoked"] is True
    owners = {c["metadata"]["owner_id"] for c in result["candidate_pool"].values()}
    assert "user_x" not in owners
    assert owners <= {"user_a"}


def test_8_scope_resolution_runs_before_and_independently_of_goal_understanding():
    """resolve_scope is the first graph node and does not depend on `goal`
    at all (architecture_final.md §11) -- confirm the same user_scope
    comes back whether or not the LLM slice fires."""
    no_llm_result = run_query("user_a", CLEAR_QUERY, llm_provider=NeverCalledProvider())
    llm_result = run_query("user_a", AMBIGUOUS_QUERY, llm_provider=FakeProvider(response=_valid_payload()))

    no_llm_owners = {c["metadata"]["owner_id"] for c in no_llm_result["candidate_pool"].values()}
    llm_owners = {c["metadata"]["owner_id"] for c in llm_result["candidate_pool"].values()}
    assert no_llm_owners <= {"user_a"}
    assert llm_owners <= {"user_a"}
