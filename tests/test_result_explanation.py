"""
Phase 5D tests -- LLM result explanation (app/llm/result_explainer.py,
ResultExplanation in app/llm/schema.py, node_explain_result in
app/graph/build.py).

Structure (numbers follow the Phase 5D spec's test list):
  1. Valid structured explanation
  2. Schema rejects extra/control fields
  3. Explanation grounded in supplied evidence
  4. Bounded evidence passed to provider
  5. No candidates -> no LLM explanation
  6. Successful result -> explanation call
  7. NullProvider -> deterministic result unchanged
  8. Provider failure -> deterministic result unchanged
  9. Malformed output -> deterministic result unchanged
 10. Explanation cannot introduce a candidate
 11. Explanation cannot modify candidate IDs
 12. Explanation cannot modify scores
 13. Explanation cannot modify user scope
 14. Phase 5C-applied result explained without another comparison
 15. Phase 5C failure / no-op handled safely
 16. Ordering unchanged
 17. End-to-end explanation path
 (18. full-suite is run separately)
"""
import json

import pytest
from pydantic import ValidationError

from app.db.connection import get_cursor
from app.final_ranking import final_rank
from app.graph.build import build_graph
from app.llm.provider import LLMUnavailableError, NullProvider
from app.llm.result_explainer import (
    SYSTEM_PROMPT,
    build_explanation_evidence,
    build_user_prompt,
    explain_result_with_llm,
    should_explain,
)
from app.llm.schema import (
    MAX_EVIDENCE_POINTS,
    LLMResponseInvalidError,
    ResultExplanation,
    parse_result_explanation_response,
)
from app.run import _initial_state
from app.tools.registry import ToolResult

QUERY = "quarterly budget planning"


# --- helpers -----------------------------------------------------------------

def _entry(file_id=1, scores=None, **metadata):
    md = {"filename": "budget_2024.docx", "file_type": "docx", "tags": ["budget", "finance"],
          "summary": "Quarterly budget summary.", "owner_id": "user_a",
          "file_path": "/secret/store/budget_2024.docx"}
    md.update(metadata)
    scores = scores or {"hybrid_search": 0.9}
    return {"file_id": file_id, "evidence_sources": list(scores), "scores": scores, "metadata": md,
            "retrieval_score": 0.9, "personalization_score": 0.0, "final_score": 0.765}


def _reply(summary="Matches your budget query via its title and tags.", points=None, confidence=0.8):
    return json.dumps({"summary": summary,
                       "evidence_points": points if points is not None else ["Tagged 'budget'."],
                       "confidence": confidence})


class ExplainProvider:
    """Answers the explanation prompt with `response` (or raises). The 5C
    comparison prompt is answered with `compare_response` if given (else
    fails), and 5A/5B prompts always fail -- isolating 5D. Records calls."""

    name = "fake"

    def __init__(self, response=None, raises=None, compare_response=None):
        self.response, self.raises, self.compare_response = response, raises, compare_response
        self.explain_calls, self.compare_calls, self.other_calls = [], [], 0

    def complete(self, system_prompt, user_prompt):
        if "user-facing explanation" in system_prompt:
            self.explain_calls.append((system_prompt, user_prompt))
            if self.raises is not None:
                raise self.raises
            return self.response
        if "already-retrieved candidate files" in system_prompt:
            self.compare_calls.append((system_prompt, user_prompt))
            if self.compare_response is None:
                raise RuntimeError("comparison intentionally unavailable")
            return self.compare_response
        self.other_calls += 1
        raise RuntimeError("other LLM slices intentionally unavailable in this test")


def _ids(owner, n):
    with get_cursor() as cur:
        cur.execute("SELECT file_id FROM files WHERE owner_id = %s ORDER BY file_id LIMIT %s", (owner, n))
        return [r["file_id"] for r in cur.fetchall()]


class TiedRegistry:
    def __init__(self, file_ids, scores):
        self.file_ids, self.scores = file_ids, scores

    def execute(self, tool, args, user_scope):
        return ToolResult(tool=tool, candidates=[
            {"file_id": fid, "score": sc, "source": tool,
             "metadata": {"filename": f"f{fid}.docx", "file_type": "docx", "tags": ["budget"],
                          "summary": f"summary {fid}", "owner_id": "user_a",
                          "file_path": f"/private/f{fid}.docx"}}
            for fid, sc in zip(self.file_ids, self.scores)])


def _run(provider, file_ids=None, scores=(0.90, 0.85, 0.80)):
    file_ids = file_ids or _ids("user_a", 3)
    graph = build_graph(registry=TiedRegistry(file_ids, scores), max_iterations=1, llm_provider=provider)
    return graph.invoke(_initial_state("user_a", QUERY, 1), config={"recursion_limit": 100}), file_ids


def _compare_reply(file_id):
    return json.dumps({"selected_file_id": file_id, "confidence": 0.9, "rationale": "closer topical match"})


def _deterministic_view(result):
    return (result["final_ranking"], result["candidate_pool"], result["user_scope"])


# =============================================================================
# 1. Valid structured explanation
# =============================================================================

def test_1_valid_payload_parses():
    parsed = parse_result_explanation_response(_reply("S.", ["a", "b"], 0.7))
    assert isinstance(parsed, ResultExplanation)
    assert (parsed.summary, parsed.evidence_points, parsed.confidence) == ("S.", ["a", "b"], 0.7)


def test_1_fenced_json_tolerated_and_points_optional():
    parsed = parse_result_explanation_response('```json\n{"summary": "S", "confidence": 0.5}\n```')
    assert parsed.evidence_points == []


def test_1_explainer_returns_explanation_bound_to_file_id():
    exp, meta = explain_result_with_llm(QUERY, {}, _entry(file_id=42), None, ExplainProvider(_reply()))
    assert exp is not None
    assert meta["explanation_status"] == "generated"
    assert meta["explained_file_id"] == 42
    assert meta["explanation_provider"] == "fake"
    assert meta["explanation_summary"] == exp.summary
    assert meta["explanation_evidence_points"] == exp.evidence_points
    assert meta["explanation_confidence"] == 0.8


# =============================================================================
# 2. Schema rejects extra / control fields; bounds enforced
# =============================================================================

def test_2_schema_fields_are_exactly_the_whitelist():
    assert set(ResultExplanation.model_fields) == {"summary", "evidence_points", "confidence"}


@pytest.mark.parametrize("field", ["file_id", "selected_file_id", "ranking", "score", "final_score", "sql",
                                   "tool_call", "user_scope", "owner_id", "reasoning", "chain_of_thought"])
def test_2_extra_and_control_fields_rejected(field):
    with pytest.raises(ValidationError):
        ResultExplanation.model_validate({"summary": "S", "confidence": 0.5, field: "x"})


@pytest.mark.parametrize("payload", [
    {"summary": "", "confidence": 0.5},
    {"summary": "   ", "confidence": 0.5},
    {"summary": "x" * 401, "confidence": 0.5},
    {"summary": "S", "confidence": 1.1},
    {"summary": "S", "confidence": 0.5, "evidence_points": ["p"] * (MAX_EVIDENCE_POINTS + 1)},
    {"summary": "S", "confidence": 0.5, "evidence_points": ["x" * 201]},
    {"summary": "S", "confidence": 0.5, "evidence_points": [1]},
    {"confidence": 0.5},
])
def test_2_bounds_and_types_enforced(payload):
    with pytest.raises(LLMResponseInvalidError):
        parse_result_explanation_response(json.dumps(payload))


# =============================================================================
# 3. Grounded in supplied evidence
# =============================================================================

def test_3_evidence_contains_only_real_signals():
    entry = _entry(scores={"bm25_search": 0.8, "rerank_candidates": 0.95}, snippet="bm25 words",
                   matched_chunk="semantic words")
    ev = build_explanation_evidence(QUERY, {"filters": {"file_type": "docx"}, "topic_tokens": ["budget"]}, entry)
    r = ev["result"]
    assert r["retrieval_sources"] == ["bm25_search", "rerank_candidates"]  # exactly what ran
    assert r["reranked"] is True and r["reranker_score"] == 0.95
    assert r["bm25_snippet"] == "bm25 words" and r["semantic_matched_chunk"] == "semantic words"
    assert ev["goal"]["file_type"] == "docx" and ev["goal"]["topic_tokens"] == ["budget"]
    assert ev["selection_basis"] == "deterministic_ranking"


def test_3_unused_methods_are_not_claimed_in_evidence():
    ev = build_explanation_evidence(QUERY, {}, _entry(scores={"metadata_search": 0.9}))
    r = ev["result"]
    assert r["retrieval_sources"] == ["metadata_search"]
    assert r["reranked"] is False and r["reranker_score"] is None
    assert r["bm25_snippet"] is None and r["semantic_matched_chunk"] is None
    assert "personalization" not in ev and "comparison" not in ev


def test_3_personalization_only_when_meaningful():
    good = _entry(scores={"hybrid_search": 0.9, "get_user_activity": 0.6})
    good["metadata"]["personalization"] = {"affinity_score": 0.6, "cold_start": False,
                                           "edit_count": 3, "preferred_file_type_match": True, "secret": "x"}
    ev = build_explanation_evidence(QUERY, {}, good)
    assert ev["personalization"] == {"affinity_score": 0.6, "preferred_file_type_match": True, "edit_count": 3}
    assert "get_user_activity" not in ev["result"]["retrieval_sources"]

    cold = _entry()
    cold["metadata"]["personalization"] = {"affinity_score": 0.6, "cold_start": True}
    zero = _entry()
    zero["metadata"]["personalization"] = {"affinity_score": 0.0, "cold_start": False}
    assert "personalization" not in build_explanation_evidence(QUERY, {}, cold)
    assert "personalization" not in build_explanation_evidence(QUERY, {}, zero)


def test_3_prompt_instructs_grounding_and_insufficient_evidence():
    for phrase in ("ONLY the evidence", "do not invent", "insufficient", "not your reasoning steps"):
        assert phrase.lower() in SYSTEM_PROMPT.lower()


@pytest.mark.parametrize("leak", ["/secret/store/budget_2024.docx", "USER_A owns it", "user_a"])
def test_3_explanation_quoting_withheld_values_is_rejected(leak):
    provider = ExplainProvider(_reply(summary=f"Found in {leak}."))
    exp, meta = explain_result_with_llm(QUERY, {}, _entry(), None, provider, user_id="user_a", user_scope=["user_a"])
    assert exp is None and meta["explanation_status"] == "invalid_output"


# =============================================================================
# 4. Bounded evidence passed to provider
# =============================================================================

def test_4_prompt_excludes_sensitive_and_internal_fields():
    entry = _entry(file_id=987654)
    prompt = build_user_prompt(build_explanation_evidence(QUERY, {}, entry))
    for forbidden in ("owner_id", "user_a", "/secret/store", "file_path", "user_scope", "987654", "file_id",
                      "created_at", "updated_at"):
        assert forbidden not in prompt


def test_4_long_text_and_tags_bounded():
    entry = _entry(summary="s" * 5000, snippet="n" * 5000, matched_chunk="c" * 5000,
                   tags=[f"t{i}" for i in range(50)])
    ev = build_explanation_evidence("q" * 5000, {}, entry)
    r = ev["result"]
    assert len(r["summary"]) <= 400 and len(r["bm25_snippet"]) <= 400 and len(r["semantic_matched_chunk"]) <= 400
    assert len(r["tags"]) <= 10 and len(ev["user_query"]) <= 500


def test_4_only_top_result_is_sent_no_other_candidates():
    provider = ExplainProvider(_reply())
    _, ids = _run(provider)
    assert len(provider.explain_calls) == 1
    payload = json.loads(provider.explain_calls[0][1].split("\n", 1)[1])
    assert set(payload) == {"user_query", "goal", "result", "selection_basis"}
    assert payload["result"]["filename"] == f"f{ids[0]}.docx"
    for other in ids[1:]:
        assert f"f{other}.docx" not in provider.explain_calls[0][1]


# =============================================================================
# 5. No candidates -> no LLM explanation
# =============================================================================

def test_5_should_explain_gate():
    assert should_explain([]) is False
    assert should_explain([_entry()]) is True
    assert should_explain([_entry()], {"explanation_status": "generated"}) is False
    assert should_explain([_entry()], {"explanation_status": "unavailable"}) is True


def test_5_no_result_run_never_calls_llm():
    class Empty:
        def execute(self, tool, args, user_scope):
            return ToolResult(tool=tool, candidates=[])

    provider = ExplainProvider(_reply())
    graph = build_graph(registry=Empty(), max_iterations=2, llm_provider=provider)
    result = graph.invoke(_initial_state("user_a", QUERY, 2), config={"recursion_limit": 100})
    assert result["final_ranking"] == []
    assert provider.explain_calls == []
    assert result["candidate_explanation"] == {"explanation_invoked": False, "explanation_status": "not_needed"}
    assert result["explanations"] == {}


def test_5_clarification_run_never_calls_llm():
    provider = ExplainProvider(_reply())
    graph = build_graph(llm_provider=provider)
    result = graph.invoke(_initial_state("user_a", "budget", 6), config={"recursion_limit": 100})
    assert result["stopping_decision"]["status"] == "clarify"
    assert provider.explain_calls == []
    assert not result.get("candidate_explanation")
    assert not result.get("explanations")


# =============================================================================
# 6. Successful result -> explanation call
# =============================================================================

def test_6_successful_result_calls_llm_exactly_once_and_stores_explanation():
    provider = ExplainProvider(_reply("Because budget.", ["tag match"], 0.9))
    result, ids = _run(provider, scores=(1.0, 0.2, 0.1))  # dominant: no 5C
    assert len(provider.explain_calls) == 1
    meta = result["candidate_explanation"]
    assert meta["explanation_status"] == "generated" and meta["explained_file_id"] == ids[0]
    assert meta["explanation_summary"] == "Because budget."
    assert result["explanations"] == {ids[0]: "Because budget."}


def test_6_single_candidate_is_explained():
    provider = ExplainProvider(_reply())
    result, ids = _run(provider, file_ids=_ids("user_a", 1), scores=(0.9,))
    assert len(provider.explain_calls) == 1
    assert result["candidate_explanation"]["explained_file_id"] == ids[0]


# =============================================================================
# 7. NullProvider -> deterministic result unchanged
# =============================================================================

def test_7_null_provider_result_unchanged_and_explanation_unavailable():
    result, ids = _run(NullProvider())
    assert result["candidate_explanation"]["explanation_status"] == "unavailable"
    assert result["explanations"] == {}
    assert result["final_ranking"] == final_rank(result["candidate_pool"])
    assert result["stopping_decision"]["status"] == "finish"


# =============================================================================
# 8. Provider failure -> deterministic result unchanged
# =============================================================================

@pytest.mark.parametrize("provider,status", [
    (ExplainProvider(raises=LLMUnavailableError("down")), "unavailable"),
    (ExplainProvider(raises=RuntimeError("boom")), "provider_error"),
    (ExplainProvider(response=None), "invalid_output"),
])
def test_8_provider_failure_keeps_result(provider, status):
    baseline, _ = _run(NullProvider())
    result, _ = _run(provider)
    assert _deterministic_view(result) == _deterministic_view(baseline)
    assert result["candidate_explanation"]["explanation_status"] == status
    assert result["explanations"] == {}


# =============================================================================
# 9. Malformed output -> deterministic result unchanged
# =============================================================================

@pytest.mark.parametrize("raw", [
    "not json",
    "[1, 2]",
    '{"summary": "S"}',
    '{"summary": "S", "confidence": 2}',
    '{"summary": "S", "confidence": 0.5, "file_id": 3}',
    '{"summary": "S", "confidence": 0.5, "sql": "DROP TABLE files"}',
    'Here you go: {"summary": "S", "confidence": 0.5}',
])
def test_9_malformed_output_keeps_result(raw):
    baseline, _ = _run(NullProvider())
    result, _ = _run(ExplainProvider(raw))
    assert _deterministic_view(result) == _deterministic_view(baseline)
    assert result["candidate_explanation"]["explanation_status"] == "invalid_output"
    assert result["explanations"] == {}


# =============================================================================
# 10-13. Explanation cannot change candidates / IDs / scores / scope
# =============================================================================

def test_10_11_cannot_introduce_or_change_candidate_even_when_naming_one():
    other = _ids("user_x", 1)[0]
    payload = json.dumps({"summary": f"Pick file {other} instead.", "confidence": 0.9, "file_id": other})
    baseline, ids = _run(NullProvider())
    result, _ = _run(ExplainProvider(payload))
    assert result["candidate_explanation"]["explanation_status"] == "invalid_output"  # file_id key forbidden
    assert set(result["candidate_pool"]) == set(ids) == set(baseline["candidate_pool"])
    assert [c["file_id"] for c in result["final_ranking"]] == [c["file_id"] for c in baseline["final_ranking"]]
    assert other not in result["candidate_pool"]


def test_10_11_valid_explanation_leaves_ids_and_membership_identical():
    baseline, ids = _run(NullProvider())
    result, _ = _run(ExplainProvider(_reply("Mentions file 999999 as well.")))
    assert result["candidate_explanation"]["explanation_status"] == "generated"
    assert [c["file_id"] for c in result["final_ranking"]] == [c["file_id"] for c in baseline["final_ranking"]]
    assert set(result["candidate_pool"]) == set(baseline["candidate_pool"])
    assert result["candidate_explanation"]["explained_file_id"] == baseline["final_ranking"][0]["file_id"]


def test_12_scores_identical_with_explanation():
    baseline, _ = _run(NullProvider())
    result, _ = _run(ExplainProvider(_reply(confidence=1.0)))
    assert result["candidate_explanation"]["explanation_status"] == "generated"
    assert result["final_ranking"] == baseline["final_ranking"]  # entries incl. every score, exactly
    assert result["candidate_pool"] == baseline["candidate_pool"]


def test_13_user_scope_identical_with_explanation():
    baseline, _ = _run(NullProvider())
    result, _ = _run(ExplainProvider(_reply()))
    assert result["user_scope"] == baseline["user_scope"] == ["user_a"]


# =============================================================================
# 14. Phase 5C applied -> explained without another comparison
# =============================================================================

def test_14_applied_comparison_is_explained_with_rationale_and_no_second_comparison():
    ids = _ids("user_a", 3)
    provider = ExplainProvider(_reply("Chosen after comparing close matches."), compare_response=_compare_reply(ids[1]))
    result, _ = _run(provider, file_ids=ids)

    assert result["candidate_comparison"]["comparison_applied"] is True
    assert result["final_ranking"][0]["file_id"] == ids[1]
    assert len(provider.compare_calls) == 1               # only the 5C call, none added by 5D
    assert len(provider.explain_calls) == 1

    assert result["candidate_explanation"]["explained_file_id"] == ids[1]  # the post-5C top result
    payload = json.loads(provider.explain_calls[0][1].split("\n", 1)[1])
    assert payload["selection_basis"] == "llm_comparison_of_near_tied_candidates"
    assert payload["comparison"] == {"rationale": "closer topical match"}
    assert payload["result"]["filename"] == f"f{ids[1]}.docx"
    assert "selected_file_id" not in provider.explain_calls[0][1]


def test_14_explanation_does_not_undo_or_alter_comparison_order():
    ids = _ids("user_a", 3)
    without, _ = _run(ExplainProvider(response=None, compare_response=_compare_reply(ids[1])), file_ids=ids)
    with_exp, _ = _run(ExplainProvider(_reply(), compare_response=_compare_reply(ids[1])), file_ids=ids)
    assert with_exp["final_ranking"] == without["final_ranking"]
    assert [c["file_id"] for c in with_exp["final_ranking"]][0] == ids[1]


# =============================================================================
# 15. Phase 5C failure / no-op handled safely
# =============================================================================

def test_15_comparison_failed_explains_deterministic_result_normally():
    ids = _ids("user_a", 3)
    provider = ExplainProvider(_reply(), compare_response="garbage")  # 5C invalid_output
    result, _ = _run(provider, file_ids=ids)
    assert result["candidate_comparison"]["comparison_status"] == "invalid_output"
    assert result["final_ranking"][0]["file_id"] == ids[0]
    payload = json.loads(provider.explain_calls[0][1].split("\n", 1)[1])
    assert payload["selection_basis"] == "deterministic_ranking"
    assert "comparison" not in payload
    assert result["candidate_explanation"]["explained_file_id"] == ids[0]


def test_15_comparison_noop_confirming_top_is_not_treated_as_applied():
    ids = _ids("user_a", 3)
    provider = ExplainProvider(_reply(), compare_response=_compare_reply(ids[0]))
    result, _ = _run(provider, file_ids=ids)
    assert result["candidate_comparison"]["comparison_applied"] is False
    payload = json.loads(provider.explain_calls[0][1].split("\n", 1)[1])
    assert payload["selection_basis"] == "deterministic_ranking" and "comparison" not in payload


def test_15_comparison_not_invoked_still_explains():
    provider = ExplainProvider(_reply())
    result, ids = _run(provider, scores=(1.0, 0.2, 0.1))
    assert result["candidate_comparison"]["comparison_invoked"] is False
    assert result["candidate_explanation"]["explanation_status"] == "generated"


# =============================================================================
# 16. Ordering unchanged
# =============================================================================

def test_16_ordering_unchanged_across_all_explanation_outcomes():
    baseline, _ = _run(NullProvider())
    order = [c["file_id"] for c in baseline["final_ranking"]]
    for provider in (ExplainProvider(_reply()), ExplainProvider(raises=RuntimeError("x")),
                     ExplainProvider("bad"), NullProvider()):
        result, _ = _run(provider)
        assert [c["file_id"] for c in result["final_ranking"]] == order


def test_16_explain_node_writes_only_explanation_keys():
    result, _ = _run(ExplainProvider(_reply()))
    trace = [t for t in result["trace"] if t["event"] == "explain_result"]
    assert len(trace) == 1 and trace[0]["explanation"]["explanation_status"] == "generated"
    assert "explanation_evidence_points" not in trace[0]["explanation"]


# =============================================================================
# 17. End-to-end
# =============================================================================

def test_17_end_to_end_run_query_exposes_explanation(monkeypatch):
    from app import run as run_module

    ids = _ids("user_a", 3)
    real_build = run_module.build_graph
    monkeypatch.setattr(
        run_module, "build_graph",
        lambda registry=None, max_iterations=None, llm_provider=None: real_build(
            registry=TiedRegistry(ids, (0.9, 0.85, 0.8)), max_iterations=1, llm_provider=llm_provider))
    provider = ExplainProvider(_reply("It matches your budget request.", ["Tagged 'budget'.", "Docx file."], 0.85),
                               compare_response=_compare_reply(ids[2]))
    out = run_module.run_query("user_a", QUERY, max_iterations=1, llm_provider=provider)

    assert out["final_ranking"][0]["file_id"] == ids[2]
    exp = out["candidate_explanation"]
    assert exp["explanation_status"] == "generated" and exp["explained_file_id"] == ids[2]
    assert exp["explanation_evidence_points"] == ["Tagged 'budget'.", "Docx file."]
    assert out["explanations"] == {ids[2]: "It matches your budget request."}
    assert out["candidate_comparison"]["comparison_applied"] is True


def test_17_run_query_default_provider_still_functional():
    out = __import__("app.run", fromlist=["run_query"]).run_query(
        "user_a", "Find the slide deck about AI architecture I edited last week.")
    assert out["status"] == "finish" and out["final_ranking"]
    assert out["candidate_explanation"]["explanation_status"] == "unavailable"
    assert out["explanations"] == {}
