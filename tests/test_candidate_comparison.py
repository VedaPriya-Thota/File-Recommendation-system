"""
Phase 5C tests -- LLM-assisted candidate comparison
(app/llm/candidate_comparer.py, CandidateComparison in app/llm/schema.py,
node_compare_candidates in app/graph/build.py).

Structure:
  1. Valid structured comparison
  2. Candidate ID must belong to the supplied candidates
  3. Unknown / generated candidate ID rejected
  4. Bounded candidate input
  5. Evidence passed correctly
  6. Clear / dominant candidate -> no LLM call
  7. Close / ambiguous candidates -> LLM call
  8. Provider failure -> deterministic result unchanged
  9. Invalid output -> deterministic result unchanged
 10. Access control cannot be changed by the LLM
 11. Candidate pool cannot expand
 12. Existing retrieval/reranker/personalization/final-ranking scores preserved
 13. End-to-end ambiguous comparison path (real graph, real scope)
"""
import json

import pytest
from pydantic import ValidationError

from app.config import COMPARISON_MIN_CONFIDENCE, COMPARISON_TOP_N, DOMINANT_SCORE_GAP
from app.db.connection import get_cursor
from app.final_ranking import final_rank
from app.graph.build import build_graph
from app.llm.candidate_comparer import (
    SYSTEM_PROMPT,
    build_user_prompt,
    candidate_evidence,
    compare_candidates_with_llm,
    reorder_with_selection,
    select_comparison_set,
)
from app.llm.provider import LLMUnavailableError, NullProvider
from app.llm.schema import (
    CandidateComparison,
    LLMResponseInvalidError,
    parse_candidate_comparison_response,
)
from app.run import _initial_state
from app.tools.registry import ToolResult

QUERY = "quarterly budget planning"


# --- helpers -----------------------------------------------------------------

def _entry(score, file_id_hint=None, **metadata):
    md = {"filename": f"file_{file_id_hint}.docx", "file_type": "docx", "tags": ["budget"],
          "summary": "A summary.", "owner_id": "user_a", "file_path": "/secret/path"}
    md.update(metadata)
    return {
        "evidence_sources": ["hybrid_search"],
        "scores": {"hybrid_search": score},
        "metadata": md,
        "first_seen_iteration": 1,
        "last_updated_iteration": 1,
    }


def _ranking(*scores, **extra_md):
    """A real deterministic final ranking (via final_rank) for file ids 1..n."""
    pool = {i + 1: _entry(s, i + 1, **extra_md) for i, s in enumerate(scores)}
    return final_rank(pool)


def _order(ranking):
    return [c["file_id"] for c in ranking]


class CompareProvider:
    """Answers the comparison prompt with `response` (or raises); any other
    prompt (5A/5B) fails, isolating 5C. Records every comparison call."""

    name = "fake"

    def __init__(self, response=None, raises=None):
        self.response = response
        self.raises = raises
        self.calls = []
        self.other_calls = 0

    def complete(self, system_prompt, user_prompt):
        if "already-retrieved candidate files" in system_prompt:
            self.calls.append((system_prompt, user_prompt))
            if self.raises is not None:
                raise self.raises
            return self.response
        self.other_calls += 1
        raise RuntimeError("other LLM slices intentionally unavailable in this test")


def _reply(file_id, confidence=0.9, rationale="better topical match"):
    return json.dumps({"selected_file_id": file_id, "confidence": confidence, "rationale": rationale})


# --- DB-backed ids for the end-to-end tests ----------------------------------

def _ids(owner, n):
    with get_cursor() as cur:
        cur.execute("SELECT file_id FROM files WHERE owner_id = %s ORDER BY file_id LIMIT %s", (owner, n))
        return [r["file_id"] for r in cur.fetchall()]


class TiedRegistry:
    """Every tool returns the same near-tied in-scope candidates, tagged with
    the tool's own name as source. Real scope resolution still runs."""

    def __init__(self, file_ids, scores):
        self.file_ids, self.scores = file_ids, scores

    def execute(self, tool, args, user_scope):
        cands = [
            {"file_id": fid, "score": sc, "source": tool,
             "metadata": {"filename": f"f{fid}.docx", "file_type": "docx", "tags": ["budget"],
                          "summary": f"summary {fid}", "owner_id": "user_a"}}
            for fid, sc in zip(self.file_ids, self.scores)
        ]
        return ToolResult(tool=tool, candidates=cands)


def _run(provider, file_ids=None, scores=(0.90, 0.85, 0.80)):
    file_ids = file_ids or _ids("user_a", 3)
    graph = build_graph(registry=TiedRegistry(file_ids, scores), max_iterations=1, llm_provider=provider)
    return graph.invoke(_initial_state("user_a", QUERY, 1), config={"recursion_limit": 100}), file_ids


# =============================================================================
# 1. Valid structured comparison
# =============================================================================

def test_1_valid_payload_parses():
    parsed = parse_candidate_comparison_response(_reply(7, 0.8, "gist"))
    assert isinstance(parsed, CandidateComparison)
    assert (parsed.selected_file_id, parsed.confidence, parsed.rationale) == (7, 0.8, "gist")


def test_1_fenced_json_tolerated_and_rationale_optional():
    parsed = parse_candidate_comparison_response('```json\n{"selected_file_id": 2, "confidence": 1}\n```')
    assert parsed.selected_file_id == 2 and parsed.rationale is None


def test_1_comparer_returns_selection_and_metadata():
    ranking = _ranking(0.9, 0.85)
    sel, meta = compare_candidates_with_llm(QUERY, {}, ranking, CompareProvider(_reply(2)))
    assert sel == 2
    assert meta["comparison_status"] == "selected"
    assert meta["compared_file_ids"] == [1, 2]
    assert meta["comparison_confidence"] == 0.9
    assert meta["comparison_rationale"] == "better topical match"


def test_1_schema_fields_are_exactly_the_whitelist():
    assert set(CandidateComparison.model_fields) == {"selected_file_id", "confidence", "rationale"}


# =============================================================================
# 2. Candidate ID must belong to the supplied candidates
# =============================================================================

def test_2_out_of_set_id_rejected_even_if_it_is_a_real_ranked_file():
    ranking = _ranking(0.9, 0.85, 0.2)  # file 3 exists in the ranking but is NOT close -> not supplied
    close = select_comparison_set(ranking)
    assert _order(close) == [1, 2]
    sel, meta = compare_candidates_with_llm(QUERY, {}, close, CompareProvider(_reply(3)))
    assert sel is None and meta["comparison_status"] == "unknown_candidate"


@pytest.mark.parametrize("raw", [
    json.dumps({"selected_file_id": "2", "confidence": 0.9}),    # string
    json.dumps({"selected_file_id": 2.0, "confidence": 0.9}),    # float
    json.dumps({"selected_file_id": True, "confidence": 0.9}),   # bool
    json.dumps({"selected_file_id": None, "confidence": 0.9}),
    json.dumps({"confidence": 0.9}),                             # missing
])
def test_2_non_integer_ids_fail_schema(raw):
    with pytest.raises(LLMResponseInvalidError):
        parse_candidate_comparison_response(raw)


# =============================================================================
# 3. Unknown / generated candidate ID rejected
# =============================================================================

def test_3_generated_id_rejected_and_order_unchanged():
    ranking = _ranking(0.9, 0.85)
    sel, meta = compare_candidates_with_llm(QUERY, {}, ranking, CompareProvider(_reply(424242)))
    assert sel is None
    assert meta["comparison_status"] == "unknown_candidate"
    assert meta["selected_file_id"] is None


def test_3_e2e_generated_id_leaves_deterministic_ranking():
    baseline, _ = _run(NullProvider())
    result, _ = _run(CompareProvider(_reply(424242)))
    assert _order(result["final_ranking"]) == _order(baseline["final_ranking"])
    assert result["candidate_comparison"]["comparison_status"] == "unknown_candidate"
    assert result["candidate_comparison"]["comparison_applied"] is False


# =============================================================================
# 4. Bounded candidate input
# =============================================================================

def test_4_comparison_set_capped_at_top_n():
    ranking = _ranking(0.9, 0.89, 0.88, 0.87, 0.86, 0.85)  # six near-tied
    close = select_comparison_set(ranking)
    assert len(close) == COMPARISON_TOP_N
    assert _order(close) == _order(ranking)[:COMPARISON_TOP_N]


def test_4_only_close_candidates_within_top_n_are_supplied():
    ranking = _ranking(0.9, 0.85, 0.30, 0.29)
    assert _order(select_comparison_set(ranking)) == [1, 2]


def test_4_llm_prompt_contains_only_supplied_ids():
    ranking = _ranking(0.9, 0.89, 0.88, 0.87, 0.86)
    provider = CompareProvider(_reply(1))
    compare_candidates_with_llm(QUERY, {}, select_comparison_set(ranking), provider)
    payload = json.loads(provider.calls[0][1].split("\n", 1)[1])
    assert [c["file_id"] for c in payload["candidates"]] == [1, 2, 3][:COMPARISON_TOP_N]


def test_4_long_evidence_text_is_truncated():
    ranking = _ranking(0.9, 0.85, summary="s" * 5000, snippet="n" * 5000, matched_chunk="c" * 5000)
    ev = candidate_evidence(ranking[0])
    assert len(ev["summary"]) <= 400 and len(ev["bm25_snippet"]) <= 400 and len(ev["semantic_matched_chunk"]) <= 400


# =============================================================================
# 5. Evidence passed correctly
# =============================================================================

def test_5_prompt_carries_query_goal_and_full_candidate_evidence():
    pool = {
        1: _entry(0.9, 1, snippet="bm25 words", matched_chunk="semantic words",
                  personalization={"affinity_score": 0.7, "preferred_file_type_match": True,
                                   "edit_count": 3, "cold_start": False, "secret_extra": "x"}),
        2: _entry(0.85, 2),
    }
    pool[1]["scores"]["rerank_candidates"] = 0.95
    ranking = final_rank(pool)
    goal = {"filters": {"file_type": "docx", "filename_candidate": None}, "topic_tokens": ["budget"],
            "expanded_terms": ["fiscal"], "ambiguity_score": 0.4, "intent": "file_discovery"}
    prompt = build_user_prompt(QUERY, goal, ranking)
    payload = json.loads(prompt.split("\n", 1)[1])

    assert payload["user_query"] == QUERY
    assert payload["goal"]["file_type"] == "docx"
    assert payload["goal"]["topic_tokens"] == ["budget"]
    assert payload["goal"]["expanded_terms"] == ["fiscal"]

    top = next(c for c in payload["candidates"] if c["file_id"] == 1)
    assert top["filename"] == "file_1.docx" and top["file_type"] == "docx"
    assert top["tags"] == ["budget"] and top["summary"] == "A summary."
    assert top["bm25_snippet"] == "bm25 words"
    assert top["semantic_matched_chunk"] == "semantic words"
    assert top["source_scores"] == {"hybrid_search": 0.9, "rerank_candidates": 0.95}
    assert top["retrieval_score"] == ranking[0]["retrieval_score"]
    assert top["final_score"] == ranking[0]["final_score"]
    assert top["personalization_score"] == ranking[0]["personalization_score"]
    assert top["personalization"] == {"preferred_file_type_match": True, "edit_count": 3,
                                      "affinity_score": 0.7, "cold_start": False}


def test_5_hybrid_metadata_keys_are_recognised():
    ranking = _ranking(0.9, 0.85, bm25_snippet="hy bm25", semantic_matched_chunk="hy sem")
    ev = candidate_evidence(ranking[0])
    assert ev["bm25_snippet"] == "hy bm25" and ev["semantic_matched_chunk"] == "hy sem"


def test_5_prompt_excludes_owner_and_path():
    prompt = build_user_prompt(QUERY, {}, _ranking(0.9, 0.85))
    assert "owner_id" not in prompt and "/secret/path" not in prompt and "user_a" not in prompt


def test_5_system_prompt_forbids_new_ids_and_requests_no_reasoning():
    assert "Never invent a file_id" in SYSTEM_PROMPT
    assert "not your reasoning steps" in SYSTEM_PROMPT


# =============================================================================
# 6. Clear / dominant candidate -> no LLM call
# =============================================================================

def test_6_gate_empty_for_dominant_single_and_empty():
    assert select_comparison_set(_ranking(0.9, 0.9 - DOMINANT_SCORE_GAP - 0.2)) == []
    assert select_comparison_set(_ranking(0.9)) == []
    assert select_comparison_set([]) == []


def test_6_e2e_dominant_candidate_never_calls_llm():
    provider = CompareProvider(_reply(1))
    result, _ = _run(provider, scores=(1.0, 0.2, 0.1))
    assert provider.calls == []
    assert result["candidate_comparison"] == {
        "comparison_invoked": False, "comparison_status": "not_needed", "comparison_applied": False}


def test_6_e2e_single_candidate_never_calls_llm():
    provider = CompareProvider(_reply(1))
    result, ids = _run(provider, file_ids=_ids("user_a", 1), scores=(0.9,))
    assert provider.calls == [] and result["candidate_comparison"]["comparison_invoked"] is False


# =============================================================================
# 7. Close / ambiguous candidates -> LLM call
# =============================================================================

def test_7_gate_returns_near_tied_set():
    assert _order(select_comparison_set(_ranking(0.9, 0.85))) == [1, 2]


def test_7_e2e_near_tie_calls_llm_exactly_once():
    provider = CompareProvider(_reply(_ids("user_a", 3)[1]))
    _run(provider)
    assert len(provider.calls) == 1


# =============================================================================
# 8. Provider failure -> deterministic result unchanged
# =============================================================================

@pytest.mark.parametrize("provider,status", [
    (CompareProvider(raises=LLMUnavailableError("down")), "unavailable"),
    (CompareProvider(raises=RuntimeError("boom")), "provider_error"),
    (CompareProvider(response=None), "invalid_output"),  # non-str return
])
def test_8_provider_failure_keeps_deterministic_ranking(provider, status):
    baseline, _ = _run(NullProvider())
    result, _ = _run(provider)
    assert result["final_ranking"] == baseline["final_ranking"]
    assert result["candidate_comparison"]["comparison_status"] == status
    assert result["candidate_comparison"]["comparison_applied"] is False


def test_8_default_null_provider_is_unavailable_and_unchanged():
    result, _ = _run(NullProvider())
    assert result["candidate_comparison"]["comparison_status"] == "unavailable"
    assert result["final_ranking"] == final_rank(result["candidate_pool"])


# =============================================================================
# 9. Invalid output -> deterministic result unchanged
# =============================================================================

@pytest.mark.parametrize("raw", [
    "not json at all",
    "[1, 2]",
    '{"selected_file_id": 1}',                                              # no confidence
    '{"selected_file_id": 1, "confidence": 1.5}',                           # out of range
    '{"selected_file_id": 1, "confidence": 0.9, "sql": "DROP TABLE files"}',  # extra key
    '{"selected_file_id": 1, "confidence": 0.9, "tool_call": "metadata_search"}',
    '{"selected_file_id": 1, "confidence": 0.9, "rationale": "' + "x" * 201 + '"}',
    'Sure! {"selected_file_id": 1, "confidence": 0.9}',                     # prose around JSON
])
def test_9_invalid_output_keeps_deterministic_ranking(raw):
    baseline, _ = _run(NullProvider())
    result, _ = _run(CompareProvider(raw))
    assert result["final_ranking"] == baseline["final_ranking"]
    assert result["candidate_comparison"]["comparison_status"] == "invalid_output"


def test_9_low_confidence_preference_is_ignored():
    ids = _ids("user_a", 3)
    baseline, _ = _run(NullProvider())
    result, _ = _run(CompareProvider(_reply(ids[1], confidence=COMPARISON_MIN_CONFIDENCE - 0.1)))
    assert result["final_ranking"] == baseline["final_ranking"]
    assert result["candidate_comparison"]["comparison_status"] == "low_confidence"


# =============================================================================
# 10. Access control cannot be changed by the LLM
# =============================================================================

def test_10_llm_cannot_select_another_users_file():
    other = _ids("user_x", 1)
    assert other, "seed data must contain user_x files"
    result, ids = _run(CompareProvider(_reply(other[0])))
    assert result["candidate_comparison"]["comparison_status"] == "unknown_candidate"
    assert _order(result["final_ranking"])[0] == ids[0]


def test_10_user_scope_is_unchanged_by_comparison():
    other = _ids("user_x", 1)[0]
    with_llm, ids = _run(CompareProvider(_reply(_ids("user_a", 3)[1])))
    baseline, _ = _run(NullProvider())
    assert with_llm["user_scope"] == baseline["user_scope"] == ["user_a"]
    assert other not in _order(with_llm["final_ranking"])


def test_10_schema_has_no_scope_or_user_field():
    for forbidden in ("user_scope", "user_id", "owner_id", "scope"):
        with pytest.raises(ValidationError):
            CandidateComparison.model_validate(
                {"selected_file_id": 1, "confidence": 0.5, forbidden: "user_x"})


# =============================================================================
# 11. Candidate pool cannot expand
# =============================================================================

def test_11_pool_and_ranking_membership_identical_with_and_without_llm():
    baseline, ids = _run(NullProvider())
    result, _ = _run(CompareProvider(_reply(ids[2])))
    assert result["candidate_pool"] == baseline["candidate_pool"]
    assert set(result["candidate_pool"]) == set(ids)
    assert sorted(_order(result["final_ranking"])) == sorted(_order(baseline["final_ranking"]))
    assert len(result["final_ranking"]) == len(baseline["final_ranking"])


def test_11_reorder_never_adds_removes_or_copies():
    ranking = _ranking(0.9, 0.85, 0.3)
    out = reorder_with_selection(ranking, 2)
    assert _order(out) == [2, 1, 3]
    assert all(a is b for a, b in zip(sorted(out, key=lambda c: c["file_id"]),
                                      sorted(ranking, key=lambda c: c["file_id"])))
    assert _order(reorder_with_selection(ranking, 999)) == _order(ranking)


# =============================================================================
# 12. Existing scores preserved
# =============================================================================

def test_12_scores_and_entries_unchanged_only_order_may_differ():
    baseline, ids = _run(NullProvider())
    result, _ = _run(CompareProvider(_reply(ids[1])))
    by_id = lambda r: {c["file_id"]: c for c in r["final_ranking"]}  # noqa: E731
    assert by_id(result) == by_id(baseline)
    assert result["final_ranking"] != baseline["final_ranking"]  # order did change
    recomputed = {c["file_id"]: c for c in final_rank(result["candidate_pool"])}
    for entry in result["final_ranking"]:
        assert entry["final_score"] == recomputed[entry["file_id"]]["final_score"]
        assert entry["retrieval_score"] == recomputed[entry["file_id"]]["retrieval_score"]
        assert entry["scores"] == result["candidate_pool"][entry["file_id"]]["scores"]


def test_12_deterministic_order_recorded_for_audit():
    result, ids = _run(CompareProvider(_reply(_ids("user_a", 3)[1])))
    meta = result["candidate_comparison"]
    assert meta["deterministic_order"] == _order(final_rank(result["candidate_pool"]))


# =============================================================================
# 13. End-to-end ambiguous comparison path
# =============================================================================

def test_13_e2e_llm_preference_promotes_candidate_within_near_tie():
    ids = _ids("user_a", 3)
    deterministic, _ = _run(NullProvider(), file_ids=ids)
    assert _order(deterministic["final_ranking"])[0] == ids[0]

    result, _ = _run(CompareProvider(_reply(ids[1], 0.9, "closer to budget planning")), file_ids=ids)
    assert _order(result["final_ranking"])[0] == ids[1]
    meta = result["candidate_comparison"]
    assert meta["comparison_invoked"] and meta["comparison_applied"]
    assert meta["comparison_status"] == "selected"
    assert meta["selected_file_id"] == ids[1]
    assert meta["compared_file_ids"] == ids
    assert meta["deterministic_order"][0] == ids[0]
    assert meta["comparison_rationale"] == "closer to budget planning"
    trace = [t for t in result["trace"] if t["event"] == "compare_candidates"]
    assert len(trace) == 1 and trace[0]["comparison"]["comparison_applied"] is True


def test_13_e2e_llm_confirming_the_top_candidate_is_a_noop():
    ids = _ids("user_a", 3)
    baseline, _ = _run(NullProvider(), file_ids=ids)
    result, _ = _run(CompareProvider(_reply(ids[0])), file_ids=ids)
    assert result["final_ranking"] == baseline["final_ranking"]
    assert result["candidate_comparison"]["comparison_status"] == "selected"
    assert result["candidate_comparison"]["comparison_applied"] is False


def test_13_run_query_exposes_candidate_comparison(monkeypatch):
    from app import run as run_module

    ids = _ids("user_a", 3)
    real_build = run_module.build_graph
    monkeypatch.setattr(
        run_module, "build_graph",
        lambda registry=None, max_iterations=None, llm_provider=None: real_build(
            registry=TiedRegistry(ids, (0.9, 0.85, 0.8)), max_iterations=1, llm_provider=llm_provider),
    )
    out = run_module.run_query("user_a", QUERY, max_iterations=1, llm_provider=CompareProvider(_reply(ids[2])))
    assert out["candidate_comparison"]["comparison_applied"] is True
    assert out["final_ranking"][0]["file_id"] == ids[2]
