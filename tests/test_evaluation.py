"""
Phase 7 -- tests for the evaluation framework (evaluation/*).

These exercise the evaluation LOGIC (metrics, aggregation, success judging,
counting, variant wiring, dataset validation, scope audit, reproducibility)
against hand-computed values and against real run_query() executions. None of
them assert the benchmark's headline numbers: those are findings, not
invariants.
"""
import copy
import json

import pytest

from app.db.connection import get_cursor
from app.llm.provider import LLMUnavailableError, NullProvider
from app.llm.schema import (
    LLMResponseInvalidError,
    parse_candidate_comparison_response,
    parse_llm_goal_response,
    parse_query_expansion_response,
    parse_result_explanation_response,
)
from app.tools.registry import ToolResult

import evaluation.runner as runner
from evaluation import benchmark_dataset as ds
from evaluation import metrics as m
from evaluation import report
from evaluation.providers import STAGE_PROMPTS, CountingProvider, ScriptedProvider, identify_stage
from evaluation.variants import (
    CORE_VARIANT_NAMES,
    VARIANTS,
    VARIANTS_BY_NAME,
    build_variant_provider,
    build_variant_registry,
    null_personalization,
)

A, B, C, D = (VARIANTS_BY_NAME[n] for n in CORE_VARIANT_NAMES)


@pytest.fixture
def telemetry():
    """Function-scoped on purpose: run_benchmark() installs its own fixture and
    (correctly) refuses to start if telemetry rows already exist, so no test may
    hold this open while a benchmark runs."""
    with runner.personalization_fixture():
        yield


@pytest.fixture(scope="module")
def full_output():
    """One full benchmark execution shared by the schema/smoke tests."""
    return runner.run_benchmark()


def _rec(**kw):
    base = dict(
        query_id="X", query="q", user_id="user_a", category="A_exact", variant="V",
        expected_outcome="results", personalization_context="none",
        expected_file_ids=[1], highly_relevant_file_ids=[1], forbidden_file_ids=[],
        returned_file_ids=[1], pool_file_ids=[1], pool_ranking_file_ids=[1],
        first_relevant_rank=1, precision_at_1=1.0, precision_at_3=1 / 3, precision_at_5=0.2,
        recall_at_1=1.0, recall_at_3=1.0, recall_at_5=1.0, reciprocal_rank=1.0, ndcg_at_5=1.0,
        pool_ranking_first_relevant_rank=1, pool_ranking_reciprocal_rank=1.0,
        highly_relevant_at_1=1.0, pool_ranking_highly_relevant_at_1=1.0,
        success=True, status="finish", clarified=False, no_result=False, degraded_mode=False,
        hard_failure=False, confidence=1.0, scope_violations=0, scope_violation_file_ids=[],
        tool_calls=1, retrieval_actions=1,
        tool_counts={t: 0 for t in report.TOOL_FIELDS}, tool_sequence=["metadata_search"],
        iterations=1, goal_ambiguity_score=0.1, llm_calls=0, llm_attempts=0,
        llm_calls_by_stage={}, llm_attempts_by_stage={}, llm_stage_status={}, top3_final_scores=[0.85],
        error_count=0,
    )
    base.update(kw)
    return base


# --- 1. ground-truth dataset validity ----------------------------------------
def test_1_dataset_is_valid_and_covers_every_category():
    assert ds.validate_dataset() == []
    assert 20 <= len(ds.QUERIES) <= 40
    assert {q.category for q in ds.QUERIES} == set(ds.CATEGORIES)
    for q in ds.QUERIES:
        assert q.notes, f"{q.query_id}: ground truth must be justified"


def test_1b_dataset_matches_live_database():
    with get_cursor() as cur:
        cur.execute("SELECT file_id, filename, owner_id FROM files ORDER BY file_id;")
        rows = [dict(r) for r in cur.fetchall()]
    assert ds.validate_against_db(rows) == []
    drifted = [dict(r) for r in rows]
    drifted[0]["filename"] = "Renamed.pptx"
    assert any("file_id 1" in p for p in ds.validate_against_db(drifted))
    assert any("missing" in p for p in ds.validate_against_db(rows[1:]))


# --- 2. unknown file id detection --------------------------------------------
def test_2_unknown_and_inaccessible_ids_are_detected():
    bad_unknown = ds.BenchmarkQuery("A9", "q one", "user_a", "A_exact", {999: 2}, "results")
    bad_forbidden_unknown = ds.BenchmarkQuery("A8", "q two", "user_a", "A_exact", {1: 2}, "results", forbidden_file_ids=(998,))
    bad_owner = ds.BenchmarkQuery("A7", "q three", "user_a", "A_exact", {8: 2}, "results")  # file 8 is user_b's
    both = ds.BenchmarkQuery("A6", "q four", "user_a", "A_exact", {1: 2}, "results", forbidden_file_ids=(1,))
    problems = ds.validate_dataset([bad_unknown, bad_forbidden_unknown, bad_owner, both])
    text = " | ".join(problems)
    assert "unknown relevant file_id 999" in text
    assert "unknown forbidden file_id 998" in text
    assert "not accessible to user_a" in text
    assert "actually owned by user_a" in text


def test_2b_outcome_and_relevance_consistency_is_enforced():
    no_result_with_rel = ds.BenchmarkQuery("F9", "x y", "user_a", "F_no_result", {1: 1}, "no_result")
    results_without_rel = ds.BenchmarkQuery("A9", "x z", "user_a", "A_exact", {}, "results")
    bad_grade = ds.BenchmarkQuery("A8", "x w", "user_a", "A_exact", {1: 5}, "results")
    problems = " | ".join(ds.validate_dataset([no_result_with_rel, results_without_rel, bad_grade]))
    assert "no_result query must not list relevant files" in problems
    assert "has no relevant files" in problems
    assert "invalid grade 5" in problems


# --- 3. duplicate detection ---------------------------------------------------
def test_3_duplicate_queries_and_ids_are_detected():
    q = ds.QUERIES[0]
    same_text = ds.BenchmarkQuery("Z1", "  " + q.query.upper() + " ", q.user_id, q.category, dict(q.relevance), q.expected_outcome)
    dup_id = ds.BenchmarkQuery(q.query_id, "something else entirely", q.user_id, q.category, dict(q.relevance), q.expected_outcome)
    problems = " | ".join(ds.validate_dataset([q, same_text, dup_id]))
    assert "duplicate (user_id, query)" in problems
    assert "duplicate query_id" in problems
    # same text for a DIFFERENT user is legitimate
    other_user = ds.BenchmarkQuery("Z2", q.query, "user_b", q.category, {8: 2}, "results")
    assert ds.validate_dataset([other_user]) == []


# --- 4/5/6. metrics: hand-computed ---------------------------------------------
def test_4_precision_at_k_standard_definition():
    ret, rel = [1, 2, 3, 4, 5], {2, 5}
    assert m.precision_at_k(ret, rel, 1) == 0.0
    assert m.precision_at_k(ret, rel, 3) == pytest.approx(1 / 3)
    assert m.precision_at_k(ret, rel, 5) == pytest.approx(2 / 5)
    # K is the denominator even when fewer than K files were returned
    assert m.precision_at_k([2], {2}, 5) == pytest.approx(1 / 5)
    assert m.precision_at_k([], {2}, 3) == 0.0
    assert m.precision_at_k([1], set(), 3) is None  # undefined, not 0


def test_5_recall_at_k_standard_definition():
    ret, rel = [1, 2, 3, 4, 5], {2, 5, 9}
    assert m.recall_at_k(ret, rel, 1) == 0.0
    assert m.recall_at_k(ret, rel, 3) == pytest.approx(1 / 3)
    assert m.recall_at_k(ret, rel, 5) == pytest.approx(2 / 3)
    assert m.recall_at_k([], rel, 5) == 0.0
    assert m.recall_at_k(ret, set(), 5) is None


def test_6_mrr_reciprocal_rank_and_duplicates():
    assert m.reciprocal_rank([9, 8, 2], {2}) == pytest.approx(1 / 3)
    assert m.reciprocal_rank([9, 8], {2}) == 0.0
    assert m.reciprocal_rank([2, 9], {2}) == 1.0
    assert m.reciprocal_rank([9], set()) is None
    assert m.first_relevant_rank([9, 9, 2], {2}) == 2  # a duplicated id cannot push the rank down
    recs = [_rec(reciprocal_rank=1.0), _rec(query_id="Y", reciprocal_rank=0.5), _rec(query_id="Z", reciprocal_rank=0.0)]
    assert report.retrieval_summary(recs)["MRR"] == pytest.approx(0.5, abs=1e-4)


def test_6b_ndcg_uses_graded_relevance():
    rel = {1: 2, 2: 1}
    perfect = m.ndcg_at_k([1, 2], rel, 5)
    swapped = m.ndcg_at_k([2, 1], rel, 5)
    assert perfect == pytest.approx(1.0)
    assert 0 < swapped < perfect


# --- 7. aggregation by category / variant ----------------------------------------
def test_7_category_and_variant_aggregation():
    recs = [
        _rec(query_id="a1", category="A_exact", variant="V1", reciprocal_rank=1.0, precision_at_1=1.0),
        _rec(query_id="a2", category="A_exact", variant="V1", reciprocal_rank=0.0, precision_at_1=0.0, success=False),
        _rec(query_id="c1", category="C_topic", variant="V1", reciprocal_rank=0.5, precision_at_1=0.0),
        _rec(query_id="a1", category="A_exact", variant="V2", reciprocal_rank=1.0, precision_at_1=1.0),
    ]
    s = report.build_summary(recs)
    assert s["by_category"]["A_exact"]["V1"]["MRR"] == pytest.approx(0.5)
    assert s["by_category"]["C_topic"]["V1"]["MRR"] == pytest.approx(0.5)
    assert s["by_category"]["A_exact"]["V2"]["MRR"] == pytest.approx(1.0)
    assert s["overall"]["V1"]["MRR"] == pytest.approx(0.5, abs=1e-4)  # (1+0+0.5)/3
    assert s["overall"]["V1"]["n"] == 3 and s["overall"]["V2"]["n"] == 1
    assert s["by_variant"]["V1"]["by_category"]["A_exact"]["success_rate"] == pytest.approx(0.5)
    assert set(s["by_variant"]) == {"V1", "V2"}


# --- 8. no-result handling ------------------------------------------------------------
def test_8_no_result_judging_and_aggregation_never_drops_queries():
    assert m.judge_success("no_result", [], False, {}, 0) is True
    assert m.judge_success("no_result", [8], False, {}, 0) is False  # fabricated result
    assert m.judge_success("no_result", [], True, {}, 0) is True  # clarifying with nothing returned is fine
    # undefined retrieval metrics are None (not 0), and the query is still counted in n
    metrics = m.retrieval_metrics([], {})
    assert metrics["precision_at_1"] is None and metrics["reciprocal_rank"] is None
    recs = [_rec(), _rec(query_id="F", expected_file_ids=[], expected_outcome="no_result", returned_file_ids=[], no_result=True,
                          precision_at_1=None, reciprocal_rank=None, recall_at_3=None, recall_at_1=None, recall_at_5=None,
                          precision_at_3=None, precision_at_5=None, ndcg_at_5=None, first_relevant_rank=None)]
    s = report.retrieval_summary(recs)
    assert s["n"] == 2 and s["n_scored"] == 1
    assert s["success_rate"] == 1.0  # the no-result query still counts toward success
    assert report.behavior_summary(recs)["no_result_false_positive_rate"] == 0.0


def test_8b_real_no_result_query_returns_nothing(telemetry):
    rec = runner.run_one(ds.query_by_id("F1"), B)
    assert rec["returned_file_ids"] == [] and rec["success"] is True
    assert rec["precision_at_1"] is None and rec["expected_file_ids"] == []


# --- 9. clarification handling ------------------------------------------------------------
def test_9_clarification_judging():
    rel = {2: 1, 3: 1}
    assert m.judge_success("clarify_or_results", [], True, rel, 0) is True
    assert m.judge_success("clarify_or_results", [3], False, rel, 0) is True
    assert m.judge_success("clarify_or_results", [9], False, rel, 0) is False
    assert m.judge_success("results", [], True, rel, 0) is False  # clarifying is not an answer when one was expected
    assert m.judge_success("clarify_or_results", [2], False, rel, 1) is False  # scope violation always fails


def test_9b_clarified_real_query_is_scored_zero_not_dropped(telemetry):
    rec = runner.run_one(ds.query_by_id("D1"), B)
    assert rec["clarified"] is True and rec["returned_file_ids"] == []
    assert rec["precision_at_1"] == 0.0 and rec["reciprocal_rank"] == 0.0  # zero, not None
    assert rec["success"] is True  # acceptable outcome for an ambiguous query
    s = report.retrieval_summary([rec])
    assert s["n_scored"] == 1 and s["abstention_rate_scored"] == 1.0
    assert s["n_answered"] == 0 and s["MRR_answered_only"] is None


# --- 10. cross-user isolation ---------------------------------------------------------------
def test_10_cross_user_isolation_in_every_variant(telemetry):
    for q in (q for q in ds.QUERIES if q.category == "G_cross_user" or q.forbidden_file_ids):
        for v in VARIANTS:
            rec = runner.run_one(q, v)
            assert rec["scope_violations"] == 0, (q.query_id, v.name, rec["scope_violation_file_ids"])
            assert not set(rec["returned_file_ids"]) & set(q.forbidden_file_ids)
            assert not set(rec["pool_file_ids"]) & set(q.forbidden_file_ids)


def test_10b_scope_audit_detects_a_leak(monkeypatch, telemetry):
    """The audit itself must work: feed runner a fabricated leaking result."""
    leak = {
        "status": "finish", "final_ranking": [{"file_id": 13, "metadata": {"owner_id": "user_x"}, "final_score": 0.9}],
        "candidate_pool": {13: {"metadata": {"owner_id": "user_x"}, "scores": {}}},
        "committed_actions": [], "iteration_count": 0, "trace": [],
    }
    monkeypatch.setattr(runner, "run_query", lambda *a, **k: leak)
    rec = runner.run_one(ds.query_by_id("G3"), B)
    assert rec["scope_violations"] == 1 and rec["scope_violation_file_ids"] == [13]
    assert rec["success"] is False


# --- 11. variant configuration -------------------------------------------------------------------
def test_11_variant_configuration_is_wired_as_documented(telemetry):
    assert [v.name for v in VARIANTS if v.core] == list(CORE_VARIANT_NAMES)
    assert A.max_iterations == 1 and B.max_iterations is None
    assert (A.personalization, B.personalization, C.personalization, D.personalization) == (False, False, True, True)
    assert (A.llm_mode, B.llm_mode, C.llm_mode, D.llm_mode) == (None, None, None, "useful")

    # personalization disabled <=> pass-through tool; enabled <=> the production tool
    assert build_variant_registry(B).get("get_user_activity").fn is null_personalization
    assert build_variant_registry(C).get("get_user_activity").fn is not null_personalization
    assert set(build_variant_registry(B).names()) == set(build_variant_registry(C).names())

    assert isinstance(build_variant_provider(B)._inner, NullProvider)
    assert isinstance(build_variant_provider(D)._inner, ScriptedProvider)

    # variant A really is single-shot metadata_search, even where B escalates
    q = ds.query_by_id("D1")
    ra, rb = runner.run_one(q, A), runner.run_one(q, B)
    assert ra["tool_sequence"] == ["metadata_search"] and ra["iterations"] == 1
    assert len(rb["tool_sequence"]) > 1 and rb["tool_sequence"][0] == "metadata_search"


def test_11b_null_personalization_is_neutral_passthrough():
    cands = [{"file_id": 2, "metadata": {"filename": "x"}}, {"file_id": 7, "metadata": {}}]
    res = null_personalization({"candidates": cands}, None)
    assert [c["file_id"] for c in res.candidates] == [2, 7]  # never adds/removes a candidate
    assert all(c["score"] == 0.0 and c["source"] == "get_user_activity" for c in res.candidates)


def test_11c_personalization_ablation_changes_only_ranking_evidence(telemetry):
    """B vs C: same corpus/user/query/candidate pool; only affinity differs."""
    q = ds.query_by_id("E2")
    rb, rc = runner.run_one(q, B), runner.run_one(q, C)
    assert rb["pool_file_ids"] == rc["pool_file_ids"]
    assert rb["tool_sequence"] == rc["tool_sequence"]
    assert rb["pool_ranking_file_ids"] != rc["pool_ranking_file_ids"]  # fixture history reorders the pool
    assert rc["pool_ranking_file_ids"][0] == 7


# --- 12. deterministic fake provider ---------------------------------------------------------------
def _sys(stage):
    return STAGE_PROMPTS[stage]


GOAL_PROMPT = "User query: 'financial overview'\nThe deterministic system already detected: x"
CMP_PROMPT = "Pick the single best-matching candidate for the user's request.\n" + json.dumps({
    "user_query": "budget slides",
    "candidates": [
        {"file_id": 2, "filename": "Q3_Budget_Review.xlsx", "tags": ["budget"], "summary": "spreadsheet"},
        {"file_id": 7, "filename": "Budget_Slides_Draft.pptx", "tags": ["budget", "slides"], "summary": "Draft slides"},
    ]})
EXPL_PROMPT = "Explain why this result was returned.\n" + json.dumps({"result": {"filename": "Q3_Budget_Review.xlsx", "file_type": "xlsx"}})


def test_12_scripted_provider_is_deterministic_and_emits_schema_valid_output():
    p1, p2 = ScriptedProvider("useful"), ScriptedProvider("useful")
    for stage, prompt, parser in (
        ("goal", GOAL_PROMPT, parse_llm_goal_response),
        ("expansion", GOAL_PROMPT, parse_query_expansion_response),
        ("comparison", CMP_PROMPT, parse_candidate_comparison_response),
        ("explanation", EXPL_PROMPT, parse_result_explanation_response),
    ):
        out1, out2 = p1.complete(_sys(stage), prompt), p2.complete(_sys(stage), prompt)
        assert out1 == out2, f"{stage} not deterministic"
        parser(out1)  # validated by the PRODUCTION parsers
    assert parse_llm_goal_response(p1.complete(_sys("goal"), GOAL_PROMPT)).topic_tokens == ["finance"]
    sel = parse_candidate_comparison_response(p1.complete(_sys("comparison"), CMP_PROMPT))
    assert sel.selected_file_id in (2, 7) and sel.selected_file_id == 7  # more query-word overlap


def test_12b_failure_modes_and_stage_selective_failure():
    with pytest.raises(LLMUnavailableError):
        ScriptedProvider("unavailable").complete(_sys("goal"), GOAL_PROMPT)
    bad = ScriptedProvider("invalid").complete(_sys("comparison"), CMP_PROMPT)
    with pytest.raises(LLMResponseInvalidError):
        parse_candidate_comparison_response(bad)
    partial = ScriptedProvider("useful", fail_stages=("explanation",))
    partial.complete(_sys("goal"), GOAL_PROMPT)  # other stages still work
    with pytest.raises(LLMUnavailableError):
        partial.complete(_sys("explanation"), EXPL_PROMPT)
    with pytest.raises(ValueError):
        ScriptedProvider("bogus")


def test_12c_comparison_never_invents_an_id_and_low_overlap_is_low_confidence():
    prompt = "x\n" + json.dumps({"user_query": "zzz qqq", "candidates": [{"file_id": 4, "filename": "A.docx", "tags": [], "summary": "s"}]})
    cmp = parse_candidate_comparison_response(ScriptedProvider("useful").complete(_sys("comparison"), prompt))
    assert cmp.selected_file_id == 4 and cmp.confidence < 0.5  # below production floor -> would be ignored


# --- 13. LLM call counting -------------------------------------------------------------------------------
def test_13_counting_provider_separates_attempts_from_calls():
    unavailable = CountingProvider(ScriptedProvider("unavailable"))
    with pytest.raises(LLMUnavailableError):
        unavailable.complete(_sys("goal"), GOAL_PROMPT)
    assert (unavailable.total_attempts, unavailable.total_calls) == (1, 0)

    invalid = CountingProvider(ScriptedProvider("invalid"))
    invalid.complete(_sys("expansion"), GOAL_PROMPT)  # returns text (billable) even though unparseable
    assert (invalid.total_attempts, invalid.total_calls) == (1, 1)

    useful = CountingProvider(ScriptedProvider("useful"))
    useful.complete(_sys("goal"), GOAL_PROMPT)
    useful.complete(_sys("goal"), GOAL_PROMPT)
    useful.complete(_sys("explanation"), EXPL_PROMPT)
    assert useful.calls["goal"] == 2 and useful.calls["explanation"] == 1 and useful.total_calls == 3
    assert identify_stage("something unrelated") == "other"


def test_13b_llm_counts_in_real_runs(telemetry):
    clear = runner.run_one(ds.query_by_id("A1"), D)
    # gate closed (low ambiguity): only the presentation-only explanation stage runs, exactly once
    assert clear["llm_calls_by_stage"] == {"goal": 0, "expansion": 0, "comparison": 0, "explanation": 1}
    assert clear["llm_calls"] == 1 == clear["llm_attempts"]

    ambiguous = runner.run_one(ds.query_by_id("D1"), D)
    assert ambiguous["goal_ambiguity_score"] >= 0.65
    assert ambiguous["llm_calls_by_stage"]["goal"] == 1 and ambiguous["llm_calls_by_stage"]["expansion"] == 1

    # NullProvider variants: stages are attempted but nothing is billable
    nb = runner.run_one(ds.query_by_id("D1"), B)
    assert nb["llm_calls"] == 0 and nb["llm_attempts"] >= 2

    outage = runner.run_one(ds.query_by_id("D1"), VARIANTS_BY_NAME["D_llm_unavailable"])
    assert outage["llm_calls"] == 0 and outage["llm_attempts"] >= 2


# --- 14. tool-call counting ----------------------------------------------------------------------------------
def test_14_tool_call_counting_matches_committed_actions(telemetry):
    rec = runner.run_one(ds.query_by_id("F1"), B)  # zero results everywhere -> metadata, bm25, semantic
    assert rec["tool_sequence"] == ["metadata_search", "bm25_search", "semantic_search"]
    assert rec["tool_calls"] == 3 == rec["iterations"]
    assert rec["tool_counts"]["bm25_search"] == 1 and rec["tool_counts"]["semantic_search"] == 1
    assert rec["tool_counts"]["rerank_candidates"] == 0
    assert rec["retrieval_actions"] == 3

    flat = runner.run_one(ds.query_by_id("E1"), C)  # flat pool -> metadata, rerank, get_user_activity
    assert flat["tool_sequence"] == ["metadata_search", "rerank_candidates", "get_user_activity"]
    assert flat["retrieval_actions"] == 1  # rerank / activity are not retrieval actions
    assert sum(flat["tool_counts"].values()) == flat["tool_calls"]

    one = runner.run_one(ds.query_by_id("A1"), B)
    assert one["tool_calls"] == one["retrieval_actions"] == one["iterations"] == 1


# --- 15. output schema ---------------------------------------------------------------------------------------------
REQUIRED_RECORD_FIELDS = {
    "query", "user_id", "category", "variant", "expected_file_ids", "returned_file_ids", "first_relevant_rank",
    "precision_at_1", "precision_at_3", "precision_at_5", "recall_at_1", "recall_at_3", "recall_at_5",
    "reciprocal_rank", "success", "clarified", "tool_calls", "iterations", "llm_calls", "retrieval_actions",
}


def test_15_output_schema_and_privacy(full_output):
    assert set(full_output) == {"metadata", "results", "summary", "scenarios"}
    for rec in full_output["results"]:
        assert REQUIRED_RECORD_FIELDS <= set(rec), REQUIRED_RECORD_FIELDS - set(rec)
        assert "latency_ms_environment_specific" not in rec  # off by default
    blob = json.dumps(full_output, default=str)
    json.loads(blob)  # JSON-serialisable
    assert "owner_id" not in blob and "file_path" not in blob and "/files/" not in blob
    assert set(full_output["summary"]) >= {"overall", "by_category", "by_variant", "behavior"}
    for v, per in full_output["summary"]["by_variant"].items():
        assert set(per) == {"retrieval", "behavior", "by_category"}
    md = report.render_markdown(full_output)
    assert "Limitations" in md and "Scenarios" in md


def test_15b_latency_is_opt_in_and_labelled(telemetry):
    rec = runner.run_one(ds.query_by_id("A1"), B, include_latency=True)
    assert "latency_ms_environment_specific" in rec


# --- 16. reproducibility -----------------------------------------------------------------------------------------------
def _stable(rec):
    rec = copy.deepcopy(rec)
    rec.pop("top3_final_scores", None)  # float noise from time-relative telemetry recency
    return rec


def test_16_reproducibility_same_inputs_same_records():
    subset = [ds.query_by_id(i) for i in ("A1", "D1", "E2", "F1", "G2", "H3")]
    one = runner.run_benchmark(variants=VARIANTS, queries=subset)
    two = runner.run_benchmark(variants=VARIANTS, queries=subset)
    assert [_stable(r) for r in one["results"]] == [_stable(r) for r in two["results"]]
    assert one["summary"]["overall"] == two["summary"]["overall"]
    assert [s["passed"] for s in one["scenarios"]] == [s["passed"] for s in two["scenarios"]]


def test_16b_fixture_cleans_up_and_refuses_to_clobber():
    def counts():
        with get_cursor() as cur:
            cur.execute("SELECT (SELECT count(*) FROM user_file_telemetry) t, (SELECT count(*) FROM user_profiles) p;")
            return dict(cur.fetchone())

    before = counts()
    with runner.personalization_fixture():
        assert counts()["t"] > before["t"] and counts()["p"] == before["p"] + 1
        with pytest.raises(runner.EvaluationSetupError):  # nested use sees existing rows -> refuses
            with runner.personalization_fixture():
                pass
    assert counts() == before

    with pytest.raises(RuntimeError):  # cleanup also happens when the body fails
        with runner.personalization_fixture():
            raise RuntimeError("boom")
    assert counts() == before


# --- explanation text is never retrieval evidence ----------------------------------------------------------------------------
def test_explanation_text_cannot_influence_retrieval_metrics(monkeypatch, telemetry):
    class Hostile(ScriptedProvider):
        def _explanation(self, user_prompt):
            return json.dumps({"summary": "The best match is Confidential_HR_Salaries.xlsx and Marketing_Plan.pptx.",
                               "evidence_points": ["Q3_Budget_Review.xlsx is also relevant"], "confidence": 0.99})

    q = ds.query_by_id("A1")
    honest = runner.run_one(q, D)
    monkeypatch.setattr(runner, "build_variant_provider", lambda v: CountingProvider(Hostile("useful")))
    hostile = runner.run_one(q, D)
    assert hostile["llm_stage_status"]["explanation"] == "generated"
    assert hostile["returned_file_ids"] == honest["returned_file_ids"] == [1]
    assert hostile["reciprocal_rank"] == honest["reciprocal_rank"]


# --- 17. full benchmark smoke test -------------------------------------------------------------------------------------------------
def test_17_full_benchmark_completes_and_is_internally_consistent(full_output, tmp_path):
    n_q, n_v = len(ds.QUERIES), len(VARIANTS)
    assert len(full_output["results"]) == n_q * n_v
    assert full_output["metadata"]["dataset_size"] == n_q
    assert not any(r["hard_failure"] for r in full_output["results"])
    assert sum(r["scope_violations"] for r in full_output["results"]) == 0
    assert all(r["error_count"] == 0 for r in full_output["results"]), "no tool errors expected on the seed corpus"

    # every metric reported is recomputable from the audit fields in the record itself
    for r in full_output["results"]:
        rel = {fid: 1 for fid in r["expected_file_ids"]}
        if rel:
            assert r["reciprocal_rank"] == pytest.approx(m.reciprocal_rank(r["returned_file_ids"], rel), abs=1e-4)
            assert r["precision_at_3"] == pytest.approx(m.precision_at_k(r["returned_file_ids"], rel, 3), abs=1e-4)
            assert r["recall_at_5"] == pytest.approx(m.recall_at_k(r["returned_file_ids"], rel, 5), abs=1e-4)
        else:
            assert r["reciprocal_rank"] is None
        assert r["tool_calls"] == len(r["tool_sequence"])

    # scenarios that are invariants of the architecture must hold; findings may fail without breaking the harness
    by_name = {s["name"]: s for s in full_output["scenarios"]}
    for name in ("cross_user_isolation", "llm_outage", "explanation_never_affects_ranking", "near_tie_gating"):
        assert by_name[name]["passed"], [c for c in by_name[name]["checks"] if not c["passed"]][:3]

    out = tmp_path / "res.json"
    runner.write_results(full_output, str(out))
    assert json.loads(out.read_text())["metadata"]["dataset_size"] == n_q
    with get_cursor() as cur:  # fixture rows were removed
        cur.execute("SELECT count(*) AS c FROM user_file_telemetry WHERE user_id = 'user_a';")
        assert cur.fetchone()["c"] == 0


def test_17b_benchmark_does_not_modify_production_modules():
    """The evaluation package must stay outside the production path."""
    import pathlib
    import re
    for path in pathlib.Path("app").rglob("*.py"):
        assert not re.search(r"^\s*(from|import)\s+evaluation", path.read_text(), re.M), path
