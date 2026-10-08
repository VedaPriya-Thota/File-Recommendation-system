"""
Phase 12 -- the Phase 11 failures S3/S6 on the REAL corpus through the real agent, plus the evaluation additions
(standalone stage probes, provider-vs-agent layer attribution, coverage sensitivity, separate diagnostics).

Quality numbers that depend on the embedding model / reranker the environment resolves are not asserted; what is
asserted is behaviour that must hold with ANY provider: the agent consults semantic evidence when lexical evidence is
weak, does not when it is strong, flags what it cannot support, and attributes failures to the right layer.
"""
import pytest

from app.embeddings import get_provider_name
from app.llm.provider import NullProvider
from app.run import run_query

from evaluation.content import runner
from evaluation.content.queries import DIAGNOSTIC_QUERIES, QUERIES, validate
from evaluation.content.corpus_spec import CORPUS

Q = {q.qid: q for q in QUERIES}


@pytest.fixture(scope="module")
def output():
    return runner.run_content_eval()


def by_id(out):
    return {r["qid"]: r for r in out["results"]}


# ============================================================ S3 / S6 through the real agent
def test_weak_lexical_hits_no_longer_preempt_semantic_retrieval_on_the_real_corpus(output):
    x = by_id(output)
    for qid in ("S3", "S6"):
        assert x[qid]["tool_sequence"][:3] == ["metadata_search", "bm25_search", "semantic_search"], qid     # semantic is now consulted
        assert x[qid]["probes"]["top"]["bm25_top_coverage"] == 0.25, qid                                   # the lexical hit explained 1 of 4 terms
    assert "when" in x["S3"]["goal"]["topic_tokens"] and "sunlight" in x["S6"]["goal"]["topic_tokens"]


def test_the_weak_lexical_candidate_never_claims_more_confidence_than_it_explains(output):
    r = by_id(output)["S3"]
    if r["decision_basis"] == "weak_lexical_evidence":
        assert r["degraded_mode"] is True and "weak lexical" in r["reason"].lower()


def test_strong_lexical_queries_are_not_escalated_so_the_agent_is_not_a_fixed_pipeline(output):
    x = by_id(output)
    for qid in ("C1", "C2", "C3", "C4", "L1", "L2", "L3", "L4"):
        assert x[qid]["tool_sequence"] in (["metadata_search"], ["metadata_search", "bm25_search"]), qid
        assert x[qid]["tool_counts"]["semantic_search"] == 0, qid
        assert x[qid]["success"], qid


def test_semantic_s6_is_answered_correctly_when_the_embedding_provider_is_semantic(output):
    if "hashing_fallback" in get_provider_name():
        pytest.skip("hashing stub is not semantic")
    r = by_id(output)["S6"]
    assert r["probes"]["first_relevant_rank"]["semantic"] == 1             # the provider alone finds it...
    assert r["success"] and r["returned_files"][0] == "alice/solar_install_quote.txt"   # ...and the agent no longer loses it


def test_cross_user_isolation_still_holds_for_every_query_and_diagnostic(output):
    for r in output["results"] + output["diagnostics"]["results"]:
        assert r["scope_violations"] == 0, r["qid"]
        assert all(f.startswith(r["user"] + "/") for f in r["returned_files"] + r["pool_files"]), r["qid"]


# ============================================================ evaluation additions
def test_every_record_carries_standalone_probes_for_each_retrieval_stage(output):
    for r in output["results"]:
        assert set(r["probes"]["top"]) >= {"bm25", "semantic", "hybrid", "rerank_of_semantic", "bm25_top_coverage"}, r["qid"]
        assert set(r["probes"]["first_relevant_rank"]) == {"bm25", "semantic", "hybrid", "rerank_of_semantic"}
        assert r["layer"]["layer"] and (r["layer"]["layer"] == "ok") == r["success"], r["qid"]


def test_provider_failures_and_agent_failures_are_attributed_to_different_layers(output):
    for r in output["results"]:
        layer = r["layer"]["layer"]
        if layer == "semantic_provider":
            assert r["probes"]["first_relevant_rank"]["semantic"] != 1 and r["expect_semantic"], r["qid"]    # semantic ALONE was wrong
        if layer in ("agent_decision", "reranker", "hybrid", "final_ranking", "personalization"):
            path = "semantic" if r["expect_semantic"] else "bm25"
            assert r["probes"]["first_relevant_rank"][path] == 1, r["qid"]                                    # the provider had it right


def _rec(**kw):
    base = {"success": False, "scope_violations": 0, "expected_outcome": "results", "goal_blocks_relevant": False,
            "expected_files": ["alice/a.txt"], "expect_bm25": False, "expect_semantic": True,
            "tool_counts": {"semantic_search": 1, "bm25_search": 1, "hybrid_search": 0, "rerank_candidates": 0, "get_user_activity": 0, "metadata_search": 1},
            "probes": {"top": {"semantic": [["alice/a.txt", 1.0]], "bm25": []}, "first_relevant_rank": {"semantic": 1, "bm25": None}},
            "counterfactual": {"final": ["alice/b.txt", "alice/a.txt"], "retrieval_only": ["alice/a.txt"], "without_rerank": ["alice/a.txt"],
                               "without_personalization": ["alice/b.txt"], "without_hybrid": ["alice/b.txt"]},
            "returned_files": ["alice/b.txt"], "pool_files": ["alice/a.txt", "alice/b.txt"], "top_evidence_source": {}}
    base.update(kw)
    return base


def test_attribute_layer_rules():
    A = runner.attribute_layer
    assert A(_rec(success=True)) == {"layer": "ok"}
    # semantic ALONE ranks the right file below another -> the provider, never the agent
    wrong = {"top": {"semantic": [["alice/z.txt", 1.0], ["alice/a.txt", 0.99]], "bm25": []}, "first_relevant_rank": {"semantic": 2, "bm25": None}}
    assert A(_rec(probes=wrong))["layer"] == "semantic_provider"
    # provider right, but the agent never consulted it
    assert A(_rec(tool_counts={**_rec()["tool_counts"], "semantic_search": 0}))["layer"] == "agent_decision"
    # provider right, consulted, retrieval ordering right, rerank flips it
    assert A(_rec())["layer"] == "reranker"
    # provider right, consulted, hybrid is what breaks it
    cf = {"final": ["alice/b.txt"], "retrieval_only": ["alice/a.txt"], "without_rerank": ["alice/b.txt"], "without_personalization": ["alice/b.txt"], "without_hybrid": ["alice/a.txt"]}
    assert A(_rec(counterfactual=cf, tool_counts={**_rec()["tool_counts"], "hybrid_search": 1}))["layer"] == "hybrid"
    # no-result queries
    assert A(_rec(expected_outcome="no_result", expected_files=[]))["layer"] == "no_relevance_floor"
    assert A(_rec(expected_outcome="no_result", expected_files=[], tool_counts={**_rec()["tool_counts"], "semantic_search": 0}))["layer"] == "lexical_false_positive"
    assert A(_rec(goal_blocks_relevant=True))["layer"] == "goal" and A(_rec(scope_violations=1))["layer"] == "isolation"


def test_coverage_sensitivity_counts_correct_and_wrong_hits_per_boundary():
    recs = [{"qid": q, "expected_files": ["alice/ok.txt"], "probes": {"top": {"bm25": [[lab, 1.0]], "bm25_top_coverage": cov}}}
            for q, lab, cov in (("a", "alice/ok.txt", 1.0), ("b", "alice/ok.txt", 0.5), ("c", "alice/bad.txt", 0.25), ("d", "alice/bad.txt", 0.75))]
    recs.append({"qid": "e", "expected_files": [], "probes": {"top": {"bm25": [], "bm25_top_coverage": None}}})   # no hit: ignored
    s = runner.coverage_sensitivity(recs)["by_boundary"]
    assert s["0.5"] == {"correct_flagged_weak": 0, "wrong_flagged_weak": 1, "n_correct": 2, "n_wrong": 2}
    assert s["0.6"]["correct_flagged_weak"] == 1 and s["0.3"]["wrong_flagged_weak"] == 1 and s["0.8"]["wrong_flagged_weak"] == 2


def test_committed_sensitivity_shows_the_shipped_boundary_is_at_the_edge_of_the_safe_region(output):
    sens = output["summary"]["coverage_sensitivity"]["by_boundary"]
    assert sens["0.5"]["correct_flagged_weak"] == 0 and sens["0.5"]["wrong_flagged_weak"] >= 1
    assert sens["0.6"]["correct_flagged_weak"] > 0, "a boundary above one half would start rejecting correct lexical hits"


# ============================================================ diagnostics are separate and verify the personalization path
def test_diagnostics_are_separate_from_the_frozen_queries_and_overall_metrics(output):
    assert len(output["results"]) == 31 and {r["qid"] for r in output["diagnostics"]["results"]} == {"D1"}
    assert "D1" not in {q.qid for q in QUERIES} and output["summary"]["overall"]["retrieval"]["n"] == 31
    assert validate(CORPUS) == [] and len(DIAGNOSTIC_QUERIES) == 1


def test_the_existing_personalization_path_is_reachable_on_real_content(output):
    d1 = output["diagnostics"]["results"][0]
    assert d1["tool_counts"]["get_user_activity"] >= 1                     # metadata did NOT dominate: personalization ran
    assert d1["tool_sequence"][0] == "metadata_search" and d1["scope_violations"] == 0
    assert set(d1["returned_files"]) <= {"alice/thai_curry_guide.txt", "alice/recipe_pasta.md"}


def test_frozen_phase11_ground_truth_was_not_edited():
    assert len(QUERIES) == 31
    assert Q["S3"].relevant == {"lease_agreement_summary.docx": 2} and Q["S6"].relevant == {"solar_install_quote.txt": 2}
    assert Q["P1"].relevant == {"thai_curry_guide.txt": 2, "recipe_pasta.md": 1}


def test_evaluation_with_probes_is_deterministic(output):
    again = runner.run_content_eval()
    assert again["results"] == output["results"] and again["summary"] == output["summary"]
    assert again["diagnostics"] == output["diagnostics"]


# ============================================================ honesty about the providers actually used
def test_the_report_states_exactly_which_providers_ran_and_whether_production_models_were_evaluated(output):
    from app.embeddings import PREFERRED_MODEL_NAME as EMB
    from app.reranker import PREFERRED_MODEL_NAME as RR
    m = output["metadata"]["production_models"]
    assert m["embedding"]["intended"] == EMB == "BAAI/bge-small-en-v1.5" and m["reranker"]["intended"] == RR == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert m["embedding"]["used"] == output["metadata"]["embedding_provider"] and m["reranker"]["used"] == output["metadata"]["reranker_provider"]
    both = m["embedding"]["intended_model_used"] and m["reranker"]["intended_model_used"]
    assert (m["production_model_evaluation"] == "available") == both
    if not both:
        assert m["production_model_evaluation"].startswith("BLOCKED")
    from evaluation.content.report import render_markdown
    assert "Production-model evaluation" in render_markdown(output)


def test_a_fallback_provider_is_never_reported_as_the_intended_model():
    info = {"embedding_provider": "spacy:en_core_web_md (sandbox fallback)", "reranker_provider": "lexical_overlap_fallback (NOT a cross-encoder)"}
    s = runner.production_model_status(info)
    assert not s["embedding"]["intended_model_used"] and not s["reranker"]["intended_model_used"]
    assert s["production_model_evaluation"].startswith("BLOCKED")
    ok = runner.production_model_status({"embedding_provider": "sentence-transformers:BAAI/bge-small-en-v1.5",
                                         "reranker_provider": "cross-encoder:cross-encoder/ms-marco-MiniLM-L-6-v2"})
    assert ok["production_model_evaluation"] == "available"


# ============================================================ semantic relevance floor: evidence, not a threshold
def test_semantic_floor_evidence_is_computed_from_raw_cosines_and_never_sets_a_threshold():
    mk = lambda outcome, top, rel: {"expected_outcome": outcome, "expected_files": rel, "raw_semantic_top": [[top[0], top[1]]]}  # noqa: E731
    overlapping = [mk("no_result", ("alice/x", 0.77), []), mk("results", ("alice/a", 0.73), ["alice/a"]), mk("results", ("alice/b", 0.88), ["alice/b"]),
                   mk("results", ("alice/z", 0.50), ["alice/c"])]
    e = runner.semantic_floor_evidence(overlapping)
    assert e["ranges_overlap"] is True and e["floor_supported"] is False and e["n"] == {"no_relevant": 1, "correct": 2, "wrong": 1}
    separable = [mk("no_result", ("alice/x", 0.40), []), mk("results", ("alice/a", 0.73), ["alice/a"])]
    assert runner.semantic_floor_evidence(separable)["floor_supported"] is True
    assert runner.semantic_floor_evidence([])["floor_supported"] is False


def test_with_this_provider_a_floor_is_not_supported_and_none_was_added(output):
    from app import config
    e = output["summary"]["semantic_floor_evidence"]
    assert e["n"]["no_relevant"] == 4 and e["n"]["correct"] >= 1
    assert e["ranges_overlap"] == (e["no_relevant_top1_raw_cosine"][1] >= e["correct_top1_raw_cosine"][0])
    if "hashing_fallback" not in get_provider_name():
        assert e["floor_supported"] is False                              # measured overlap -> no defensible boundary
    assert not any("FLOOR" in k.upper() or "MIN_SIMILARITY" in k.upper() for k in vars(config))   # and no such knob exists
    nr = [r for r in output["results"] if r["category"] == "N_no_result"]
    assert all(r["layer"]["layer"] == "no_relevance_floor" for r in nr)    # unchanged, documented behaviour
