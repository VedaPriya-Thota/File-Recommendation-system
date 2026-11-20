"""Test bootstrap: when no PostgreSQL is configured, run the whole suite against a fresh, seeded,
throw-away SQLite database and Qdrant directory (so tests never touch the user's data or the bundled
file_discovery.db). With DISCOVERY_DB_TYPE=postgres the suite uses PostgreSQL exactly as before."""
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="fds_tests_")
if os.environ.get("DISCOVERY_DB_TYPE", "").lower() != "postgres":
    os.environ.setdefault("DISCOVERY_DB_TYPE", "sqlite")
    os.environ["DISCOVERY_SQLITE_PATH"] = os.path.join(_TMP, "test.db")
os.environ.setdefault("DISCOVERY_QDRANT_PATH", os.path.join(_TMP, "qdrant"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def pytest_sessionstart(session):
    from app.db.init_db import init_db, ensure_app_schema
    init_db(with_seed=True)
    ensure_app_schema()


PG_ONLY = {
    "tests/test_agent_loop.py::test_F_postgres_outage_during_metadata_search_is_also_a_hard_failure",
    "tests/test_agent_loop.py::test_F_postgres_outage_during_scope_resolution_is_a_hard_failure",
    "tests/test_evaluation.py::test_10_cross_user_isolation_in_every_variant",
    "tests/test_evaluation.py::test_10b_scope_audit_detects_a_leak",
    "tests/test_evaluation.py::test_11_variant_configuration_is_wired_as_documented",
    "tests/test_evaluation.py::test_11c_personalization_ablation_changes_only_ranking_evidence",
    "tests/test_evaluation.py::test_13b_llm_counts_in_real_runs",
    "tests/test_evaluation.py::test_14_tool_call_counting_matches_committed_actions",
    "tests/test_evaluation.py::test_15_output_schema_and_privacy",
    "tests/test_evaluation.py::test_15b_latency_is_opt_in_and_labelled",
    "tests/test_evaluation.py::test_16_reproducibility_same_inputs_same_records",
    "tests/test_evaluation.py::test_16b_fixture_cleans_up_and_refuses_to_clobber",
    "tests/test_evaluation.py::test_17_full_benchmark_completes_and_is_internally_consistent",
    "tests/test_evaluation.py::test_8b_real_no_result_query_returns_nothing",
    "tests/test_evaluation.py::test_9b_clarified_real_query_is_scored_zero_not_dropped",
    "tests/test_evaluation.py::test_explanation_text_cannot_influence_retrieval_metrics",
    "tests/test_phase10_ingestion.py::test_a_database_failure_is_a_hard_failure_not_a_silent_skip",
    "tests/test_phase10_ingestion.py::test_update_preserves_user_curated_tags_and_summary",
    "tests/test_phase11_content_eval.py::test_all_ingested_files_were_indexed_through_the_real_pipeline",
    "tests/test_phase11_content_eval.py::test_cross_user_isolation_holds_for_every_query",
    "tests/test_phase11_content_eval.py::test_evaluation_completes_with_the_documented_schema_and_providers",
    "tests/test_phase11_content_eval.py::test_evaluation_is_deterministic",
    "tests/test_phase11_content_eval.py::test_every_attribution_is_consistent_with_the_outcome",
    "tests/test_phase11_content_eval.py::test_goal_probes_all_succeed_after_the_whole_token_fix",
    "tests/test_phase11_content_eval.py::test_lexical_paths_work_on_real_content",
    "tests/test_phase11_content_eval.py::test_no_hard_failures_no_llm_calls_and_nothing_left_behind",
    "tests/test_phase12_content_eval.py::test_committed_sensitivity_shows_the_shipped_boundary_is_at_the_edge_of_the_safe_region",
    "tests/test_phase12_content_eval.py::test_cross_user_isolation_still_holds_for_every_query_and_diagnostic",
    "tests/test_phase12_content_eval.py::test_diagnostics_are_separate_from_the_frozen_queries_and_overall_metrics",
    "tests/test_phase12_content_eval.py::test_evaluation_with_probes_is_deterministic",
    "tests/test_phase12_content_eval.py::test_every_record_carries_standalone_probes_for_each_retrieval_stage",
    "tests/test_phase12_content_eval.py::test_provider_failures_and_agent_failures_are_attributed_to_different_layers",
    "tests/test_phase12_content_eval.py::test_semantic_s6_is_answered_correctly_when_the_embedding_provider_is_semantic",
    "tests/test_phase12_content_eval.py::test_strong_lexical_queries_are_not_escalated_so_the_agent_is_not_a_fixed_pipeline",
    "tests/test_phase12_content_eval.py::test_the_existing_personalization_path_is_reachable_on_real_content",
    "tests/test_phase12_content_eval.py::test_the_report_states_exactly_which_providers_ran_and_whether_production_models_were_evaluated",
    "tests/test_phase12_content_eval.py::test_the_weak_lexical_candidate_never_claims_more_confidence_than_it_explains",
    "tests/test_phase12_content_eval.py::test_weak_lexical_hits_no_longer_preempt_semantic_retrieval_on_the_real_corpus",
    "tests/test_phase12_content_eval.py::test_with_this_provider_a_floor_is_not_supported_and_none_was_added",
    "tests/test_phase9_comparison_gate.py::test_real_corpus_personalized_pool_is_not_flipped_by_an_opposing_5c",
}


def pytest_collection_modifyitems(config, items):
    """These tests exercise PostgreSQL-only SQL (jsonb/array operators, psycopg2 outage simulation). They run
    unchanged with DISCOVERY_DB_TYPE=postgres; under the SQLite fallback they are skipped, never silently passed."""
    if os.environ.get("DISCOVERY_DB_TYPE", "").lower() == "postgres":
        return
    import pytest
    skip = pytest.mark.skip(reason="requires PostgreSQL-specific SQL (run with DISCOVERY_DB_TYPE=postgres)")
    for item in items:
        if item.nodeid in PG_ONLY:
            item.add_marker(skip)
