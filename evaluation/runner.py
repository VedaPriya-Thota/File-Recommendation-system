"""
Benchmark runner. Sits OUTSIDE the production path:

    benchmark -> run_query() -> collect result + telemetry -> metrics

Nothing here is imported by app/*. It only reads what `run_query()` already
returns (status, final_ranking, committed_actions, candidate_pool, trace, LLM
metadata) plus counters from the wrapping `CountingProvider`.
"""
import json
import time
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Sequence

from app.config import AMBIGUITY_CLARIFY_THRESHOLD, DOMINANT_SCORE_GAP, COMPARISON_TOP_N
from app.db.connection import get_cursor
from app.final_ranking import final_rank
from app.run import run_query

from evaluation import benchmark_dataset as ds
from evaluation.metrics import judge_success, retrieval_metrics
from evaluation.providers import STAGES
from evaluation.variants import VARIANTS, Variant, build_variant_provider, build_variant_registry

SCHEMA_VERSION = 1
RETRIEVAL_TOOLS = ("metadata_search", "bm25_search", "semantic_search", "hybrid_search")
TRACKED_TOOLS = (*RETRIEVAL_TOOLS, "rerank_candidates", "get_user_activity")


class EvaluationSetupError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Environment setup
# --------------------------------------------------------------------------
def check_database() -> None:
    """Static dataset validation + live DB check. Raises EvaluationSetupError."""
    problems = ds.validate_dataset()
    if problems:
        raise EvaluationSetupError("benchmark dataset invalid: " + "; ".join(problems))
    with get_cursor() as cur:
        cur.execute("SELECT file_id, filename, owner_id FROM files ORDER BY file_id;")
        rows = [dict(r) for r in cur.fetchall()]
    problems = ds.validate_against_db(rows)
    if problems:
        raise EvaluationSetupError(
            "database does not match the seed corpus the benchmark was written against "
            "(run `python -m app.db.init_db`): " + "; ".join(problems)
        )


def rebuild_semantic_index() -> dict:
    """Uses the existing public API so the (possibly empty) index reflects the DB."""
    from app.tools.semantic_search import rebuild_index

    return rebuild_index(force=True)


def provider_info() -> dict:
    info = {}
    try:
        from app.embeddings import get_provider_name
        info["embedding_provider"] = get_provider_name()
    except Exception as exc:  # noqa: BLE001
        info["embedding_provider"] = f"unavailable ({exc})"
    try:
        from app.reranker import get_reranker_provider_name
        info["reranker_provider"] = get_reranker_provider_name()
    except Exception as exc:  # noqa: BLE001
        info["reranker_provider"] = f"unavailable ({exc})"
    return info


@contextmanager
def personalization_fixture():
    """Inserts PERSONALIZATION_FIXTURE telemetry/profile rows for the run and
    removes exactly those rows afterwards (even on error). Refuses to run if
    the fixture users already have rows, so it can never clobber real data."""
    inserted_events: List[int] = []
    inserted_profiles: List[str] = []
    with get_cursor() as cur:
        for user in ds.PERSONALIZATION_FIXTURE:
            cur.execute("SELECT count(*) AS c FROM user_file_telemetry WHERE user_id = %s;", (user,))
            n_ev = cur.fetchone()["c"]
            cur.execute("SELECT count(*) AS c FROM user_profiles WHERE user_id = %s;", (user,))
            n_pr = cur.fetchone()["c"]
            if n_ev or n_pr:
                raise EvaluationSetupError(
                    f"{user} already has telemetry/profile rows; refusing to overwrite. "
                    "Re-seed with `python -m app.db.init_db` first."
                )
    try:
        with get_cursor(commit=True) as cur:
            for user, spec in ds.PERSONALIZATION_FIXTURE.items():
                cur.execute(
                    "INSERT INTO user_profiles (user_id, preferred_file_types, frequent_topics) "
                    "VALUES (%s, %s, %s::jsonb);",
                    (user, spec["profile"]["preferred_file_types"], json.dumps(spec["profile"]["frequent_topics"])),
                )
                inserted_profiles.append(user)
                for file_id, events in spec["events"].items():
                    for action, seconds, days_ago in events:
                        cur.execute(
                            "INSERT INTO user_file_telemetry (user_id, file_id, action_type, time_spent_seconds, timestamp) "
                            "VALUES (%s, %s, %s, %s, now() - make_interval(days => %s)) RETURNING event_id;",
                            (user, file_id, action, seconds, days_ago),
                        )
                        inserted_events.append(cur.fetchone()["event_id"])
        yield
    finally:
        with get_cursor(commit=True) as cur:
            if inserted_events:
                cur.execute("DELETE FROM user_file_telemetry WHERE event_id = ANY(%s);", (inserted_events,))
            for user in inserted_profiles:
                cur.execute("DELETE FROM user_profiles WHERE user_id = %s;", (user,))


# --------------------------------------------------------------------------
# One (query, variant) run
# --------------------------------------------------------------------------
def _round(x, n=4):
    return None if x is None else round(float(x), n)


def run_one(query: ds.BenchmarkQuery, variant: Variant, include_latency: bool = False) -> dict:
    provider = build_variant_provider(variant)
    registry = build_variant_registry(variant)

    t0 = time.perf_counter()
    result = run_query(
        query.user_id, query.query,
        max_iterations=variant.max_iterations, registry=registry, llm_provider=provider,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    hard_failure = result.get("status") == "hard_failure"
    ranking = result.get("final_ranking") or []
    returned = [r["file_id"] for r in ranking]
    status = result.get("status")
    clarified = status == "clarify"
    pool = result.get("candidate_pool") or {}
    actions = result.get("committed_actions") or []
    sequence = [a["tool"] for a in actions]

    # scope audit: anything returned or pooled that is not the caller's own file
    seen_ids = set(returned) | set(pool)
    violations = sorted(
        {fid for fid in seen_ids if fid in query.forbidden_file_ids}
        | {fid for fid, e in pool.items() if (e.get("metadata") or {}).get("owner_id") not in (None, query.user_id)}
        | {r["file_id"] for r in ranking if (r.get("metadata") or {}).get("owner_id") not in (None, query.user_id)}
    )

    goal_entry = next((t for t in result.get("trace", []) if t.get("event") == "understand_goal"), {})
    ambiguity = (goal_entry.get("goal") or {}).get("ambiguity_score")

    goal_meta = result.get("llm_goal_metadata") or {}
    exp_meta = result.get("query_expansion_metadata") or {}
    cmp_meta = result.get("candidate_comparison") or {}
    expl_meta = result.get("candidate_explanation") or {}

    metrics = retrieval_metrics(returned, query.relevance)
    # DIAGNOSTIC (not what the system returned): how the existing deterministic final_rank() would order the
    # accumulated candidate pool. Separates "quality of the evidence/ranking" from "decision to abstain".
    pool_ranking = [e["file_id"] for e in final_rank(pool)] if pool else []
    pool_metrics = retrieval_metrics(pool_ranking, query.relevance)
    hr = set(query.highly_relevant_ids)  # graded view: is the INTENDED (grade-2) file ranked first?
    hr_at_1 = (float(bool(returned) and returned[0] in hr)) if hr else None
    pool_hr_at_1 = (float(bool(pool_ranking) and pool_ranking[0] in hr)) if hr else None
    rec = {
        "query_id": query.query_id,
        "query": query.query,
        "user_id": query.user_id,
        "category": query.category,
        "variant": variant.name,
        "expected_outcome": query.expected_outcome,
        "personalization_context": query.personalization_context,
        "expected_file_ids": query.relevant_ids,
        "highly_relevant_file_ids": query.highly_relevant_ids,
        "forbidden_file_ids": list(query.forbidden_file_ids),
        "returned_file_ids": returned,
        "pool_file_ids": sorted(pool),
        "pool_ranking_file_ids": pool_ranking,
        "highly_relevant_at_1": hr_at_1,
        "pool_ranking_highly_relevant_at_1": pool_hr_at_1,
        "pool_ranking_first_relevant_rank": pool_metrics["first_relevant_rank"],
        "pool_ranking_reciprocal_rank": _round(pool_metrics["reciprocal_rank"]),
        **{k: (_round(v) if k != "first_relevant_rank" else v) for k, v in metrics.items()},
        "success": False if hard_failure else judge_success(
            query.expected_outcome, returned, clarified, query.relevance, len(violations)),
        "status": status,
        "clarified": clarified,
        "no_result": len(returned) == 0,
        "degraded_mode": bool(result.get("degraded_mode")),
        "hard_failure": hard_failure,
        "confidence": _round(result.get("confidence")),
        "scope_violations": len(violations),
        "scope_violation_file_ids": violations,
        "tool_calls": len(actions),
        "retrieval_actions": sum(1 for t in sequence if t in RETRIEVAL_TOOLS),
        "tool_counts": {t: sequence.count(t) for t in TRACKED_TOOLS},
        "tool_sequence": sequence,
        "iterations": result.get("iteration_count", 0),
        "goal_ambiguity_score": ambiguity,
        "llm_calls": provider.total_calls,
        "llm_attempts": provider.total_attempts,
        "llm_calls_by_stage": {s: provider.calls[s] for s in STAGES},
        "llm_attempts_by_stage": {s: provider.attempts[s] for s in STAGES},
        "llm_stage_status": {
            "goal": goal_meta.get("llm_status"),
            "expansion": exp_meta.get("expansion_status"),
            "comparison": cmp_meta.get("comparison_status"),
            "comparison_applied": bool(cmp_meta.get("comparison_applied")),
            "explanation": expl_meta.get("explanation_status"),
        },
        "top3_final_scores": [_round(r.get("final_score"), 3) for r in ranking[:3]],
        "error_count": len(result.get("errors") or []),
    }
    if include_latency:
        rec["latency_ms_environment_specific"] = round(elapsed_ms, 1)
    return rec


# --------------------------------------------------------------------------
# Whole benchmark
# --------------------------------------------------------------------------
def run_benchmark(
    variants: Optional[Sequence[Variant]] = None,
    queries: Optional[Sequence[ds.BenchmarkQuery]] = None,
    include_latency: bool = False,
    rebuild_index: bool = True,
) -> dict:
    from evaluation.report import build_summary
    from evaluation.scenarios import evaluate_scenarios

    variants = list(variants if variants is not None else VARIANTS)
    queries = list(queries if queries is not None else ds.QUERIES)
    check_database()
    index_stats = rebuild_semantic_index() if rebuild_index else None

    records: List[dict] = []
    with personalization_fixture():
        for q in queries:
            for v in variants:
                records.append(run_one(q, v, include_latency=include_latency))

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset_size": len(queries),
        "categories": sorted({q.category for q in queries}),
        "variants": [{"name": v.name, "description": v.description, "core": v.core} for v in variants],
        "semantic_index": index_stats,
        "thresholds_in_effect": {
            "AMBIGUITY_CLARIFY_THRESHOLD": AMBIGUITY_CLARIFY_THRESHOLD,
            "DOMINANT_SCORE_GAP": DOMINANT_SCORE_GAP,
            "COMPARISON_TOP_N": COMPARISON_TOP_N,
        },
        "latency_note": "latency fields (if present) are environment-specific and not comparable across machines",
        **provider_info(),
        "limitations": LIMITATIONS,
    }
    return {
        "metadata": metadata,
        "results": records,
        "summary": build_summary(records, queries),
        "scenarios": evaluate_scenarios(records),
    }


LIMITATIONS = [
    "Seed corpus has no extracted_text: the semantic index is empty, so semantic_search and the semantic half of hybrid_search cannot contribute evidence here.",
    "Seed corpus has no telemetry/profiles: personalization uses the explicit PERSONALIZATION_FIXTURE (user_a only); its 'intended file' labels are defined by that fixture, so E-category personalization gains are by construction a test of the mechanism, not of real-user value.",
    "No two users share an accessible file, so personalization is ablated within-user (history vs. disabled), not across users.",
    "Variant D uses a scripted provider with documented simple policies; it measures architecture behaviour under controlled LLM assistance, not real-model quality.",
    "Embedding/reranker providers are whatever the environment resolves (see embedding_provider / reranker_provider); a sandbox fallback is NOT the intended production model.",
    "Small corpus (13 files) and 31 queries: differences of a single query are within noise; treat rates as descriptive.",
]


def write_results(output: dict, path: str) -> None:
    import json
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
