# Phase 7 — Evaluation benchmark + ablation study

Measures the existing system from the outside (`benchmark -> run_query() -> metrics`).
Nothing under `app/` imports this package, and no production code was changed.

## Run

```bash
python -m app.db.init_db          # normal project setup: schema + seed corpus
python -m evaluation              # writes evaluation/output/benchmark_{results.json,report.md}
python -m pytest tests/test_evaluation.py
```

Options: `--output-dir DIR`, `--latency` (adds environment-specific timings, off by default),
`--qdrant-path DIR` (default: fresh temp dir), `--init-db` (**destructive** re-seed).
The runner refuses to start if the database does not match the seed corpus the ground
truth was written against, and if `user_a` already has telemetry rows.

Fully offline: LLM stages use scripted local providers, `HF_HUB_OFFLINE=1` is set so an
uncached embedding/reranker model falls back immediately instead of downloading. The
report records which embedding/reranker provider actually loaded — a fallback is **not**
the intended production model.

## Layout

| file | role |
|---|---|
| `benchmark_dataset.py` | 31 hand-labelled queries (IDs 1–13 from `app/db/seed.py`), personalization telemetry fixture, validators |
| `metrics.py` | P@1/3/5, R@1/3/5, MRR, nDCG@5, success rule (pure functions) |
| `variants.py` | ablation variants, expressed only via `run_query(max_iterations, registry, llm_provider)` |
| `providers.py` | scripted offline LLM provider + call-counting wrapper |
| `runner.py` | executes (query × variant), audits scope, collects per-run records |
| `scenarios.py` | behavioural scenario checks (reported apart from retrieval metrics) |
| `report.py` | aggregation (overall / by category / by variant / behaviour) + markdown |

## Variants

| variant | what differs |
|---|---|
| `A_single_shot_metadata` | `max_iterations=1`: one `metadata_search`, no replanning |
| `B_agentic_deterministic` | full loop + all retrieval/rerank tools; no LLM; personalization disabled |
| `C_agentic_personalized` | B + real `get_user_activity` over the telemetry fixture |
| `D_full_system` | C + scripted goal/expansion/comparison/explanation providers |
| `D_llm_unavailable`, `D_llm_invalid_output`, `D_no_explanation` | scenario variants (outage / garbage / 5D off) |
| `B/C/D_cap3_diagnostic` | `max_iterations=3`: forces flat pools to finish. **Diagnostic only** — exposes finish-time stages; not a proposed default |

Personalization is disabled by swapping `get_user_activity` for a pass-through tool
returning affinity 0.0, so control flow is identical between B and C.

## Reading the numbers

* Success rule (single definition): `results` → top-1 relevant; `clarify_or_results` →
  clarified or top-1 relevant; `no_result` → nothing returned; always AND zero scope violations.
* A clarified query returns **nothing** and scores 0 on P/R/MRR (never dropped). The
  *abstention-aware* table (answered-only, pool-ranking diagnostic) separates "bad evidence"
  from "chose to ask". `pool_ranking_*` is how `final_rank()` would order the pool — a
  diagnostic, not what the system returned.
* Queries with no relevant files have undefined P/R/MRR (`null`); they are judged by the
  success rule and counted in `n` but not `n_scored`.
* `llm_attempts` counts `complete()` invocations; `llm_calls` counts those that returned a
  response (billable-equivalent). A `NullProvider` yields attempts > 0, calls = 0.
* A failing scenario is a **finding**, not a harness error. Scenario checks of kind
  `observation` are recorded but do not gate pass/fail.

## Limitations

See `metadata.limitations` in the JSON. Chiefly: the seed corpus has no `extracted_text`
(semantic index is empty), no telemetry (a fixture is used), and 13 files / 31 queries is
small — treat rates as descriptive.
