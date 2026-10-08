# Phase 11 — real-content evaluation

Measures whether the *complete agent* returns the right **real file** (ingested through the Phase 10
pipeline), and where a correct answer becomes incorrect. It is **separate** from the Phase 7/8/9 regression
benchmark (`python -m evaluation`, 310 records), which it never touches.

```bash
python -m app.db.init_db        # PostgreSQL schema + seed (the evaluation only needs the DB to exist)
python -m evaluation.content    # writes evaluation/content/output/content_eval_results.{json,md}
python -m pytest tests/test_phase11_content_eval.py tests/test_phase11_goal.py
```

Offline and deterministic: no LLM (`NullProvider`), an isolated in-memory Qdrant, users `p11_alice`/`p11_bob`
created and removed per run. The report records the embedding and reranker providers actually used. **A
`lexical_overlap_fallback` reranker is not a CrossEncoder and `spacy:` embeddings are the sandbox fallback; numbers
from such an environment do not represent production-model quality.**

## Design

| | |
|---|---|
| Corpus | `corpus_spec.py` is the single source of text; `corpus/` holds the committed files (txt/md/pdf/docx) made by `make_corpus.py`. 29 files for `alice`, 3 for `bob` (isolation). |
| Queries | `queries.py`: 31 hand-labelled queries in 8 categories (filename, content-only, semantic, goal probes, ambiguous, personalization, no-result, cross-user). **Labels were written and frozen before the first measurement run**; ground truth is `(owner, relative path)` of fixture files only. |
| Success rule | the single Phase 7 rule (`evaluation.metrics.judge_success`). |
| Stage data | per query: goal, expansion, agent actions, and each source's own ordering (metadata / BM25 / semantic / hybrid / rerank) taken from the agent's candidate pool, plus counterfactual rankings from the production `final_rank()` with evidence removed (without rerank, without personalization, retrieval-only). |
| Attribution | `runner.attribute()` (pure): `goal`, `retrieval_miss`, `retrieval_ordering`, `reranker`, `personalization`, `final_ranking`, `stopping`, `post_ranking`, `false_positive`, `isolation`. "Semantic wrong" and "semantic right, reranker wrong" are reported as different verdicts. |
| Diagnostic | `raw_semantic_top`: raw cosine of the best chunk per file, queried from Qdrant outside production (the tool hides it by max-normalising). |

## Outputs

- `output/phase11_results.*` — the Phase 11 result (reference for the Phase 12 comparison).
- `output/content_eval_results.*` — current code, now with standalone probes, layer attribution, boundary sensitivity, semantic-floor evidence and diagnostics.

- `output/content_eval_before_fixes.*` — Phase 10 production code.
- `output/content_eval_results.*` — current code (goal fix).
- `output/experiments/` — a reranker fix that was **evaluated and deliberately not shipped** (patch + results + reasons).

## Known limitations of this evaluation

- Small (31 queries, 32 files): one query moves a rate by ~3 points; treat rates as descriptive.
- The personalization queries did **not** exercise personalization (a lexical filename match made `metadata_search`
  dominant, so `get_user_activity` ran 0 times); personalization's contribution on real content is unmeasured here.
- Queries were not edited after seeing results, including those that turned out to be poor probes.
- `DIAGNOSTIC_QUERIES` (D1) were pre-registered in Phase 12 and are reported separately; they are never part of the 31 frozen queries or any overall metric.
- Production-model evaluation (BGE embeddings, CrossEncoder) is BLOCKED in this environment; `metadata.production_models` records this explicitly.
