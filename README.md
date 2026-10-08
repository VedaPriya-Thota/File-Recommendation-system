# Project 5 — Agentic AI-based Intelligent File Discovery

A natural-language file-discovery agent. Given a goal, a bounded LangGraph loop chooses a retrieval
strategy, observes what it returned, and changes strategy mid-query (replan / finish / clarify). Retrieval,
scoring and access control are deterministic; LLMs are optional, narrow, and never mandatory.

## Architecture (through Phase 10)

```text
 local files (--root)                                              user query
      │                                                                │
  LocalDirectorySource ── discovery: metadata only, no reads           ▼
      │                                         resolve_scope ─ immutable user_scope, applied inside every query
  ContentExtractor  (.txt .md .pdf .docx)                              ▼
      │                                         understand_goal ──(5A optional LLM interpretation)
  normalize_text  (NFKC, line endings, whitespace)                     ▼
      │                                         ┌──────────── agent loop (LangGraph, bounded) ────────────┐
      ▼                                         │ plan_next_action ─► execute_tool ─► candidate_pool      │
 PostgreSQL: files.extracted_text               │   (5B optional query expansion)        │                │
          + file_ingestion_state                │                       reflect ◄────────┘                │
      │                                         │  replan │ finish (basis recorded) │ clarify             │
      ├─► BM25 (bm25s; filename×3+tags×2+summary+text)  └──────────────────────────────────────────────────┘
      └─► chunks ─► embeddings ─► Qdrant                               ▼
                    (payload carries owner_id)   tools: metadata_search · bm25_search · semantic_search ·
                                                         hybrid_search (RRF) · rerank_candidates · get_user_activity
                                                                       ▼
                                          final_rank (α·retrieval + γ·affinity, deterministic)
                                                                       ▼
                                  compare_stage (5C, near-ties only) ─► explain_stage (5D, presentation only)
```

Design rules that hold throughout: access control is a query-time predicate (SQL `WHERE owner_id = ANY(..)`,
Qdrant payload filter), never a post-filter; personalization breaks ties among candidates that already have
retrieval evidence and can never introduce one; LLM stages can never add candidates or change scope; a missing
or failing LLM, embedding model or Qdrant degrades to the deterministic path.

## Setup

```bash
pip install -r requirements.txt     # + the spaCy model noted there if the HF embedding model is unreachable
python3 -m app.db.init_db           # schema + seed corpus (needs PostgreSQL; see app/config.py for env overrides)
python3 -m pytest tests/            # full suite
python3 -m app.run user_a "Find the slide deck about AI architecture I edited last week."
```

## Real-file ingestion (Phase 10)

```bash
export DISCOVERY_INGEST_ROOT=/path/to/documents DISCOVERY_INGEST_OWNER_ID=<existing user_id>
python3 -m app.ingestion            # or: --root DIR --owner USER [--no-semantic]   (re-run any time: it is incremental)
```

Populates the tables the existing tools already read — no second retrieval path. Per file: new → extract →
index; unchanged (size+mtime) → skipped without reading; modified → re-extracted, old semantic points deleted
first; removed from disk → its row, telemetry and semantic points are removed (only for files ingestion itself
manages, and never if discovery was incomplete). A bad file (malformed PDF, unreadable, too large, embedding or
Qdrant outage) is recorded in `file_ingestion_state` with a short error code and retried next run; it never aborts
the run. The report holds counts and relative paths only — never document text or absolute paths.
Lifecycle states: `metadata_indexed → content_extracted → content_indexed`, or `unsupported` / `failed`.

## Layout

- `app/` — agent: `goal`, `planning`, `reflect`, `pool_resolution`, `final_ranking`, `candidate_pool`, `graph/build`, `run`
- `app/tools/` — retrieval tools (`metadata`, `bm25`, `semantic`, `hybrid`, `rerank`, `get_user_activity`) + registry
- `app/llm/` — optional 5A–5D stages behind a provider protocol (`NullProvider` by default; no network required)
- `app/ingestion/` — source, extraction, normalization, incremental pipeline, CLI
- `app/db/` — schema, seed corpus, connection helper
- `evaluation/` — Phase 7 benchmark/ablation framework (unchanged by Phase 10); `tests/` — unit + integration suites

## Known limitations

- Embeddings/reranker: if `sentence-transformers`/Hugging Face is unreachable the spaCy `en_core_web_md` fallback and a
  lexical-overlap reranker are used (real but weak). On the real-content evaluation the spaCy scores are nearly flat
  across documents: the semantic stage alone ranks the right file first for 3 of 6 paraphrase queries (S1/S2/S5 are
  provider failures), and the agent's final answer is right for 2 of 6 (S3 is lost afterwards to the lexical reranker).
- `semantic_search` has no relevance floor: any query returns nearest neighbours. Measured (Phase 11/12): raw top-1
  cosine of no-relevant queries (0.42-0.77) overlaps that of correct hits (0.61-0.88); no floor is supported.
- `rerank_candidates` min-max normalisation hands its best candidate 1.0 whatever its absolute strength, and 1.0 to
  *all* candidates when raw scores are identical; with the lexical fallback this can re-promote a weak lexical hit
  (S3). A zero-spread fix was evaluated and not shipped (`evaluation/content/output/experiments/`).
- The weak-lexical boundary ("less than half") rests on thin evidence: on the Phase 11 data any boundary in
  (0.25, 0.5] separates the 2 spurious from 18 correct lexical hits, and boundaries above 0.5 reject correct hits.
- BM25's stop-word list (`bm25s` "en") omits function words such as "when"/"so"/"can"; such a hit is now treated as
  weak evidence instead of ending the search.

## Phase 11 -- real-content evaluation

`python -m evaluation.content` measures the whole agent on ingested real files and attributes each failure to a
pipeline stage (see `evaluation/content/README.md`). One proven defect was fixed: file-type inference in
`understand_goal` matched keywords as bare substrings (`representations`→`presentation`→pptx, `doctor`→docx,
`photosynthesis`→png, `excellent`→xlsx, `landslides`→pptx), and the resulting hard filter removed every relevant
file; it now matches whole tokens (alphanumeric boundaries, optional plural), which raised the goal-probe success
from 1/5 to 5/5 with no other query changed and the 310-record regression benchmark unchanged.

## Phase 12 -- weak lexical evidence no longer preempts semantic retrieval

Phase 11 showed that for conceptual queries whose only lexical hit was a single function word or a distractor term
(`when`, `sunlight`), the agent stopped after BM25 although semantic retrieval alone ranked the right file first.
Cause: `bm25_search` divides scores by the query's best hit, so its top candidate is always 1.0, and a lone candidate
counts as dominant at any strength. The fix keeps the loop observation-driven (not a fixed pipeline):

- `bm25_search` also reports `coverage` -- the fraction of the query's distinct content terms the document contains
  (carried per source in `candidate_pool[...]["coverage"]`; metadata is replaced on merge, coverage is not).
- A lexical match covering a **minority** (< half) of the query is partial evidence: it counts as `score × coverage`
  (`final_ranking.retrieval_evidence_score`), and if it is a candidate's *only* retrieval evidence it is "weakly
  lexical" (`final_ranking.is_weak_lexical`; corroboration by metadata or semantic evidence makes it not weak).
- `reflect` will not stop on a weak lexical leader: it replans with the `weak_evidence` hint and the planner answers
  with `semantic_search` once (architecture §7: "bm25 weak -> semantic"). Strong lexical evidence still stops
  immediately. If no independent strategy remains, the agent finishes with the candidate flagged low-confidence
  (`decision_basis = weak_lexical_evidence`, confidence capped at its coverage) -- it never clarifies or discards it.

No ranking weight, threshold or LLM stage changed. Also fixed: `hybrid_search` ordered RRF-tied candidates by set
iteration order, which varied with database ids; ties now break by ascending `file_id`.

Evaluation: `python -m evaluation.content` now also probes each retrieval stage standalone and attributes every
failure to a layer (`semantic_provider` vs `agent_decision` vs `reranker` ...), reports the boundary sensitivity of the
coverage rule, whether a semantic relevance floor is supported by the data, and the production-model status.
**The intended BAAI/bge-small-en-v1.5 embedding model and cross-encoder reranker are unavailable in this environment
(no `sentence-transformers`, no cache, Hugging Face unreachable); every number here comes from the spaCy and
lexical-overlap fallbacks, and production-model quality remains unmeasured.**

## Evaluation (Phase 7)

```bash
python -m evaluation                          # benchmark + ablation -> evaluation/output/
python -m pytest tests/test_evaluation.py     # tests for the evaluation framework
```

Benchmark data lives in `evaluation/benchmark_dataset.py`; results are written to
`evaluation/output/benchmark_results.json` (+ `benchmark_report.md`). See `evaluation/README.md`.

## Phase 8 -- behavioural fixes found by the Phase 7 benchmark

Four production defects were fixed (no new features, no threshold/weight changes):

1. **Personalization-aware termination** (`app/pool_resolution.py` Rule A, hooked in `reflect()`): if retrieval
   evidence is flat but `get_user_activity` separates the final-ranking leader by an affinity margin of at least
   `DOMINANT_SCORE_GAP`, the agent finishes (a distinct, labelled path -- never counted as retrieval dominance).
2. **Clarify vs finish** (Rule B, used when no strategy is left): clarify only if retrieval evidence leaves the top
   candidates genuinely *tied* (and personalization did not resolve them); with a strict retrieval leader the agent
   finishes with the ranked shortlist, flagged low-confidence, instead of discarding it.
3. **Exact filenames** (`app/planning.py`): a filename with an extension typed in the user's own query is a hard
   constraint -- never relaxed away and never escalated to bm25/semantic. A miss is a safe "nothing found".
4. **Short tokens** (`app/tools/metadata_search.py`): tokens of <= `METADATA_SHORT_TOKEN_MAX_LEN` (3) characters match
   only as whole tokens, so `ai` no longer matches "campaign"; longer tokens keep substring matching.

Hardening: `update_candidate_pool` now refuses to let the personalization source create a pool entry.

Before/after benchmark: `python -m evaluation && python -m evaluation.compare` ->
`evaluation/output/phase8_vs_phase7.md` (Phase 7 baseline kept as `phase7_baseline_*`).

## Phase 9 -- Phase 5C no longer re-adjudicates a personalization-resolved decision

Phase 8 let personalization resolve a retrieval-flat pool, but 5C's near-tie gate looked only at final-score
gaps, and a personalization-resolved pool always has a gap below `DOMINANT_SCORE_GAP` (affinity is 15% of the
score) -- so the LLM comparison could reverse the deterministic decision. The loop now records *why* it finished
in `stopping_decision["basis"]` (also returned by `run_query` as `decision_basis`):

| basis | meaning | 5C |
|---|---|---|
| `retrieval_dominance` | one candidate clearly dominates on retrieval evidence | existing gate |
| `personalization_resolved` | retrieval flat; Phase 8 affinity-margin rule resolved it | **skipped** (provider not called) |
| `near_tie_unresolved` | finished at the cap / on exhaustion with candidates still near-tied | existing gate (may run) |

Nothing is inferred from score gaps, no threshold or weight changed, and 5D is unaffected (a skipped comparison is
described as `deterministic_ranking`). Measure it with `python -m evaluation && python -m evaluation.stage5c`;
Phase 8 results are kept as `evaluation/output/phase8_baseline_*`, the comparison is `phase9_vs_phase8.md`.
