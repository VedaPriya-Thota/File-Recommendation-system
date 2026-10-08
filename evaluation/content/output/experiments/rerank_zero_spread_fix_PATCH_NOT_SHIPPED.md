# Reranker zero-spread fix -- evaluated, measured, deliberately NOT shipped in Phase 11

In `app/tools/rerank_candidates.py`, replace
    normalized = round((raw_score - lo) / spread, 4) if spread > 0 else 1.0
with: spread > 0 -> unchanged; exactly one candidate -> 1.0; **all candidates scored identically (n > 1) -> 0.0**
(and mark `rerank_zero_spread` in metadata), so a reranker that expressed no preference contributes no evidence
instead of lifting every candidate to 1.0 through final ranking's `max()` merge.

Measured effect (see `rerank_zero_spread_fix_RESULTS_NOT_SHIPPED.json`):
| | without fix | with fix |
|---|---|---|
| new content eval: P@1 / success | 0.741 / 0.677 | 0.741 / 0.677 (unchanged) |
| new content eval: MRR / Recall@3 | 0.777 / 0.741 | 0.789 / 0.778 |
| S1 correct file, first relevant rank | 6 | 2 |
| OLD 310-record benchmark | baseline | statuses, files, tool sequences, success and summary identical; **`confidence` differs in 3 records** (H3 under D, D_no_explanation, D_cap3: 0.5 -> 0.1667, because the old 1.0 reranker score inflated the retrieval evidence confidence is derived from) |

Why it was not shipped: the benefit is small and the brief requires the 310 old records to stay identical.
The change is tiny and tested; it needs an explicit decision to accept the 3-record `confidence` change.
