"""
Pure, dependency-free evaluation metrics (no DB, no app imports).

Definitions (standard):
    Precision@K = (# relevant files in the top K returned) / K
                  -- K is the fixed denominator even if fewer than K files
                     were returned (a system that returns 1 file can never
                     exceed 1/K; this is the usual strict definition).
    Recall@K    = (# relevant files in the top K) / (# relevant in ground truth)
    RR          = 1 / rank of the first relevant file; 0 if none returned.
    nDCG@K      = DCG / ideal DCG with graded gains (grade 2 -> 3, grade 1 -> 1,
                  log2 position discount). Reported as an extra, graded view.

A file is "relevant" when its grade is >= 1 ("highly relevant" is grade 2).

Queries with NO relevant files (no-result / pure cross-user-isolation cases)
have undefined Precision/Recall/MRR: those fields are None, and aggregation
reports how many queries were scored (`n_scored`) versus total (`n`). Those
queries are judged by `judge_success` (nothing returned, no scope violation)
instead. Queries on which the system asked for clarification return nothing and
therefore score 0 on every retrieval metric -- they are NEVER dropped.
"""
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

KS = (1, 3, 5)
_GAIN = {1: 1.0, 2: 3.0}


def _dedup(returned: Sequence[int]) -> List[int]:
    seen, out = set(), []
    for fid in returned:
        if fid not in seen:
            seen.add(fid)
            out.append(fid)
    return out


def first_relevant_rank(returned: Sequence[int], relevant: Iterable[int]) -> Optional[int]:
    relevant = set(relevant)
    for i, fid in enumerate(_dedup(returned), start=1):
        if fid in relevant:
            return i
    return None


def precision_at_k(returned: Sequence[int], relevant: Iterable[int], k: int) -> Optional[float]:
    relevant = set(relevant)
    if not relevant:
        return None
    hits = sum(1 for fid in _dedup(returned)[:k] if fid in relevant)
    return hits / k


def recall_at_k(returned: Sequence[int], relevant: Iterable[int], k: int) -> Optional[float]:
    relevant = set(relevant)
    if not relevant:
        return None
    hits = sum(1 for fid in _dedup(returned)[:k] if fid in relevant)
    return hits / len(relevant)


def reciprocal_rank(returned: Sequence[int], relevant: Iterable[int]) -> Optional[float]:
    relevant = set(relevant)
    if not relevant:
        return None
    rank = first_relevant_rank(returned, relevant)
    return 1.0 / rank if rank else 0.0


def ndcg_at_k(returned: Sequence[int], relevance: Mapping[int, int], k: int) -> Optional[float]:
    if not relevance:
        return None
    dcg = sum(
        _GAIN.get(relevance.get(fid, 0), 0.0) / math.log2(i + 1)
        for i, fid in enumerate(_dedup(returned)[:k], start=1)
    )
    ideal = sorted((_GAIN.get(g, 0.0) for g in relevance.values()), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def retrieval_metrics(returned: Sequence[int], relevance: Mapping[int, int]) -> Dict[str, Optional[float]]:
    relevant = set(relevance)
    out: Dict[str, Optional[float]] = {"first_relevant_rank": first_relevant_rank(returned, relevant) if relevant else None}
    for k in KS:
        out[f"precision_at_{k}"] = precision_at_k(returned, relevant, k)
    for k in KS:
        out[f"recall_at_{k}"] = recall_at_k(returned, relevant, k)
    out["reciprocal_rank"] = reciprocal_rank(returned, relevant)
    out["ndcg_at_5"] = ndcg_at_k(returned, relevance, 5)
    return out


def judge_success(
    expected_outcome: str,
    returned: Sequence[int],
    clarified: bool,
    relevance: Mapping[int, int],
    scope_violations: int,
) -> bool:
    """The single success rule used everywhere (see benchmark_dataset docstring)."""
    if scope_violations:
        return False
    returned = list(returned)
    if expected_outcome == "no_result":
        return len(returned) == 0
    top1_relevant = bool(returned) and returned[0] in relevance
    if expected_outcome == "clarify_or_results":
        return clarified or top1_relevant
    return top1_relevant  # "results"


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
