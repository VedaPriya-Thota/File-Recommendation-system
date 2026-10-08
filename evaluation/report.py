"""Aggregation + human-readable rendering of benchmark records. Pure functions."""
from typing import Dict, List, Optional, Sequence

from evaluation.metrics import mean

RETRIEVAL_FIELDS = (
    "precision_at_1", "precision_at_3", "precision_at_5",
    "recall_at_1", "recall_at_3", "recall_at_5",
    "reciprocal_rank", "ndcg_at_5",
)
TOOL_FIELDS = ("metadata_search", "bm25_search", "semantic_search", "hybrid_search",
               "rerank_candidates", "get_user_activity")


def _r(x, n=4):
    return None if x is None else round(x, n)


def retrieval_summary(recs: Sequence[dict]) -> dict:
    """Means over queries that HAVE relevant files (n_scored). Clarified /
    empty-result queries among them score 0 and are included, never dropped."""
    scored = [r for r in recs if r["expected_file_ids"]]
    out = {"n": len(recs), "n_scored": len(scored)}
    for f in RETRIEVAL_FIELDS:
        out["MRR" if f == "reciprocal_rank" else f] = _r(mean(r[f] for r in scored))
    out["success_rate"] = _r(mean(1.0 if r["success"] else 0.0 for r in recs))
    # Abstention-aware views (a clarified query returns nothing and scores 0 above; these show why).
    answered = [r for r in scored if r["returned_file_ids"]]
    out["abstention_rate_scored"] = _r(mean(1.0 if r["clarified"] else 0.0 for r in scored))
    out["n_answered"] = len(answered)
    out["MRR_answered_only"] = _r(mean(r["reciprocal_rank"] for r in answered))
    out["P@1_answered_only"] = _r(mean(r["precision_at_1"] for r in answered))
    # DIAGNOSTIC: ordering final_rank() would give the accumulated pool (even if the system clarified).
    out["MRR_pool_ranking_diagnostic"] = _r(mean(r["pool_ranking_reciprocal_rank"] for r in scored))
    # Graded: is the intended (grade-2) file ranked first? (None-valued for queries with no grade-2 file.)
    out["HR@1"] = _r(mean(r["highly_relevant_at_1"] for r in scored))
    out["HR@1_pool_ranking_diagnostic"] = _r(mean(r["pool_ranking_highly_relevant_at_1"] for r in scored))
    return out


def behavior_summary(recs: Sequence[dict]) -> dict:
    n = len(recs)
    if n == 0:
        return {"n": 0}
    out = {
        "n": n,
        "success_rate": _r(mean(1.0 if r["success"] else 0.0 for r in recs)),
        "clarification_rate": _r(mean(1.0 if r["clarified"] else 0.0 for r in recs)),
        "no_result_rate": _r(mean(1.0 if r["no_result"] else 0.0 for r in recs)),
        "degraded_rate": _r(mean(1.0 if r["degraded_mode"] else 0.0 for r in recs)),
        "average_tool_calls": _r(mean(r["tool_calls"] for r in recs), 3),
        "average_retrieval_actions": _r(mean(r["retrieval_actions"] for r in recs), 3),
        "average_iterations": _r(mean(r["iterations"] for r in recs), 3),
        "average_llm_calls": _r(mean(r["llm_calls"] for r in recs), 3),
        "average_llm_attempts": _r(mean(r["llm_attempts"] for r in recs), 3),
        "escalation_rate": _r(mean(1.0 if r["tool_calls"] > 1 else 0.0 for r in recs)),
        "scope_violations": sum(r["scope_violations"] for r in recs),
        "hard_failures": sum(1 for r in recs if r["hard_failure"]),
    }
    for t in TOOL_FIELDS:
        out[f"average_{t}_calls"] = _r(mean(r["tool_counts"].get(t, 0) for r in recs), 3)
    no_res = [r for r in recs if r["expected_outcome"] == "no_result"]
    out["no_result_false_positive_rate"] = _r(mean(0.0 if r["no_result"] else 1.0 for r in no_res)) if no_res else None
    return out


def _group(recs, key):
    groups: Dict[str, List[dict]] = {}
    for r in recs:
        groups.setdefault(r[key], []).append(r)
    return groups


def escalation_vs_baseline(recs: Sequence[dict], baseline: str, target: str) -> dict:
    """For each query with relevant files, did extra tool calls beyond the
    baseline's help (higher RR), change nothing, or hurt? Answers 'is the agent
    escalating unnecessarily?' without any production hook."""
    base = {r["query_id"]: r for r in recs if r["variant"] == baseline}
    tgt = {r["query_id"]: r for r in recs if r["variant"] == target}
    counts = {"not_escalated": 0, "helped": 0, "no_change": 0, "hurt": 0}
    detail = {}
    for qid, t in tgt.items():
        b = base.get(qid)
        if b is None or not t["expected_file_ids"]:
            continue
        if t["tool_calls"] <= b["tool_calls"]:
            outcome = "not_escalated"
        else:
            d = (t["reciprocal_rank"] or 0.0) - (b["reciprocal_rank"] or 0.0)
            outcome = "helped" if d > 1e-9 else "hurt" if d < -1e-9 else "no_change"
        counts[outcome] += 1
        detail[qid] = outcome
    counts["unnecessary_escalation_rate"] = _r(
        (counts["no_change"] + counts["hurt"]) / max(1, counts["helped"] + counts["no_change"] + counts["hurt"])
    )
    return {"baseline": baseline, "target": target, "counts": counts, "per_query": detail}


def build_summary(records: Sequence[dict], queries=None) -> dict:
    by_variant = _group(records, "variant")
    summary = {
        "n_results": len(records),
        "overall": {v: retrieval_summary(rs) for v, rs in by_variant.items()},
        "by_category": {},
        "by_variant": {},
        "behavior": {v: behavior_summary(rs) for v, rs in by_variant.items()},
        "ablation_deltas": [],
        "escalation_vs_single_shot": {},
    }
    for v, rs in by_variant.items():
        summary["by_variant"][v] = {
            "retrieval": retrieval_summary(rs),
            "behavior": behavior_summary(rs),
            "by_category": {c: retrieval_summary(crs) for c, crs in sorted(_group(rs, "category").items())},
        }
        for c, crs in _group(rs, "category").items():
            summary["by_category"].setdefault(c, {})[v] = {
                **retrieval_summary(crs),
                "average_tool_calls": _r(mean(r["tool_calls"] for r in crs), 3),
            }
    chain = [("A_single_shot_metadata", "B_agentic_deterministic"),
             ("B_agentic_deterministic", "C_agentic_personalized"),
             ("C_agentic_personalized", "D_full_system")]
    for a, b in chain:
        if a in summary["overall"] and b in summary["overall"]:
            delta = {"from": a, "to": b}
            for k in ("precision_at_1", "precision_at_3", "recall_at_3", "recall_at_5", "MRR", "ndcg_at_5", "success_rate"):
                x, y = summary["overall"][a].get(k), summary["overall"][b].get(k)
                delta[k] = None if x is None or y is None else round(y - x, 4)
            for k in ("average_tool_calls", "average_iterations", "average_llm_calls"):
                x, y = summary["behavior"][a].get(k), summary["behavior"][b].get(k)
                delta[k] = None if x is None or y is None else round(y - x, 3)
            summary["ablation_deltas"].append(delta)
    for target in ("B_agentic_deterministic", "C_agentic_personalized", "D_full_system"):
        if "A_single_shot_metadata" in by_variant and target in by_variant:
            summary["escalation_vs_single_shot"][target] = escalation_vs_baseline(
                records, "A_single_shot_metadata", target)
    return summary


def _fmt(x):
    return "  -  " if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def render_markdown(output: dict) -> str:
    s, m = output["summary"], output["metadata"]
    core = [v["name"] for v in m["variants"] if v["core"]]
    lines = [f"# Phase 7 benchmark report", "",
             f"{m['dataset_size']} queries x {len(m['variants'])} variants. "
             f"Embedding: `{m.get('embedding_provider')}`; reranker: `{m.get('reranker_provider')}`; "
             f"semantic index: {m.get('semantic_index')}.", "",
             "## Overall retrieval (core ablation)", "",
             "| variant | n_scored | P@1 | P@3 | P@5 | R@1 | R@3 | R@5 | MRR | nDCG@5 | success |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in core:
        o = s["overall"][v]
        lines.append(f"| {v} | {o['n_scored']} | " + " | ".join(_fmt(o[k]) for k in (
            "precision_at_1", "precision_at_3", "precision_at_5", "recall_at_1", "recall_at_3",
            "recall_at_5", "MRR", "ndcg_at_5", "success_rate")) + " |")
    lines += ["", "## Abstention-aware view (core ablation)", "",
              "Clarified queries return nothing and score 0 in the table above. `answered-only` restricts to queries where a ranking was returned; "
              "`pool-ranking` is a DIAGNOSTIC of how final_rank() would order the accumulated pool (not what the system returned).", "",
              "| variant | abstention rate | n_answered | P@1 answered-only | MRR answered-only | MRR pool-ranking (diag.) | HR@1 | HR@1 pool-ranking (diag.) |", "|---|---|---|---|---|---|---|---|"]
    for v in core:
        o = s["overall"][v]
        lines.append(f"| {v} | " + " | ".join(_fmt(o[k]) for k in (
            "abstention_rate_scored", "n_answered", "P@1_answered_only", "MRR_answered_only", "MRR_pool_ranking_diagnostic",
            "HR@1", "HR@1_pool_ranking_diagnostic")) + " |")
    lines += ["", "## Behaviour (all variants)", "",
              "| variant | success | clarify | no_result | tool calls | retr. actions | iterations | LLM calls | LLM attempts | escalation | scope viol. |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v, b in s["behavior"].items():
        lines.append(f"| {v} | " + " | ".join(_fmt(b[k]) for k in (
            "success_rate", "clarification_rate", "no_result_rate", "average_tool_calls",
            "average_retrieval_actions", "average_iterations", "average_llm_calls",
            "average_llm_attempts", "escalation_rate", "scope_violations")) + " |")
    lines += ["", "## MRR / success by category", "", "| category | " + " | ".join(core) + " |",
              "|---|" + "---|" * len(core)]
    for c, per in sorted(s["by_category"].items()):
        lines.append(f"| {c} | " + " | ".join(
            f"MRR {_fmt(per[v]['MRR'])} / succ {_fmt(per[v]['success_rate'])}" for v in core) + " |")
    lines += ["", "## Scenarios", ""]
    for sc in output["scenarios"]:
        lines.append(f"- **{sc['name']}**: {'PASS' if sc['passed'] else 'FAIL'} -- {sc['expectation']}")
        for chk in sc["checks"]:
            if not chk["passed"]:
                label = "OBSERVATION" if chk.get("kind") == "observation" else "FAIL"
                lines.append(f"    - {label} `{chk['scope']}` {chk['check']}: {chk['detail']}")
    lines += ["", "## Limitations", ""] + [f"- {x}" for x in m["limitations"]]
    return "\n".join(lines) + "\n"
