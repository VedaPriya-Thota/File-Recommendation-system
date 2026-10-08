"""
Behavioural scenario evaluation -- reported separately from retrieval metrics.

Each scenario states an EXPECTATION taken from the Phase 7 brief / architecture,
evaluates it over already-collected records, and reports PASS/FAIL per check.
A FAIL is a finding about the system, not something to be tuned away here.
"""
from typing import Callable, Dict, List, Sequence

from app.config import AMBIGUITY_CLARIFY_THRESHOLD, COMPARISON_TOP_N, DOMINANT_SCORE_GAP

from evaluation.variants import CORE_VARIANT_NAMES

LLM_VARIANTS = ("D_full_system", "D_llm_unavailable", "D_llm_invalid_output", "D_no_explanation", "D_cap3_diagnostic")


def _near_tie(top3_scores: Sequence[float]) -> bool:
    """Mirror of select_comparison_set's gate, evaluated on recorded scores.
    Uses the max of the top-N because Phase 5C only reorders within that set."""
    scores = [s for s in top3_scores[:COMPARISON_TOP_N] if s is not None]
    if len(scores) < 2:
        return False
    top = max(scores)
    return sum(1 for s in scores if (top - s) < DOMINANT_SCORE_GAP) >= 2


def _scenario(name: str, expectation: str, checks: List[dict]) -> dict:
    """Pass/fail is decided by `expectation` checks only. `observation`
    checks are recorded (and can show FAIL) for transparency but describe
    by-design behaviour worth costing, so they do not gate the scenario."""
    gating = [c for c in checks if c["kind"] == "expectation"]
    return {"name": name, "expectation": expectation, "checks": checks,
            "passed": bool(gating) and all(c["passed"] for c in gating), "n_checks": len(checks),
            "n_observations": len(checks) - len(gating)}


def _chk(scope: str, check: str, passed: bool, detail: str = "", observation: bool = False) -> dict:
    return {"scope": scope, "check": check, "passed": bool(passed), "detail": detail,
            "kind": "observation" if observation else "expectation"}


def evaluate_scenarios(records: Sequence[dict]) -> List[dict]:
    idx: Dict[tuple, dict] = {(r["variant"], r["query_id"]): r for r in records}
    variants_present = {r["variant"] for r in records}
    have = lambda *vs: all(v in variants_present for v in vs)  # noqa: E731
    out: List[dict] = []

    # 1. clear query: direct retrieval, no unnecessary LLM, correct result
    if have("D_full_system"):
        checks = []
        for qid in ("A1", "B2"):
            r = idx.get(("D_full_system", qid))
            if not r:
                continue
            checks += [
                _chk(qid, "single iteration / single tool call", r["iterations"] == 1 and r["tool_calls"] == 1, f"iterations={r['iterations']} tools={r['tool_sequence']}"),
                _chk(qid, "no decision-making LLM stage (goal/expansion/comparison) attempted",
                     sum(r["llm_attempts_by_stage"][st] for st in ("goal", "expansion", "comparison")) == 0,
                     f"attempts_by_stage={r['llm_attempts_by_stage']}"),
                # Added after the first run: Phase 5D (explanation) runs on EVERY non-empty result by design
                # (app/graph/build.py node_explain_result / should_explain), so a clear query still costs one
                # presentation-only LLM call. Recorded as an observation, not hidden and not gating.
                _chk(qid, "no LLM call at all (including presentation-only 5D explanation)",
                     r["llm_attempts"] == 0, f"attempts={r['llm_attempts']} by_stage={r['llm_attempts_by_stage']}",
                     observation=True),
                _chk(qid, "top-1 is relevant", r["first_relevant_rank"] == 1, f"returned={r['returned_file_ids']}"),
            ]
        out.append(_scenario("clear_query", "Direct retrieval in one action, zero LLM stages, correct top result.", checks))

    # 2. ambiguous queries
    if have("D_full_system"):
        checks = []
        for qid in ("D1", "D2", "D3"):
            r = idx.get(("D_full_system", qid))
            if not r:
                continue
            guessed = r["status"] == "finish" and not r["degraded_mode"] and len(r["returned_file_ids"]) == 1
            checks += [
                _chk(qid, "no confident single guess", not guessed, f"status={r['status']} degraded={r['degraded_mode']} returned={r['returned_file_ids']}"),
                _chk(qid, "LLM goal/expansion gated on ambiguity score",
                     (r["llm_attempts_by_stage"]["goal"] > 0) == ((r["goal_ambiguity_score"] or 0) >= AMBIGUITY_CLARIFY_THRESHOLD),
                     f"ambiguity={r['goal_ambiguity_score']} goal_attempts={r['llm_attempts_by_stage']['goal']}"),
                _chk(qid, "no scope violation", r["scope_violations"] == 0),
            ]
        out.append(_scenario("ambiguous_query", "Ambiguity handled: clarify or flagged/low-confidence result, never a silent confident guess; LLM only past the ambiguity gate.", checks))

    # 3. weak retrieval / observation-driven escalation
    checks = []
    for v in ("B_agentic_deterministic",):
        for qid in ("H2", "D1", "E1"):
            r = idx.get((v, qid))
            if r:
                seq = r["tool_sequence"]
                checks.append(_chk(f"{v}:{qid}", "escalates beyond first action using a different tool",
                                   len(seq) > 1 and len(set(seq)) > 1, f"sequence={seq}"))
                checks.append(_chk(f"{v}:{qid}", "one committed action per iteration", r["iterations"] == r["tool_calls"], f"iterations={r['iterations']} calls={r['tool_calls']}"))
        for qid in ("F1", "F2", "F3"):
            r = idx.get((v, qid))
            if r:
                seq = r["tool_sequence"]
                checks.append(_chk(f"{v}:{qid}", "zero-result observation escalates metadata -> bm25 -> semantic",
                                   seq[:1] == ["metadata_search"] and "bm25_search" in seq and "semantic_search" in seq, f"sequence={seq}"))
    if checks:
        out.append(_scenario("weak_retrieval_escalation", "Strategy changes mid-query in response to flat or empty observations.", checks))

    # 4. near tie: 5C only when the deterministic gate says so
    checks = []
    exercised = 0
    suppressed_count = 0
    for v in LLM_VARIANTS:
        for r in records:
            if r["variant"] != v:
                continue
            gate = _near_tie(r["top3_final_scores"]) and not r["clarified"]
            # Phase 9: a pool already resolved by the deterministic personalization path is
            # intentionally NOT re-adjudicated by 5C even though the score gate is open.
            suppressed = r["llm_stage_status"]["comparison"] == "skipped_personalization_resolved"
            attempted = r["llm_attempts_by_stage"]["comparison"] > 0
            if attempted:
                exercised += 1
            if suppressed:
                suppressed_count += 1
            checks.append(_chk(f"{v}:{r['query_id']}", "comparison attempted iff near-tie gate (and not personalization-resolved)",
                               attempted == (gate and not suppressed),
                               f"gate={gate} suppressed={suppressed} attempted={attempted} top3={r['top3_final_scores']}"))
            if suppressed:
                checks.append(_chk(f"{v}:{r['query_id']}", "suppression only ever applies where 5C would otherwise have been eligible",
                                   gate and not attempted, f"gate={gate} attempted={attempted}"))
    if checks:
        sc = _scenario("near_tie_gating", "Phase 5C is consulted only when the deterministic near-tie gate fires.", checks)
        sc["comparison_stage_exercised_count"] = exercised
        sc["comparison_suppressed_personalization_resolved_count"] = suppressed_count
        out.append(sc)

    # 5. no result
    checks = []
    for v in CORE_VARIANT_NAMES:
        for qid in ("F1", "F2", "F3", "G1", "G2", "G5"):
            r = idx.get((v, qid))
            if r:
                checks.append(_chk(f"{v}:{qid}", "no fabricated result", r["no_result"] and r["scope_violations"] == 0,
                                   f"status={r['status']} returned={r['returned_file_ids']}"))
    if checks:
        out.append(_scenario("no_result", "Nothing relevant is accessible: no file is returned.", checks))

    # 6. cross-user isolation (every record, every variant)
    checks = [_chk(f"{r['variant']}:{r['query_id']}", "no inaccessible file returned or pooled",
                   r["scope_violations"] == 0, f"violations={r['scope_violation_file_ids']}") for r in records]
    out.append(_scenario("cross_user_isolation", "No file owned by another user ever appears in a result or the candidate pool.", checks))

    # 6b. personalization must be able to influence what the user is actually shown
    checks = []
    hist = [q for q in sorted({r["query_id"] for r in records if r["personalization_context"] == "history"})]
    if have("B_agentic_deterministic", "C_agentic_personalized") and hist:
        for q in hist:
            b, c = idx[("B_agentic_deterministic", q)], idx[("C_agentic_personalized", q)]
            checks.append(_chk(f"{q}", "C returns a non-empty ranking that differs from B (personalization reaches the result)",
                               bool(c["returned_file_ids"]) and c["returned_file_ids"] != b["returned_file_ids"],
                               f"B={b['returned_file_ids']}/{b['status']} C={c['returned_file_ids']}/{c['status']}"))
            checks.append(_chk(f"{q}", "mechanism: final_rank(pool) ordering differs between B and C",
                               c["pool_ranking_file_ids"] != b["pool_ranking_file_ids"],
                               f"B={b['pool_ranking_file_ids']} C={c['pool_ranking_file_ids']}", observation=True))
            if have("B_cap3_diagnostic", "C_cap3_diagnostic"):
                bb, cc = idx[("B_cap3_diagnostic", q)], idx[("C_cap3_diagnostic", q)]
                checks.append(_chk(f"{q}", "diagnostic cap-3: forced finish lets personalization change the returned ranking",
                                   bool(cc["returned_file_ids"]) and cc["returned_file_ids"] != bb["returned_file_ids"],
                                   f"B_cap3={bb['returned_file_ids']} C_cap3={cc['returned_file_ids']}", observation=True))
        out.append(_scenario("personalization_end_to_end", "For history-sensitive queries, personalization changes the ranking the user actually receives.", checks))

    # 7. LLM outage: deterministic result unchanged
    checks = []
    if have("C_agentic_personalized", "D_llm_unavailable", "D_llm_invalid_output"):
        for q in sorted({r["query_id"] for r in records}):
            base = idx.get(("C_agentic_personalized", q))
            for v in ("D_llm_unavailable", "D_llm_invalid_output"):
                r = idx.get((v, q))
                if base and r:
                    same = r["returned_file_ids"] == base["returned_file_ids"] and r["status"] == base["status"]
                    checks.append(_chk(f"{v}:{q}", "result identical to deterministic variant C", same,
                                       f"C={base['returned_file_ids']}/{base['status']} {v}={r['returned_file_ids']}/{r['status']}"))
                    if v == "D_llm_unavailable":
                        checks.append(_chk(f"{v}:{q}", "no billable LLM call", r["llm_calls"] == 0))
        out.append(_scenario("llm_outage", "With the LLM unavailable or returning garbage, results equal the deterministic system's.", checks))

    # 8. explanation (5D) never affects ranking
    checks = []
    if have("D_full_system", "D_no_explanation"):
        for q in sorted({r["query_id"] for r in records}):
            a, b = idx.get(("D_full_system", q)), idx.get(("D_no_explanation", q))
            if a and b:
                checks.append(_chk(q, "ranking identical with and without the explanation stage",
                                   a["returned_file_ids"] == b["returned_file_ids"],
                                   f"with={a['returned_file_ids']} without={b['returned_file_ids']}"))
        out.append(_scenario("explanation_never_affects_ranking", "Phase 5D is presentation-only.", checks))
    return out
