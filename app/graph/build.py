"""
LangGraph orchestration -- architecture_final.md §2 / §6 / prompt §6.

    START -> resolve_scope -> understand_goal -> expand_query -> plan_next_action
          -> execute_tool -> update_candidate_pool -> reflect
          -> {replan -> plan_next_action,
              finish -> finish_stage -> compare_stage -> explain_stage -> END,
              clarify -> clarify_stage -> END}

This is a real cycle: reflect's decision determines whether
plan_next_action runs again with a different strategy (genuine
observation-driven replanning), not a fixed sequence.

Phase 5A/5B: two narrow, optional LLM slices (goal interpretation,
query expansion) run in understand_goal/expand_query, each gated on the
deterministic ambiguity heuristic and each falling back safely to pure
Phase 1-4 deterministic behavior on any failure -- see
app/llm/goal_interpreter.py and app/llm/query_expander.py.

Phase 5C: a third narrow slice, compare_stage, runs AFTER the deterministic
final_rank (finish_stage) and only when the top candidates are genuinely
near-tied; it may promote one already-ranked candidate to the front and
nothing else -- see app/llm/candidate_comparer.py.

Phase 5D: explain_stage is the terminal, presentation-only slice: it writes
explanatory text for the already-final top result and never touches
final_ranking, the pool, scores or scope -- see app/llm/result_explainer.py.
Every other node remains zero-LLM, exactly as in Phase 1.
"""
from datetime import datetime, timezone
from typing import Optional

from langgraph.graph import END, StateGraph

from app.candidate_pool import update_candidate_pool as merge_candidate_pool
from app.config import AMBIGUITY_CLARIFY_THRESHOLD, MAX_ITERATIONS
from app.final_ranking import final_rank
from app.goal import understand_goal as understand_goal_fn
from app.llm.candidate_comparer import (
    compare_candidates_with_llm,
    reorder_with_selection,
    select_comparison_set,
)
from app.llm.goal_interpreter import augment_goal_with_llm
from app.llm.result_explainer import explain_result_with_llm, should_explain
from app.llm.provider import get_default_provider
from app.llm.query_expander import expand_query_with_llm
from app.planning import args_signature, plan_next_action as plan_next_action_fn
from app.final_ranking import leading_is_weak_lexical, weak_lexical_confidence
from app.pool_resolution import (
    BASIS_NEAR_TIE_UNRESOLVED,
    BASIS_PERSONALIZATION_RESOLVED,
    BASIS_WEAK_LEXICAL,
    assess_exhausted_pool,
)
from app.reflect import reflect as reflect_fn
from app.scope import UnknownUserError, resolve_scope as resolve_scope_fn
from app.state import AgentState
from app.tools import build_registry


def _trace(state: AgentState, event: str, **fields) -> dict:
    entry = {"event": event, "iteration": state.get("iteration_count", 0), **fields}
    trace = list(state.get("trace", []))
    trace.append(entry)
    return {"trace": trace}


def build_graph(registry=None, max_iterations: Optional[int] = None, llm_provider=None):
    registry = registry or build_registry()
    max_iterations = max_iterations if max_iterations is not None else MAX_ITERATIONS
    # Phase 5A: defaults to app.llm.provider.NullProvider (no I/O, no
    # network) unless DISCOVERY_LLM_PROVIDER is explicitly set or a
    # provider is injected directly (e.g. a fake in tests) -- see
    # app/llm/provider.py.
    llm_provider = llm_provider or get_default_provider()

    # --- Nodes -----------------------------------------------------

    def node_resolve_scope(state: AgentState) -> dict:
        scope = resolve_scope_fn(state["user_id"])  # raises on hard failure -- propagates out of graph
        return {
            "user_scope": list(scope.owner_ids),
            **_trace(state, "resolve_scope", owner_ids=list(scope.owner_ids)),
        }

    def node_understand_goal(state: AgentState) -> dict:
        goal = understand_goal_fn(state["raw_query"]).to_dict()
        # Phase 5A: the deterministic heuristic always runs first and is
        # always what gets returned by default. The LLM slice is only
        # invoked when that heuristic already flagged genuine ambiguity
        # (requirement 7 -- never call it for every simple query); a
        # provider failure or invalid output falls back to this same
        # deterministic `goal` untouched (requirement 6).
        llm_meta = {"llm_invoked": False, "llm_status": "not_needed"}
        if goal["ambiguity_score"] >= AMBIGUITY_CLARIFY_THRESHOLD:
            goal, llm_meta = augment_goal_with_llm(state["raw_query"], goal, llm_provider)
        return {
            "goal": goal,
            "llm_goal_metadata": llm_meta,
            **_trace(state, "understand_goal", goal=goal, llm=llm_meta),
        }

    def node_expand_query(state: AgentState) -> dict:
        # Phase 5B: runs AFTER understand_goal (so it sees any Phase-5A
        # goal augmentation) but BEFORE plan_next_action ever runs
        # (requirement 9), gated on the same deterministic ambiguity
        # signal as Phase 5A (requirement 6 -- never call it for every
        # simple query). `goal["raw_query"]`/`topic_tokens`/`concept_text`
        # are never modified here; expanded terms land in a separate
        # `goal["expanded_terms"]` key that app/planning.py optionally
        # folds into individual tool queries (requirement 5).
        goal = dict(state["goal"])
        expansion_meta = {"expansion_invoked": False, "expansion_status": "not_needed"}
        if goal.get("ambiguity_score", 0.0) >= AMBIGUITY_CLARIFY_THRESHOLD:
            expanded_terms, expansion_meta = expand_query_with_llm(state["raw_query"], goal, llm_provider)
            goal["expanded_terms"] = expanded_terms
        else:
            goal["expanded_terms"] = []
        return {
            "goal": goal,
            "query_expansion_metadata": expansion_meta,
            **_trace(state, "expand_query", expanded_terms=goal["expanded_terms"], expansion=expansion_meta),
        }

    def node_plan_next_action(state: AgentState) -> dict:
        iteration = state.get("iteration_count", 0) + 1
        # observation_hint comes from the previous reflection (if any);
        # absent on the very first planning call.
        observation_hint = state.get("observation_hint")
        action = plan_next_action_fn(
            goal=state["goal"],
            committed_actions=state.get("committed_actions", []),
            failed_strategies=state.get("failed_strategies", []),
            iteration=iteration,
            observation_hint=observation_hint,
            candidate_pool=state.get("candidate_pool", {}),
        )
        if action is None:
            # No viable strategy left. Resolve based on why we got here:
            #  - "zero_results": genuinely nothing to find -> finish (empty).
            #  - "too_many" with a non-empty pool (Phase 8, Rule B in
            #    app/pool_resolution.py): clarify ONLY if the top candidates
            #    are genuinely tied after every evidence source; if final_rank
            #    yields a strict leader, finish with the ranked shortlist
            #    (flagged low-confidence) instead of discarding the pool.
            #  - anything else: ask.
            pool = state.get("candidate_pool", {})
            verdict = assess_exhausted_pool(pool) if (observation_hint == "too_many" and pool) else None
            # A weak lexical leader whose independent search then found nothing (hint "zero_results") is the
            # same situation as "weak_evidence" with no strategy left -- report it honestly, not as "nothing found".
            weak_finish = observation_hint == "weak_evidence" or (
                observation_hint == "zero_results" and leading_is_weak_lexical(pool)
            )
            if weak_finish:
                resolution = "finish"
                reason = ("Only weak lexical evidence (a match covering a minority of the query's terms) was found "
                          "and no independent strategy remains; returning it flagged low-confidence.")
                # confidence derives from the weak candidate itself (coverage-capped). Not inherited from reflect():
                # after an EMPTY semantic search reflect() reports 0.0 ("nothing found"), which would be wrong for a
                # result that is actually being returned.
                extra = {"degraded_mode": True, "confidence": weak_lexical_confidence(pool)}
            elif observation_hint == "zero_results":
                resolution = "finish"
                reason = "No further metadata_search strategy available; nothing found."
                extra = {"degraded_mode": True, "confidence": 0.0}
            elif verdict is not None and verdict["decision"] == "finish":
                resolution = "finish"
                reason = verdict["reason"]
                # confidence is left as reflect() computed it (retrieval-derived); never invented here.
                extra = {"degraded_mode": True}
            else:
                resolution = "clarify"
                reason = (verdict["reason"] if verdict is not None
                          else "No further metadata_search strategy available to disambiguate.")
                extra = {}
            return {
                "next_action": None,
                "stopping_decision": {
                    "status": resolution,
                    "reason": reason,
                    # Rule B finishes leave candidates near-tied by construction (retrieval lead < dominance).
                    **({"basis": BASIS_NEAR_TIE_UNRESOLVED} if (verdict is not None and resolution == "finish") else {}),
                    **({"basis": BASIS_WEAK_LEXICAL} if (weak_finish and resolution == "finish") else {}),
                },
                "clarification_question": (
                    _build_clarification_question(state, verdict) if resolution == "clarify" else None
                ),
                **extra,
                **_trace(
                    state, "plan_next_action_exhausted", resolution=resolution, reason=reason,
                    **({"leader_id": verdict["leader_id"], "gap": verdict["gap"], "tied_ids": verdict["tied_ids"]}
                       if verdict is not None else {}),
                ),
            }
        return {
            "next_action": action,
            "iteration_count": iteration,
            **_trace(state, "plan_next_action", action=action),
        }

    def node_execute_tool(state: AgentState) -> dict:
        action = state["next_action"]
        user_scope = _scope_obj(state)
        result = registry.execute(action["tool"], action["args"], user_scope)
        result_dict = {
            "tool": result.tool,
            "candidates": result.candidates,
            "error": result.error,
            "degraded": result.degraded,
        }
        committed = list(state.get("committed_actions", []))
        committed.append(action)
        return {
            "committed_actions": committed,
            "last_tool_result": result_dict,
            **_trace(
                state,
                "execute_tool",
                tool=action["tool"],
                args=_jsonable(action["args"]),
                candidate_count=len(result.candidates),
                error=result.error,
            ),
        }

    def node_update_candidate_pool(state: AgentState) -> dict:
        result = state["last_tool_result"]
        from app.tools.registry import ToolResult

        tool_result_obj = ToolResult(
            tool=result["tool"],
            candidates=result["candidates"],
            error=result["error"],
            degraded=result["degraded"],
        )
        pool = merge_candidate_pool(
            state.get("candidate_pool", {}), tool_result_obj, state.get("iteration_count", 0)
        )
        return {
            "candidate_pool": pool,
            **_trace(state, "update_candidate_pool", pool_size=len(pool)),
        }

    def node_reflect(state: AgentState) -> dict:
        last_action = state["committed_actions"][-1]
        rr = reflect_fn(
            last_action=last_action,
            last_tool_result=state["last_tool_result"],
            candidate_pool=state.get("candidate_pool", {}),
            failed_strategies=state.get("failed_strategies", []),
            errors=state.get("errors", []),
            iteration_count=state.get("iteration_count", 0),
            max_iterations=state.get("max_iterations", max_iterations),
        )
        update = {
            "confidence": rr.confidence,
            "failed_strategies": rr.failed_strategies,
            "errors": rr.errors,
            "stopping_decision": {
                "status": rr.status,
                "reason": rr.reason,
                **({"basis": rr.resolution_basis} if rr.resolution_basis else {}),
            },
            "observation_hint": rr.observation_hint,
            **_trace(
                state,
                "reflect",
                status=rr.status,
                reason=rr.reason,
                basis=rr.resolution_basis,
                confidence=rr.confidence,
                observation_hint=rr.observation_hint,
            ),
        }
        if rr.low_confidence:
            update["degraded_mode"] = True
        return update

    def node_finish(state: AgentState) -> dict:
        # Phase 4B: the actual final-ranking policy lives in
        # app/final_ranking.py (architecture_final.md §8 touchpoint 3 /
        # §10) -- it separates retrieval/reranker evidence from
        # personalization evidence instead of taking an unweighted
        # max() over both. See that module's docstring for the formula.
        pool = state.get("candidate_pool", {})
        ranking = final_rank(pool)
        return {
            "final_ranking": ranking,
            **_trace(state, "finish", result_count=len(ranking)),
        }

    def node_compare_candidates(state: AgentState) -> dict:
        # Phase 5C: runs after the deterministic final_rank, gated on
        # genuine near-ties among the top candidates (never for a clearly
        # dominant or single candidate -- the LLM is not called at all).
        # `final_ranking` entries/scores are never modified; on success
        # the chosen candidate is only moved to the front, and the
        # original deterministic order is recorded for audit. Any failure
        # leaves final_ranking exactly as final_rank produced it.
        ranking = state.get("final_ranking", [])
        comparison_set = select_comparison_set(ranking)
        # Phase 9: if the pool was already meaningfully RESOLVED by the deterministic
        # personalization path (Rule A, recorded explicitly in stopping_decision["basis"]),
        # 5C must not re-adjudicate that decision. Such a pool always has a final-score gap
        # below DOMINANT_SCORE_GAP, so the near-tie gate alone cannot tell it apart from an
        # ordinary near-tie -- hence the explicit basis rather than any score inference.
        # The provider is not called; ordinary near-ties and retrieval-dominant results keep
        # their existing 5C behaviour.
        if (state.get("stopping_decision") or {}).get("basis") == BASIS_PERSONALIZATION_RESOLVED:
            meta = {
                "comparison_invoked": False,
                "comparison_status": "skipped_personalization_resolved",
                "comparison_applied": False,
                "near_tie_gate_open": bool(comparison_set),  # audit: would 5C otherwise have been eligible?
                "deterministic_order": [c["file_id"] for c in ranking],
            }
            return {"candidate_comparison": meta, **_trace(state, "compare_candidates", comparison=meta)}
        if not comparison_set:
            meta = {"comparison_invoked": False, "comparison_status": "not_needed", "comparison_applied": False}
            return {"candidate_comparison": meta, **_trace(state, "compare_candidates", comparison=meta)}

        selected_id, meta = compare_candidates_with_llm(
            state["raw_query"], state.get("goal", {}), comparison_set, llm_provider
        )
        deterministic_order = [c["file_id"] for c in ranking]
        meta["deterministic_order"] = deterministic_order
        meta["comparison_applied"] = False
        update: dict = {}
        if selected_id is not None and selected_id != deterministic_order[0]:
            update["final_ranking"] = reorder_with_selection(ranking, selected_id)
            meta["comparison_applied"] = True
        update["candidate_comparison"] = meta
        update.update(_trace(state, "compare_candidates", comparison=meta))
        return update

    def node_explain_result(state: AgentState) -> dict:
        # Phase 5D: terminal presentation stage. Returns ONLY
        # candidate_explanation/explanations -- never final_ranking,
        # candidate_pool, scores or user_scope -- so whatever the LLM does
        # (or fails to do), the result is exactly what final_rank + Phase 5C
        # produced.
        ranking = state.get("final_ranking", [])
        if not should_explain(ranking, state.get("candidate_explanation")):
            meta = {"explanation_invoked": False, "explanation_status": "not_needed"}
            return {"candidate_explanation": meta, **_trace(state, "explain_result", explanation=meta)}

        explanation, meta = explain_result_with_llm(
            state["raw_query"],
            state.get("goal", {}),
            ranking[0],
            state.get("candidate_comparison"),
            llm_provider,
            user_id=state.get("user_id"),
            user_scope=state.get("user_scope"),
        )
        update: dict = {"candidate_explanation": meta}
        if explanation is not None:
            update["explanations"] = {**state.get("explanations", {}), meta["explained_file_id"]: explanation.summary}
        trace_meta = {k: v for k, v in meta.items() if k != "explanation_evidence_points"}
        update.update(_trace(state, "explain_result", explanation=trace_meta))
        return update

    def node_clarify(state: AgentState) -> dict:
        question = state.get("clarification_question") or _build_clarification_question(state)
        return {
            "clarification_question": question,
            **_trace(state, "clarify", question=question),
        }

    def _scope_obj(state: AgentState):
        from app.scope import UserScope

        return UserScope(user_id=state["user_id"], owner_ids=tuple(state["user_scope"]))

    def _jsonable(d: dict) -> dict:
        out = {}
        for k, v in d.items():
            out[k] = v.isoformat() if isinstance(v, datetime) else v
        return out

    def _build_clarification_question(state: AgentState, verdict: Optional[dict] = None) -> str:
        goal = state.get("goal", {})
        topic = " ".join(goal.get("topic_tokens", [])) or "that"
        pool = state.get("candidate_pool", {})
        if pool:
            # Phase 8: list the genuinely tied contenders (in final-ranking order) when known.
            ranked = final_rank(pool)
            tied = set((verdict or {}).get("tied_ids") or [])
            shown = [e for e in ranked if e["file_id"] in tied] if len(tied) > 1 else ranked
            names = [e["metadata"]["filename"] for e in shown[:5]]
            return (
                f"I found multiple possible matches for '{topic}' "
                f"({', '.join(names)}) and can't confidently pick one. "
                "Could you narrow it down (file type, approximate date, or filename)?"
            )
        return (
            f"I couldn't confidently interpret the request for '{topic}'. "
            "Could you give a file type, approximate date, or part of the filename?"
        )

    # --- Conditional routing ----------------------------------------

    def route_after_plan(state: AgentState) -> str:
        if state.get("next_action") is None:
            status = state.get("stopping_decision", {}).get("status")
            return "clarify" if status == "clarify" else "finish"
        return "execute"

    def route_after_reflect(state: AgentState) -> str:
        status = state["stopping_decision"]["status"]
        if status == "finish":
            return "finish"
        if status == "clarify":
            return "clarify"
        return "replan"  # -> plan_next_action again

    # --- Graph wiring --------------------------------------------------
    graph = StateGraph(AgentState)
    graph.add_node("resolve_scope", node_resolve_scope)
    graph.add_node("understand_goal", node_understand_goal)
    graph.add_node("expand_query", node_expand_query)
    graph.add_node("plan_next_action", node_plan_next_action)
    graph.add_node("execute_tool", node_execute_tool)
    graph.add_node("update_candidate_pool", node_update_candidate_pool)
    graph.add_node("reflect", node_reflect)
    graph.add_node("finish_stage", node_finish)
    graph.add_node("compare_stage", node_compare_candidates)
    graph.add_node("explain_stage", node_explain_result)
    graph.add_node("clarify_stage", node_clarify)

    graph.set_entry_point("resolve_scope")
    graph.add_edge("resolve_scope", "understand_goal")
    graph.add_edge("understand_goal", "expand_query")
    graph.add_edge("expand_query", "plan_next_action")

    graph.add_conditional_edges(
        "plan_next_action",
        route_after_plan,
        {"execute": "execute_tool", "finish": "finish_stage", "clarify": "clarify_stage"},
    )
    graph.add_edge("execute_tool", "update_candidate_pool")
    graph.add_edge("update_candidate_pool", "reflect")
    graph.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {"replan": "plan_next_action", "finish": "finish_stage", "clarify": "clarify_stage"},
    )
    graph.add_edge("finish_stage", "compare_stage")
    graph.add_edge("compare_stage", "explain_stage")
    graph.add_edge("explain_stage", END)
    graph.add_edge("clarify_stage", END)

    return graph.compile()
