"""
Phase 1 entrypoint: run_query(user_id, raw_query) -> result dict.

Wires resolve_scope/understand_goal/plan_next_action/execute_tool/
update_candidate_pool/reflect together via the LangGraph skeleton in
app/graph/build.py, and translates hard failures (DB errors, unknown user)
into an explicit error result rather than letting them crash silently.
"""
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from app.config import MAX_ITERATIONS
from app.db.connection import DatabaseError
from app.graph.build import build_graph
from app.scope import UnknownUserError

logger = logging.getLogger("file_discovery_agent")


def _initial_state(user_id: str, raw_query: str, max_iterations: int) -> dict:
    return {
        "query_id": str(uuid.uuid4()),
        "user_id": user_id,
        "raw_query": raw_query,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "committed_actions": [],
        "next_action": None,
        "candidate_pool": {},
        "failed_strategies": [],
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "confidence": 0.0,
        "stopping_decision": {},
        "clarification_question": None,
        "final_ranking": [],
        "explanations": {},
        "errors": [],
        "degraded_mode": False,
        "cold_start": False,
        "llm_goal_metadata": {},
        "query_expansion_metadata": {},
        "candidate_comparison": {},
        "candidate_explanation": {},
        "trace": [],
    }


def run_query(
    user_id: str,
    raw_query: str,
    max_iterations: Optional[int] = None,
    registry=None,
    llm_provider=None,
) -> dict:
    max_iterations = max_iterations if max_iterations is not None else MAX_ITERATIONS
    graph = build_graph(registry=registry, max_iterations=max_iterations, llm_provider=llm_provider)
    initial_state = _initial_state(user_id, raw_query, max_iterations)

    try:
        final_state = graph.invoke(initial_state, config={"recursion_limit": 100})
    except (DatabaseError, UnknownUserError) as exc:
        # Hard failures (architecture_final.md §7/§11/§14): never degrade,
        # never silently swallow -- surface clearly.
        result = {
            "query_id": initial_state["query_id"],
            "user_id": user_id,
            "raw_query": raw_query,
            "status": "hard_failure",
            "error": str(exc),
        }
        _log_run(result, trace=[])
        return result

    status = final_state["stopping_decision"].get("status", "unknown")
    result = {
        "query_id": final_state["query_id"],
        "user_id": user_id,
        "raw_query": raw_query,
        "status": status,
        "reason": final_state["stopping_decision"].get("reason"),
        "decision_basis": final_state["stopping_decision"].get("basis"),  # Phase 9: why a finish was decided
        "confidence": final_state.get("confidence", 0.0),
        "degraded_mode": final_state.get("degraded_mode", False),
        "clarification_question": final_state.get("clarification_question"),
        "llm_goal_metadata": final_state.get("llm_goal_metadata", {}),
        "query_expansion_metadata": final_state.get("query_expansion_metadata", {}),
        "candidate_comparison": final_state.get("candidate_comparison", {}),
        "candidate_explanation": final_state.get("candidate_explanation", {}),
        "explanations": final_state.get("explanations", {}),
        "final_ranking": final_state.get("final_ranking", []),
        "iteration_count": final_state.get("iteration_count", 0),
        "committed_actions": final_state.get("committed_actions", []),
        "candidate_pool": final_state.get("candidate_pool", {}),
        "errors": final_state.get("errors", []),
        "trace": final_state.get("trace", []),
    }
    _log_run(result, trace=final_state.get("trace", []))
    return result


def _log_run(result: dict, trace: list) -> None:
    """Structured, per-run/per-iteration logging (architecture_final.md §12,
    prompt §11). No chain-of-thought/hidden reasoning is persisted -- only
    the observable action/observation/decision fields.
    """
    for entry in trace:
        if entry["event"] == "execute_tool":
            logger.info(
                "iteration=%s tool=%s args=%s candidate_count=%s error=%s query_id=%s user_id=%s",
                entry["iteration"],
                entry["tool"],
                entry["args"],
                entry["candidate_count"],
                entry["error"],
                result["query_id"],
                result["user_id"],
            )
        elif entry["event"] == "reflect":
            logger.info(
                "iteration=%s reflect_status=%s reason=%s confidence=%s query_id=%s",
                entry["iteration"],
                entry["status"],
                entry["reason"],
                entry["confidence"],
                result["query_id"],
            )
    logger.info(
        "run_complete query_id=%s user_id=%s status=%s degraded_mode=%s iterations=%s",
        result["query_id"],
        result["user_id"],
        result["status"],
        result.get("degraded_mode"),
        result.get("iteration_count"),
    )


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Run the Phase 1 file discovery agent.")
    parser.add_argument("user_id")
    parser.add_argument("query")
    args = parser.parse_args()

    out = run_query(args.user_id, args.query)
    print(json.dumps(out, default=str, indent=2))
