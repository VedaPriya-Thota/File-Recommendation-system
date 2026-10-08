"""
Agent state -- architecture_final.md §4, Phase 1 subset.

Only the fields Phase 1 actually uses are populated; the rest of the
architecture's state shape (final_ranking/explanations/degraded_mode etc.)
is represented so later phases don't need a state-shape migration, but
Phase 1 leaves them at their defaults.

Implemented as a plain dict (a TypedDict) rather than a class because
LangGraph's StateGraph works most naturally with dict-like state that
nodes return partial updates for.
"""
from typing import Any, Dict, List, Optional, TypedDict


class Action(TypedDict):
    tool: str
    args: dict
    reason: str
    iteration: int


class ToolError(TypedDict):
    tool: str
    error: str
    fallback_taken: str


class StoppingDecision(TypedDict, total=False):
    status: str  # "finish" | "replan" | "clarify"
    reason: str
    # Phase 9: WHY a finish was decided (see app/pool_resolution.py BASIS_*).
    # Unset for replan/clarify and for finishes with no ranking basis (nothing found, tool error).
    basis: str


class AgentState(TypedDict, total=False):
    # --- identity / input ---
    query_id: str
    user_id: str
    raw_query: str
    timestamp: str

    # --- goal (populated by understand_goal; possibly LLM-augmented, see
    # app/llm/goal_interpreter.py -- always deterministic-first) ---
    goal: Dict[str, Any]
    # Phase 5A: decision metadata only (no chain-of-thought) about whether/
    # how the LLM goal-interpretation slice ran for this query. Always
    # present once understand_goal has run; {"llm_invoked": False, ...} for
    # the (majority) unambiguous-query case where the LLM was never called.
    llm_goal_metadata: Dict[str, Any]
    # Phase 5B: same pattern, for LLM query expansion (app/llm/
    # query_expander.py). goal["expanded_terms"] (a plain list[str],
    # separate from goal["topic_tokens"]) carries the actual terms;
    # this is decision metadata only, mirroring llm_goal_metadata.
    query_expansion_metadata: Dict[str, Any]
    # Phase 5C: decision metadata (no chain-of-thought) for the LLM
    # candidate-comparison slice (app/llm/candidate_comparer.py). Always
    # present once the finish path has run; {"comparison_invoked": False,
    # ...} when the top candidates were not genuinely near-tied.
    candidate_comparison: Dict[str, Any]
    # Phase 5D: presentation-only explanation of the final top result
    # (app/llm/result_explainer.py). Holds the bound `explained_file_id`,
    # status (generated | unavailable | provider_error | invalid_output |
    # not_needed), provider name, and the summary/evidence points/
    # confidence when generated. Never influences ranking.
    candidate_explanation: Dict[str, Any]

    # --- scope (populated by resolve_scope, immutable after) ---
    user_scope: List[str]

    # --- plan ---
    committed_actions: List[Action]
    next_action: Optional[Action]

    # --- candidate pool ---
    candidate_pool: Dict[int, dict]

    # --- loop bookkeeping ---
    failed_strategies: List[str]  # serialized (tool, args_signature) pairs
    iteration_count: int
    max_iterations: int

    # --- reflection ---
    confidence: float
    stopping_decision: StoppingDecision
    clarification_question: Optional[str]
    observation_hint: Optional[str]  # "zero_results" | "too_many" | "error" | None -- set by reflect, read by plan_next_action

    # --- last tool execution (transient; consumed by update_candidate_pool/reflect) ---
    last_tool_result: Optional[dict]

    # --- final output ---
    final_ranking: List[dict]
    explanations: Dict[int, str]
    errors: List[ToolError]
    degraded_mode: bool
    cold_start: bool

    # --- observability ---
    trace: List[dict]
