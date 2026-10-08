"""
plan_next_action -- architecture_final.md §5.

Selects exactly ONE next action at a time. Never produces a multi-step
pipeline in advance. Reads the evolving plan (committed_actions,
failed_strategies) plus current goal/candidate_pool state.

Phase 1 has exactly one tool (metadata_search), so "choosing a strategy"
means choosing *which combination of metadata_search filters* to try. The
combinations are generated from the goal's available signals and ordered
from most-specific (all available filters applied) to progressively more
relaxed, matching the adaptation table's "relax weakest filter, retry"
guidance (§7). Filters are relaxed in this order (weakest/least reliable
signal dropped first): file_type -> date_range -> filename_candidate.
topic_tokens are never dropped -- they are the actual relevance signal.

If a query is broad enough that "too many/flat" candidates come back and
there is no narrower, untried, available strategy left, plan_next_action
returns None -- signalling that Phase 1's capabilities are exhausted (the
architecture's escalation to hybrid_search/rerank is deferred to a later
phase), and the graph should move to `clarify`.
"""
import json
from dataclasses import asdict
from datetime import datetime
from typing import List, Optional

from app.goal import EXPLICIT_FILE_RE
from app.state import Action
from app.tools.rerank_candidates import candidates_from_pool

RELAXATION_ORDER = ["file_type", "date_range", "filename_candidate"]


def is_exact_filename_request(goal: dict) -> bool:
    """True iff the USER'S OWN QUERY names a complete filename (with an
    extension, e.g. "Report_Final.xlsx") and that is the filename_candidate.

    Phase 8: this distinguishes "find THIS exact file" from "find files related
    to this name/topic" (e.g. `named Meeting_Notes`, no extension). An exact
    filename is a hard constraint that must never be relaxed away or replaced by
    a broader lexical/semantic search -- doing so degraded a miss into an
    unrelated same-extension result (Phase 7 finding G1).

    Derived deterministically from goal["raw_query"] only, never from an
    LLM-suggested filename_candidate (Phase 5A boundary: the LLM can never turn
    a filename into a non-relaxable constraint). Not tied to any particular
    filename.
    """
    candidate = (goal.get("filters") or {}).get("filename_candidate")
    if not candidate:
        return False
    m = EXPLICIT_FILE_RE.search(goal.get("raw_query") or "")
    return bool(m) and m.group(0).lower() == str(candidate).lower()


def _json_default(o):
    if isinstance(o, datetime):
        return o.isoformat()
    return str(o)


def args_signature(tool: str, args: dict) -> str:
    """Canonical (tool, args) signature used for failed_strategies tracking."""
    return f"{tool}:{json.dumps(args, sort_keys=True, default=_json_default)}"


def _merged_topic_tokens(goal: dict) -> List[str]:
    """topic_tokens with Phase 5B's LLM-suggested expanded_terms folded in
    (requirement 9). Purely additive and order-preserving; goal["topic_tokens"]
    itself (the deterministic signal) is never mutated -- this only builds
    the list used to construct a given tool's args. Goals without an
    "expanded_terms" key (e.g. built directly by app.goal.understand_goal,
    bypassing the graph's expand_query node) behave exactly as before."""
    tokens = list(goal.get("topic_tokens") or [])
    for term in goal.get("expanded_terms") or []:
        if term not in tokens:
            tokens.append(term)
    return tokens


def _query_text_with_expansion(base_text: Optional[str], goal: dict) -> Optional[str]:
    """Appends any Phase 5B expanded_terms not already present in
    `base_text` (case-insensitively) as extra space-separated words --
    natural for both bm25/lexical matching and embedding-based semantic
    matching. Returns `base_text` unchanged when there is nothing to add,
    including when the goal has no "expanded_terms" key at all."""
    expanded = goal.get("expanded_terms") or []
    if not expanded:
        return base_text
    base_lower = (base_text or "").lower()
    extra = [t for t in expanded if t not in base_lower]
    if not extra:
        return base_text
    return " ".join(([base_text] if base_text else []) + extra)


def _base_args(goal: dict, drop: List[str]) -> dict:
    """Builds metadata_search args from the goal, omitting any filter named
    in `drop`. topic_tokens (merged with any Phase 5B expanded_terms) are
    always included when present -- they are never a candidate for
    relaxation."""
    args = {}
    filters = goal["filters"]

    if "filename_candidate" not in drop and filters.get("filename_candidate"):
        args["filename_contains"] = filters["filename_candidate"]

    if "file_type" not in drop and filters.get("file_type"):
        args["file_type"] = filters["file_type"]

    if "date_range" not in drop and filters.get("date_range"):
        args["date_from"] = filters["date_range"]["from"]
        args["date_to"] = filters["date_range"]["to"]

    merged_topic_tokens = _merged_topic_tokens(goal)
    if merged_topic_tokens:
        args["topic_tokens"] = merged_topic_tokens

    return args


def _bm25_args(goal: dict) -> Optional[dict]:
    """Builds bm25_search args from the goal. Returns None when there is no
    lexical query text to search on -- bm25_search is a fallback for
    evidence metadata_search's structural filters couldn't find, not a
    structural-filter tool itself, so it needs actual topic_tokens (Phase
    5B expanded_terms alone, with no deterministic topic_tokens, are not
    enough to trigger it -- they only enrich an already-real query)."""
    topic_tokens = goal.get("topic_tokens") or []
    if not topic_tokens:
        return None
    base_text = goal.get("concept_text") or " ".join(topic_tokens)
    args = {"query": _query_text_with_expansion(base_text, goal)}
    filters = goal["filters"]
    if filters.get("file_type"):
        args["file_type"] = filters["file_type"]
    if filters.get("date_range"):
        args["date_from"] = filters["date_range"]["from"]
        args["date_to"] = filters["date_range"]["to"]
    return args


def _semantic_args(goal: dict) -> Optional[dict]:
    """Builds semantic_search args from the goal. Unlike _bm25_args, this
    uses the raw natural-language query rather than the stripped
    topic-token list: embedding models are trained on natural phrasing, so
    keeping stopwords/word order tends to help rather than hurt (unlike
    bm25/metadata_search's substring/token matching, where stripped
    keywords are the whole point). Falls back to concept_text if
    raw_query is somehow absent (shouldn't happen in practice). Phase 5B
    expanded_terms are appended after the natural-language query."""
    query_text = goal.get("raw_query") or goal.get("concept_text")
    if not query_text:
        return None
    args = {"query": _query_text_with_expansion(query_text, goal)}
    filters = goal["filters"]
    if filters.get("file_type"):
        args["file_type"] = filters["file_type"]
    if filters.get("date_range"):
        args["date_from"] = filters["date_range"]["from"]
        args["date_to"] = filters["date_range"]["to"]
    return args


def _hybrid_args(goal: dict) -> Optional[dict]:
    """Builds hybrid_search args from the goal. Same query-text choice as
    _semantic_args (raw_query, with Phase 5B expanded_terms appended) --
    hybrid_search forwards this one query string unchanged to both
    bm25_search and semantic_search, so both rankings being fused answer
    the same (possibly expanded) question."""
    query_text = goal.get("raw_query") or goal.get("concept_text")
    if not query_text:
        return None
    args = {"query": _query_text_with_expansion(query_text, goal)}
    filters = goal["filters"]
    if filters.get("file_type"):
        args["file_type"] = filters["file_type"]
    if filters.get("date_range"):
        args["date_from"] = filters["date_range"]["from"]
        args["date_to"] = filters["date_range"]["to"]
    return args


def _already_tried(tool: str, args: dict, failed_strategies: List[str], committed_actions: List[Action]) -> bool:
    sig = args_signature(tool, args)
    if sig in failed_strategies:
        return True
    return any(args_signature(a["tool"], a["args"]) == sig for a in committed_actions)


def _was_committed(tool: str, committed_actions: List[Action]) -> bool:
    """Whether `tool` was executed at all in this run, regardless of which
    exact args it was called with (bm25_search/semantic_search may have
    been tried more than once with different filters during escalation --
    hybrid_search's trigger only cares that each was genuinely exercised,
    not with which specific arguments)."""
    return any(a["tool"] == tool for a in committed_actions)


def _candidate_strategies(goal: dict) -> List[dict]:
    """Ordered list of args dicts, from most-specific (nothing dropped) to
    progressively relaxed (RELAXATION_ORDER items dropped one at a time,
    cumulatively). Strategies that end up identical to a previous one (e.g.
    because the dropped filter was never present) are de-duplicated, and
    the fully-empty (no filters at all) strategy is never offered -- an
    unconstrained scan is not a valid Phase 1 action.
    """
    strategies = []
    seen_sigs = set()
    dropped: List[str] = []

    # Phase 8: an exact filename request is a hard constraint -- file_type and
    # date_range may still be relaxed, but the filename itself never is.
    relaxable = [r for r in RELAXATION_ORDER if not (r == "filename_candidate" and is_exact_filename_request(goal))]

    # Level 0: everything available applied.
    for level in range(0, len(relaxable) + 1):
        args = _base_args(goal, dropped)
        # Skip if this strategy has neither structural filters nor topic
        # tokens -- that would be an unconstrained scan.
        has_signal = bool(args)
        sig = args_signature("metadata_search", args)
        if has_signal and sig not in seen_sigs:
            strategies.append(args)
            seen_sigs.add(sig)
        if level < len(relaxable):
            dropped = dropped + [relaxable[level]]

    return strategies


def plan_next_action(
    goal: dict,
    committed_actions: List[Action],
    failed_strategies: List[str],
    iteration: int,
    observation_hint: Optional[str] = None,
    candidate_pool: Optional[dict] = None,
) -> Optional[Action]:
    """
    observation_hint:
        None            -- first action of the run
        "zero_results"  -- previous action returned nothing; the strategy
                            list is walked FORWARD (toward progressively
                            relaxed/broader strategies), per the adaptation
                            table's "relax weakest filter, retry".
        "too_many"      -- previous action returned too many/flat
                            candidates. metadata_search strategies are
                            always tried narrowest-first, so there is by
                            construction nothing narrower left there.
                            However (Phase 2C), if both bm25_search AND
                            semantic_search have already independently
                            contributed evidence to this run and are
                            leaving a flat/tied pool, hybrid_search's RRF
                            fusion is a genuine, different strategy for
                            resolving that ambiguity -- not a narrower
                            metadata filter, but a different combination
                            of existing lexical+semantic signals. This is
                            NOT a fixed metadata->bm25->semantic->hybrid
                            chain: hybrid_search is only ever proposed
                            here, in response to an actual "too_many"
                            observation, and only once both of its inputs
                            have genuinely been exercised in this run.

                            (Phase 3) If hybrid_search isn't available/isn't
                            new, and the accumulated candidate_pool still
                            has multiple candidates, deterministic
                            cross-encoder reranking of that already-
                            retrieved pool (rerank_candidates) is the next
                            escalation -- a reordering of existing evidence,
                            not a new search, and tried at most once per run.

                            (Phase 4A) If rerank_candidates has already run
                            this turn and the pool is STILL flat/tied,
                            get_user_activity is the next and final
                            escalation in this chain -- exactly the
                            adaptation table's "Reranked, still tied
                            (Δ < ε) -> get_user_activity on those specific
                            candidates" (§7). Only proposed once rerank has
                            genuinely been tried (never as a substitute for
                            reranking, and never before it), and at most
                            once per run.

    candidate_pool: the agent's current accumulated candidate_pool (§4),
        needed only to build rerank_candidates'/get_user_activity's args
        (both operate on candidates already retrieved, not a fresh
        search). Optional/None for callers (existing tests, non-"too_many"
        paths) that don't need either escalation.
    """
    if observation_hint == "weak_evidence":
        # Phase 12: only a minority-coverage lexical match exists. Independent semantic evidence is the
        # adaptation table's answer ("bm25 weak -> semantic_search"); proposed at most once, and only in
        # response to this observation (not a fixed pipeline). Nothing else to try -> None, and the graph
        # finishes with the weak candidate flagged low-confidence.
        semantic_args = _semantic_args(goal)
        if (
            semantic_args is not None
            and not _was_committed("semantic_search", committed_actions)
            and not _already_tried("semantic_search", semantic_args, failed_strategies, committed_actions)
        ):
            return Action(
                tool="semantic_search",
                args=semantic_args,
                reason=(
                    "Only weak lexical evidence (a minority of the query's terms) so far; "
                    "seeking independent semantic evidence"
                ),
                iteration=iteration,
            )
        return None
    if observation_hint == "too_many":
        hybrid_args = _hybrid_args(goal)
        if (
            hybrid_args is not None
            and _was_committed("bm25_search", committed_actions)
            and _was_committed("semantic_search", committed_actions)
            and not _already_tried("hybrid_search", hybrid_args, failed_strategies, committed_actions)
        ):
            return Action(
                tool="hybrid_search",
                args=hybrid_args,
                reason=(
                    "Flat/tied candidate pool after independent bm25_search and "
                    "semantic_search evidence; fusing both via RRF (hybrid_search)"
                ),
                iteration=iteration,
            )

        if (
            candidate_pool
            and len(candidate_pool) > 1
            and not _was_committed("rerank_candidates", committed_actions)
        ):
            query_text = goal.get("raw_query") or goal.get("concept_text")
            if query_text:
                return Action(
                    tool="rerank_candidates",
                    args={
                        "query": query_text,
                        "candidates": candidates_from_pool(candidate_pool),
                    },
                    reason=(
                        "Flat/tied candidate pool with no further retrieval "
                        "strategy available; reranking the already-retrieved "
                        "candidates with a deterministic cross-encoder"
                    ),
                    iteration=iteration,
                )

        if (
            candidate_pool
            and len(candidate_pool) > 1
            and _was_committed("rerank_candidates", committed_actions)
            and not _was_committed("get_user_activity", committed_actions)
        ):
            return Action(
                tool="get_user_activity",
                args={"candidates": candidates_from_pool(candidate_pool)},
                reason=(
                    "Still flat/tied after cross-encoder reranking; pulling "
                    "user activity/profile signals for these specific "
                    "candidates as a tie-breaker"
                ),
                iteration=iteration,
            )

        return None

    strategies = _candidate_strategies(goal)

    for args in strategies:
        sig = args_signature("metadata_search", args)
        if sig in failed_strategies:
            continue
        already_tried = any(
            args_signature(a["tool"], a["args"]) == sig for a in committed_actions
        )
        if already_tried:
            continue

        reason = _describe_reason(args, goal, committed_actions)
        return Action(tool="metadata_search", args=args, reason=reason, iteration=iteration)

    # Phase 2B: metadata_search's structural strategies are exhausted.
    # "too_many" already returned above (no narrowing tool for that yet).
    # For "zero_results", the adaptation table (§7) calls for escalating
    # through progressively less-structural, more-semantic retrieval --
    # bm25_search (genuine lexical), then semantic_search (genuine
    # embedding similarity) -- each only reached after the previous tier
    # was tried and observed to fail. This is a real, observation-driven
    # escalation chain, not an unconditional metadata->bm25->semantic
    # pipeline: any tier can be, and typically is, skipped entirely
    # whenever an earlier tier already found something (see test_agent_loop
    # test_A/test_E, which never touch bm25_search or semantic_search).
    if observation_hint == "zero_results" and is_exact_filename_request(goal):
        # Phase 8: bm25/semantic take free query text and cannot honour an exact
        # filename constraint, so escalating would silently turn "this exact
        # file" into "files vaguely like it". A missing exact filename is a
        # genuine nothing-found (the graph finishes safely with no result).
        return None
    if observation_hint == "zero_results":
        bm25_args = _bm25_args(goal)
        if bm25_args is not None and not _already_tried(
            "bm25_search", bm25_args, failed_strategies, committed_actions
        ):
            return Action(
                tool="bm25_search",
                args=bm25_args,
                reason=(
                    "metadata_search strategies exhausted with zero results; "
                    "escalating to genuine BM25 lexical retrieval"
                ),
                iteration=iteration,
            )

        semantic_args = _semantic_args(goal)
        if semantic_args is not None and not _already_tried(
            "semantic_search", semantic_args, failed_strategies, committed_actions
        ):
            return Action(
                tool="semantic_search",
                args=semantic_args,
                reason=(
                    "metadata_search and bm25_search both exhausted with zero results; "
                    "escalating to semantic (embedding) retrieval"
                ),
                iteration=iteration,
            )

    return None  # no untried, available strategy -- caller should clarify


def _describe_reason(args: dict, goal: dict, committed_actions: List[Action]) -> str:
    if not committed_actions:
        parts = []
        if args.get("filename_contains"):
            parts.append("filename candidate")
        if args.get("file_type"):
            parts.append("file_type filter")
        if args.get("date_from"):
            parts.append("date range")
        if args.get("topic_tokens"):
            parts.append("topic tokens")
        return "Initial strategy using available signals: " + ", ".join(parts)

    prev_args = committed_actions[-1]["args"]
    dropped = [k for k in prev_args if k not in args and k != "topic_tokens"]
    added = [k for k in args if k not in prev_args]
    if dropped:
        return f"Relaxing over-narrow filter(s) from previous attempt: {', '.join(dropped)}"
    if added:
        return f"Narrowing with additional filter(s): {', '.join(added)}"
    return "Trying alternate metadata_search strategy"
