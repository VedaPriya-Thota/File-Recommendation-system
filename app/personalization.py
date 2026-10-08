"""
Deterministic personalization signal computation (architecture_final.md
§8; Phase 4A -- "Deterministic Personalization Signals" only).

Pure functions: no DB access here (that's app/tools/get_user_activity.py's
job) and no LLM anywhere. Computes, per candidate, a small set of
interpretable signals from:
    - the candidate's own metadata (file_type, tags, filename, summary) --
      already retrieved/scoped, never looked up fresh here
    - the calling user's user_profiles row (preferred_file_types,
      frequent_topics) -- may be None (cold start / no profile yet)
    - the calling user's user_file_telemetry rows for this one file --
      may be empty (cold start / never interacted with this file)

Cold start (architecture_final.md §8 "Cold start"): a missing profile
and/or empty telemetry never raises and never contaminates the result --
every signal defaults to its neutral value (False / 0 / 0.0) and
`cold_start` is set True whenever there is genuinely nothing (no profile
AND no telemetry) to go on. Phase 4A stops at computing/attaching/exposing
these signals -- the architecture's cold-start "redistribute affinity
weight (γ) into retrieval weight (α)" mechanism belongs to the later final-
weighted-scoring phase (§10), which is explicitly NOT implemented here.

`affinity_score` is a single [0, 1] composite, for convenience/observability
and as an optional tie-break input ONLY. It is never written into
candidate_pool's shared `scores` dict in a way that overwrites or sums with
retrieval scores (see app/tools/get_user_activity.py's module docstring):
it becomes its own distinct per-source score, exactly like every other
tool's, and `personalized_order` below only ever uses it to break ties
among candidates already ranked by real retrieval evidence -- never as a
substitute for that evidence (architecture_final.md §8's anti-circularity
guarantee).
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional

# Recency decay: an interaction "today" contributes full weight; the
# contribution decays linearly to 0 by RECENCY_HORIZON_DAYS. Deliberately
# simple/explicit (a real decay-rate config, per architecture_final.md
# §19's "λ (recency decay)", is a later-phase tuning knob) -- this is
# Phase 4A's fixed default.
RECENCY_HORIZON_DAYS = 90

# Caps activity-event counts before normalizing to [0, 1], so a handful of
# extremely active files can't blow out the whole [0, 1] range -- an
# explicit, documented deterministic choice, not a learned parameter.
ACTIVITY_COUNT_CAP = 10


def _file_type_match(candidate_metadata: dict, preferred_file_types: List[str]) -> bool:
    file_type = candidate_metadata.get("file_type")
    return bool(file_type) and file_type in (preferred_file_types or [])


def _topic_matches(candidate_metadata: dict, frequent_topics: dict) -> Dict[str, float]:
    """Returns {topic: weight} for every frequent_topics entry that matches
    this candidate's tags/filename/summary -- a plain
    substring/membership check (deterministic, no embeddings/LLM)."""
    if not frequent_topics:
        return {}
    tags = {str(t).lower() for t in (candidate_metadata.get("tags") or [])}
    filename = (candidate_metadata.get("filename") or "").lower()
    summary = (candidate_metadata.get("summary") or "").lower()

    matches: Dict[str, float] = {}
    for topic, weight in frequent_topics.items():
        topic_l = str(topic).lower()
        if topic_l in tags or topic_l in filename or topic_l in summary:
            try:
                matches[topic] = float(weight)
            except (TypeError, ValueError):
                matches[topic] = 1.0
    return matches


def _activity_summary(telemetry_events: List[dict]) -> dict:
    counts: Dict[str, int] = {}
    total_time = 0
    last_ts = None
    for ev in telemetry_events:
        action = ev.get("action_type") or "unknown"
        counts[action] = counts.get(action, 0) + 1
        total_time += ev.get("time_spent_seconds") or 0
        ts = ev.get("timestamp")
        if ts is not None and (last_ts is None or ts > last_ts):
            last_ts = ts

    days_since_last = None
    if last_ts is not None:
        now = datetime.now(timezone.utc)
        days_since_last = max((now - last_ts).total_seconds() / 86400.0, 0.0)

    return {
        "open_count": counts.get("open", 0),
        "view_count": counts.get("view", 0),
        "edit_count": counts.get("edit", 0),
        "total_events": len(telemetry_events),
        "total_time_spent_seconds": total_time,
        "last_interaction_at": last_ts.isoformat() if last_ts else None,
        "days_since_last_interaction": days_since_last,
    }


def _recency_score(days_since_last: Optional[float]) -> float:
    if days_since_last is None:
        return 0.0
    if days_since_last >= RECENCY_HORIZON_DAYS:
        return 0.0
    return round(1.0 - (days_since_last / RECENCY_HORIZON_DAYS), 4)


def _frequency_score(total_events: int) -> float:
    if total_events <= 0:
        return 0.0
    return round(min(total_events, ACTIVITY_COUNT_CAP) / ACTIVITY_COUNT_CAP, 4)


def compute_personalization_signals(
    candidate_metadata: dict,
    telemetry_events: List[dict],
    profile: Optional[dict],
) -> dict:
    """Pure computation -- no DB access. `profile` is the user_profiles row
    (or None -- no profile yet); `telemetry_events` is this one file's
    user_file_telemetry rows for the calling user (or [] -- never
    interacted with this file, or no telemetry at all)."""
    preferred_file_types = (profile or {}).get("preferred_file_types") or []
    frequent_topics = (profile or {}).get("frequent_topics") or {}

    file_type_match = _file_type_match(candidate_metadata, preferred_file_types)
    topic_matches = _topic_matches(candidate_metadata, frequent_topics)
    topic_score = round(min(sum(topic_matches.values()), 1.0), 4) if topic_matches else 0.0

    activity = _activity_summary(telemetry_events)
    recency_score = _recency_score(activity["days_since_last_interaction"])
    frequency_score = _frequency_score(activity["total_events"])

    cold_start = profile is None and activity["total_events"] == 0

    # Simple, transparent, deterministic composite -- an equal-weighted
    # mean of the four [0,1] signals actually available. This is NOT the
    # architecture's deferred alpha/beta/gamma final-weighted formula
    # (§10) -- that combines a whole affinity term against retrieval/
    # recency at the finish step, a later phase. This is just a single
    # self-contained personalization score for observability/tie-breaking
    # within this phase.
    components = [
        1.0 if file_type_match else 0.0,
        topic_score,
        recency_score,
        frequency_score,
    ]
    affinity_score = 0.0 if cold_start else round(sum(components) / len(components), 4)

    return {
        "preferred_file_type_match": file_type_match,
        "frequent_topic_matches": topic_matches,
        "frequent_topic_score": topic_score,
        **activity,
        "recency_score": recency_score,
        "frequency_score": frequency_score,
        "affinity_score": affinity_score,
        "cold_start": cold_start,
    }


def attach_personalization_signals(candidate_pool: dict, signals_by_file_id: Dict[int, dict]) -> dict:
    """Functional update (mirrors app/candidate_pool.py's style): returns a
    NEW candidate_pool with `entry["personalization"]` attached for every
    file_id present in `signals_by_file_id`. Never adds a new file_id --
    architecture_final.md §8's anti-circularity guarantee ("can never
    introduce a candidate with zero retrieval evidence"): any signal for a
    file_id not already in candidate_pool is silently ignored, not
    inserted. Existing `scores`/`metadata`/`evidence_sources` on every
    entry are left completely untouched."""
    pool = dict(candidate_pool)
    for file_id, signals in signals_by_file_id.items():
        if file_id not in pool:
            continue  # never introduce a candidate that wasn't already retrieved
        entry = dict(pool[file_id])
        entry["personalization"] = signals
        pool[file_id] = entry
    return pool


def personalized_order(candidate_pool: dict) -> List[int]:
    """Returns file_ids in final-ranking order.

    Phase 4B note: this used to contain its own standalone
    (max-retrieval-score, affinity) tie-break sort, kept separate from the
    graph's actual `finish_stage` logic (which, before Phase 4B, wrongly
    used `max(entry["scores"].values())` over ALL scores including
    personalization's -- see app/final_ranking.py's module docstring). That
    left two different, divergent ranking policies in the codebase. Phase
    4B introduces app/final_ranking.py as the one real policy (now wired
    into app/graph/build.py's finish_stage) and this function becomes a
    thin re-export of it, so there is exactly one ranking implementation
    -- this name is kept only for backward compatibility with existing
    callers/tests."""
    from app.final_ranking import personalized_order as _personalized_order

    return _personalized_order(candidate_pool)
