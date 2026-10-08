"""
get_user_activity -- read-only telemetry/profile lookup, scoped to an
already-retrieved candidate set (architecture_final.md §6 `get_user_activity`,
§8 touchpoint 2 "investigation/disambiguation"; Phase 4A).

Like rerank_candidates, this tool NEVER performs a new corpus search and
NEVER expands the candidate set: it receives the candidates already
produced by prior (authorized) retrieval actions -- via `args["candidates"]`
(the same `candidates_from_pool` shape rerank_candidates uses) -- and
returns, for each supplied file_id, deterministic personalization signals.
No file_id outside the supplied set can ever appear in the result: there is
no code path here that looks up files by anything other than the exact
file_ids it was handed.

Two read-only queries, BOTH always filtered by the calling user's own
`user_id` (architecture_final.md §11's immutable-scope discipline extended
to personalization data, prompt "keep user scope immutable and enforce
user_id filtering in every telemetry/profile query"):
    - `user_file_telemetry`, restricted to
      `user_id = %(user_id)s AND file_id = ANY(%(file_ids)s)` -- the
      CALLING user's OWN interaction history with exactly the supplied
      files, never another user's telemetry and never a file outside the
      supplied set.
    - `user_profiles`, restricted to `user_id = %(user_id)s` -- the
      calling user's own preferred_file_types / frequent_topics, fetched
      once per call (profile-level, not per-file).

`user_scope.user_id` (not `owner_ids`) is what's used for both queries:
`owner_ids` describes which files' *content* this user may see (Phase 1
access control, already enforced by whichever tool originally retrieved
these candidates); `user_id` is whose *activity/preferences* personalize
the ranking. For Phase 1's single-tenant scope these coincide, but the
distinction is deliberate.

Cold start: a user with no user_profiles row and/or no telemetry rows at
all produces neutral signals (see app/personalization.py) -- never an
error, never a KeyError/None crash.

Personalization signal computation itself lives in app/personalization.py
(a separate, directly unit-testable, DB-free module) -- this tool is a
thin DB-access + orchestration layer over it, mirroring how
bm25_search/semantic_search separate "fetch scoped rows" from "score".

Score handling: exactly like rerank_candidates, the composite
affinity_score enters candidate_pool as its own per-source score
(`scores["get_user_activity"]`) via the existing generic
`update_candidate_pool` merge -- ADDED alongside, never overwriting or
summed with, bm25/semantic/hybrid/reranker scores (architecture_final.md
§8's anti-circularity guarantee + this phase's explicit "do not overwrite
or numerically mix existing scores yet"). The full signal breakdown
(file-type match, topic matches, raw activity counts, recency, cold_start)
is preserved in candidate metadata (`metadata["personalization"]`) for
observability, alongside all prior evidence (matched_chunk/snippet/
rerank_raw_score/etc), which is copied forward unchanged.
"""
from typing import Dict, List, Optional

from app.db.connection import get_cursor
from app.personalization import compute_personalization_signals
from app.scope import UserScope
from app.tools.registry import Tool, ToolResult

TOOL_NAME = "get_user_activity"


def _fetch_telemetry(user_id: str, file_ids: List[int]) -> Dict[int, List[dict]]:
    if not file_ids:
        return {}
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT file_id, action_type, time_spent_seconds, timestamp
            FROM user_file_telemetry
            WHERE user_id = %(user_id)s AND file_id = ANY(%(file_ids)s);
            """,
            {"user_id": user_id, "file_ids": file_ids},
        )
        rows = [dict(row) for row in cur.fetchall()]
    by_file: Dict[int, List[dict]] = {}
    for row in rows:
        by_file.setdefault(row["file_id"], []).append(row)
    return by_file


def _fetch_profile(user_id: str) -> Optional[dict]:
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT user_id, preferred_file_types, frequent_topics
            FROM user_profiles
            WHERE user_id = %(user_id)s;
            """,
            {"user_id": user_id},
        )
        row = cur.fetchone()
        return dict(row) if row else None


def get_user_activity(args: dict, user_scope: UserScope) -> ToolResult:
    candidates = args.get("candidates") or []
    if not candidates:
        return ToolResult(tool=TOOL_NAME, candidates=[])

    file_ids = [c["file_id"] for c in candidates]
    telemetry_by_file = _fetch_telemetry(user_scope.user_id, file_ids)
    profile = _fetch_profile(user_scope.user_id)

    out = []
    for c in candidates:
        fid = c["file_id"]
        metadata = dict(c.get("metadata") or {})  # copy prior evidence forward, untouched
        signals = compute_personalization_signals(
            candidate_metadata=metadata,
            telemetry_events=telemetry_by_file.get(fid, []),
            profile=profile,
        )
        metadata["personalization"] = signals
        out.append(
            {
                "file_id": fid,
                "metadata": metadata,
                "score": signals["affinity_score"],
                "source": TOOL_NAME,
            }
        )
    return ToolResult(tool=TOOL_NAME, candidates=out)


def make_tool() -> Tool:
    return Tool(
        name=TOOL_NAME,
        description=(
            "Read-only, user-scoped telemetry/profile lookup for an "
            "already-retrieved candidate set. Never performs a new corpus "
            "search and cannot introduce a file outside the supplied "
            "candidates. Computes deterministic personalization signals "
            "(preferred file type match, frequent topic match, activity "
            "frequency/recency) from user_file_telemetry and user_profiles, "
            "scoped to the calling user_id. Cold-start users (no profile/"
            "telemetry) get neutral signals, never an error. Best used as "
            "a tie-breaker after reranking still leaves a flat/tied "
            "candidate pool. Requires `candidates` (file_id + metadata "
            "pairs, e.g. from the current candidate_pool)."
        ),
        fn=get_user_activity,
    )
