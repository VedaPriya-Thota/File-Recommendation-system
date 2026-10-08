"""
metadata_search -- deterministic SQL predicate lookup (architecture_final.md
§6, §10).

Supported args (all optional, combined with AND):
    file_id             int
    filename_contains   str   -- ILIKE '%...%' fuzzy match
    filename_exact       str   -- exact filename match
    file_type            str
    date_from / date_to  datetime -- inclusive range on updated_at
    topic_tokens          list[str] -- matched against tags (overlap) OR
                                        summary/filename (ILIKE any token)

Access control (architecture_final.md §11, non-negotiable):
    `owner_id = ANY(%s)` is always injected into the WHERE clause, built
    from `user_scope.owner_ids`. There is no code path in this function
    that can return a row whose owner_id is outside user_scope -- filtering
    never happens after the fact in Python.

Note on matching: filename/summary matching here is plain ILIKE substring
matching (unindexed at Phase 1's data scale). This is NOT BM25 and is not
a stand-in for it -- it exists only so metadata_search can react to
topic_tokens deterministically. Genuine lexical retrieval (`bm25_search`,
architecture_final.md §6, using `bm25s` per §10) is a separate, later tool
that this registry can accommodate without changes to this file.
"""
import re
from typing import Optional

from app.config import METADATA_SHORT_TOKEN_MAX_LEN
from app.db.connection import get_cursor
from app.scope import UserScope
from app.tools.registry import Tool, ToolResult

TOOL_NAME = "metadata_search"


def _is_short_token(token: str) -> bool:
    return len(token) <= METADATA_SHORT_TOKEN_MAX_LEN


def _token_pattern(token: str) -> str:
    """Whole-token regex (Postgres ARE / Python `re` compatible): the token
    delimited by a non-alphanumeric character or the string boundary. Note
    `_` and `.` count as delimiters, so AI_Architecture.pptx matches `ai`."""
    return rf"(^|[^a-z0-9]){re.escape(token.lower())}([^a-z0-9]|$)"


def _token_in_text(token: str, text: str) -> bool:
    """The ONE definition of "token appears in text", shared by the SQL
    predicate's semantics and the Python match-strength scorer so they can
    never disagree. Short tokens: whole-token. Longer: substring."""
    token, text = token.lower(), text.lower()
    if _is_short_token(token):
        return re.search(_token_pattern(token), text) is not None
    return token in text


def _build_query(args: dict, owner_ids):
    where = ["owner_id = ANY(%(owner_ids)s)"]  # access control -- always present, first
    params = {"owner_ids": list(owner_ids)}

    if args.get("file_id") is not None:
        where.append("file_id = %(file_id)s")
        params["file_id"] = args["file_id"]

    if args.get("filename_exact"):
        where.append("filename ILIKE %(filename_exact)s")
        params["filename_exact"] = args["filename_exact"]

    if args.get("filename_contains"):
        where.append("filename ILIKE %(filename_contains)s")
        params["filename_contains"] = f"%{args['filename_contains']}%"

    if args.get("file_type"):
        where.append("file_type = %(file_type)s")
        params["file_type"] = args["file_type"]

    if args.get("date_from"):
        where.append("updated_at >= %(date_from)s")
        params["date_from"] = args["date_from"]

    if args.get("date_to"):
        where.append("updated_at <= %(date_to)s")
        params["date_to"] = args["date_to"]

    topic_tokens = args.get("topic_tokens") or []
    if topic_tokens:
        # A candidate matches if it has tag overlap OR any token appears in
        # its filename/summary. Built as one OR-group, still AND-ed with
        # every other filter (including the mandatory owner_id predicate).
        clauses = []
        for i, tok in enumerate(topic_tokens):
            if _is_short_token(tok):
                # whole-token match only (Phase 8) -- never a bare substring
                clauses.append(f"filename ~* %(token_{i})s OR summary ~* %(token_{i})s")
                params[f"token_{i}"] = _token_pattern(tok)
            else:
                clauses.append(f"filename ILIKE %(token_{i})s OR summary ILIKE %(token_{i})s")
                params[f"token_{i}"] = f"%{tok}%"
        where.append("(tags && %(topic_tokens)s OR " + " OR ".join(clauses) + ")")
        params["topic_tokens"] = topic_tokens
    sql = f"""
        SELECT file_id, filename, file_type, file_path, owner_id,
               created_at, updated_at, tags, summary
        FROM files
        WHERE {' AND '.join(where)}
        ORDER BY updated_at DESC
        LIMIT 50;
    """
    return sql, params


def _score_row(row: dict, args: dict) -> float:
    """Simple, explainable match-strength score (0..1).

    Not a ranking model -- a deterministic count of "how many of the
    filters that were actually applied does this row satisfy strongly",
    used only so Tier-1 reflection can tell a dominant match from a flat
    tie (architecture_final.md §7).
    """
    criteria_applied = 0
    criteria_hit = 0

    if args.get("filename_exact"):
        criteria_applied += 1
        criteria_hit += 1  # SQL already enforced exact match
    if args.get("filename_contains"):
        criteria_applied += 1
        criteria_hit += 1
    if args.get("file_type"):
        criteria_applied += 1
        criteria_hit += 1
    if args.get("date_from") or args.get("date_to"):
        criteria_applied += 1
        criteria_hit += 1

    topic_tokens = args.get("topic_tokens") or []
    if topic_tokens:
        criteria_applied += 1
        tags = set(t.lower() for t in (row.get("tags") or []))
        tag_hits = len(tags.intersection(t.lower() for t in topic_tokens))
        text = f"{row.get('filename', '')} {row.get('summary', '')}".lower()
        text_hits = sum(1 for t in topic_tokens if _token_in_text(t, text))
        strength = min(1.0, (tag_hits + text_hits) / max(1, len(topic_tokens)))
        criteria_hit += strength

    if criteria_applied == 0:
        return 0.5  # no filters at all -- neutral score, shouldn't normally happen
    return round(criteria_hit / criteria_applied, 4)


def metadata_search(args: dict, user_scope: UserScope) -> ToolResult:
    sql, params = _build_query(args, user_scope.owner_ids)
    with get_cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    candidates = []
    for row in rows:
        row = dict(row)
        candidates.append(
            {
                "file_id": row["file_id"],
                "metadata": {
                    "filename": row["filename"],
                    "file_type": row["file_type"],
                    "file_path": row["file_path"],
                    "owner_id": row["owner_id"],
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat(),
                    "tags": row["tags"],
                    "summary": row["summary"],
                },
                "score": _score_row(row, args),
                "source": TOOL_NAME,
            }
        )

    return ToolResult(tool=TOOL_NAME, candidates=candidates)


def make_tool() -> Tool:
    return Tool(
        name=TOOL_NAME,
        description=(
            "Exact/predicate SQL lookup over file metadata (filename, "
            "file_type, date range, topic tokens against tags/summary). "
            "Access control (owner_id scope) is enforced inside the query."
        ),
        fn=metadata_search,
    )
