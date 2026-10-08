"""
resolve_scope -- architecture_final.md §11.

Runs once, first, per run. Produces an immutable `user_scope`: the list of
owner_ids this user is allowed to see. For Phase 1 the access model is the
simple owner-based model described in §19 ("single-tenant/owner-based
access model sufficient for V1; auth/user identity managed upstream") --
a user's scope is just their own owner_id. No team/sharing abstraction is
introduced since nothing in the architecture calls for it yet.

A Postgres failure here is a hard failure (never degraded around), because
every downstream retrieval tool depends on this scope for access control.
"""
from dataclasses import dataclass
from typing import Tuple

from app.db.connection import DatabaseError, get_cursor


class UnknownUserError(Exception):
    """Raised when the user_id does not exist. Treated as a hard failure
    (not silently returning an empty scope), since resolving to an empty
    scope + '0 results' would look identical to 'this user has no files'."""


@dataclass(frozen=True)
class UserScope:
    user_id: str
    owner_ids: Tuple[str, ...]  # immutable -- never mutated after resolution


def resolve_scope(user_id: str) -> UserScope:
    try:
        with get_cursor() as cur:
            cur.execute("SELECT 1 FROM users WHERE user_id = %s;", (user_id,))
            row = cur.fetchone()
    except DatabaseError:
        # Explicit re-raise to make the "no degrade" contract obvious at
        # the call site, per architecture_final.md §7 / §14.
        raise

    if row is None:
        raise UnknownUserError(f"Unknown user_id: {user_id!r}")

    # Owner-based scope: a user's accessible owner_ids is just themself.
    return UserScope(user_id=user_id, owner_ids=(user_id,))
