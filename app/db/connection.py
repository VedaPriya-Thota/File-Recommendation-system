"""
Thin Postgres connection helper.

Deliberately minimal: Phase 1 has exactly one deterministic SQL tool
(`metadata_search`) plus scope resolution, so a full ORM/connection-pool
abstraction would be premature. A context-managed connection/cursor is
enough and keeps every SQL statement auditable.
"""
from contextlib import contextmanager

import psycopg2
import psycopg2.extras

from app.config import DSN


class DatabaseError(Exception):
    """Raised on a hard Postgres failure.

    Per architecture_final.md §7 / §11 / §14, Postgres errors are NEVER
    degraded around (access control depends on Postgres) -- they must
    propagate as hard failures.
    """


@contextmanager
def get_connection():
    conn = None
    try:
        conn = psycopg2.connect(DSN)
        yield conn
    except psycopg2.Error as exc:
        raise DatabaseError(str(exc)) from exc
    finally:
        if conn is not None:
            conn.close()


@contextmanager
def get_cursor(commit: bool = False):
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            yield cur
            if commit:
                conn.commit()
        except psycopg2.Error as exc:
            conn.rollback()
            raise DatabaseError(str(exc)) from exc
        finally:
            cur.close()
