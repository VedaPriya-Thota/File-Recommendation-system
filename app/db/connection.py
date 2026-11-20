"""
Dual-engine database connection helper.

Supports PostgreSQL when reachable, and falls back seamlessly to Python's built-in
SQLite (`sqlite3`) so the project runs out of the box with zero external installation.
"""
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from app.config import DSN

_LEGACY_DB_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "file_discovery.db",
)
# DISCOVERY_SQLITE_PATH lets the application (api_server.py) keep its own database under ./data
# while the legacy tests/evaluation keep using the bundled file_discovery.db.
DB_FILE = os.environ.get("DISCOVERY_SQLITE_PATH") or _LEGACY_DB_FILE


class DatabaseError(Exception):
    """Raised on a hard database failure."""


_PG_PROBE = {}


def _use_postgres() -> bool:
    db_type = os.environ.get("DISCOVERY_DB_TYPE", "").lower()
    if db_type == "sqlite":
        return False
    if db_type == "postgres":
        return True
    # Probe PostgreSQL once per process (a failed probe is cached for 30 s so a late-started server is
    # picked up without paying a connection attempt on every query).
    import time
    cached = _PG_PROBE.get("v")
    if cached is not None and (cached[0] or time.monotonic() - cached[1] < 30):
        return cached[0]
    try:
        import psycopg2
        conn = psycopg2.connect(DSN, connect_timeout=1)
        conn.close()
        ok = True
    except Exception:
        ok = False
    _PG_PROBE["v"] = (ok, time.monotonic())
    return ok


def engine_name() -> str:
    return "postgresql" if _use_postgres() else "sqlite"


class SQLiteCursorWrapper:
    """Wrapper around sqlite3.Cursor to provide a psycopg2-compatible interface.
    Automatically translates Postgres syntax (%s, %(key)s, ILIKE, = ANY(...), ~*)
    into SQLite compatible query structures.
    """
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.cur = conn.cursor()
        self.last_description = None
        self._returning_id = None

    def execute(self, query: str, params=None):
        wants_returning = bool(re.search(r"RETURNING\s+file_id", query or "", flags=re.IGNORECASE))
        self._returning_id = None
        query, params = self._translate_query(query, params)
        try:
            # Handle multiple SQL statements if present
            statements = [s.strip() for s in query.split(";") if s.strip()]
            if len(statements) > 1 and params is None:
                for stmt in statements:
                    self.cur.execute(stmt)
            elif len(statements) > 0:
                if params is not None:
                    self.cur.execute(statements[0], params)
                else:
                    self.cur.execute(statements[0])
            self.last_description = self.cur.description
            if wants_returning:
                self._returning_id = self.cur.lastrowid
        except Exception as exc:
            raise DatabaseError(str(exc)) from exc

    def executemany(self, query: str, seq):
        for params in seq:
            self.execute(query, params)

    def fetchone(self):
        if self._returning_id is not None:  # emulates `INSERT ... RETURNING file_id`
            rid, self._returning_id = self._returning_id, None
            return {"file_id": rid}
        row = self.cur.fetchone()
        if row is None:
            return None
        return self._format_row(dict(row))

    def fetchall(self):
        rows = self.cur.fetchall()
        return [self._format_row(dict(r)) for r in rows]

    @property
    def rowcount(self):
        return self.cur.rowcount

    def close(self):
        self.cur.close()

    def _format_row(self, d: dict) -> dict:
        """Converts strings back to datetime objects and lists where expected."""
        formatted = {}
        for k, v in d.items():
            if k in ("created_at", "updated_at", "indexed_at", "timestamp", "last_viewed_at") and isinstance(v, str):
                try:
                    formatted[k] = datetime.fromisoformat(v)
                except Exception:
                    formatted[k] = v
            elif k in ("frequent_topics", "preferred_file_types") and isinstance(v, str):
                try:
                    formatted[k] = json.loads(v)
                except Exception:
                    formatted[k] = v
            elif k == "tags" and isinstance(v, str):
                try:
                    formatted[k] = json.loads(v)
                except Exception:
                    formatted[k] = [t.strip() for t in v.split(",") if t.strip()]
            else:
                formatted[k] = v
        return formatted

    @staticmethod
    def _expand_any(query: str, params):
        if isinstance(params, dict):
            params = dict(params)

            def repl(m):
                col, key = m.group(1), m.group(2)
                values = list(params.pop(key))
                if not values:
                    return "1 = 0"
                names = []
                for i, v in enumerate(values):
                    nm = f"{key}__{i}"
                    params[nm] = v
                    names.append(f":{nm}")
                return f"{col} IN ({', '.join(names)})"

            query = re.sub(r"([A-Za-z_][\w.]*)\s*=\s*ANY\(\s*%\((\w+)\)s\s*\)", repl, query)
            return query, params
        if isinstance(params, (list, tuple)):
            params = list(params)
            out, new_params, pi, pos = [], [], 0, 0
            pattern = re.compile(r"([A-Za-z_][\w.]*)\s*=\s*ANY\(\s*%s\s*\)|%s")
            for m in pattern.finditer(query):
                out.append(query[pos:m.start()])
                pos = m.end()
                if m.group(1):
                    values = list(params[pi])
                    pi += 1
                    if not values:
                        out.append("1 = 0")
                    else:
                        out.append(f"{m.group(1)} IN ({', '.join(['%s'] * len(values))})")
                        new_params.extend(values)
                else:
                    out.append("%s")
                    new_params.append(params[pi])
                    pi += 1
            out.append(query[pos:])
            return "".join(out), tuple(new_params)
        return query, params

    def _translate_query(self, query: str, params):
        if not query or params is None:
            query = re.sub(r"::\s*[A-Za-z_]+(\[\])?", "", query or "")
            # Simple cleanup for TRUNCATE, ON CONFLICT, RETURNING
            query = query.replace("TRUNCATE files, users RESTART IDENTITY CASCADE;", "DELETE FROM user_file_telemetry; DELETE FROM file_ingestion_state; DELETE FROM files; DELETE FROM users;")
            return query, params

        # 1. Handle `ON CONFLICT (user_id) DO NOTHING` -> `INSERT OR IGNORE`
        if "ON CONFLICT" in query:
            query = re.sub(r"INSERT\s+INTO", "INSERT OR IGNORE INTO", query, flags=re.IGNORECASE)
            query = re.sub(r"ON\s+CONFLICT\s*\([^\)]+\)\s*DO\s+NOTHING;?", ";", query, flags=re.IGNORECASE)

        # 2. Handle `<col> = ANY(%(name)s)` / `<col> = ANY(%s)` for any column (list params expand to IN (...))
        if "= ANY(" in query:
            query, params = self._expand_any(query, params)

        # 3. Handle `tags && %(topic_tokens)s`
        if "tags && %(topic_tokens)s OR" in query:
            # PostgreSQL array overlap -> SQLite: tags are stored as a JSON array, so test membership with json_each
            tokens = list(params.pop("topic_tokens", [])) if isinstance(params, dict) else []
            if tokens:
                names = []
                for i, t in enumerate(tokens):
                    params[f"topic_tok_{i}"] = t
                    names.append(f":topic_tok_{i}")
                repl = f"EXISTS (SELECT 1 FROM json_each(CASE WHEN json_valid(tags) THEN tags ELSE '[]' END) WHERE value IN ({', '.join(names)})) OR"
            else:
                repl = "0 OR"
            query = query.replace("tags && %(topic_tokens)s OR", repl)
            if isinstance(params, dict) and "topic_tokens" in params:
                del params["topic_tokens"]

        # 3b. Postgres casts (`?::jsonb`, `%s::text[]`) have no SQLite equivalent and are unnecessary there
        query = re.sub(r"::\s*[A-Za-z_]+(\[\])?", "", query)

        # 4. Handle ILIKE -> LIKE (SQLite LIKE is case-insensitive for ASCII)
        query = query.replace("ILIKE", "LIKE")

        # 5. Handle Postgres regex `~*` -> `LIKE`
        if "~*" in query:
            query = re.sub(r"([a-zA-Z0-9_]+)\s+~\*\s+%\(([a-zA-Z0-9_]+)\)s", r"LOWER(\1) LIKE :\2", query)
            if isinstance(params, dict):
                for k, v in list(params.items()):
                    if isinstance(v, str) and (v.startswith("(^|") or v.startswith("r(")):
                        clean_tok = re.sub(r"[^a-zA-Z0-9]", "", v)
                        params[k] = f"%{clean_tok}%"

        # 6. Handle RETURNING file_id;
        query = re.sub(r"RETURNING\s+file_id;?", ";", query, flags=re.IGNORECASE)

        # 7. Convert params format: %(key)s -> :key if dict, %s -> ? if tuple/list
        if isinstance(params, dict):
            for k, v in list(params.items()):
                if isinstance(v, list):
                    params[k] = json.dumps(v)
                elif isinstance(v, datetime):
                    params[k] = v.isoformat()
            query = re.sub(r"%\(([a-zA-Z0-9_]+)\)s", r":\1", query)
        elif isinstance(params, (list, tuple)):
            new_params = []
            for v in params:
                if isinstance(v, list):
                    new_params.append(json.dumps(v))
                elif isinstance(v, datetime):
                    new_params.append(v.isoformat())
                else:
                    new_params.append(v)
            params = tuple(new_params)
            query = query.replace("%s", "?")

        return query, params


@contextmanager
def get_connection():
    if _use_postgres():
        import psycopg2
        conn = None
        try:
            conn = psycopg2.connect(DSN)
            yield conn
        except Exception as exc:
            raise DatabaseError(str(exc)) from exc
        finally:
            if conn is not None:
                conn.close()
    else:
        conn = None
        try:
            os.makedirs(os.path.dirname(os.path.abspath(DB_FILE)), exist_ok=True)
            conn = sqlite3.connect(DB_FILE, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.create_function("now", 0, lambda: datetime.now(timezone.utc).isoformat())
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA foreign_keys=ON")
            except sqlite3.Error:
                pass
            yield conn
        except Exception as exc:
            raise DatabaseError(str(exc)) from exc
        finally:
            if conn is not None:
                conn.close()


@contextmanager
def get_cursor(commit: bool = False):
    if _use_postgres():
        import psycopg2
        import psycopg2.extras
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            try:
                yield cur
                if commit:
                    conn.commit()
            except Exception as exc:
                conn.rollback()
                raise DatabaseError(str(exc)) from exc
            finally:
                cur.close()
    else:
        with get_connection() as conn:
            wrapper = SQLiteCursorWrapper(conn)
            try:
                yield wrapper
                if commit:
                    conn.commit()
            except Exception as exc:
                conn.rollback()
                raise DatabaseError(str(exc)) from exc
            finally:
                wrapper.close()
