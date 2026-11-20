"""Apply schema.sql to the configured database, then seed it."""
import os

from app.db.connection import get_cursor, _use_postgres, get_connection
from app.db.seed import seed

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS files (
    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f+00:00','now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f+00:00','now')),
    tags TEXT DEFAULT '[]',
    summary TEXT DEFAULT '',
    extracted_text TEXT DEFAULT '',
    FOREIGN KEY (owner_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS file_ingestion_state (
    file_id INTEGER PRIMARY KEY REFERENCES files(file_id) ON DELETE CASCADE,
    owner_id TEXT NOT NULL,
    rel_path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('discovered','metadata_indexed','content_extracted','content_indexed','unsupported','failed')),
    extraction_status TEXT CHECK (extraction_status IN ('ok','empty','unsupported','failed')),
    error_code TEXT,
    fingerprint TEXT,
    chunk_count INTEGER DEFAULT 0,
    indexed_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_file_telemetry (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    file_id INTEGER NOT NULL,
    action_type TEXT NOT NULL,
    time_spent_seconds INTEGER,
    timestamp TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f+00:00','now'))
);

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT PRIMARY KEY,
    preferred_file_types TEXT DEFAULT '[]',
    frequent_topics TEXT DEFAULT '{}',
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

# Application tables (connected folders, settings, assistant history). Written to be valid on both
# SQLite and PostgreSQL so the same statements run everywhere.
_APP_TABLES_SQLITE = [
    """CREATE TABLE IF NOT EXISTS connected_folders (
        folder_id INTEGER PRIMARY KEY AUTOINCREMENT,
        owner_id TEXT NOT NULL,
        path TEXT NOT NULL,
        label TEXT NOT NULL,
        exclusions TEXT NOT NULL DEFAULT '[]',
        status TEXT NOT NULL DEFAULT 'idle',
        watch INTEGER NOT NULL DEFAULT 1,
        last_error TEXT,
        last_scan_started TEXT,
        last_scan_finished TEXT,
        last_scan_summary TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        UNIQUE (owner_id, path)
    )""",
    """CREATE TABLE IF NOT EXISTS app_settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS assistant_runs (
        run_id TEXT PRIMARY KEY,
        owner_id TEXT NOT NULL,
        query TEXT NOT NULL,
        intent TEXT,
        status TEXT NOT NULL,
        result TEXT,
        created_at TEXT NOT NULL,
        finished_at TEXT
    )""",
]
# (table, column, ddl) -- added when missing (idempotent migration for databases created by earlier phases)
APP_COLUMNS = [
    ("file_ingestion_state", "folder_id", "INTEGER"),
    ("file_ingestion_state", "meta_json", "TEXT NOT NULL DEFAULT '{}'"),
    ("file_ingestion_state", "duplicate_of", "INTEGER"),
]


def _app_tables(pg: bool):
    if not pg:
        return _APP_TABLES_SQLITE
    return [t.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY") for t in _APP_TABLES_SQLITE]


DEFAULT_USER_ID = "local"
DEFAULT_USER_NAME = "You"


def _existing_columns(cur, pg: bool, table: str) -> set:
    if pg:
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = %s;", (table,))
        return {r["column_name"] for r in cur.fetchall()}
    cur.cur.execute(f"PRAGMA table_info({table})")
    return {r[1] for r in cur.cur.fetchall()}


def ensure_app_schema() -> None:
    """Idempotent, NON-destructive schema bring-up used by the application (never drops data):
    creates missing base tables, the application tables, adds missing columns to databases created by
    earlier phases, and makes sure the default local user exists. Safe to call on every start."""
    pg = _use_postgres()
    if pg:
        with get_cursor() as cur:
            cur.execute("SELECT 1 FROM information_schema.tables WHERE table_name = 'users';")
            has_base = cur.fetchone() is not None
        if not has_base:
            with open(SCHEMA_PATH, encoding="utf-8") as f:
                schema_sql = f.read()
            with get_cursor(commit=True) as cur:
                cur.execute(schema_sql)
    else:
        with get_connection() as conn:
            conn.executescript(SQLITE_SCHEMA)
            conn.commit()

    with get_cursor(commit=True) as cur:
        for ddl in _app_tables(pg):
            cur.execute(ddl)
        for table, column, ddl in APP_COLUMNS:
            if column not in _existing_columns(cur, pg, table):
                cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
        if pg:
            cur.execute("ALTER TABLE file_ingestion_state DROP CONSTRAINT IF EXISTS file_ingestion_state_owner_id_rel_path_key")
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_ingest_folder_rel ON file_ingestion_state (folder_id, rel_path)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ingest_folder ON file_ingestion_state (folder_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ingest_fingerprint ON file_ingestion_state (owner_id, fingerprint)")
        cur.execute("SELECT 1 FROM users WHERE user_id = %s;", (DEFAULT_USER_ID,))
        if cur.fetchone() is None:
            cur.execute("INSERT INTO users (user_id, display_name) VALUES (%s, %s);", (DEFAULT_USER_ID, DEFAULT_USER_NAME))


def init_db(with_seed: bool = True):
    if _use_postgres():
        with open(SCHEMA_PATH) as f:
            schema_sql = f.read()
        with get_cursor(commit=True) as cur:
            cur.execute(schema_sql)
    else:
        with get_connection() as conn:
            conn.executescript(SQLITE_SCHEMA)
            conn.commit()

    if with_seed:
        seed()


if __name__ == "__main__":
    init_db()
    print("Schema applied and seed data loaded successfully.")
