-- Phase 1.1 schema -- reconciled toward the intended project baseline
-- (files / user_file_telemetry / user_profiles), while keeping only what
-- Phase 1's deterministic loop actually reads/writes.
--
-- Reconciliation notes (see project report for full detail):
--   * `path` -> `file_path`, `modified_at` -> `updated_at` to match the
--     intended baseline's column names.
--   * `user_file_telemetry` and `user_profiles` are now created to match
--     the baseline shape. Phase 1 code does not read or write either
--     table -- no personalization logic exists yet (architecture_final.md
--     §9/§17: personalization/telemetry-driven planning is explicitly
--     deferred). They exist now so later phases are a data-population
--     step, not a schema migration.
--   * `pg_trgm` and its trigram index were REMOVED (see below).
--   * `extracted_text` is KEPT on `files` (see below).
--   * `users` is KEPT as a Phase-1-only simplification (see below).

DROP TABLE IF EXISTS file_ingestion_state CASCADE;
DROP TABLE IF EXISTS user_file_telemetry CASCADE;
DROP TABLE IF EXISTS user_profiles CASCADE;
DROP TABLE IF EXISTS files CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- --- users -----------------------------------------------------------
-- NOT part of the intended three-table baseline. Kept as a deliberate
-- Phase-1-only simplification: architecture_final.md §19 assumes
-- "auth/user identity managed upstream" -- in a real deployment this
-- table wouldn't exist here at all, identity would come from an external
-- IdP/auth service. Phase 1 has no such service to call, so this table
-- is a local stand-in that gives resolve_scope() something authoritative
-- to check "does this user_id exist" against (needed for the hard-fail-
-- on-unknown-user behavior), and gives files.owner_id / user_profiles.
-- user_id / user_file_telemetry.user_id a real FK target instead of an
-- unconstrained free-text column. It carries no columns beyond identity
-- (no roles/permissions/ACLs) precisely because sharing/ACL systems are
-- out of scope for this pass.
CREATE TABLE users (
    user_id      TEXT PRIMARY KEY,
    display_name TEXT NOT NULL
);

-- --- files -------------------------------------------------------------
CREATE TABLE files (
    file_id        SERIAL PRIMARY KEY,
    filename       TEXT NOT NULL,
    file_path      TEXT NOT NULL,
    file_type      TEXT NOT NULL,               -- e.g. 'pptx', 'xlsx', 'docx', 'pdf', 'png'
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    owner_id       TEXT NOT NULL REFERENCES users(user_id),
    tags           TEXT[] NOT NULL DEFAULT '{}',
    summary        TEXT NOT NULL DEFAULT '',

    -- Reserved for later phases, KEPT deliberately (not Phase-2 leakage):
    -- architecture_final.md §10 makes inspect_file_content depend on
    -- chunk-level extracted text already being stored somewhere -- for
    -- Qdrant-backed chunks that's the payload's content_text, but a
    -- file-level fallback/staging column here means later ingestion has
    -- somewhere to land text before it's chunked/embedded, without a
    -- schema migration. Phase 1 never reads or writes this column.
    -- IMPORTANT for later ingestion: this column is a placeholder for
    -- the FULL extracted text of the file, not chunks. When the semantic
    -- layer is built, chunk-level text must be preserved separately
    -- (e.g. in Qdrant payloads, per §10) -- do not assume this single
    -- column is sufficient for chunk-level retrieval/inspection.
    extracted_text TEXT NOT NULL DEFAULT ''
);

CREATE INDEX idx_files_owner      ON files (owner_id);
CREATE INDEX idx_files_file_type  ON files (file_type);
CREATE INDEX idx_files_updated_at ON files (updated_at);
CREATE INDEX idx_files_tags       ON files USING GIN (tags);

-- --- user_file_telemetry -------------------------------------------------
-- Raw per-event telemetry. Not read by any Phase 1 code path -- exists so
-- later phases (get_user_activity, cold-start/affinity scoring) have a
-- populated table to build against rather than a schema change.
CREATE TABLE user_file_telemetry (
    event_id           SERIAL PRIMARY KEY,
    user_id            TEXT NOT NULL REFERENCES users(user_id),
    file_id            INTEGER NOT NULL REFERENCES files(file_id),
    action_type        TEXT NOT NULL,             -- e.g. 'open', 'edit', 'view'
    time_spent_seconds INTEGER,
    timestamp          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_telemetry_user ON user_file_telemetry (user_id);
CREATE INDEX idx_telemetry_file ON user_file_telemetry (file_id);

-- --- user_profiles --------------------------------------------------------
-- Derived/aggregated personalization inputs (§8/§16). Not read or written
-- by any Phase 1 code path -- personalization touchpoints are explicitly
-- deferred (§9/§17). `frequent_topics` is JSONB so per-topic weights can
-- be represented once that phase exists.
CREATE TABLE user_profiles (
    user_id               TEXT PRIMARY KEY REFERENCES users(user_id),
    preferred_file_types  TEXT[] NOT NULL DEFAULT '{}',
    frequent_topics       JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- --- file_ingestion_state (Phase 10) ----------------------------------------
-- One row per file that the local-file ingestion pipeline manages (see
-- app/ingestion/). Files WITHOUT a row here (e.g. the seed corpus) are never
-- touched by ingestion -- that is what makes "file disappeared from disk ->
-- remove it" safe. Holds only what incremental indexing needs:
--   * identity:   (owner_id, rel_path) -- path relative to the source root,
--                 never an absolute machine path
--   * change detection: size_bytes + mtime_ns (cheap, no read) and a sha256
--                 `fingerprint` of the bytes (computed only when the cheap
--                 check says "maybe changed")
--   * lifecycle:  state, extraction_status, error_code (a short code -- never
--                 an exception message, which could contain paths/content)
CREATE TABLE file_ingestion_state (
    file_id            INTEGER PRIMARY KEY REFERENCES files(file_id) ON DELETE CASCADE,
    owner_id           TEXT NOT NULL,
    rel_path           TEXT NOT NULL,
    size_bytes         BIGINT NOT NULL,
    mtime_ns           BIGINT NOT NULL,
    fingerprint        TEXT,
    state              TEXT NOT NULL CHECK (state IN
        ('discovered', 'metadata_indexed', 'content_extracted', 'content_indexed', 'unsupported', 'failed')),
    extraction_status  TEXT CHECK (extraction_status IN ('ok', 'empty', 'unsupported', 'failed')),
    error_code         TEXT,
    chunk_count        INTEGER NOT NULL DEFAULT 0,
    indexed_at         TIMESTAMPTZ,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (owner_id, rel_path)
);
CREATE INDEX idx_ingestion_state_owner ON file_ingestion_state (owner_id);
