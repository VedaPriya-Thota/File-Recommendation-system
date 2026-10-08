"""
Local-file ingestion pipeline (Phase 10).

    source.discover()  ->  per file: metadata -> read -> extract -> normalize
        -> PostgreSQL (files + file_ingestion_state)
        -> BM25   : implicit -- bm25_search already indexes files.extracted_text
        -> chunks + embeddings + Qdrant : semantic_search.index_file()

There is no second retrieval path: this module only POPULATES the tables and
the Qdrant collection that the existing tools already read.

Lifecycle (file_ingestion_state.state):
    metadata_indexed -> content_extracted -> content_indexed
    metadata_indexed -> unsupported | failed
(`discovered` is the in-memory step before the first row exists.)

Incremental rules
    new        : insert files + state, extract, index
    unchanged  : (size, mtime) equal AND state is terminal-OK -> skipped with NO read
    touched    : (size, mtime) differ but sha256 equal and already indexed -> bookkeeping only
    modified   : sha256 differs -> re-extract, replace text, old semantic points deleted first
    deleted    : state row present but file gone -> semantic points, telemetry and the files row removed
    retried    : failed / content_extracted (e.g. embedding outage) files are retried on the next run

Safety
    * Only files WITH a state row are ever updated/deleted -- seed and other
      files are untouchable by ingestion.
    * Identity is (owner_id, rel_path); every file is owned by exactly the owner
      passed in, and that owner_id travels into the Qdrant payload, so
      query-time scope filtering applies exactly as for every other file.
    * Deletion is skipped entirely when discovery was incomplete.
    * One bad file never aborts the run; a PostgreSQL error does (access
      control depends on the database -- fail clearly, architecture §14).
    * Logs/reports carry counts, relative paths and short error codes only --
      never document text, exception messages or absolute paths.
"""
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from app.config import INGEST_MAX_FILE_BYTES
from app.db.connection import get_cursor
from app.ingestion.extract import EMPTY, FAILED, OK, UNSUPPORTED, ContentExtractor
from app.ingestion.source import FileSource, FileTooLargeError, LocalDirectorySource

logger = logging.getLogger("file_discovery_agent.ingestion")

TERMINAL_OK = ("content_indexed", "unsupported")


class UnknownOwnerError(ValueError):
    """The owner must already exist (identity is managed upstream; ingestion never creates users)."""


@dataclass
class IngestionReport:
    discovered: int = 0
    new: int = 0
    updated: int = 0
    unchanged: int = 0          # skipped: nothing to do
    deleted: int = 0
    indexed: int = 0            # content_indexed with >= 1 chunk
    empty: int = 0
    unsupported: int = 0
    failed: int = 0
    semantic_failed: int = 0    # text stored (BM25 works) but chunks/embeddings not written; retried next run
    duration_ms: float = 0.0
    deletion_skipped: bool = False
    files: List[dict] = field(default_factory=list)  # per file: rel_path, action, state, extraction_status, error_code

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("files")
        return d

    def to_dict(self) -> dict:
        return asdict(self)


def _ts(ns: int) -> datetime:
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)


def _owner_exists(owner_id: str) -> bool:
    with get_cursor() as cur:
        cur.execute("SELECT 1 FROM users WHERE user_id = %s;", (owner_id,))
        return cur.fetchone() is not None


def _load_state(owner_id: str) -> Dict[str, dict]:
    with get_cursor() as cur:
        cur.execute("SELECT * FROM file_ingestion_state WHERE owner_id = %s;", (owner_id,))
        return {r["rel_path"]: dict(r) for r in cur.fetchall()}


def _set_state(file_id: int, **cols) -> None:
    cols["updated_at"] = datetime.now(timezone.utc)
    assignments = ", ".join(f"{k} = %({k})s" for k in cols)
    with get_cursor(commit=True) as cur:
        cur.execute(f"UPDATE file_ingestion_state SET {assignments} WHERE file_id = %(file_id)s;", {**cols, "file_id": file_id})


def _set_text(file_id: int, text: str) -> None:
    with get_cursor(commit=True) as cur:
        cur.execute("UPDATE files SET extracted_text = %s WHERE file_id = %s;", (text, file_id))


def _upsert_metadata(owner_id: str, f, st: Optional[dict]) -> int:
    """files row (+ state at 'metadata_indexed'). tags/summary are never touched on update."""
    file_type = f.extension or "unknown"
    with get_cursor(commit=True) as cur:
        if st is None:
            cur.execute(
                """INSERT INTO files (filename, file_path, file_type, created_at, updated_at, owner_id)
                   VALUES (%s, %s, %s, %s, %s, %s) RETURNING file_id;""",
                (f.filename, f.rel_path, file_type, _ts(f.created_ns), _ts(f.mtime_ns), owner_id),
            )
            file_id = cur.fetchone()["file_id"]
            cur.execute(
                """INSERT INTO file_ingestion_state (file_id, owner_id, rel_path, size_bytes, mtime_ns, state)
                   VALUES (%s, %s, %s, %s, %s, 'metadata_indexed');""",
                (file_id, owner_id, f.rel_path, f.size_bytes, f.mtime_ns),
            )
            return file_id
        file_id = st["file_id"]
        cur.execute(
            "UPDATE files SET filename = %s, file_path = %s, file_type = %s, updated_at = %s WHERE file_id = %s AND owner_id = %s;",
            (f.filename, f.rel_path, file_type, _ts(f.mtime_ns), file_id, owner_id),
        )
        cur.execute(
            "UPDATE file_ingestion_state SET size_bytes = %s, mtime_ns = %s, state = 'metadata_indexed', error_code = NULL, updated_at = now() WHERE file_id = %s;",
            (f.size_bytes, f.mtime_ns, file_id),
        )
        return file_id


def ingest(
    source: FileSource,
    owner_id: str,
    *,
    extractor: Optional[ContentExtractor] = None,
    index_semantic: bool = True,
) -> IngestionReport:
    from app.tools import semantic_search as sem  # lazy: keeps `import app.ingestion` light

    t0 = time.perf_counter()
    extractor = extractor or ContentExtractor()
    if not _owner_exists(owner_id):
        raise UnknownOwnerError("owner_id does not exist")
    report = IngestionReport()
    discovered = source.discover()  # SourceUnavailableError propagates: nothing was changed
    report.discovered = len(discovered)
    existing = _load_state(owner_id)

    def record(f, action, state, status=None, code=None):
        report.files.append({"rel_path": f.rel_path, "action": action, "state": state,
                             "extraction_status": status, "error_code": code})

    for f in discovered:
        st = existing.get(f.rel_path)
        if st and st["state"] in TERMINAL_OK and st["size_bytes"] == f.size_bytes and st["mtime_ns"] == f.mtime_ns:
            report.unchanged += 1
            record(f, "unchanged", st["state"], st["extraction_status"])
            continue
        action = "new" if st is None else "updated"
        file_id = None
        try:
            file_id = _upsert_metadata(owner_id, f, st)
            state, status, code = _process_content(source, extractor, sem, f, file_id, st, index_semantic, report)
        except Exception as exc:  # noqa: BLE001
            from app.db.connection import DatabaseError
            if isinstance(exc, DatabaseError):
                raise  # hard fail: scope/access control depends on the database
            state, status, code = "failed", "failed", "unexpected_error"
            report.failed += 1
            logger.warning("ingestion: unexpected per-file error (%s)", type(exc).__name__)
            if file_id is not None:  # persist it like every other failure: report and database must agree
                _record_unexpected_failure(sem, file_id, index_semantic)
        if state == "unchanged_content":  # touched file, same bytes: bookkeeping only
            report.unchanged += 1
            record(f, "unchanged", "content_indexed", status)
            continue
        if action == "new":
            report.new += 1
        else:
            report.updated += 1
        record(f, action, state, status, code)

    # --- deletions: only for files this pipeline manages, and only if discovery was complete
    gone = [st for rel, st in existing.items() if rel not in {f.rel_path for f in discovered}]
    if gone and getattr(source, "last_errors", 0):
        report.deletion_skipped = True
        logger.warning("ingestion: discovery incomplete; skipping deletion of %d file(s)", len(gone))
    else:
        for st in gone:
            _delete_file(sem, st, index_semantic)
            report.deleted += 1
            report.files.append({"rel_path": st["rel_path"], "action": "deleted", "state": None,
                                 "extraction_status": None, "error_code": None})

    report.duration_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    logger.info("ingestion_run %s", json.dumps(report.summary(), sort_keys=True))
    return report


def _process_content(source, extractor, sem, f, file_id, st, index_semantic, report):
    """Returns (state, extraction_status, error_code)."""
    def drop_semantic():
        if index_semantic:
            try:
                sem.delete_file_points(file_id)
            except Exception:  # noqa: BLE001 -- DB row is the authority; orphan points are dropped at query time
                logger.warning("ingestion: semantic cleanup failed for file_id=%s", file_id)

    if not extractor.supports(f.extension):
        _set_text(file_id, "")
        drop_semantic()
        _set_state(file_id, state="unsupported", extraction_status=UNSUPPORTED, error_code=None, chunk_count=0,
                   fingerprint=None, indexed_at=None)
        report.unsupported += 1
        return "unsupported", UNSUPPORTED, None

    try:
        data = source.read_bytes(f.rel_path, INGEST_MAX_FILE_BYTES)
    except FileTooLargeError:
        return _fail(file_id, "too_large", _set_text, drop_semantic, report)
    except OSError:
        return _fail(file_id, "unreadable", _set_text, drop_semantic, report)

    fingerprint = hashlib.sha256(data).hexdigest()
    if st and st["fingerprint"] == fingerprint and st["state"] == "content_indexed":
        # mtime/size changed but the bytes did not: nothing to re-extract or re-embed
        _set_state(file_id, state="content_indexed", size_bytes=f.size_bytes, mtime_ns=f.mtime_ns)
        return "unchanged_content", st["extraction_status"], None

    content = extractor.extract(f.extension, data)
    if content.status == FAILED:
        return _fail(file_id, content.error_code or "extract_failed", _set_text, drop_semantic, report)

    _set_text(file_id, content.text)  # BM25 sees this immediately (it reads files.extracted_text)
    _set_state(file_id, state="content_extracted", extraction_status=content.status, error_code=None,
               fingerprint=fingerprint, chunk_count=0)
    if content.status == EMPTY:
        drop_semantic()
        _set_state(file_id, state="content_indexed", indexed_at=datetime.now(timezone.utc))
        report.empty += 1
        return "content_indexed", EMPTY, None

    chunks = 0
    if index_semantic:
        try:
            chunks = sem.index_file(file_id)
        except Exception as exc:  # noqa: BLE001 -- embedding / Qdrant outage: keep text + BM25, retry next run
            _set_state(file_id, error_code="semantic_index_failed")
            report.semantic_failed += 1
            logger.warning("ingestion: semantic indexing failed for file_id=%s (%s)", file_id, type(exc).__name__)
            return "content_extracted", OK, "semantic_index_failed"
    _set_state(file_id, state="content_indexed", chunk_count=chunks, error_code=None,
               indexed_at=datetime.now(timezone.utc))
    report.indexed += 1
    return "content_indexed", OK, None


def _record_unexpected_failure(sem, file_id, index_semantic) -> None:
    """Same end state as _fail(): the previous version's text/points must not stay searchable."""
    _set_text(file_id, "")
    if index_semantic:
        try:
            sem.delete_file_points(file_id)
        except Exception:  # noqa: BLE001
            logger.warning("ingestion: semantic cleanup failed for file_id=%s", file_id)
    _set_state(file_id, state="failed", extraction_status=FAILED, error_code="unexpected_error", chunk_count=0,
               fingerprint=None)


def _fail(file_id, code, set_text, drop_semantic, report):
    set_text(file_id, "")   # never leave the previous version's text searchable under a file we cannot read now
    drop_semantic()
    _set_state(file_id, state="failed", extraction_status=FAILED, error_code=code, chunk_count=0, fingerprint=None)
    report.failed += 1
    return "failed", FAILED, code


def _delete_file(sem, st, index_semantic) -> None:
    file_id = st["file_id"]
    if index_semantic:
        try:
            sem.delete_file_points(file_id)
        except Exception:  # noqa: BLE001
            logger.warning("ingestion: semantic cleanup failed for deleted file_id=%s", file_id)
    with get_cursor(commit=True) as cur:
        cur.execute("DELETE FROM user_file_telemetry WHERE file_id = %s;", (file_id,))  # FK to files
        cur.execute("DELETE FROM files WHERE file_id = %s AND owner_id = %s;", (file_id, st["owner_id"]))  # state cascades


def ingest_directory(root, owner_id: str, **kwargs) -> IngestionReport:
    return ingest(LocalDirectorySource(root), owner_id, **kwargs)
