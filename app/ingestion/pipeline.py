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
    moved: int = 0              # renamed / moved files re-linked WITHOUT re-extraction or re-embedding
    duplicates: int = 0         # byte-identical copies: vectors cloned instead of re-embedded
    excluded: int = 0           # files/dirs skipped by exclusion rules
    cancelled: bool = False
    discovery_errors: int = 0
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


def _load_state(owner_id: str, folder_id: Optional[int] = None) -> Dict[str, dict]:
    with get_cursor() as cur:
        if folder_id is None:
            cur.execute("SELECT * FROM file_ingestion_state WHERE owner_id = %s AND folder_id IS NULL;", (owner_id,))
        else:
            cur.execute("SELECT * FROM file_ingestion_state WHERE owner_id = %s AND folder_id = %s;", (owner_id, folder_id))
        return {r["rel_path"]: dict(r) for r in cur.fetchall()}


def _set_state(file_id: int, **cols) -> None:
    cols["updated_at"] = datetime.now(timezone.utc)
    assignments = ", ".join(f"{k} = %({k})s" for k in cols)
    with get_cursor(commit=True) as cur:
        cur.execute(f"UPDATE file_ingestion_state SET {assignments} WHERE file_id = %(file_id)s;", {**cols, "file_id": file_id})


def _set_text(file_id: int, text: str) -> None:
    with get_cursor(commit=True) as cur:
        cur.execute("UPDATE files SET extracted_text = %s WHERE file_id = %s;", (text, file_id))


def _upsert_metadata(owner_id: str, f, st: Optional[dict], folder_id: Optional[int] = None) -> int:
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
                """INSERT INTO file_ingestion_state (file_id, owner_id, rel_path, size_bytes, mtime_ns, state, folder_id)
                   VALUES (%s, %s, %s, %s, %s, 'metadata_indexed', %s);""",
                (file_id, owner_id, f.rel_path, f.size_bytes, f.mtime_ns, folder_id),
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


def _structure_current(st: dict) -> bool:
    """Files indexed by an older chunker (no passage structure) are re-extracted and re-embedded once."""
    from app.ingestion.structure import STRUCTURE_VERSION
    if st.get("state") in ("unsupported", "failed") or st.get("extraction_status") == "empty":
        return True
    try:
        return json.loads(st.get("meta_json") or "{}").get("_sv") == STRUCTURE_VERSION
    except ValueError:
        return False


def _emit(cb, **kw):
    if cb:
        try:
            cb(kw)
        except Exception:  # noqa: BLE001 -- a UI progress hook must never break indexing
            pass


def _find_moves(source, extractor, existing, discovered_paths, discovered, max_bytes):
    """Renamed/moved files: a NEW path whose bytes equal those of a state row that has DISAPPEARED.
    Only same-size candidates are hashed, so the cost is bounded by actual rename candidates."""
    gone = {rel: st for rel, st in existing.items() if rel not in discovered_paths and st.get("fingerprint")}
    if not gone:
        return {}
    by_size: Dict[int, List[dict]] = {}
    for st in gone.values():
        by_size.setdefault(st["size_bytes"], []).append(st)
    moves, claimed = {}, set()
    for f in discovered:
        if f.rel_path in existing or f.size_bytes not in by_size or f.size_bytes > max_bytes:
            continue
        try:
            fp = hashlib.sha256(source.read_bytes(f.rel_path, max_bytes)).hexdigest()
        except Exception:  # noqa: BLE001
            continue
        for st in by_size[f.size_bytes]:
            if st["file_id"] not in claimed and st["fingerprint"] == fp:
                moves[f.rel_path] = st
                claimed.add(st["file_id"])
                break
    return moves


def ingest(
    source: FileSource,
    owner_id: str,
    *,
    extractor: Optional[ContentExtractor] = None,
    index_semantic: bool = True,
    folder_id: Optional[int] = None,
    progress=None,
    should_cancel=None,
    record_unchanged: bool = True,
    max_file_bytes: Optional[int] = None,
) -> IngestionReport:
    from app.tools import semantic_search as sem  # lazy: keeps `import app.ingestion` light

    t0 = time.perf_counter()
    extractor = extractor or ContentExtractor()
    max_bytes = max_file_bytes or INGEST_MAX_FILE_BYTES
    if not _owner_exists(owner_id):
        raise UnknownOwnerError("owner_id does not exist")
    report = IngestionReport()
    _emit(progress, phase="discovering", total=0, processed=0)
    discovered = source.discover()  # SourceUnavailableError propagates: nothing was changed
    report.discovered = len(discovered)
    report.excluded = getattr(source, "excluded", 0)
    report.discovery_errors = getattr(source, "last_errors", 0)
    existing = _load_state(owner_id, folder_id)
    discovered_paths = {f.rel_path for f in discovered}
    moves = _find_moves(source, extractor, existing, discovered_paths, discovered, max_bytes)
    moved_ids = {st["file_id"] for st in moves.values()}

    def record(f, action, state, status=None, code=None):
        if action == "unchanged" and not record_unchanged:
            return
        report.files.append({"rel_path": f.rel_path, "action": action, "state": state,
                             "extraction_status": status, "error_code": code})

    total = len(discovered)
    for i, f in enumerate(discovered):
        if should_cancel and should_cancel():
            report.cancelled = True
            break
        if i % 5 == 0 or i == total - 1:
            _emit(progress, phase="indexing", total=total, processed=i, current=f.rel_path, report=report.summary())
        st = existing.get(f.rel_path)
        if st and st["state"] in TERMINAL_OK and st["size_bytes"] == f.size_bytes and st["mtime_ns"] == f.mtime_ns and _structure_current(st):
            report.unchanged += 1
            record(f, "unchanged", st["state"], st["extraction_status"])
            continue
        if f.rel_path in moves:  # rename / move: re-link the existing row, no extraction, no embedding
            mst = moves[f.rel_path]
            try:
                _apply_move(sem, owner_id, f, mst, index_semantic)
                report.moved += 1
                record(f, "moved", mst["state"], mst.get("extraction_status"))
                continue
            except Exception as exc:  # noqa: BLE001
                from app.db.connection import DatabaseError
                if isinstance(exc, DatabaseError):
                    raise
                moved_ids.discard(mst["file_id"])  # fall through: treat as an ordinary new file
        action = "new" if st is None else "updated"
        file_id = None
        try:
            file_id = _upsert_metadata(owner_id, f, st, folder_id)
            state, status, code = _process_content(source, extractor, sem, f, file_id, st, index_semantic, report, max_bytes)
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
    gone = [st for rel, st in existing.items()
            if rel not in discovered_paths and st["file_id"] not in moved_ids]
    if report.cancelled:
        pass  # a cancelled run is incomplete: never delete on partial information
    elif gone and getattr(source, "last_errors", 0):
        report.deletion_skipped = True
        logger.warning("ingestion: discovery incomplete; skipping deletion of %d file(s)", len(gone))
    else:
        _emit(progress, phase="removing", total=total, processed=total)
        for st in gone:
            _delete_file(sem, st, index_semantic)
            report.deleted += 1
            report.files.append({"rel_path": st["rel_path"], "action": "deleted", "state": None,
                                 "extraction_status": None, "error_code": None})

    report.duration_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    _emit(progress, phase="done", total=total, processed=total, report=report.summary())
    logger.info("ingestion_run %s", json.dumps(report.summary(), sort_keys=True))
    return report


def _apply_move(sem, owner_id, f, st, index_semantic) -> None:
    file_id = st["file_id"]
    with get_cursor(commit=True) as cur:
        cur.execute("UPDATE files SET filename = %s, file_path = %s, file_type = %s, updated_at = %s WHERE file_id = %s AND owner_id = %s;",
                    (f.filename, f.rel_path, f.extension or "unknown", _ts(f.mtime_ns), file_id, owner_id))
        cur.execute("UPDATE file_ingestion_state SET rel_path = %s, size_bytes = %s, mtime_ns = %s, updated_at = now() WHERE file_id = %s;",
                    (f.rel_path, f.size_bytes, f.mtime_ns, file_id))
    if index_semantic and st.get("state") == "content_indexed":
        try:
            sem.update_file_payload(file_id, filename=f.filename, file_type=f.extension or "unknown")
        except Exception:  # noqa: BLE001 -- payload only affects display; DB row is the authority
            logger.warning("ingestion: payload refresh failed for moved file_id=%s", file_id)


def _process_content(source, extractor, sem, f, file_id, st, index_semantic, report, max_bytes=None):
    max_bytes = max_bytes or INGEST_MAX_FILE_BYTES
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
        data = source.read_bytes(f.rel_path, max_bytes)
    except FileTooLargeError:
        return _fail(file_id, "too_large", _set_text, drop_semantic, report)
    except OSError:
        return _fail(file_id, "unreadable", _set_text, drop_semantic, report)

    fingerprint = hashlib.sha256(data).hexdigest()
    if st and st["fingerprint"] == fingerprint and st["state"] == "content_indexed" and _structure_current(st):
        # mtime/size changed but the bytes did not: nothing to re-extract or re-embed
        _set_state(file_id, state="content_indexed", size_bytes=f.size_bytes, mtime_ns=f.mtime_ns)
        return "unchanged_content", st["extraction_status"], None

    content = extractor.extract(f.extension, data)
    if content.status == FAILED:
        return _fail(file_id, content.error_code or "extract_failed", _set_text, drop_semantic, report)

    _set_text(file_id, content.text)  # BM25 sees this immediately (it reads files.extracted_text)
    _set_state(file_id, state="content_extracted", extraction_status=content.status, error_code=None,
               fingerprint=fingerprint, chunk_count=0, meta_json=json.dumps(content.metadata, default=str),
               duplicate_of=None)
    if content.status == EMPTY:
        drop_semantic()
        _set_state(file_id, state="content_indexed", indexed_at=datetime.now(timezone.utc))
        report.empty += 1
        return "content_indexed", EMPTY, None

    chunks = 0
    if index_semantic:
        try:
            twin = _find_twin(st_owner(file_id), fingerprint, file_id)
            if twin is not None:  # byte-identical content already embedded: clone vectors, do not re-embed
                chunks = sem.clone_file_points(twin, file_id)
                if chunks:
                    _set_state(file_id, duplicate_of=twin)
                    report.duplicates += 1
                else:
                    chunks = sem.index_file(file_id)
            else:
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


def st_owner(file_id: int) -> str:
    with get_cursor() as cur:
        cur.execute("SELECT owner_id FROM files WHERE file_id = %s;", (file_id,))
        row = cur.fetchone()
        return row["owner_id"] if row else ""


def _find_twin(owner_id: str, fingerprint: str, file_id: int) -> Optional[int]:
    """Another already-indexed file of the same owner with identical bytes (and chunks to clone)."""
    with get_cursor() as cur:
        cur.execute("""SELECT file_id FROM file_ingestion_state
                       WHERE owner_id = %s AND fingerprint = %s AND state = 'content_indexed'
                         AND chunk_count > 0 AND file_id <> %s AND duplicate_of IS NULL ORDER BY file_id;""",
                    (owner_id, fingerprint, file_id))
        row = cur.fetchone()
        return row["file_id"] if row else None


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
        cur.execute("UPDATE file_ingestion_state SET duplicate_of = NULL WHERE duplicate_of = %s;", (file_id,))
        cur.execute("DELETE FROM file_ingestion_state WHERE file_id = %s;", (file_id,))
        cur.execute("DELETE FROM user_file_telemetry WHERE file_id = %s;", (file_id,))  # FK to files
        cur.execute("DELETE FROM files WHERE file_id = %s AND owner_id = %s;", (file_id, st["owner_id"]))  # state cascades


def ingest_directory(root, owner_id: str, exclusions=(), **kwargs) -> IngestionReport:
    return ingest(LocalDirectorySource(root, exclusions=exclusions), owner_id, **kwargs)
