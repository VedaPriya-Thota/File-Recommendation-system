"""
Connected folders: explicit, user-authorized local directories + the background scanner that keeps them
indexed.

Security model
  * Only directories the user explicitly connected are ever read. Every file access goes through
    `resolve_file` which re-checks that the folder is still connected and that the resolved path (symlinks
    resolved) is INSIDE the folder root -- path traversal and symlink escapes are rejected.
  * The system is read-only with respect to user files: nothing in this module (or anywhere in the app)
    opens a user file for writing, renames, moves or deletes it. "Remove folder" only deletes the INDEX.
  * Nothing is uploaded anywhere: extraction, chunking, BM25 and (by default) embeddings are all local.
"""
import json
import logging
import os
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.appconfig import get_settings
from app.db.connection import get_cursor
from app.ingestion.extract import ContentExtractor
from app.ingestion.pipeline import ingest
from app.ingestion.source import LocalDirectorySource, SourceUnavailableError

logger = logging.getLogger("file_discovery_agent.folders")


class FolderError(ValueError):
    """User-facing validation error (bad path, overlap, not found...)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- path validation
_BLOCKED_POSIX = ("/proc", "/sys", "/dev", "/boot", "/etc", "/bin", "/sbin", "/lib", "/lib64", "/usr", "/var", "/run", "/snap", "/root/.ssh")
_BLOCKED_WIN = ("c:\\windows", "c:\\program files", "c:\\program files (x86)", "c:\\programdata", "c:\\$recycle.bin")


def _is_within(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


def validate_directory(path: str) -> Path:
    if not path or not str(path).strip():
        raise FolderError("Choose a directory.")
    raw = os.path.expanduser(str(path).strip())
    if not os.path.isabs(raw):
        raise FolderError("Use an absolute path (for example /home/me/Documents or C:\\Users\\me\\Documents).")
    p = Path(raw).resolve()
    if not p.exists():
        raise FolderError("That directory does not exist.")
    if not p.is_dir():
        raise FolderError("That path is not a directory.")
    if not os.access(p, os.R_OK | os.X_OK):
        raise FolderError("This application has no permission to read that directory.")
    if p == Path(p.anchor):
        raise FolderError("Connecting an entire drive / filesystem root is not allowed. Pick a specific folder.")
    low = str(p).lower()
    if os.name == "nt":
        if any(low == b or low.startswith(b + "\\") for b in _BLOCKED_WIN):
            raise FolderError("System directories cannot be connected.")
    elif any(low == b or low.startswith(b + "/") for b in _BLOCKED_POSIX):
        raise FolderError("System directories cannot be connected.")
    if p == Path.home().resolve():
        raise FolderError("Your whole home directory is too broad (it contains credentials and app data). Pick a subfolder such as Documents.")
    return p


# --------------------------------------------------------------------------- rows
def _row_to_folder(r: dict) -> dict:
    return {
        "folder_id": r["folder_id"], "owner_id": r["owner_id"], "path": r["path"], "label": r["label"],
        "exclusions": json.loads(r["exclusions"] or "[]"), "status": r["status"], "watch": bool(r["watch"]),
        "last_error": r["last_error"], "last_scan_started": r["last_scan_started"],
        "last_scan_finished": r["last_scan_finished"], "last_scan_summary": json.loads(r["last_scan_summary"] or "{}"),
        "created_at": r["created_at"], "exists": os.path.isdir(r["path"]),
    }


def get_folder(owner_id: str, folder_id: int) -> dict:
    with get_cursor() as cur:
        cur.execute("SELECT * FROM connected_folders WHERE folder_id = %s AND owner_id = %s;", (folder_id, owner_id))
        r = cur.fetchone()
    if r is None:
        raise FolderError("Folder not found.")
    return _row_to_folder(dict(r))


def list_folders(owner_id: str) -> List[dict]:
    with get_cursor() as cur:
        cur.execute("SELECT * FROM connected_folders WHERE owner_id = %s ORDER BY folder_id;", (owner_id,))
        folders = [_row_to_folder(dict(r)) for r in cur.fetchall()]
        cur.execute("""SELECT s.folder_id AS folder_id, s.state AS state, COUNT(*) AS n, COALESCE(SUM(s.size_bytes), 0) AS bytes,
                              COALESCE(SUM(s.chunk_count), 0) AS chunks
                       FROM file_ingestion_state s WHERE s.owner_id = %s AND s.folder_id IS NOT NULL
                       GROUP BY s.folder_id, s.state;""", (owner_id,))
        agg = [dict(r) for r in cur.fetchall()]
    by = {f["folder_id"]: {"files": 0, "bytes": 0, "chunks": 0, "states": {}} for f in folders}
    for a in agg:
        d = by.get(a["folder_id"])
        if d is None:
            continue
        d["files"] += a["n"]
        d["bytes"] += int(a["bytes"] or 0)
        d["chunks"] += int(a["chunks"] or 0)
        d["states"][a["state"]] = a["n"]
    for f in folders:
        c = by[f["folder_id"]]
        st = c["states"]
        f["counts"] = {"files": c["files"], "indexed": st.get("content_indexed", 0), "unsupported": st.get("unsupported", 0),
                       "failed": st.get("failed", 0), "pending": st.get("metadata_indexed", 0) + st.get("content_extracted", 0),
                       "bytes": c["bytes"], "chunks": c["chunks"]}
        f["scan"] = SCANS.job_for(f["folder_id"])
    return folders


def add_folder(owner_id: str, path: str, label: Optional[str] = None, exclusions: Optional[List[str]] = None, scan: bool = True) -> dict:
    p = validate_directory(path)
    with get_cursor() as cur:
        cur.execute("SELECT folder_id, path FROM connected_folders WHERE owner_id = %s;", (owner_id,))
        existing = [(r["folder_id"], Path(r["path"])) for r in cur.fetchall()]
    for _fid, ep in existing:
        if _is_within(p, ep):
            raise FolderError("This directory is already covered by the connected folder " + str(ep) + ".")
        if _is_within(ep, p):
            raise FolderError("This directory contains the already-connected folder " + str(ep) + ". Remove it first or pick the parent only.")
    label = (label or p.name or str(p)).strip()[:80]
    with get_cursor(commit=True) as cur:
        cur.execute("""INSERT INTO connected_folders (owner_id, path, label, exclusions, status, watch, created_at)
                       VALUES (%s, %s, %s, %s, 'idle', 1, %s);""", (owner_id, str(p), label, json.dumps(exclusions or []), _now()))
        cur.execute("SELECT folder_id FROM connected_folders WHERE owner_id = %s AND path = %s;", (owner_id, str(p)))
        folder_id = cur.fetchone()["folder_id"]
    if scan:
        SCANS.enqueue(folder_id, owner_id, reason="connected")
    return get_folder(owner_id, folder_id)


def update_folder(owner_id: str, folder_id: int, label=None, exclusions=None, watch=None) -> dict:
    get_folder(owner_id, folder_id)
    sets, params = [], {"fid": folder_id, "oid": owner_id}
    if label is not None:
        sets.append("label = %(label)s"); params["label"] = str(label).strip()[:80] or "Folder"
    if exclusions is not None:
        sets.append("exclusions = %(ex)s"); params["ex"] = json.dumps([str(x).strip() for x in exclusions if str(x).strip()][:200])
    if watch is not None:
        sets.append("watch = %(w)s"); params["w"] = 1 if watch else 0
    if sets:
        with get_cursor(commit=True) as cur:
            cur.execute(f"UPDATE connected_folders SET {', '.join(sets)} WHERE folder_id = %(fid)s AND owner_id = %(oid)s;", params)
    if exclusions is not None:
        SCANS.enqueue(folder_id, owner_id, reason="settings changed")
    return get_folder(owner_id, folder_id)


def remove_folder(owner_id: str, folder_id: int) -> dict:
    """Disconnects a folder and deletes its INDEX (rows, text, vectors). User files on disk are never touched."""
    get_folder(owner_id, folder_id)
    SCANS.cancel(folder_id, wait=True)
    from app.tools import semantic_search as sem
    with get_cursor() as cur:
        cur.execute("SELECT file_id FROM file_ingestion_state WHERE folder_id = %s AND owner_id = %s;", (folder_id, owner_id))
        ids = [r["file_id"] for r in cur.fetchall()]
    removed = 0
    for fid in ids:
        try:
            sem.delete_file_points(fid)
        except Exception:  # noqa: BLE001
            logger.warning("vector cleanup failed for file_id=%s", fid)
    with get_cursor(commit=True) as cur:
        for fid in ids:
            cur.execute("DELETE FROM user_file_telemetry WHERE file_id = %s;", (fid,))
            cur.execute("DELETE FROM file_ingestion_state WHERE file_id = %s;", (fid,))
            cur.execute("DELETE FROM files WHERE file_id = %s AND owner_id = %s;", (fid, owner_id))
            removed += 1
        cur.execute("DELETE FROM connected_folders WHERE folder_id = %s AND owner_id = %s;", (folder_id, owner_id))
    return {"removed_files": removed}


def clear_folder_index(owner_id: str, folder_id: int) -> int:
    """Deletes a folder's INDEX only (rows, text, vectors); the folder stays connected."""
    from app.tools import semantic_search as sem
    with get_cursor() as cur:
        cur.execute("SELECT file_id FROM file_ingestion_state WHERE folder_id = %s AND owner_id = %s;", (folder_id, owner_id))
        ids = [r["file_id"] for r in cur.fetchall()]
    for fid in ids:
        try:
            sem.delete_file_points(fid)
        except Exception:  # noqa: BLE001
            logger.warning("vector cleanup failed for file_id=%s", fid)
    with get_cursor(commit=True) as cur:
        for fid in ids:
            cur.execute("DELETE FROM user_file_telemetry WHERE file_id = %s;", (fid,))
            cur.execute("DELETE FROM file_ingestion_state WHERE file_id = %s;", (fid,))
            cur.execute("DELETE FROM files WHERE file_id = %s AND owner_id = %s;", (fid, owner_id))
    return len(ids)


# --------------------------------------------------------------------------- source / safe file access
class _AllowedFormats(ContentExtractor):
    def __init__(self, allowed):
        self.allowed = set(allowed)

    def supports(self, extension: str) -> bool:
        return (extension or "").lower() in self.allowed and super().supports(extension)


def make_source(folder: dict, settings: Optional[dict] = None) -> LocalDirectorySource:
    s = settings or get_settings()
    patterns = list(s["indexing"]["exclusions"]) + list(folder["exclusions"])
    return LocalDirectorySource(folder["path"], exclusions=patterns, include_hidden=s["indexing"]["include_hidden"])


def resolve_file(owner_id: str, file_id: int) -> dict:
    """Authorized, traversal-safe lookup of one indexed file -> {record..., abs_path: Path}.
    Raises FolderError if the file is unknown, not owned, its folder is no longer connected, or the resolved
    path leaves the connected root."""
    with get_cursor() as cur:
        cur.execute("""SELECT f.file_id, f.filename, f.file_path, f.file_type, f.owner_id, c.folder_id AS folder_id,
                              c.path AS folder_path, c.label AS folder_label
                       FROM files f
                       JOIN file_ingestion_state s ON s.file_id = f.file_id
                       JOIN connected_folders c ON c.folder_id = s.folder_id
                       WHERE f.file_id = %s AND f.owner_id = %s AND c.owner_id = %s;""", (file_id, owner_id, owner_id))
        r = cur.fetchone()
    if r is None:
        raise FolderError("File not found.")
    r = dict(r)
    root = Path(r["folder_path"]).resolve()
    target = (root / r["file_path"]).resolve()
    if not _is_within(target, root):
        raise FolderError("Path escapes the connected folder.")
    if not target.is_file():
        raise FolderError("The file no longer exists on disk (it will disappear from the index on the next scan).")
    r["abs_path"] = target
    r["root"] = root
    return r


# --------------------------------------------------------------------------- scanning
class ScanManager:
    """One worker thread drains a queue of folder scans (one at a time, so embedding model + SQLite are never
    contended); a second thread watches connected folders and enqueues incremental re-scans. Job state is kept
    in memory for the UI (polled) -- counts come from the real pipeline report, never estimated."""

    def __init__(self):
        self.q: "queue.Queue" = queue.Queue()
        self.jobs: Dict[int, dict] = {}
        self.cancel_flags: Dict[int, threading.Event] = {}
        self.lock = threading.Lock()
        self.worker: Optional[threading.Thread] = None
        self.watcher: Optional[threading.Thread] = None
        self.stop = threading.Event()
        self.last_checked: Dict[int, float] = {}

    # lifecycle
    def start(self):
        if self.worker and self.worker.is_alive():
            return
        self.stop.clear()
        with get_cursor(commit=True) as cur:  # crash recovery
            cur.execute("UPDATE connected_folders SET status = 'idle' WHERE status = 'scanning';")
        self.worker = threading.Thread(target=self._run, name="scan-worker", daemon=True)
        self.watcher = threading.Thread(target=self._watch, name="scan-watcher", daemon=True)
        self.worker.start()
        self.watcher.start()
        with get_cursor() as cur:  # catch up on changes made while the app was closed
            cur.execute("SELECT folder_id, owner_id FROM connected_folders;")
            for r in cur.fetchall():
                self.enqueue(r["folder_id"], r["owner_id"], reason="startup")

    def shutdown(self):
        self.stop.set()
        for ev in self.cancel_flags.values():
            ev.set()

    # api
    def enqueue(self, folder_id: int, owner_id: str, reason: str = "manual") -> dict:
        with self.lock:
            job = self.jobs.get(folder_id)
            if job and job["phase"] in ("queued", "discovering", "indexing", "removing", "loading_model"):
                return job
            job = {"folder_id": folder_id, "owner_id": owner_id, "phase": "queued", "reason": reason, "total": 0, "processed": 0,
                   "current": None, "counts": {}, "started_at": _now(), "finished_at": None, "error": None, "report": None}
            self.jobs[folder_id] = job
            self.cancel_flags[folder_id] = threading.Event()
        self.q.put((folder_id, owner_id))
        return job

    def cancel(self, folder_id: int, wait: bool = False):
        ev = self.cancel_flags.get(folder_id)
        if ev:
            ev.set()
        if wait:
            for _ in range(100):
                j = self.jobs.get(folder_id)
                if not j or j["phase"] not in ("discovering", "indexing", "removing", "loading_model"):
                    break
                time.sleep(0.1)

    def job_for(self, folder_id: int) -> Optional[dict]:
        with self.lock:
            j = self.jobs.get(folder_id)
            return dict(j) if j else None

    def active(self) -> List[dict]:
        with self.lock:
            return [dict(j) for j in self.jobs.values() if j["phase"] in ("queued", "discovering", "indexing", "removing", "loading_model")]

    # worker
    def _run(self):
        while not self.stop.is_set():
            try:
                folder_id, owner_id = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._scan(folder_id, owner_id)
            except Exception:  # noqa: BLE001
                logger.exception("scan worker error")

    def _scan(self, folder_id: int, owner_id: str):
        job = self.jobs[folder_id]
        cancel = self.cancel_flags[folder_id]
        try:
            folder = get_folder(owner_id, folder_id)
        except FolderError:
            job.update(phase="cancelled", finished_at=_now())
            return
        settings = get_settings()
        quiet = job["reason"] == "watch"
        if not quiet:
            with get_cursor(commit=True) as cur:
                cur.execute("UPDATE connected_folders SET status = 'scanning', last_scan_started = %s, last_error = NULL WHERE folder_id = %s;", (_now(), folder_id))
        job["phase"] = "discovering"
        sem_on = settings["indexing"]["semantic_enabled"]

        def progress(info: dict):
            job["phase"] = info.get("phase", job["phase"])
            for k in ("total", "processed", "current"):
                if k in info:
                    job[k] = info[k]
            if "report" in info:
                job["counts"] = info["report"]

        try:
            if sem_on:
                job["phase"] = "loading_model"
                from app.embeddings import get_provider
                get_provider()
            report = ingest(make_source(folder, settings), owner_id,
                            extractor=_AllowedFormats(settings["indexing"]["enabled_formats"]),
                            index_semantic=sem_on, folder_id=folder_id, progress=progress,
                            should_cancel=lambda: cancel.is_set() or self.stop.is_set(),
                            record_unchanged=False, max_file_bytes=settings["indexing"]["max_file_mb"] * 1024 * 1024)
            errors = [{"rel_path": f["rel_path"], "error_code": f["error_code"]} for f in report.files if f.get("error_code")][:100]
            summary = {**report.summary(), "errors": errors}
            changed = report.new + report.updated + report.deleted + report.moved + report.failed
            job.update(phase="cancelled" if report.cancelled else "done", finished_at=_now(), report=summary, counts=report.summary(),
                       processed=job["total"], current=None, changed=changed)
            if not (quiet and changed == 0):
                with get_cursor(commit=True) as cur:
                    cur.execute("""UPDATE connected_folders SET status = %s, last_scan_finished = %s, last_scan_summary = %s, last_error = NULL
                                   WHERE folder_id = %s;""", ("idle", _now(), json.dumps(summary), folder_id))
            elif quiet:
                with get_cursor(commit=True) as cur:
                    cur.execute("UPDATE connected_folders SET status = 'idle' WHERE folder_id = %s;", (folder_id,))
        except SourceUnavailableError:
            job.update(phase="error", error="The directory is not accessible (moved, unmounted or permissions changed). Nothing was deleted from the index.", finished_at=_now())
            self._store_error(folder_id, job["error"], "unavailable")
        except Exception as exc:  # noqa: BLE001
            logger.exception("scan failed")
            job.update(phase="error", error=f"Scan failed: {type(exc).__name__}", finished_at=_now())
            self._store_error(folder_id, job["error"], "error")

    @staticmethod
    def _store_error(folder_id, message, status):
        with get_cursor(commit=True) as cur:
            cur.execute("UPDATE connected_folders SET status = %s, last_error = %s, last_scan_finished = %s WHERE folder_id = %s;",
                        (status, message, _now(), folder_id))

    # watcher
    def _watch(self):
        while not self.stop.wait(2.0):
            try:
                s = get_settings()["indexing"]
                if not s["watch_enabled"]:
                    continue
                interval = s["watch_interval_s"]
                now = time.monotonic()
                with get_cursor() as cur:
                    cur.execute("SELECT folder_id, owner_id, watch FROM connected_folders;")
                    rows = [dict(r) for r in cur.fetchall()]
                for r in rows:
                    if not r["watch"]:
                        continue
                    j = self.job_for(r["folder_id"])
                    if j and j["phase"] in ("queued", "discovering", "indexing", "removing", "loading_model"):
                        continue
                    if now - self.last_checked.get(r["folder_id"], 0) >= interval:
                        self.last_checked[r["folder_id"]] = now
                        self.enqueue(r["folder_id"], r["owner_id"], reason="watch")
            except Exception:  # noqa: BLE001
                logger.exception("watcher error")


SCANS = ScanManager()


# --------------------------------------------------------------------------- directory browser (for picking)
def browse_directory(path: Optional[str]) -> dict:
    """Lists SUB-DIRECTORIES only (never files) so the user can pick a folder to connect. Nothing is indexed or
    read; the API only listens on localhost."""
    if not path:
        roots = []
        if os.name == "nt":
            import string
            roots = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
        else:
            roots = [str(Path.home()), "/"]
        home = Path.home()
        shortcuts = [str(home / n) for n in ("Documents", "Desktop", "Downloads", "Pictures", "Projects", "Work") if (home / n).is_dir()]
        return {"path": "", "parent": None, "entries": [{"name": r, "path": r} for r in shortcuts + roots], "shortcuts": True}
    p = Path(os.path.expanduser(path))
    if not p.is_absolute():
        raise FolderError("Use an absolute path.")
    p = p.resolve()
    if not p.is_dir():
        raise FolderError("Not a directory.")
    entries = []
    try:
        with os.scandir(p) as it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False) and not e.name.startswith("."):
                        entries.append({"name": e.name, "path": str(Path(e.path))})
                except OSError:
                    continue
    except PermissionError:
        raise FolderError("Permission denied.")
    entries.sort(key=lambda x: x["name"].lower())
    parent = str(p.parent) if p.parent != p else None
    return {"path": str(p), "parent": parent, "entries": entries[:1000], "shortcuts": False}


def pick_native_directory() -> Optional[str]:
    """Opens the operating system's own folder dialog (tkinter) in a child process. Returns None when no
    display / tkinter is available -- the UI then falls back to the in-app browser."""
    import subprocess
    code = ("import tkinter as tk\nfrom tkinter import filedialog\nr=tk.Tk(); r.withdraw(); r.attributes('-topmost',True)\n"
            "print(filedialog.askdirectory(title='Select a folder to connect') or '')")
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    except Exception:  # noqa: BLE001
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or ""
