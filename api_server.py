"""
Local API + web UI for the Agentic File Discovery assistant.

    python api_server.py            # http://127.0.0.1:8000  (serves the built frontend from frontend/dist)

The server binds to 127.0.0.1 only. Requests whose Host/Origin are not local are rejected (DNS-rebinding and
cross-site request protection). Only directories explicitly connected through /api/folders are ever read, and
user files are never written to.
"""
import os
import platform
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DISCOVERY_DATA_DIR", ROOT / "data")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("DISCOVERY_SQLITE_PATH", str(DATA_DIR / "discovery.db"))
os.environ.setdefault("DISCOVERY_QDRANT_PATH", str(DATA_DIR / "qdrant"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from fastapi import FastAPI, HTTPException, Query, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app import appconfig, folders as F  # noqa: E402
from app.assistant.engine import run_assistant  # noqa: E402
from app.db import connection as dbc  # noqa: E402
from app.db.connection import get_cursor  # noqa: E402
from app.db.init_db import DEFAULT_USER_ID, ensure_app_schema  # noqa: E402
from app.ingestion.extract import SUPPORTED_EXTENSIONS, TYPE_GROUPS  # noqa: E402
from app.llm import provider as llmp  # noqa: E402
from app.run import run_query  # noqa: E402

OWNER = DEFAULT_USER_ID
PORT = int(os.environ.get("DISCOVERY_PORT", "8000"))
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1", "testserver"}
ALLOWED_ORIGINS = [f"http://localhost:{p}" for p in (5173, 4173, PORT)] + [f"http://127.0.0.1:{p}" for p in (5173, 4173, PORT)]

@asynccontextmanager
async def lifespan(_app):
    ensure_app_schema()
    F.SCANS.start()
    yield
    F.SCANS.shutdown()


app = FastAPI(title="Agentic File Discovery", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=False, allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def local_only(request: Request, call_next):
    host = (request.headers.get("host") or "").rsplit(":", 1)[0] if not (request.headers.get("host") or "").startswith("[") else "[::1]"
    origin = request.headers.get("origin")
    if host not in ALLOWED_HOSTS or (origin and origin not in ALLOWED_ORIGINS):
        return JSONResponse({"detail": "Forbidden: this API only accepts requests from the local machine."}, status_code=403)
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    return resp


@app.exception_handler(F.FolderError)
async def folder_error(_req, exc: F.FolderError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


# ----------------------------------------------------------------------------- health / info
def _embedding_info() -> dict:
    try:
        from app.embeddings import get_provider
        p = get_provider()
        return {"name": p.name, "dimension": p.dimension, "semantic": not p.name.startswith("hashing")}
    except Exception as exc:  # noqa: BLE001
        return {"name": f"unavailable ({type(exc).__name__})", "dimension": 0, "semantic": False}


@app.get("/api/health")
def health():
    ok, err = True, None
    try:
        with get_cursor() as cur:
            cur.execute("SELECT 1 AS ok;")
            cur.fetchone()
    except Exception as exc:  # noqa: BLE001
        ok, err = False, type(exc).__name__
    return {"status": "healthy" if ok else "degraded", "database_connected": ok, "database": dbc.engine_name(), "detail": err,
            "embedding": _embedding_info() if os.environ.get("_FDS_EMB_PROBED") else {"name": "not loaded yet", "semantic": None},
            "llm": llmp.llm_status(), "platform": platform.system(), "active_scans": len(F.SCANS.active())}


@app.get("/api/embedding")
def embedding():
    os.environ["_FDS_EMB_PROBED"] = "1"
    return _embedding_info()


@app.get("/api/users")
def users():
    with get_cursor() as cur:
        cur.execute("SELECT user_id, display_name FROM users ORDER BY user_id;")
        return [dict(r) for r in cur.fetchall()]


# ----------------------------------------------------------------------------- folders
class FolderIn(BaseModel):
    path: str
    label: Optional[str] = None
    exclusions: List[str] = Field(default_factory=list)


class FolderPatch(BaseModel):
    label: Optional[str] = None
    exclusions: Optional[List[str]] = None
    watch: Optional[bool] = None


@app.get("/api/folders")
def folders_list():
    return F.list_folders(OWNER)


@app.post("/api/folders")
def folders_add(body: FolderIn):
    return F.add_folder(OWNER, body.path, body.label, body.exclusions)


@app.patch("/api/folders/{folder_id}")
def folders_patch(folder_id: int, body: FolderPatch):
    return F.update_folder(OWNER, folder_id, body.label, body.exclusions, body.watch)


@app.delete("/api/folders/{folder_id}")
def folders_delete(folder_id: int):
    return F.remove_folder(OWNER, folder_id)


@app.post("/api/folders/{folder_id}/scan")
def folders_scan(folder_id: int):
    F.get_folder(OWNER, folder_id)
    return F.SCANS.enqueue(folder_id, OWNER, reason="manual")


@app.post("/api/folders/{folder_id}/cancel")
def folders_cancel(folder_id: int):
    F.get_folder(OWNER, folder_id)
    F.SCANS.cancel(folder_id)
    return {"ok": True}


@app.post("/api/folders/{folder_id}/rebuild")
def folders_rebuild(folder_id: int):
    """Drops this folder's index (rows, text, vectors) and re-indexes from scratch. Files on disk are untouched."""
    f = F.get_folder(OWNER, folder_id)
    F.SCANS.cancel(folder_id, wait=True)
    F.clear_folder_index(OWNER, folder_id)
    return F.SCANS.enqueue(folder_id, OWNER, reason="rebuild") | {"label": f["label"]}


@app.get("/api/scans")
def scans():
    return {"active": F.SCANS.active()}


@app.get("/api/fs/browse")
def fs_browse(path: Optional[str] = None):
    return F.browse_directory(path)


@app.post("/api/fs/pick")
def fs_pick():
    chosen = F.pick_native_directory()
    if chosen is None:
        raise HTTPException(status_code=501, detail="No native folder dialog is available here; use the in-app browser or type a path.")
    return {"path": chosen}


# ----------------------------------------------------------------------------- files
_SORTS = {"name": "LOWER(f.filename)", "modified": "f.updated_at", "size": "s.size_bytes", "type": "f.file_type", "indexed": "s.indexed_at", "path": "LOWER(f.file_path)"}
_BASE = """FROM files f JOIN file_ingestion_state s ON s.file_id = f.file_id JOIN connected_folders c ON c.folder_id = s.folder_id"""


def _file_row(r: dict) -> dict:
    import json as _json
    return {
        "file_id": r["file_id"], "filename": r["filename"], "rel_path": r["file_path"], "file_type": r["file_type"],
        "size_bytes": r["size_bytes"], "modified": str(r["updated_at"]), "created": str(r["created_at"]), "state": r["state"],
        "extraction_status": r["extraction_status"], "error_code": r["error_code"], "chunks": r["chunk_count"],
        "folder_id": r["folder_id"], "folder_label": r["label"], "abs_path": str(Path(r["folder_path"]) / r["file_path"]),
        "duplicate_of": r["duplicate_of"], "indexed_at": str(r["indexed_at"]) if r["indexed_at"] else None,
        "meta": _json.loads(r["meta_json"] or "{}"),
    }


_COLS = """f.file_id, f.filename, f.file_path, f.file_type, f.created_at, f.updated_at, s.size_bytes, s.state, s.extraction_status,
           s.error_code, s.chunk_count, s.folder_id, s.duplicate_of, s.indexed_at, s.meta_json, c.path AS folder_path, c.label AS label"""


@app.get("/api/files")
def files_list(q: str = "", file_type: str = "", group: str = "", folder_id: Optional[int] = None, state: str = "",
               modified_within_days: Optional[int] = None, min_size: Optional[int] = None, max_size: Optional[int] = None,
               duplicates_only: bool = False, sort: str = "modified", direction: str = "desc",
               limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    where, p = ["f.owner_id = %(o)s"], {"o": OWNER}
    if q.strip():
        like = "%" + q.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        where.append("(f.filename ILIKE %(q)s ESCAPE '\\' OR f.file_path ILIKE %(q)s ESCAPE '\\')")
        p["q"] = like
    if file_type:
        types = [t.strip().lower().lstrip(".") for t in file_type.split(",") if t.strip()]
        where.append("f.file_type = ANY(%(types)s)"); p["types"] = types
    if group in TYPE_GROUPS:
        where.append("f.file_type = ANY(%(gtypes)s)"); p["gtypes"] = TYPE_GROUPS[group]
    if folder_id is not None:
        where.append("s.folder_id = %(fid)s"); p["fid"] = folder_id
    if state:
        where.append("s.state = %(st)s"); p["st"] = state
    if modified_within_days:
        from datetime import datetime, timedelta, timezone
        where.append("f.updated_at >= %(since)s"); p["since"] = datetime.now(timezone.utc) - timedelta(days=modified_within_days)
    if min_size is not None:
        where.append("s.size_bytes >= %(mn)s"); p["mn"] = min_size
    if max_size is not None:
        where.append("s.size_bytes <= %(mx)s"); p["mx"] = max_size
    if duplicates_only:
        where.append("s.duplicate_of IS NOT NULL")
    order = _SORTS.get(sort, _SORTS["modified"]) + (" ASC" if direction.lower() == "asc" else " DESC") + ", f.file_id ASC"
    cond = " AND ".join(where)
    with get_cursor() as cur:
        cur.execute(f"SELECT COUNT(*) AS n {_BASE} WHERE {cond};", dict(p))
        total = cur.fetchone()["n"]
        cur.execute(f"SELECT {_COLS} {_BASE} WHERE {cond} ORDER BY {order} LIMIT %(lim)s OFFSET %(off)s;", {**p, "lim": limit, "off": offset})
        items = [_file_row(dict(r)) for r in cur.fetchall()]
    return {"total": total, "items": items, "limit": limit, "offset": offset}


@app.get("/api/files/facets")
def files_facets():
    with get_cursor() as cur:
        cur.execute(f"SELECT f.file_type AS t, COUNT(*) AS n {_BASE} WHERE f.owner_id = %s GROUP BY f.file_type ORDER BY n DESC;", (OWNER,))
        types = [dict(r) for r in cur.fetchall()]
        cur.execute(f"SELECT s.state AS st, COUNT(*) AS n {_BASE} WHERE f.owner_id = %s GROUP BY s.state;", (OWNER,))
        states = [dict(r) for r in cur.fetchall()]
    return {"types": types, "states": states, "groups": {k: v for k, v in TYPE_GROUPS.items()}}


def _one(file_id: int) -> dict:
    with get_cursor() as cur:
        cur.execute(f"SELECT {_COLS} {_BASE} WHERE f.owner_id = %s AND f.file_id = %s;", (OWNER, file_id))
        r = cur.fetchone()
    if r is None:
        raise HTTPException(404, "File not found")
    return _file_row(dict(r))


def _telemetry(file_id: int, action: str):
    try:
        with get_cursor(commit=True) as cur:
            cur.execute("INSERT INTO user_file_telemetry (user_id, file_id, action_type) VALUES (%s, %s, %s);", (OWNER, file_id, action))
    except Exception:  # noqa: BLE001
        pass


@app.get("/api/files/{file_id}")
def files_get(file_id: int):
    return _one(file_id)


@app.get("/api/files/{file_id}/preview")
def files_preview(file_id: int, q: str = "", max_chars: int = Query(6000, ge=500, le=40000)):
    from app.analysis import tabular
    from app.analysis import text as T
    rec = _one(file_id)
    with get_cursor() as cur:
        cur.execute("SELECT extracted_text FROM files WHERE file_id = %s AND owner_id = %s;", (file_id, OWNER))
        text = cur.fetchone()["extracted_text"] or ""
    out = {"file": rec, "kind": "code" if rec["meta"].get("language") else "text", "text": text[:max_chars], "truncated": len(text) > max_chars,
           "total_chars": len(text), "table": None, "excerpts": []}
    if q.strip():
        out["excerpts"] = T.top_excerpts(text, T.query_terms(q), 5, 400)
    if rec["file_type"] in ("csv", "tsv", "xlsx", "xlsm"):
        try:
            out["table"] = tabular.preview_rows(F_read(file_id), rec["file_type"])
            out["kind"] = "table"
        except Exception:  # noqa: BLE001
            pass
    _telemetry(file_id, "view")
    return out


def F_read(file_id: int) -> bytes:
    info = F.resolve_file(OWNER, file_id)
    if info["abs_path"].stat().st_size > 60 * 1024 * 1024:
        raise F.FolderError("File too large to preview.")
    with open(info["abs_path"], "rb") as fh:
        return fh.read()


_INLINE_OK = {"pdf": "application/pdf", "txt": "text/plain", "md": "text/plain", "csv": "text/plain", "json": "application/json",
              "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif"}


@app.get("/api/files/{file_id}/raw")
def files_raw(file_id: int, inline: bool = False):
    info = F.resolve_file(OWNER, file_id)
    _telemetry(file_id, "open")
    ext = info["file_type"]
    if inline and ext in _INLINE_OK:
        return FileResponse(info["abs_path"], media_type=_INLINE_OK[ext], content_disposition_type="inline", filename=info["filename"])
    return FileResponse(info["abs_path"], media_type="application/octet-stream", content_disposition_type="attachment", filename=info["filename"])


def _spawn(cmd: list):
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # list argv: no shell, no injection


@app.post("/api/files/{file_id}/open")
def files_open(file_id: int):
    info = F.resolve_file(OWNER, file_id)
    path = str(info["abs_path"])
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # noqa: S606
        elif sys.platform == "darwin":
            _spawn(["open", path])
        else:
            _spawn(["xdg-open", path])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Could not open the file with the system viewer ({type(exc).__name__}).")
    _telemetry(file_id, "open")
    return {"ok": True}


@app.post("/api/files/{file_id}/reveal")
def files_reveal(file_id: int):
    info = F.resolve_file(OWNER, file_id)
    path = str(info["abs_path"])
    try:
        if sys.platform.startswith("win"):
            _spawn(["explorer", "/select,", path])
        elif sys.platform == "darwin":
            _spawn(["open", "-R", path])
        else:
            _spawn(["xdg-open", str(info["abs_path"].parent)])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Could not open the folder ({type(exc).__name__}).")
    return {"ok": True}


# ----------------------------------------------------------------------------- dashboard
@app.get("/api/stats")
def stats():
    with get_cursor() as cur:
        cur.execute(f"""SELECT s.state AS st, COUNT(*) AS n, COALESCE(SUM(s.size_bytes), 0) AS bytes, COALESCE(SUM(s.chunk_count), 0) AS chunks,
                               COALESCE(SUM(CASE WHEN s.duplicate_of IS NOT NULL THEN 1 ELSE 0 END), 0) AS dups
                        {_BASE} WHERE f.owner_id = %s GROUP BY s.state;""", (OWNER,))
        st = [dict(r) for r in cur.fetchall()]
        cur.execute(f"SELECT f.file_type AS t, COUNT(*) AS n {_BASE} WHERE f.owner_id = %s GROUP BY f.file_type ORDER BY n DESC LIMIT 12;", (OWNER,))
        types = [dict(r) for r in cur.fetchall()]
        cur.execute(f"SELECT {_COLS} {_BASE} WHERE f.owner_id = %s ORDER BY f.updated_at DESC LIMIT 6;", (OWNER,))
        recent = [_file_row(dict(r)) for r in cur.fetchall()]
        cur.execute("SELECT COUNT(*) AS n FROM assistant_runs WHERE owner_id = %s;", (OWNER,))
        runs = cur.fetchone()["n"]
        cur.execute("SELECT COUNT(*) AS n FROM connected_folders WHERE owner_id = %s;", (OWNER,))
        nfold = cur.fetchone()["n"]
    by = {r["st"]: r for r in st}
    total = sum(r["n"] for r in st)
    return {
        "folders": nfold, "files": total, "indexed": by.get("content_indexed", {}).get("n", 0), "unsupported": by.get("unsupported", {}).get("n", 0),
        "failed": by.get("failed", {}).get("n", 0), "pending": by.get("metadata_indexed", {}).get("n", 0) + by.get("content_extracted", {}).get("n", 0),
        "bytes": sum(int(r["bytes"]) for r in st), "chunks": sum(int(r["chunks"]) for r in st), "duplicates": sum(int(r["dups"]) for r in st),
        "types": types, "recent": recent, "assistant_runs": runs, "active_scans": F.SCANS.active(), "llm": llmp.llm_status(),
    }


# ----------------------------------------------------------------------------- settings
@app.get("/api/settings")
def settings_get():
    s = appconfig.get_settings()
    return {"settings": s, "llm": llmp.llm_status(), "supported_formats": list(SUPPORTED_EXTENSIONS),
            "default_exclusions": appconfig.DEFAULTS["indexing"]["exclusions"], "data_dir": str(DATA_DIR), "database": dbc.engine_name()}


@app.put("/api/settings")
def settings_put(patch: dict):
    s = appconfig.update_settings(patch)
    return {"settings": s, "llm": llmp.llm_status()}


@app.post("/api/settings/reset")
def settings_reset():
    return {"settings": appconfig.reset_settings(), "llm": llmp.llm_status()}


class KeyIn(BaseModel):
    provider: str
    key: str = ""


@app.post("/api/settings/llm-key")
def settings_key(body: KeyIn):
    """Keeps the API key in process memory ONLY (never written to disk or the database)."""
    if body.provider not in ("anthropic", "openai_compatible"):
        raise HTTPException(400, "Unknown provider")
    llmp.set_runtime_api_key(body.provider, body.key.strip())
    return {"llm": llmp.llm_status()}


@app.post("/api/settings/rescan-all")
def rescan_all():
    for f in F.list_folders(OWNER):
        F.SCANS.enqueue(f["folder_id"], OWNER, reason="manual")
    return {"queued": len(F.list_folders(OWNER))}


# ----------------------------------------------------------------------------- assistant
class AskIn(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = Field(None, max_length=40)
    mode: Optional[str] = Field(None, max_length=10)          # "find" = recommend files instead of answering a question


_RUNS: dict = {}
_RUNS_LOCK = threading.Lock()
_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="assistant")


@app.post("/api/assistant/runs")
def assistant_start(body: AskIn):
    run_id = uuid.uuid4().hex
    rec = {"run_id": run_id, "query": body.query.strip(), "status": "running", "events": [], "result": None, "error": None, "started": time.time()}
    with _RUNS_LOCK:
        _RUNS[run_id] = rec
        for old in sorted(_RUNS, key=lambda k: _RUNS[k]["started"])[:-50]:
            _RUNS.pop(old, None)

    def work():
        try:
            rec["result"] = run_assistant(OWNER, rec["query"], lambda ev: rec["events"].append(ev), run_id=run_id, conversation_id=body.conversation_id, force_find=(body.mode == "find"))
            rec["status"] = "done"
        except Exception as exc:  # noqa: BLE001
            rec["error"], rec["status"] = f"{type(exc).__name__}: {exc}"[:300], "error"

    _POOL.submit(work)
    return {"run_id": run_id}


@app.get("/api/assistant/runs/{run_id}")
def assistant_poll(run_id: str, since: int = 0):
    rec = _RUNS.get(run_id)
    if rec is None:
        with get_cursor() as cur:
            cur.execute("SELECT result FROM assistant_runs WHERE run_id = %s AND owner_id = %s;", (run_id, OWNER))
            row = cur.fetchone()
        if row is None:
            raise HTTPException(404, "Run not found")
        import json as _json
        res = _json.loads(row["result"])
        return {"run_id": run_id, "status": "done", "events": res.get("activity", []), "next": len(res.get("activity", [])), "result": res, "error": None}
    ev = rec["events"]
    return {"run_id": run_id, "status": rec["status"], "events": ev[since:], "next": len(ev), "result": rec["result"], "error": rec["error"]}


@app.get("/api/assistant/history")
def assistant_history(limit: int = Query(30, ge=1, le=200), intent: str = ""):
    import json as _json
    with get_cursor() as cur:
        if intent:
            cur.execute("SELECT run_id, query, intent, status, result, created_at FROM assistant_runs WHERE owner_id = %s AND intent = %s ORDER BY created_at DESC LIMIT %s;", (OWNER, intent, limit))
        else:
            cur.execute("SELECT run_id, query, intent, status, result, created_at FROM assistant_runs WHERE owner_id = %s ORDER BY created_at DESC LIMIT %s;", (OWNER, limit))
        rows = [dict(r) for r in cur.fetchall()]
    out = []
    for r in rows:
        res = _json.loads(r["result"] or "{}")
        out.append({"run_id": r["run_id"], "query": r["query"], "intent": r["intent"], "status": r["status"], "created_at": r["created_at"],
                    "n_refs": len(res.get("references", [])), "top": [x["filename"] for x in res.get("references", [])[:3]]})
    return out


@app.delete("/api/assistant/history")
def assistant_clear():
    with get_cursor(commit=True) as cur:
        cur.execute("DELETE FROM assistant_runs WHERE owner_id = %s;", (OWNER,))
    return {"ok": True}


# ----------------------------------------------------------------------------- legacy agent endpoint (kept)
class QueryRequest(BaseModel):
    user_id: str = Field(OWNER)
    query: str
    max_iterations: Optional[int] = 6


@app.post("/api/query")
def execute_query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(400, "query cannot be empty")
    return run_query(user_id=req.user_id.strip() or OWNER, raw_query=req.query.strip(), max_iterations=req.max_iterations or 6)


# ----------------------------------------------------------------------------- static frontend
DIST = ROOT / "frontend" / "dist"
if DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(404)
        target = (DIST / full_path).resolve()
        if full_path and DIST in target.parents and target.is_file():
            return FileResponse(target)
        return FileResponse(DIST / "index.html")


if __name__ == "__main__":
    import uvicorn
    print(f"Agentic File Discovery -> http://127.0.0.1:{PORT}")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
