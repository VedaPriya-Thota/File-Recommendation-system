"""
Modular tools. Every tool takes the shared Board, returns a plain structured result (dicts / small objects) and has an
explicit list of agents allowed to call it (see REGISTRY at the bottom). Tools never write to user files, never execute
user code, and read documents only through the access-checked helpers of app.folders.
"""
import json
import re
from typing import Dict, List, Optional

from app.agents.core import CACHE, Board, ToolSpec
from app.analysis import codeqa, dataqa, htmlqa
from app.analysis import resume as resume_mod
from app.analysis import text as T
from app.assistant import qa
from app.assistant import retrieval as R
from app.assistant.understand import Need, _make_need, _owner_files, _name_tokens
from app.assistant.verify import verify_answer
from app.db.connection import get_cursor
from app.folders import FolderError, resolve_file
from app.ingestion.extract import TYPE_GROUPS
from app.ingestion.structure import build_passages

CODE_T = set(qa.CODE_TYPES)
CONFIG_T = {"json", "yaml", "yml", "toml", "ini", "cfg"}


# ---------------------------------------------------------------------------- shared helpers
def info_for(b: Board, ids) -> Dict[int, dict]:
    need = [i for i in ids if i not in b.info]
    if need:
        b.info.update(qa._file_info(b.owner_id, list(need)))
    return {i: b.info[i] for i in ids if i in b.info}


def text_of(b: Board, file_id: int) -> str:
    inf = info_for(b, [file_id]).get(file_id)
    key = ("text", file_id, inf["fingerprint"] if inf else None)
    val, _ = CACHE.get_or_compute(key, lambda: qa._text_of(b.owner_id, file_id), None)       # internal reads are not counted as "cache hits"
    return val


def _cached(b: Board, ns: str, file_id: int, fn, extra=None):
    inf = info_for(b, [file_id]).get(file_id)
    key = (ns, file_id, inf["fingerprint"] if inf else None, extra)
    val, hit = CACHE.get_or_compute(key, fn, b.metrics)
    return val, hit


# ---------------------------------------------------------------------------- discovery tools
def search_files(b: Board, query: str, types=None, limit: int = 15, folder_id: Optional[int] = None) -> dict:
    """File-name / path / type search over the connected folders (metadata only -- names are leads, never evidence)."""
    q_flat = qa_flat(query)
    q_stems = {T._stem(t) for t in T.content_terms(query)}
    rows = _owner_files(b.owner_id)
    out = []
    for r in rows:
        if r["state"] in ("failed",) or r.get("duplicate_of"):
            continue
        if types and r["file_type"] not in types:
            continue
        name = r["filename"].lower()
        score, why = 0.0, ""
        if name in query.lower():
            score, why = 3.0, "the file name was mentioned"
        else:
            stem = qa_flat(re.sub(r"\.[A-Za-z0-9]{1,5}$", "", r["filename"]))
            if len(stem.strip()) > 2 and stem in q_flat:
                score, why = 2.0, "its name matches the request"
            else:
                toks = {T._stem(t) for t in _name_tokens(r["filename"]) + re.split(r"[^A-Za-z0-9]+", r["file_path"].lower()) if t}
                hit = q_stems & toks
                if hit:
                    score, why = 0.6 * len(hit) / max(len(q_stems), 1), "name/path contains " + ", ".join(sorted(hit)[:3])
        if score > 0:
            out.append({"file_id": r["file_id"], "filename": r["filename"], "rel_path": r["file_path"], "file_type": r["file_type"], "score": round(score, 3), "why": why})
    out.sort(key=lambda x: -x["score"])
    return {"matches": out[:limit], "scanned": len(rows)}


def qa_flat(s: str) -> str:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return " " + re.sub(r"[^a-z0-9]+", " ", s.lower()).strip() + " "


def search_contents(b: Board, need: Need, need_no: int = 0, scope_ids=None, top_n: int = 6, per_file_cap: int = 3, pool: int = 30, relaxed: bool = False) -> dict:
    ret = R.retrieve_need(b.owner_id, need, need_no, list(scope_ids) if scope_ids else None, top_n=top_n, per_file_cap=per_file_cap, pool=pool, relaxed=relaxed)
    scores: Dict[int, float] = {}
    for e in ret.evidence:
        scores[e.file_id] = max(scores.get(e.file_id, 0.0), e.score) + 0.05 * e.coverage
    return {"evidence": ret.evidence, "searched_files": ret.searched_files, "searched_passages": ret.searched_passages, "steps": ret.steps, "channels": ret.channels,
            "file_scores": sorted(scores.items(), key=lambda kv: -kv[1])}


def agent_search(b: Board, query: str, plan_find: dict, terms_out: Optional[list] = None) -> dict:
    """The original LangGraph search agent (metadata + BM25 + semantic + reflection loop), used as a discovery tool."""
    from app.assistant import engine as E
    terms = terms_out if terms_out is not None else []
    cands = E._search(b.bus, b.owner_id, plan_find, terms)
    return {"cands": cands, "goal_terms": terms}


def list_project(b: Board, question: str) -> dict:
    """Pick the project the question is about (a connected folder or its top-level directory, preferring one named in the question,
    else the one with the most source code) and list its README, manifests, configuration and source files with their role."""
    from collections import defaultdict
    rows = [r for r in _owner_files(b.owner_id) if r["state"] == "content_indexed"]
    with get_cursor() as cur:
        cur.execute("SELECT s.file_id AS file_id, c.folder_id AS folder_id, c.label AS label FROM file_ingestion_state s JOIN connected_folders c ON c.folder_id = s.folder_id WHERE s.owner_id = %s;", (b.owner_id,))
        fol = {r["file_id"]: (r["folder_id"], r["label"]) for r in cur.fetchall()}
    groups = defaultdict(list)
    for r in rows:
        top = r["file_path"].split("/")[0] if "/" in r["file_path"] else ""
        f, lab = fol.get(r["file_id"], (None, ""))
        if top:
            groups[(f, top, lab)].append(r)
        groups[(f, "", lab)].append(r)
    ql = question.lower()

    def score(k):
        files = groups[k]
        code = sum(1 for f in files if f["file_type"] in CODE_T)
        name = (k[1] or k[2]).lower()
        return (5000 if name and name in ql else 0) + code + (3 if any(re.match(r"readme", f["filename"], re.I) for f in files) else 0)
    keys = sorted(groups, key=lambda k: (-score(k), -len(k[1])))
    if not keys or score(keys[0]) <= 0:
        return {"project": None, "files": []}
    # prefer the deepest directory that still holds the code (a project folder inside a bigger connected folder)
    best = keys[0]
    out = []
    for f in groups[best]:
        n = f["filename"].lower()
        role = ("README" if n.startswith("readme") else "manifest" if n in ("package.json", "pyproject.toml", "requirements.txt", "setup.py", "cargo.toml", "go.mod", "pom.xml") else
                "configuration" if f["file_type"] in ("yaml", "yml", "toml", "ini", "cfg", "json") and re.search(r"config|param|setting|hyper", n) else "source" if f["file_type"] in CODE_T else None)
        if role:
            out.append({"file_id": f["file_id"], "filename": f["filename"], "why": f"{role} of the project", "score": 1.0 if role in ("README", "manifest") else 0.8 if role == "configuration" else 0.5, "via": ["list_project"]})
    return {"project": best[1] or best[2], "files": out}


def find_related(b: Board, file_id: int, limit: int = 5) -> dict:
    inf = info_for(b, [file_id]).get(file_id)
    if not inf:
        raise FolderError("unknown file")
    text = text_of(b, file_id)
    out: Dict[int, dict] = {}
    kws = T.keywords(text, 8)
    if kws:
        need = _make_need(" ".join(kws), set(), [])
        ret = R.retrieve_need(b.owner_id, need, 0, None, top_n=14, per_file_cap=1, pool=40)
        for e in ret.evidence:
            if e.file_id != file_id and e.coverage >= 0.4:
                out[e.file_id] = {"file_id": e.file_id, "why": f"shares topics ({', '.join(kws[:3])})", "score": round(e.score, 3)}
    if inf["file_type"] in CODE_T:                             # code: files this one imports or is imported by
        o = codeqa.outline(text, inf["file_type"])
        mods = {m.split(".")[0].split()[0].lower() for m in o["imports"] if m}
        for r in _owner_files(b.owner_id):
            stem = r["filename"].rsplit(".", 1)[0].lower()
            if r["file_id"] != file_id and r["file_type"] in CODE_T and stem in mods:
                out[r["file_id"]] = {"file_id": r["file_id"], "why": "imported by this file", "score": 0.9}
    sib = [r for r in _owner_files(b.owner_id) if r["file_id"] != file_id and r["file_path"].rsplit("/", 1)[0] == inf["file_path"].rsplit("/", 1)[0] and r["file_type"] == inf["file_type"]]
    for r in sib[:3]:
        out.setdefault(r["file_id"], {"file_id": r["file_id"], "why": "same folder and type", "score": 0.2})
    ranked = sorted(out.values(), key=lambda x: -x["score"])[:limit]
    names = info_for(b, [x["file_id"] for x in ranked])
    for x in ranked:
        x["filename"] = names.get(x["file_id"], {}).get("filename", "")
    return {"related": ranked}


def get_metadata(b: Board, file_ids) -> dict:
    inf = info_for(b, list(file_ids))
    return {"files": [{"file_id": i, "filename": f["filename"], "rel_path": f["file_path"], "file_type": f["file_type"], "size": f["size_bytes"], "modified": str(f["updated_at"]),
                       "state": f["state"], "meta": {k: v for k, v in f["meta"].items() if k not in ("page_offsets", "sheet_offsets")}} for i, f in inf.items()]}


def match_tables(b: Board, question: str) -> dict:
    with get_cursor() as cur:
        cur.execute("""SELECT f.file_id, f.filename, f.file_type, s.meta_json FROM files f JOIN file_ingestion_state s ON s.file_id = f.file_id
                       JOIN connected_folders c ON c.folder_id = s.folder_id
                       WHERE f.owner_id = %s AND s.state = 'content_indexed' AND f.file_type = ANY(%s);""", (b.owner_id, list(qa.TABULAR)))
        cands = []
        for r in cur.fetchall():
            m = json.loads(r["meta_json"] or "{}")
            cands.append({"file_id": r["file_id"], "filename": r["filename"], "file_type": r["file_type"], "columns": m.get("columns"), "sheets": m.get("sheets")})
    ordered = dataqa.pick_tables(question, cands)
    return {"tables": ordered, "all": len(cands)}


# ---------------------------------------------------------------------------- retrieval tools
def read_section(b: Board, file_id: int, page: Optional[int] = None, heading: Optional[str] = None, lines=None, max_chars: int = 3500) -> dict:
    """Read a specific part of a document: a page/slide, a section by heading, or a line range. Returns verbatim text + location."""
    inf = info_for(b, [file_id]).get(file_id)
    if not inf:
        raise FolderError("unknown file")
    text = text_of(b, file_id)
    segs = []
    for p in build_passages(text, inf["meta"], inf["file_type"]):
        if page is not None and p.page != page:
            continue
        if heading and not (p.heading and heading.lower() in p.heading.lower()):
            continue
        if lines and (p.line_end < lines[0] or p.line_start > lines[1]):
            continue
        segs.append({"text": text[p.start:p.end], "start": p.start, "end": p.end, "location": p.location(), "page": p.page, "heading": p.heading,
                     "line_start": p.line_start, "line_end": p.line_end})
    used, out = 0, []
    for s in segs:
        if used + len(s["text"]) > max_chars:
            break
        out.append(s)
        used += len(s["text"])
    return {"file_id": file_id, "segments": out, "truncated": len(out) < len(segs), "total_chars": len(text)}


def read_table(b: Board, file_id: int) -> dict:
    """Profile of a spreadsheet read from the file itself: sheets, row counts, column names/types, first rows."""
    def build():
        info = resolve_file(b.owner_id, file_id)
        if info["abs_path"].stat().st_size > qa.MAX_READ:
            raise FolderError("file too large")
        with open(info["abs_path"], "rb") as fh:                      # read-only
            tables = dataqa.load_tables(fh.read(), info["file_type"])
        return {"sheets": [{"name": t.name, "rows": len(t.rows), "columns": [{"name": h, "type": t.types.get(h)} for h in t.headers],
                            "sample": [[("" if c is None else str(c))[:30] for c in r] for r in t.rows[:3]]} for t in tables]}, tables
    (prof, tables), hit = _cached(b, "table", file_id, build)
    return {"profile": prof, "_cached": hit}


def _tables(b: Board, file_id: int):
    read_table(b, file_id)                                             # populate cache
    (prof, tables), _ = _cached(b, "table", file_id, lambda: None)
    return tables


def extract_html_structure(b: Board, file_id: int) -> dict:
    st, hit = _cached(b, "html", file_id, lambda: htmlqa.extract_structure(text_of(b, file_id)))
    return {"structure": st, "_cached": hit}


def read_code_outline(b: Board, file_id: int) -> dict:
    inf = info_for(b, [file_id]).get(file_id)
    o, hit = _cached(b, "outline", file_id, lambda: codeqa.outline(text_of(b, file_id), inf["file_type"]))
    return {"outline": o, "lines": text_of(b, file_id).count("\n") + 1, "_cached": hit}


def read_config(b: Board, file_id: int) -> dict:
    inf = info_for(b, [file_id]).get(file_id)
    kv, hit = _cached(b, "config", file_id, lambda: codeqa.read_config(text_of(b, file_id), inf["file_type"]))
    return {"values": kv, "_cached": hit}


# ---------------------------------------------------------------------------- analysis tools
def summarize_document(b: Board, file_id: int, focus: Optional[Need] = None, k: Optional[int] = None) -> dict:
    inf = info_for(b, [file_id]).get(file_id)
    text = text_of(b, file_id)
    kk = k or (3 if len(text) < 2500 else 5 if len(text) < 20000 else 6)
    key_extra = (kk, tuple(focus.stems) if focus else None)
    s, hit = _cached(b, "summary", file_id, lambda: qa.summarize_text(text, inf["meta"], inf["file_type"], kk, focus), key_extra)
    return {"summary": s, "_cached": hit}


def analyze_table(b: Board, file_id: int, question: str) -> dict:
    tables = _tables(b, file_id)
    inf = info_for(b, [file_id])[file_id]
    res = dataqa.answer(question, tables, inf["filename"])
    res["file_id"] = file_id
    return res


def compare_documents(b: Board, file_ids) -> dict:
    texts = [text_of(b, f) for f in file_ids[:2]]
    return {"comparison": T.compare_documents(texts[0], texts[1])}


def trace_workflow(b: Board, file_ids, keywords) -> dict:
    files = []
    inf = info_for(b, list(file_ids))
    for fid in file_ids:
        f = inf.get(fid)
        if f and f["file_type"] in CODE_T:
            files.append({"file_id": fid, "filename": f["filename"], "rel_path": f["file_path"], "file_type": f["file_type"], "text": text_of(b, fid)})
    return codeqa.trace_workflow(files, list(keywords))


def review_resume(b: Board, file_id: int) -> dict:
    inf = info_for(b, [file_id])[file_id]
    return {"review": resume_mod.review(text_of(b, file_id), pages=inf["meta"].get("pages"))}


def rank_sentences(b: Board, need_no: int) -> dict:
    need = b.plan_q.needs[need_no]
    evs = b.evidence.get(need_no, [])
    ranked = R.best_sentences(need, evs, 8)
    assess = R.assess(need, evs)
    return {"ranked": ranked, "assess": assess}


# ---------------------------------------------------------------------------- verification tools
def check_sufficiency(b: Board, need_no: int) -> dict:
    need = b.plan_q.needs[need_no]
    evs = b.evidence.get(need_no, [])
    a = R.assess(need, evs)
    return {"ok": a["ok"], "reason": a["reason"], "missing": a.get("missing", []), "coverage": round(a["best"]["cov"], 3) if a.get("best") else 0.0}


def detect_conflicts(b: Board, need_no: int) -> dict:
    need = b.plan_q.needs[need_no]
    evs = b.evidence.get(need_no, [])
    info_for(b, [e.file_id for e in evs])
    return {"conflict": qa.detect_conflicts(need, evs, b.info)}


def verify_claims(b: Board, claims: List[dict]) -> dict:
    """Each claim = {id, text, file_id, start, end}: is the cited span really in the file, and does it support the claim's figures/words?"""
    out = []
    for c in claims:
        full = text_of(b, c["file_id"])
        span = full[c["start"]:c["end"]]
        ok_span = bool(span) and (c.get("text", "") == "" or _norm(c["text"]) in _norm(full))
        stems = {T._stem(t) for t in T.tokens(span)}
        frac = (sum(1 for s in T.stems(c.get("text", "")) if s in stems) / max(len(T.stems(c.get("text", ""))), 1)) if c.get("text") else 1.0
        nums_ok = all(n in re.sub(r"[,]", "", span) for n in re.findall(r"\d[\d.]*", re.sub(r"[,]", "", c.get("text", ""))) if len(n) >= 1)
        out.append({"id": c["id"], "supported": bool(ok_span and nums_ok and frac >= 0.6), "span_found": ok_span, "numbers_ok": nums_ok, "support": round(frac, 2)})
    return {"claims": out}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def recompute_calc(b: Board, calc: dict) -> dict:
    """Independent recomputation of a spreadsheet answer straight from the raw cells (different code path from dataqa)."""
    info = resolve_file(b.owner_id, calc["file_id"])
    with open(info["abs_path"], "rb") as fh:
        data = fh.read()
    from app.analysis import tabular
    sheets = tabular._read_xlsx(data) if info["file_type"] in ("xlsx", "xlsm") else tabular._read_csv(data)
    rows = sheets.get(calc["sheet"], {}).get("rows") or next(iter(sheets.values()))["rows"]
    header = tabular._unique_headers(rows[0])
    col = calc.get("column_first")
    if not col or col not in header or not calc.get("keep") or calc.get("operation") not in ("sum", "avg", "min", "max", "count", "median"):
        return {"checked": False, "reason": "no simple aggregate to re-check"}
    if calc.get("groups") and calc["operation"] == "sum":              # grouped totals must add up to the ungrouped total of the same rows
        ci = header.index(col)
        total = sum(v for v in (tabular._to_number(rows[1:][i][ci]) for i in calc["keep"] if i < len(rows) - 1 and ci < len(rows[1:][i])) if v is not None)
        gsum = sum(v for _, v in calc["groups"])
        return {"checked": True, "match": abs(total - gsum) <= 1e-6 * max(1.0, abs(total)), "recomputed": total, "reported": gsum, "rows": len(calc["keep"])}
    ci = header.index(col)
    body = rows[1:]
    vals = [tabular._to_number(body[i][ci]) for i in calc["keep"] if i < len(body) and ci < len(body[i])]
    vals = [v for v in vals if v is not None]
    op = calc["operation"]
    if not vals:
        return {"checked": False, "reason": "no numeric values"}
    mine = {"sum": sum(vals), "avg": sum(vals) / len(vals), "min": min(vals), "max": max(vals), "count": len(vals), "median": sorted(vals)[len(vals) // 2] if len(vals) % 2 else (sorted(vals)[len(vals) // 2 - 1] + sorted(vals)[len(vals) // 2]) / 2}[op]
    reported = calc.get("value")
    if reported is None:
        return {"checked": False, "reason": "no scalar result"}
    ok = abs(mine - reported) <= 1e-6 * max(1.0, abs(mine))
    return {"checked": True, "match": ok, "recomputed": mine, "reported": reported, "rows": len(vals)}


def verify_text(b: Board, text: str, sources: dict, extra: str = "", min_support: float = 0.55) -> dict:
    cleaned, rep = verify_answer(text, sources, extra_facts=extra, min_support=min_support)
    return {"text": cleaned, "report": rep}


REGISTRY: Dict[str, ToolSpec] = {s.name: s for s in [
    ToolSpec("search_files", "Find files by name / path / type", search_files, {"discovery"}),
    ToolSpec("search_contents", "Search inside file contents (keyword + semantic, passage level)", search_contents, {"discovery", "retrieval"}),
    ToolSpec("agent_search", "Run the LangGraph search agent (metadata + BM25 + semantic with reflection)", agent_search, {"discovery"}),
    ToolSpec("find_related", "Find files related to a given file", find_related, {"discovery"}),
    ToolSpec("list_project", "Pick the project a question is about and list its README, manifests, configs and source", list_project, {"discovery"}),
    ToolSpec("match_tables", "Find spreadsheets whose columns fit the question", match_tables, {"discovery"}),
    ToolSpec("get_metadata", "Read file metadata", get_metadata, {"discovery", "retrieval"}),
    ToolSpec("read_section", "Read a page, section or line range", read_section, {"retrieval"}),
    ToolSpec("read_table", "Read a spreadsheet's structure and first rows", read_table, {"retrieval"}),
    ToolSpec("extract_html_structure", "Extract the structure of an HTML page", extract_html_structure, {"retrieval"}),
    ToolSpec("read_code_outline", "Outline source code (definitions, imports, calls)", read_code_outline, {"retrieval"}),
    ToolSpec("read_config", "Read configuration values", read_config, {"retrieval"}),
    ToolSpec("summarize_document", "Pick representative content from a document", summarize_document, {"analysis"}),
    ToolSpec("analyze_table", "Calculate an answer from spreadsheet rows", analyze_table, {"analysis"}),
    ToolSpec("compare_documents", "Compare two documents", compare_documents, {"analysis"}),
    ToolSpec("trace_workflow", "Trace control flow across code files", trace_workflow, {"analysis"}),
    ToolSpec("review_resume", "Checklist review of a résumé", review_resume, {"analysis"}),
    ToolSpec("rank_sentences", "Pick the sentences that answer each question", rank_sentences, {"analysis"}),
    ToolSpec("check_sufficiency", "Does the evidence cover the question?", check_sufficiency, {"verification"}),
    ToolSpec("detect_conflicts", "Do sources disagree on a figure or date?", detect_conflicts, {"verification"}),
    ToolSpec("verify_claims", "Check each claim against the exact cited span", verify_claims, {"verification"}),
    ToolSpec("recompute_calc", "Independently recompute a calculation", recompute_calc, {"verification"}),
    ToolSpec("verify_text", "Check every sentence of a draft answer against its sources", verify_text, {"verification"}),
]}
