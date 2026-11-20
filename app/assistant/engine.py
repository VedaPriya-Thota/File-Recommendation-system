"""
The assistant: understand -> search -> retrieve evidence -> analyze -> verify -> answer, over the user's REAL
indexed files.

  * Search REUSES the existing agent loop (app.run.run_query: metadata/BM25/semantic/hybrid tools, scope,
    reflection, final ranking). Its trace is streamed to the UI as real activity.
  * Evidence is read from the indexed text (access-checked by owner_id), excerpts are verbatim slices with
    offsets, and are re-verified against the stored text before being returned.
  * Analysis (résumé review, CSV/Excel statistics, comparison, summaries) is deterministic Python over the real
    content. An LLM is OPTIONAL: it is called at most once per request, only if configured and permitted by the
    privacy gate, receives sanitized excerpts + the computed analysis only, has no tools, and its output is
    checked (citations must exist, quotes must be verbatim) before it is shown.
  * Read-only: nothing here writes to, moves or deletes user files.
"""
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from app.analysis import resume as resume_mod
from app.analysis import safety
from app.analysis import tabular
from app.analysis import text as T
from app.appconfig import get_settings
from app.assistant.intent import classify
from app.db.connection import get_cursor
from app.folders import FolderError, resolve_file
from app.llm.provider import LLMUnavailableError, get_default_provider, llm_status
from app.run import run_query
from app.scope import resolve_scope
from app.tools import build_registry

logger = logging.getLogger("file_discovery_agent.assistant")
MAX_ANALYSIS_BYTES = 60 * 1024 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ago(ts) -> str:
    try:
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        d = datetime.now(timezone.utc) - (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc))
        days = d.days
        return "today" if days < 1 else "yesterday" if days == 1 else f"{days} days ago" if days < 60 else f"{days // 30} months ago"
    except Exception:  # noqa: BLE001
        return ""


class Run:
    def __init__(self, run_id: str, emit: Callable[[dict], None]):
        self.id, self._emit, self.t0 = run_id, emit, time.perf_counter()
        self.events: List[dict] = []

    def step(self, kind: str, title: str, detail: str = "", status: str = "done", **extra):
        ev = {"t": round(time.perf_counter() - self.t0, 3), "kind": kind, "title": title, "detail": detail, "status": status, **extra}
        self.events.append(ev)
        self._emit(ev)
        return ev


# ------------------------------------------------------------------ data access (all owner-scoped)
def _load_files(owner_id: str, ids: List[int]) -> Dict[int, dict]:
    if not ids:
        return {}
    with get_cursor() as cur:
        cur.execute("""SELECT f.file_id, f.filename, f.file_path, f.file_type, f.created_at, f.updated_at, f.extracted_text,
                              s.size_bytes, s.mtime_ns, s.state, s.meta_json, s.folder_id, s.duplicate_of, s.fingerprint,
                              c.path AS folder_path, c.label AS folder_label
                       FROM files f
                       JOIN file_ingestion_state s ON s.file_id = f.file_id
                       JOIN connected_folders c ON c.folder_id = s.folder_id
                       WHERE f.owner_id = %(o)s AND f.file_id = ANY(%(ids)s);""", {"o": owner_id, "ids": list(ids)})
        rows = [dict(r) for r in cur.fetchall()]
    for r in rows:
        r["meta"] = json.loads(r.pop("meta_json") or "{}")
    return {r["file_id"]: r for r in rows}


def index_overview(owner_id: str) -> Dict:
    with get_cursor() as cur:
        cur.execute("""SELECT COUNT(*) AS n FROM file_ingestion_state s JOIN connected_folders c ON c.folder_id = s.folder_id
                       WHERE s.owner_id = %s AND s.state = 'content_indexed';""", (owner_id,))
        n = cur.fetchone()["n"]
        cur.execute("SELECT COUNT(*) AS n FROM connected_folders WHERE owner_id = %s;", (owner_id,))
        f = cur.fetchone()["n"]
    return {"indexed_files": n, "folders": f}


def _read_disk(owner_id: str, file_id: int, max_bytes: int = MAX_ANALYSIS_BYTES) -> bytes:
    info = resolve_file(owner_id, file_id)
    if info["abs_path"].stat().st_size > max_bytes:
        raise FolderError("File too large for analysis.")
    with open(info["abs_path"], "rb") as fh:  # read-only
        return fh.read()


# ------------------------------------------------------------------ retrieval
def _terms_present(terms: List[str], blob: str) -> List[str]:
    toks = {T._stem(t) for t in T.tokens(blob)}
    return [t for t in terms if T._stem(t) in toks]


def _semantic_is_real() -> bool:
    try:
        from app.embeddings import get_provider_name
        return not get_provider_name().startswith("hashing")
    except Exception:  # noqa: BLE001
        return False


def _search(run: Run, owner_id: str, plan: Dict, goal_terms_out: list) -> Dict[int, dict]:
    """Returns {file_id: candidate} using the existing agent for ranking + extra hybrid retrieval for typed intents."""
    def on_trace(e: dict):
        ev = e.get("event")
        if ev == "understand_goal":
            g = e.get("goal", {})
            goal_terms_out[:] = list(g.get("topic_tokens", []))
            run.step("agent", "Interpreted the search goal", f"keywords: {', '.join(g.get('topic_tokens', [])) or '—'}"
                     + (f"; file type: {g['filters']['file_type']}" if g.get("filters", {}).get("file_type") else ""))
        elif ev == "execute_tool":
            err = e.get("error")
            run.step("tool", f"Ran {e.get('tool')}", (f"error: {err}" if err else f"{e.get('candidate_count', 0)} candidate file(s)"),
                     status="error" if err else "done", tool=e.get("tool"), iteration=e.get("iteration"))
        elif ev == "reflect":
            run.step("agent", f"Reflection: {e.get('status')}", str(e.get("reason") or ""), iteration=e.get("iteration"))
        elif ev == "expand_query" and e.get("expanded_terms"):
            run.step("llm", "Expanded query terms", ", ".join(e["expanded_terms"]))

    run.step("agent", "Searching your indexed files", f"query: “{plan['retrieval_query']}”", status="running")
    result = run_query(owner_id, plan["retrieval_query"], on_trace=on_trace)
    if result.get("status") == "hard_failure":
        raise RuntimeError(result.get("error") or "search failed")
    cands: Dict[int, dict] = {}
    for rank, item in enumerate(result.get("final_ranking", [])):
        md = item.get("metadata", {}) or {}
        cands[item["file_id"]] = {
            "file_id": item["file_id"], "score": min(1.0, float(item.get("final_score", 0)) / 0.85 if item.get("final_score") else 0.0),
            "rank": rank, "sources": sorted((item.get("scores") or {}).keys()), "sem": (item.get("scores") or {}).get("semantic_search"),
            "snippet": md.get("matched_chunk") or md.get("semantic_matched_chunk") or md.get("snippet") or md.get("bm25_snippet"),
        }
    run.step("agent", f"Agent finished: {result.get('status')}", f"{len(cands)} ranked candidate(s) in {result.get('iteration_count', 0)} iteration(s)"
             + (f"; basis: {result['decision_basis']}" if result.get("decision_basis") else ""), agent_status=result.get("status"),
             clarification=result.get("clarification_question"))
    if plan["type_group"] or plan["intent"] in ("project_files",):
        reg = build_registry()
        scope = resolve_scope(owner_id)
        extra = reg.execute("hybrid_search", {"query": plan["retrieval_query"], "top_k": 60}, scope)
        added = 0
        top = max([c["score"] for c in extra.candidates] or [1.0])
        for i, c in enumerate(extra.candidates):
            if c["file_id"] not in cands:
                cands[c["file_id"]] = {"file_id": c["file_id"], "score": 0.6 * c["score"] / top, "rank": 100 + i, "sources": ["hybrid_search"],
                                       "sem": (c["metadata"] or {}).get("semantic_score"), "snippet": (c["metadata"] or {}).get("semantic_matched_chunk") or (c["metadata"] or {}).get("bm25_snippet")}
                added += 1
        run.step("tool", "Ran hybrid_search (wider pass)", f"{len(extra.candidates)} candidate(s), {added} new", tool="hybrid_search")
    return cands


def _rank_and_filter(run: Run, owner_id: str, plan: Dict, cands: Dict[int, dict], terms: List[str], files: Dict[int, dict]) -> List[dict]:
    types = set(plan["types"])
    real_sem = _semantic_is_real()
    kept = []
    dropped_sem = 0
    for fid, c in cands.items():
        f = files.get(fid)
        if f is None:
            continue
        if types and f["file_type"] not in types:
            continue
        blob_head = f"{f['filename']} {f['file_path']}"
        present = _terms_present(terms, blob_head + " " + (f["extracted_text"] or ""))
        lexical = bool(present) or not terms
        sem_ok = real_sem and (c.get("sem") or 0) >= 0.5
        if plan["intent"] == "resume_improve" or (plan["retrieval_query"].lower().find("resume") >= 0 and "resume" in plan["retrieval_query"].lower()):
            pass
        if not lexical and not sem_ok:
            dropped_sem += 1
            continue
        c["matched_terms"] = present
        c["in_name"] = _terms_present(terms, blob_head)
        kept.append(c)
    if dropped_sem:
        run.step("verify", "Filtered weak matches", f"{dropped_sem} candidate(s) had no query term in their name/content and no trustworthy semantic similarity"
                 + ("" if real_sem else " (the active embedding model is the offline hashing fallback, so semantic-only matches are not trusted)"))
    if plan["intent"] == "resume_improve":
        for c in kept:
            f = files[c["file_id"]]
            c["score"] = c["score"] * 0.4 + resume_mod_score(f) * 0.6
    kept.sort(key=lambda c: (-c["score"], c["rank"], c["file_id"]))
    if plan["type_group"] and kept:  # typed requests: drop matches far weaker than the best one
        top = kept[0]["score"]
        kept = [c for c in kept if c["score"] >= 0.45 * top]
    return kept


def resume_mod_score(f: dict) -> float:
    name = f["filename"].lower()
    text = (f["extracted_text"] or "")[:6000].lower()
    s = 0.0
    if re.search(r"resume|résumé|\bcv\b|curriculum", name):
        s += 0.55
    hits = sum(1 for w in ("experience", "education", "skills", "summary", "projects", "certification") if w in text)
    s += min(0.45, hits * 0.09)
    if re.search(r"[\w.+-]+@[\w-]+\.\w+", text):
        s += 0.05
    return min(1.0, s)


def _dedupe(kept: List[dict], files: Dict[int, dict]) -> List[dict]:
    seen, out = {}, []
    for c in kept:
        f = files[c["file_id"]]
        fp = f.get("fingerprint")
        c["duplicates"] = []
        if fp and fp in seen:
            first = seen[fp]
            if files[first["file_id"]].get("duplicate_of") is not None and f.get("duplicate_of") is None:
                # the canonical (first-indexed) copy wins over a later duplicate; keep the better position
                idx = out.index(first)
                c["duplicates"] = first["duplicates"] + [first["file_id"]]
                c["score"] = max(c["score"], first["score"])
                out[idx] = c
                seen[fp] = c
            else:
                first["duplicates"].append(c["file_id"])
            continue
        if fp:
            seen[fp] = c
        out.append(c)
    return out


# ------------------------------------------------------------------ references
def _reason(c: dict, f: dict, terms: List[str]) -> str:
    bits = []
    if c.get("in_name"):
        bits.append(f"name/path contains {', '.join('“' + t + '”' for t in c['in_name'][:4])}")
    counts = T.term_counts(f["extracted_text"] or "", terms)
    if counts:
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
        bits.append("content mentions " + ", ".join(f"“{k}” ×{v}" for k, v in top))
    if c.get("sem") is not None and c["sem"] >= 0.5 and _semantic_is_real():
        bits.append(f"semantically similar passage (similarity {c['sem']:.2f})")
    if "get_user_activity" in (c.get("sources") or []):
        bits.append("you have opened/edited it before")
    bits.append(f"modified {_ago(f['updated_at'])}" if _ago(f["updated_at"]) else "")
    return "; ".join(b for b in bits if b).capitalize() + "."


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _build_ref(idx: int, c: dict, f: dict, terms: List[str], cfg: dict) -> dict:
    text = f["extracted_text"] or ""
    ex = T.top_excerpts(text, terms, cfg["excerpts_per_file"], cfg["excerpt_chars"]) if terms else []
    if not ex and text:
        ex = [T.head_excerpt(text, cfg["excerpt_chars"])]
    is_code = bool(f["meta"].get("language"))
    for e in ex:
        e["verified"] = text[e["start"]:e["end"]] == e["text"]
        if is_code:
            e["line_start"], e["line_end"] = _line_of(text, e["start"]), _line_of(text, e["end"])
    return {
        "ref": f"F{idx}", "file_id": f["file_id"], "filename": f["filename"], "rel_path": f["file_path"], "folder_label": f["folder_label"],
        "abs_path": str(Path(f["folder_path"]) / f["file_path"]), "file_type": f["file_type"], "size_bytes": f["size_bytes"],
        "modified": str(f["updated_at"]), "relevance": round(min(1.0, max(0.0, c["score"])), 3), "reason": _reason(c, f, terms),
        "matched_terms": c.get("matched_terms", []), "sources": c.get("sources", []), "excerpts": ex, "meta": f["meta"],
        "duplicates": c.get("duplicates", []), "language": f["meta"].get("language"),
    }


# ------------------------------------------------------------------ analyses
def _analyze_resume(run: Run, owner_id: str, f: dict) -> Dict:
    run.step("analysis", "Reviewing résumé", f["filename"], status="running")
    r = resume_mod.review(f["extracted_text"] or "", pages=f["meta"].get("pages"))
    run.step("analysis", "Résumé review complete", f"score {r['score']}/100, {len(r['suggestions'])} improvement area(s)")
    return r


def _analyze_table(run: Run, owner_id: str, f: dict) -> Dict:
    run.step("analysis", "Reading data file from disk", f"{f['filename']} (read-only)", status="running")
    data = _read_disk(owner_id, f["file_id"])
    res = tabular.analyze_file(data, f["file_type"])
    run.step("analysis", "Computed statistics & trends", f"{sum(s['rows'] for s in res['sheets'])} rows across {len(res['sheets'])} sheet(s); {len(res['findings'])} finding(s)")
    return res


def _staleness(owner_id: str, f: dict) -> Optional[str]:
    try:
        st = (Path(f["folder_path"]) / f["file_path"]).stat()
        if st.st_mtime_ns != f["mtime_ns"] or st.st_size != f["size_bytes"]:
            return "changed on disk since it was last indexed (a re-scan is pending); results may be outdated"
    except OSError:
        return "no longer on disk; it will drop out of the index on the next scan"
    return None


# ------------------------------------------------------------------ answer composition (deterministic)
def _md_list(items: List[str]) -> str:
    return "\n".join(f"- {i}" for i in items)


def _compose(plan: Dict, refs: List[dict], analyses: Dict, warnings: List[str]) -> str:
    intent = plan["intent"]
    if not refs:
        return "I couldn't find any indexed file that matches that. Try different keywords, or check **Connected Folders** to make sure the folder is indexed."
    top = refs[0]
    out: List[str] = []
    if intent == "resume_improve" and analyses.get("resume"):
        r = analyses["resume"]
        out.append(f"I found **{top['filename']}** [F1] in *{top['folder_label']}* (`{top['rel_path']}`). Résumé checklist score: **{r['score']}/100**.")
        bad = [c for c in r["checks"] if not c["ok"]]
        good = [c for c in r["checks"] if c["ok"]]
        if bad:
            out.append("**Suggested improvements** (each is based on the file's actual text):")
            for c in bad:
                line = f"**{c['title']}** — {c['detail']}"
                if c["evidence"]:
                    line += " Example: “" + c["evidence"][0][:140] + "”"
                out.append("- " + line)
        if good:
            out.append("**Already good:** " + "; ".join(c["title"].lower() for c in good) + ".")
        others = [x for x in refs[1:3]]
        if others:
            out.append("Other possible résumé files: " + ", ".join(f"{o['filename']} [{o['ref']}]" for o in others) + ".")
    elif intent == "compare" and analyses.get("comparison"):
        c = analyses["comparison"]
        a, b = c["files"]
        out.append(f"Comparing **{a['filename']}** [{a['ref']}] and **{b['filename']}** [{b['ref']}]:")
        out.append(_md_list([f"Content similarity: **{c['similarity']:.0%}** (vocabulary overlap {c['vocabulary_overlap']:.0%}); length {c['words'][0]} vs {c['words'][1]} words.",
                             "Shared topics: " + (", ".join(c["shared_terms"][:8]) or "none"),
                             f"Only in {a['filename']}: " + (", ".join(c["only_in_first"][:8]) or "—"),
                             f"Only in {b['filename']}: " + (", ".join(c["only_in_second"][:8]) or "—")]
                            + ([f"Numbers only in {a['filename']}: {', '.join(c['numbers_only_in_first'][:8])}"] if c["numbers_only_in_first"] else [])
                            + ([f"Numbers only in {b['filename']}: {', '.join(c['numbers_only_in_second'][:8])}"] if c["numbers_only_in_second"] else [])))
        for s in analyses.get("summaries", []):
            out.append(f"**{s['filename']}** [{s['ref']}] — " + " ".join(x["text"] for x in s["sentences"][:3]))
    elif intent == "summarize" and analyses.get("summaries"):
        for s in analyses["summaries"]:
            out.append(f"**{s['filename']}** [{s['ref']}]")
            out.append(_md_list([x["text"] for x in s["sentences"]]))
            if s.get("keywords"):
                out.append("Key terms: " + ", ".join(s["keywords"][:10]) + ".")
    elif intent == "analyze_data" and analyses.get("tables"):
        for t in analyses["tables"]:
            out.append(f"**{t['filename']}** [{t['ref']}] — findings from the actual data:")
            out.append(_md_list(t["findings"][:10]))
    else:
        label = {"find_dataset": "dataset(s)", "find_code": "source file(s)", "project_files": "related file(s)"}.get(intent, "relevant file(s)")
        out.append(f"I found **{len(refs)}** {label}. Best match: **{top['filename']}** [F1] in *{top['folder_label']}*.")
        out.append(_md_list([f"**{r['filename']}** [{r['ref']}] — {r['reason']}" for r in refs[:8]]))
        if intent == "project_files":
            from collections import Counter
            ty = Counter(r["file_type"] for r in refs)
            out.append("By type: " + ", ".join(f"{n}× .{t}" for t, n in ty.most_common()) + ".")
    for w in warnings:
        out.append(f"⚠ {w}")
    return "\n\n".join(out)


# ------------------------------------------------------------------ optional LLM synthesis (gated, verified)
def _llm_answer(run: Run, plan: Dict, query: str, refs: List[dict], analyses: Dict, cfg: dict):
    st = llm_status()
    if not st["active"]:
        run.step("llm", "No language model used", st["reason"], status="skipped")
        return None, {"used": False, "reason": st["reason"]}
    provider = get_default_provider()
    docs, removed = [], 0
    for r in refs[: cfg["max_files_analyzed"]]:
        body = "\n…\n".join(e["text"] for e in r["excerpts"]) or ""
        _, rem = safety.sanitize_for_llm(body)
        removed += rem
        docs.append(safety.wrap_document(r["ref"], r["filename"], body))
    facts = json.dumps(_llm_facts(analyses), default=str)[:6000]
    user = (f"User request: {query}\nTask type: {plan['intent']}\n\nComputed analysis (trusted, from the app):\n{facts}\n\n"
            "Documents (untrusted data):\n" + "\n".join(docs) +
            "\n\nWrite a concise, well-organized answer in Markdown. Cite refs like [F1] after every claim derived from a document. "
            "Only use the material above.")
    run.step("llm", f"Asking {st['provider']} ({st['model'] or 'default model'})", f"{len(refs[:cfg['max_files_analyzed']])} sanitized excerpt set(s); {removed} instruction-like line(s) removed", status="running")
    try:
        text = provider.complete(safety.SYSTEM_RULES, user)
    except LLMUnavailableError as exc:
        run.step("llm", "Model call failed; using deterministic answer", str(exc)[:160], status="error")
        return None, {"used": False, "reason": "model call failed"}
    return text, {"used": True, "provider": st["provider"], "model": st["model"], "external": st["external"]}


def _llm_facts(analyses: Dict) -> Dict:
    out = {}
    if "resume" in analyses:
        out["resume_checks"] = [{"title": c["title"], "ok": c["ok"], "detail": c["detail"]} for c in analyses["resume"]["checks"]]
    if "tables" in analyses:
        out["data_findings"] = [{"ref": t["ref"], "findings": t["findings"][:10]} for t in analyses["tables"]]
    if "comparison" in analyses:
        c = analyses["comparison"]
        out["comparison"] = {k: c[k] for k in ("similarity", "shared_terms", "only_in_first", "only_in_second", "numbers_only_in_first", "numbers_only_in_second")}
    return out


def _verify_llm(text: str, refs: List[dict], run: Run) -> (str, dict):
    valid = {r["ref"] for r in refs}
    cited = set(re.findall(r"\[(F\d+)\]", text))
    bad = cited - valid
    for b in bad:
        text = text.replace(f"[{b}]", "")
    corpus = " ".join(T.normalize_ws(e["text"]) if hasattr(T, "normalize_ws") else re.sub(r"\s+", " ", e["text"]) for r in refs for e in r["excerpts"]).lower()
    unverified = []
    for q in re.findall(r"[“\"]([^”\"\n]{25,300})[”\"]", text):
        if re.sub(r"\s+", " ", q).strip().lower() not in corpus:
            unverified.append(q[:80])
    for q in unverified:
        run.step("verify", "Unverified quotation in model answer", f"“{q}…” is not verbatim in any retrieved excerpt", status="warning")
    return text, {"citations_checked": len(cited), "invalid_citations_removed": sorted(bad), "unverified_quotes": unverified}


# ------------------------------------------------------------------ main entry
def _run_find(owner_id: str, query: str, run: Run, plan: Dict, settings: dict, extra: Optional[dict] = None) -> Dict:
    """File-finding requests (recommend / locate / datasets / code files / résumé review): the original flow."""
    cfg = settings["assistant"]
    run.step("understand", "Understood the request", f"task: {plan['intent'].replace('_', ' ')}; search for “{plan['retrieval_query']}”"
             + (f"; file family: {plan['type_group']}" if plan["type_group"] else ""), intent=plan["intent"])
    overview = index_overview(owner_id)
    if overview["folders"] == 0 or overview["indexed_files"] == 0:
        msg = ("No folders are connected yet. Open **Connected Folders** and add a directory to index." if overview["folders"] == 0
               else "Your folders are connected but nothing is indexed yet (scanning may still be in progress or all files were unsupported).")
        run.step("verify", "Nothing to search", msg, status="warning")
        return _finish(owner_id, run, query, plan, "no_index", msg, [], {}, [], {"checked": 0}, {"used": False}, extra)

    terms_from_goal: List[str] = []
    try:
        cands = _search(run, owner_id, plan, terms_from_goal)
    except Exception as exc:  # noqa: BLE001
        logger.exception("search failed")
        run.step("agent", "Search failed", f"{type(exc).__name__}", status="error")
        return _finish(owner_id, run, query, plan, "error", f"The search failed ({type(exc).__name__}).", [], {}, [], {"checked": 0}, {"used": False}, extra)

    terms = T.query_terms(plan["retrieval_query"], terms_from_goal)
    if plan["intent"] == "resume_improve":
        terms = [t for t in terms if t not in ("cv", "curriculum", "vitae")] or ["resume"]
    # compare: one best file per sub-query
    if plan["intent"] == "compare" and len(plan["compare_parts"]) >= 2:
        picked = _compare_pick(run, owner_id, plan)
        for fid, c in picked.items():
            cands.setdefault(fid, c)
            cands[fid]["score"] = max(cands[fid]["score"], c["score"]) + 1.0  # force to front
    files = _load_files(owner_id, list(cands))
    kept = _dedupe(_rank_and_filter(run, owner_id, plan, cands, terms, files), files)
    limit = plan["max_results"]
    kept = kept[:limit]
    refs = [_build_ref(i + 1, c, files[c["file_id"]], terms, cfg) for i, c in enumerate(kept)]
    run.step("retrieve", f"Retrieved evidence from {len(refs)} file(s)", "excerpts are verbatim passages from the indexed text" if refs else "no matching files", count=len(refs))

    warnings: List[str] = []
    for r in refs:
        f = files[r["file_id"]]
        stale = _staleness(owner_id, f)
        if stale:
            r["stale"] = stale
            warnings.append(f"{r['filename']} {stale}.")
        inj = safety.scan(f["extracted_text"] or "")
        if inj:
            r["injection_suspected"] = inj[:3]
            warnings.append(f"{r['filename']} contains text that looks like instructions to an AI system ({len(inj)} line(s)); it was treated as data and ignored.")
            run.step("verify", "Possible prompt injection in a document", f"{r['filename']}: {inj[0]['snippet'][:100]}", status="warning")

    analyses: Dict = {}
    try:
        _run_analyses(run, owner_id, plan, refs, files, analyses, cfg)
    except Exception as exc:  # noqa: BLE001
        logger.exception("analysis failed")
        run.step("analysis", "Analysis step failed", f"{type(exc).__name__}", status="error")
        warnings.append("Some analysis could not be completed.")

    answer = _compose(plan, refs, analyses, warnings)
    llm_meta = {"used": False}
    verification = {}
    if refs and plan["intent"] in ("resume_improve", "compare", "summarize", "analyze_data"):
        text, llm_meta = _llm_answer(run, plan, query, refs, analyses, cfg)
        if text:
            text, verification = _verify_llm(text, refs, run)
            answer = text.strip() + ("\n\n" + "\n\n".join(f"⚠ {w}" for w in warnings) if warnings else "")
    # final verification of evidence
    checked = bad = 0
    for r in refs:
        for e in r["excerpts"]:
            checked += 1
            if not e.get("verified", True):
                bad += 1
    verification.update({"excerpts_checked": checked, "excerpts_failed": bad, "files_in_scope": len(refs), "checked": checked})
    run.step("verify", "Verified evidence", f"{checked - bad}/{checked} excerpts match the stored text; all files are inside your connected folders", status="done" if not bad else "warning")
    status = "ok" if refs else "no_results"
    return _finish(owner_id, run, query, plan, status, answer, refs, analyses, warnings, verification, llm_meta, extra)


def _compare_pick(run: Run, owner_id: str, plan: Dict) -> Dict[int, dict]:
    reg, scope, out = build_registry(), resolve_scope(owner_id), {}
    for part in plan["compare_parts"]:
        res = reg.execute("hybrid_search", {"query": part, "top_k": 5}, scope)
        for c in res.candidates:
            if c["file_id"] not in out:
                out[c["file_id"]] = {"file_id": c["file_id"], "score": 0.9, "rank": 0, "sources": ["hybrid_search"], "sem": None, "snippet": None}
                run.step("tool", "Located file for comparison", f"“{part}” → {c['metadata'].get('filename')}", tool="hybrid_search")
                break
    return out


def _run_analyses(run, owner_id, plan, refs, files, analyses, cfg):
    intent = plan["intent"]
    n = cfg["max_files_analyzed"]
    if intent == "resume_improve" and refs:
        analyses["resume"] = _analyze_resume(run, owner_id, files[refs[0]["file_id"]])
    elif intent == "compare" and len(refs) >= 2:
        a, b = refs[0], refs[1]
        run.step("analysis", "Comparing documents", f"{a['filename']} vs {b['filename']}", status="running")
        comp = T.compare_documents(files[a["file_id"]]["extracted_text"] or "", files[b["file_id"]]["extracted_text"] or "")
        comp["files"] = [{"ref": a["ref"], "filename": a["filename"]}, {"ref": b["ref"], "filename": b["filename"]}]
        analyses["comparison"] = comp
        analyses["summaries"] = [{"ref": r["ref"], "filename": r["filename"], "sentences": T.extractive_summary(files[r["file_id"]]["extracted_text"] or "", 3)} for r in (a, b)]
        run.step("analysis", "Comparison complete", f"similarity {comp['similarity']:.0%}")
    elif intent == "summarize" and refs:
        take = refs[:1] if len(refs) == 1 or refs[0]["relevance"] - refs[1]["relevance"] > 0.15 else refs[:3]
        analyses["summaries"] = []
        for r in take:
            t = files[r["file_id"]]["extracted_text"] or ""
            analyses["summaries"].append({"ref": r["ref"], "filename": r["filename"], "sentences": T.extractive_summary(t, 5), "keywords": T.keywords(t), "headings": T.headings(t)})
        run.step("analysis", "Summarized", ", ".join(r["filename"] for r in take))
    elif intent == "analyze_data" and refs:
        analyses["tables"] = []
        for r in [x for x in refs if x["file_type"] in ("csv", "tsv", "xlsx", "xlsm")][: min(n, 3)]:
            try:
                res = _analyze_table(run, owner_id, files[r["file_id"]])
                analyses["tables"].append({"ref": r["ref"], "filename": r["filename"], "findings": res["findings"], "sheets": res["sheets"]})
            except Exception as exc:  # noqa: BLE001
                run.step("analysis", f"Could not analyze {r['filename']}", type(exc).__name__, status="error")
        if not analyses["tables"]:
            run.step("analysis", "No tabular file among the matches", "CSV/TSV/Excel files are required for trend analysis", status="warning")


def _finish(owner_id, run: Run, query, plan, status, answer, refs, analyses, warnings, verification, llm_meta, extra=None) -> Dict:
    result = {**(extra or {}),
        "run_id": run.id, "query": query, "intent": plan["intent"], "retrieval_query": plan["retrieval_query"], "status": status,
        "answer": answer, "references": refs, "analyses": analyses, "warnings": warnings, "verification": verification,
        "llm": llm_meta, "activity": run.events, "duration_s": round(time.perf_counter() - run.t0, 2), "created_at": _now(),
    }
    try:
        if get_settings()["privacy"]["log_queries"]:
            with get_cursor(commit=True) as cur:
                cur.execute("INSERT INTO assistant_runs (run_id, owner_id, query, intent, status, result, created_at, finished_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s);",
                            (run.id, owner_id, query, plan["intent"], status, json.dumps(result, default=str), result["created_at"], _now()))
    except Exception:  # noqa: BLE001
        logger.warning("could not persist assistant run")
    return result


# ------------------------------------------------------------------ public entry: orchestrated agents
def run_assistant(owner_id: str, query: str, emit: Callable[[dict], None], run_id: Optional[str] = None,
                  conversation_id: Optional[str] = None, force_find: bool = False) -> Dict:
    from app.agents.core import EventBus
    from app.agents.orchestrator import run_agents
    from app.assistant import conversation as convmod
    from app.assistant.understand import plan_question

    run = EventBus(run_id or uuid.uuid4().hex, emit)
    settings = get_settings()
    conv = convmod.get(conversation_id)
    qplan = plan_question(query, owner_id, conv)
    if force_find and qplan.kind not in ("file_search", "resume_improve"):   # Recommendations page: always "which files are relevant"
        qplan.kind, qplan.find_plan = "file_search", classify(query)
    find_mode = qplan.kind in ("file_search", "resume_improve")
    if find_mode and qplan.find_plan is None:
        qplan.find_plan = classify(query)
    extra = {"conversation_id": conv["id"], "mode": "find" if find_mode else "qa", "followup": qplan.inherited, "task_kind": qplan.kind}
    plan_dict = qplan.find_plan if find_mode else {"intent": qplan.kind, "retrieval_query": " ".join(qplan.all_terms) or query}
    overview = index_overview(owner_id)
    run.set_context("Orchestrator", None)
    if overview["folders"] == 0 or overview["indexed_files"] == 0:
        msg = ("No folders are connected yet, so there is nothing to answer from. Open **Connected Folders** and add a directory." if overview["folders"] == 0
               else "Your folders are connected but nothing is indexed yet (scanning may still be running).")
        run.step("verify", "Nothing to search", msg, status="warning", agent="Orchestrator")
        return _finish(owner_id, run, query, plan_dict, "no_index", msg, [], {}, [], {"checked": 0}, {"used": False}, extra)
    conv_hint = ""
    if qplan.inherited and conv.get("turns"):
        last = conv["turns"][-1]
        conv_hint = f"The user previously asked: “{last['q'][:160]}”, about {', '.join(conv.get('focus_names') or [])}."
    try:
        out = run_agents(owner_id, query, qplan, settings, conv_hint, run)
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent run failed")
        run.step("agent", "The agents could not complete the request", type(exc).__name__, status="error", agent="Orchestrator")
        return _finish(owner_id, run, query, plan_dict, "error", f"Something went wrong while answering ({type(exc).__name__}). Nothing was changed.", [], {}, [], {"checked": 0}, {"used": False}, extra)
    out.pop("_board", None)
    refs = out.get("references", [])
    warnings = list(out.get("warnings", []))
    files = _load_files(owner_id, [r["file_id"] for r in refs])
    extra.update({"agents": out.get("agents"), "coverage": out.get("coverage", {}), "conflicts": out.get("conflicts", []), "clarify_options": out.get("clarify_options", [])})
    if not find_mode:
        for r in refs:                      # staleness / injection checks on the files that back the answer
            f = files.get(r["file_id"])
            if f:
                stale = _staleness(owner_id, f)
                if stale:
                    r["stale"] = stale
                    warnings.append(f"{r['filename']} {stale}.")
            inj = [h for e in r["excerpts"] for h in safety.scan(e["text"])]
            if inj:
                r["injection_suspected"] = inj[:3]
                warnings.append(f"{r['filename']} contains text that looks like instructions to an AI system; it was treated as plain data and ignored.")
                run.step("verify", "Possible prompt injection in a source", f"{r['filename']}: {inj[0]['snippet'][:100]}", status="warning", agent="Verification Agent")
    checked = sum(len(r["excerpts"]) for r in refs)
    bad = sum(1 for r in refs for e in r["excerpts"] if e.get("verified") is False)
    verification = {**out.get("verification", {}), "excerpts_checked": checked, "excerpts_failed": bad, "checked": checked}
    status = out.get("status", "error")
    focus_ids = out.get("_focus") or [r["file_id"] for r in refs if not r.get("closest_only")][:3]
    names = [files[i]["filename"] if i in files else "" for i in focus_ids]
    convmod.record(conv, query, qplan.kind, focus_ids if status in ("ok", "partial") else [], [n for n in names if n], qplan.all_terms, status in ("ok", "partial"))
    return _finish(owner_id, run, query, plan_dict, status, out.get("answer", ""), refs, out.get("analyses", {}), list(dict.fromkeys(warnings)), verification, out.get("llm", {"used": False}), extra)
