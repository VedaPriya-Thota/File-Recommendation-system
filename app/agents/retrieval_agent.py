"""Retrieval Agent -- pulls the SPECIFIC information out of the selected files and keeps where it came from.

  retrieve     passages for one information need, scoped to the discovered files; widens (all files, then looser matching) when
               the first attempt does not cover the question -- a bounded, observable retry ladder
  read_aspect  passages about one aspect inside ONE file (used per side of a comparison, run in parallel)
  read_table   spreadsheet structure and first rows, read from the file itself
  read_code    outlines of code files, README text, manifests and configuration values
  read_html    structure of an HTML page
  read_document the relevant part of a document (a page / section named in the question, else the opening) for summarising
  excerpts     verbatim excerpts with offsets for recommended files
Output is always structured (offsets + locations); document text stays in the database / cache.
"""
import re
from types import SimpleNamespace
from typing import List

from app.agents.core import AgentFailure, Board, Task
from app.agents.tools import info_for, text_of
from app.analysis import text as T
from app.assistant import qa
from app.assistant import retrieval as R
from app.assistant.understand import _make_need


class RetrievalAgent:
    name = "retrieval"

    def run(self, task: Task, b: Board, tb) -> str:
        op = getattr(self, "op_" + task.op, None)
        if op is None:
            raise AgentFailure(f"unknown retrieval operation {task.op}", retryable=False)
        return op(task, b, tb)

    # ------------------------------------------------------------------ facts
    def op_retrieve(self, task, b: Board, tb) -> str:
        q = b.plan_q
        i = task.args["need_no"]
        need = q.needs[i]
        if not need.stems:
            b.evidence[i] = []
            b.retrieval_meta[i] = {"ok": False, "assess": {"ok": False, "best": None, "reason": "no_terms", "missing": []}, "ret": SimpleNamespace(searched_files=0, searched_passages=0, evidence=[], steps=[])}
            return "Nothing concrete to look for"
        multi = q.kind == "multi_doc"
        top_n, per_file = (8 if multi else 4), (2 if multi or len(q.needs) > 1 else 3)
        hints = b.retrieval_hints.get(i, {})
        scope = None if hints.get("global") else (b.target_ids[:8] or None)
        if q.target_file_ids:
            scope = q.target_file_ids
        steps = []

        def search(scope_ids, relaxed=False, extra_n=0):
            r = tb.call("retrieval", "search_contents", need=need, need_no=i, scope_ids=scope_ids, top_n=top_n + extra_n, per_file_cap=per_file + (1 if relaxed else 0),
                        pool=60 if relaxed else 30, relaxed=relaxed,
                        summarize=lambda d: f"{len(d['evidence'])} passage(s) from {len(d['file_scores'])} file(s)" + (f" · scoped to {len(scope_ids)} file(s)" if scope_ids else " · all indexed files"))
            if not r.ok:
                raise AgentFailure("content search failed: " + r.error[:100])
            return r.data

        d = search(None if hints.get("global") else scope, relaxed=hints.get("relaxed", False), extra_n=4 if hints.get("relaxed") else 0)
        evs = self._code_filter(b, q, d["evidence"])
        a = R.assess(need, evs)
        steps.append("scoped" if scope else "all files")
        if not a["ok"] and scope:
            b.bus.step("retry", "Not enough in the discovered files — widening to all indexed files", "", status="warning")
            d = search(None)
            evs = self._code_filter(b, q, d["evidence"])
            a = R.assess(need, evs)
            steps.append("widened")
        if not a["ok"] and hints.get("relaxed"):          # only on a round the Verification Agent asked for
            d2 = d
            a2 = R.assess(need, qa._with_context(evs))
            if a2["ok"]:
                a = a2
                steps.append("relaxed")
        with b.lock:
            b.evidence[i] = evs
            b.retrieval_meta[i] = {"ok": a["ok"], "assess": a, "steps": steps, "ret": SimpleNamespace(searched_files=d["searched_files"], searched_passages=d["searched_passages"], evidence=evs, steps=d["steps"])}
        # code questions: read the structure of the code files the passages came from
        files = list(dict.fromkeys(e.file_id for e in evs))[:3]
        if files:
            inf = info_for(b, files)
            for fid in files:
                if inf.get(fid, {}).get("file_type") in qa.CODE_TYPES:
                    r = tb.call("retrieval", "read_code_outline", file_id=fid, summarize=lambda d: f"{len(d['outline']['items'])} definition(s)")
                    if r.ok:
                        b.structures.setdefault(fid, {})["outline"] = r.data["outline"]
        return f"{len(evs)} passage(s) · coverage {round(a['best']['cov'] * 100) if a.get('best') else 0}%" if evs else "No matching passages"

    @staticmethod
    def _code_filter(b, q, evs):
        if q.kind == "code_explain":
            inf = info_for(b, [e.file_id for e in evs])
            code = [e for e in evs if inf.get(e.file_id, {}).get("file_type") in qa.CODE_TYPES]
            return code or evs
        return evs

    # ------------------------------------------------------------------ comparison side
    def op_read_aspect(self, task, b: Board, tb) -> str:
        fid, need = task.args["file_id"], task.args.get("need")
        if need is not None and need.stems:
            r = tb.call("retrieval", "search_contents", need=need, need_no=0, scope_ids=[fid], top_n=3, per_file_cap=3,
                        summarize=lambda d: f"{len(d['evidence'])} passage(s) about {', '.join(need.terms[:3])}")
            evs = r.data["evidence"] if r.ok else []
        else:
            r = tb.call("retrieval", "read_section", file_id=fid, max_chars=2500, summarize=lambda d: f"{len(d['segments'])} section(s) read")
            evs = []
            if r.ok:
                inf = info_for(b, [fid])[fid]
                for s in r.data["segments"][:3]:
                    evs.append(R.Ev(file_id=fid, idx=0, start=s["start"], end=s["end"], text=s["text"], page=s["page"], heading=s["heading"], line_start=s["line_start"], line_end=s["line_end"]))
        with b.lock:
            b.passages_read[fid] = evs
        return f"{len(evs)} passage(s) from {info_for(b, [fid])[fid]['filename']}"

    # ------------------------------------------------------------------ documents / tables / html / code
    def op_read_document(self, task, b: Board, tb) -> str:
        fid = task.args["file_id"]
        inf = info_for(b, [fid]).get(fid)
        if not inf:
            raise AgentFailure("file is not indexed", retryable=False)
        if inf["file_type"] in ("html", "htm"):
            r = tb.call("retrieval", "extract_html_structure", file_id=fid, summarize=lambda d: f"title “{d['structure']['title'][:40]}”, {len(d['structure']['headings'])} heading(s), {len(d['structure']['forms'])} form(s)")
            if not r.ok:
                raise AgentFailure("could not parse the HTML")
            b.structures.setdefault(fid, {})["html"] = r.data["structure"]
            return f"Extracted the structure of {inf['filename']}"
        q = re.search(r"\bpage (\d+)\b|\bslide (\d+)\b", b.question, re.I)
        page = int(next(g for g in q.groups() if g)) if q else None
        r = tb.call("retrieval", "read_section", file_id=fid, page=page, max_chars=3000, summarize=lambda d: f"{len(d['segments'])} section(s) · {d['total_chars']:,} characters in the file")
        if not r.ok:
            raise AgentFailure("could not read the document")
        with b.lock:
            b.passages_read[fid] = [R.Ev(file_id=fid, idx=0, start=s["start"], end=s["end"], text=s["text"], page=s["page"], heading=s["heading"], line_start=s["line_start"], line_end=s["line_end"])
                                    for s in r.data["segments"]]
            b.structures.setdefault(fid, {})["chars"] = r.data["total_chars"]
        if not r.data["segments"]:
            raise AgentFailure("the document has no readable text", retryable=False)
        return f"Read {inf['filename']} ({r.data['total_chars']:,} characters)"

    def op_read_table(self, task, b: Board, tb) -> str:
        done = []
        for fid in task.args["file_ids"]:
            r = tb.call("retrieval", "read_table", file_id=fid, summarize=lambda d: ", ".join(f"{s['name']}: {s['rows']:,} rows × {len(s['columns'])} columns" for s in d["profile"]["sheets"][:2]))
            if r.ok:
                b.structures.setdefault(fid, {})["table"] = r.data["profile"]
                done.append(fid)
        if not done:
            raise AgentFailure("the spreadsheet could not be read")
        return f"Read {len(done)} spreadsheet(s)"

    def op_read_code(self, task, b: Board, tb) -> str:
        n = 0
        inf = info_for(b, task.args["file_ids"])
        for fid in task.args["file_ids"]:
            f = inf.get(fid)
            if not f:
                continue
            role = task.args.get("role", "source")
            if role == "config" or f["file_type"] in ("json", "yaml", "yml", "toml", "ini", "cfg"):
                r = tb.call("retrieval", "read_config", file_id=fid, summarize=lambda d: f"{len(d['values'])} setting(s)")
                if r.ok:
                    b.structures.setdefault(fid, {})["config"] = r.data["values"]
                    n += 1
            elif f["file_type"] in qa.CODE_TYPES:
                r = tb.call("retrieval", "read_code_outline", file_id=fid, summarize=lambda d: f"{len(d['outline']['items'])} definition(s), {len(d['outline']['imports'])} import(s)")
                if r.ok:
                    b.structures.setdefault(fid, {})["outline"] = r.data["outline"]
                    n += 1
            else:                                     # README / manifest / docs: the opening text
                r = tb.call("retrieval", "read_section", file_id=fid, max_chars=3500, summarize=lambda d: f"{len(d['segments'])} section(s)")
                if r.ok:
                    b.structures.setdefault(fid, {})["doc"] = text_of(b, fid)[:6000]
                    n += 1
        if not n:
            raise AgentFailure("nothing could be read", retryable=False)
        return f"Read {n} {task.args.get('role', 'source')} file(s)"

    # ------------------------------------------------------------------ recommendation excerpts
    def op_excerpts(self, task, b: Board, tb) -> str:
        from app.assistant import engine as E
        kept, files, terms = b.discovery_meta["find_kept"], b.discovery_meta["find_files"], b.discovery_meta["find_terms"]
        cfg = b.settings["assistant"]
        refs = [E._build_ref(i + 1, c, files[c["file_id"]], terms, cfg) for i, c in enumerate(kept)]
        with b.lock:
            b.findings["find_refs"] = refs
        return f"Pulled verbatim excerpts for {len(refs)} file(s)"
