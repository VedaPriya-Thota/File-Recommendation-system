"""Analysis Agent -- turns retrieved material into FINDINGS (structured claims that remember their source span).

  answer_facts   best answering sentence(s) per information need, with their exact spans
  summarize      representative content spread over a document (opening + one key sentence per section)
  explain_html   what a web page is and does, from its parsed structure (title, headings, forms, scripts, events)
  compare        similarities / differences of two documents, plus the aspect-specific statements of each
  calculate      real calculation over spreadsheet rows (sum / average / grouping / change / ranking ...)
  explain_code   which definitions implement the asked-for behaviour (signatures, docstrings, calls)
  trace          control flow across files from the functions that match the topic; config values; dependencies
  explain_project what a project is for and how it is organised (README + code structure)
  review_resume  checklist review with quoted evidence
No language model is used here: findings must be reproducible from the files.
"""
import re
from typing import Dict, List

from app.agents.core import AgentFailure, Board, Task
from app.agents.tools import info_for, text_of
from app.analysis import codeqa, htmlqa
from app.analysis import text as T
from app.assistant import qa
from app.assistant import retrieval as R
from app.assistant.understand import _make_need, synonym_stems

GENERIC_PROJECT = set("""project projects code codebase repo repository app application system software program tool analyze analyse explain describe review walk
through break down trace machine learning folder files file my our the this model-based works work does do""".split())


class AnalysisAgent:
    name = "analysis"

    def run(self, task: Task, b: Board, tb) -> str:
        op = getattr(self, "op_" + task.op, None)
        if op is None:
            raise AgentFailure(f"unknown analysis operation {task.op}", retryable=False)
        return op(task, b, tb)

    # ------------------------------------------------------------------ facts
    def op_answer_facts(self, task, b: Board, tb) -> str:
        q = b.plan_q
        results = []
        for i, need in enumerate(q.needs):
            evs = b.evidence.get(i, [])
            if not need.stems:
                results.append({"need": need, "ok": False, "assess": {"ok": False, "best": None, "reason": "no_terms", "missing": []}, "ranked": [], "evs": [], "ret": b.retrieval_meta.get(i, {}).get("ret")})
                continue
            r = tb.call("analysis", "rank_sentences", need_no=i, summarize=lambda d: f"{len(d['ranked'])} candidate sentence(s); evidence {'covers' if d['assess']['ok'] else 'does NOT cover'} the question")
            if not r.ok:
                raise AgentFailure("sentence ranking failed")
            results.append({"need": need, "ok": r.data["assess"]["ok"], "assess": r.data["assess"], "ranked": r.data["ranked"], "evs": evs, "ret": b.retrieval_meta.get(i, {}).get("ret")})
        # source code has no sentences: judge a passage by how well its names, comments and body cover the question
        all_evs = [e for r in results for e in r["evs"]]
        inf = info_for(b, [e.file_id for e in all_evs])
        code_mode = q.kind == "code_explain" or (all_evs and sum(1 for e in all_evs[:4] if inf.get(e.file_id, {}).get("file_type") in qa.CODE_TYPES) >= max(2, len(all_evs[:4]) - 1))
        b.findings["code_mode"] = bool(code_mode)
        if code_mode:
            for r in results:
                if not r["evs"]:
                    continue
                n_t = len(r["need"].stems)
                thr = 0.9 if n_t <= 3 else 0.75 if n_t == 4 else 0.6
                best = max(r["evs"], key=lambda e: e.coverage)
                is_code = inf.get(best.file_id, {}).get("file_type") in qa.CODE_TYPES
                if r["ok"] and is_code and best.coverage < thr:
                    r["ok"] = False
                if not r["ok"] and best.coverage >= thr and is_code:
                    r["ok"] = True
                    r["assess"] = {"ok": True, "best": {"ev": best, "cov": best.coverage, "score": best.coverage, "sentence": "", "start": best.start, "end": best.end}, "reason": "code", "missing": []}
        with b.lock:
            b.results = results
        ok = sum(1 for r in results if r["ok"])
        return f"{ok} of {len(results)} information need(s) answered from the retrieved passages"

    # ------------------------------------------------------------------ code
    def op_explain_code(self, task, b: Board, tb) -> str:
        """Definitions that implement the asked-for behaviour (reads code structure; never executes anything)."""
        answered = [r for r in b.results if r["ok"]]
        need_stems = list(dict.fromkeys(s for r in answered for s in r["need"].stems))
        inf = info_for(b, [e.file_id for r in answered for e in r["evs"]])
        named = {t.lower() for r in answered for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", r["need"].text) if "_" in t or any(c.isupper() for c in t[1:])}
        cands = []
        for r in answered:
            for ev in r["evs"]:
                f = inf.get(ev.file_id)
                if not f or f["file_type"] not in qa.CODE_TYPES:
                    continue
                full = text_of(b, ev.file_id)
                o = b.structures.get(ev.file_id, {}).get("outline") or codeqa.outline(full, f["file_type"])
                lines = full.split("\n")
                for it in o["items"]:
                    if it["line_end"] < ev.line_start or it["line_start"] > ev.line_end:
                        continue
                    body = "\n".join(lines[it["line_start"] - 1: it["line_end"]])
                    cov = R.coverage(need_stems, R._stem_set(it["name"].replace("_", " ") + " " + it["doc"] + " " + body[:1500]))
                    name_cov = R.coverage(need_stems, R._stem_set(it["name"].replace("_", " ") + " " + it["doc"]))
                    boost = 2.0 if it["name"].lower() in named else 0.0                 # the question names this exact function/class
                    cands.append({"score": cov + 0.5 * name_cov + boost, "ev": ev, "file": f, "item": it, "body": body, "imports": o["imports"]})
        if named:
            for fid in {c["file"]["file_id"] for c in cands}:
                pass
        cands.sort(key=lambda x: -x["score"])
        seen, out = set(), []
        for c in cands:
            key = (c["file"]["file_id"], c["item"]["name"])
            if key in seen:
                continue
            seen.add(key)
            c["desc"] = codeqa.describe_item(c["item"])
            out.append(c)
            if len(out) >= 4:
                break
        if not out:
            raise AgentFailure("no definition matches the question", retryable=False)
        with b.lock:
            b.findings["code_items"] = out
        return f"{len(out)} matching definition(s): " + ", ".join(f"`{c['item']['name']}`" for c in out[:3])

    # ------------------------------------------------------------------ summaries
    def op_summarize(self, task, b: Board, tb) -> str:
        q = b.plan_q
        focus = None
        ft = [t for t in q.all_terms if t not in ("summary", "summarize", "overview", "points", "point", "findings", "finding", "takeaways", "gist", "highlights", "result", "results")]
        if q.target_names:
            drop = {w for n in q.target_names for w in re.split(r"[^a-z0-9]+", n.lower())}
            ft = [t for t in ft if t not in drop]
        if ft:
            focus = _make_need(" ".join(ft), set(), [])
        done, tool_errors = {}, 0
        for fid in task.args["file_ids"]:
            r = tb.call("analysis", "summarize_document", file_id=fid, focus=focus,
                        summarize=lambda d: f"{len(d['summary']['points'])} key point(s) + opening")
            if not r.ok:
                tool_errors += 1
            elif r.data["summary"]["opening"] or r.data["summary"]["points"]:
                done[fid] = r.data["summary"]
        if not done:
            raise AgentFailure("the summarising tool failed" if tool_errors else "nothing to summarise: the document has no readable text", retryable=bool(tool_errors))
        with b.lock:
            b.findings["summaries"] = done
            b.findings["summary_focus"] = focus
        return f"Summarised {len(done)} document(s)"

    def op_explain_html(self, task, b: Board, tb) -> str:
        out = {}
        for fid in task.args["file_ids"]:
            st = b.structures.get(fid, {}).get("html")
            if st is None:
                continue
            out[fid] = {"facts": htmlqa.describe(st), "structure": st}
        if not out:
            raise AgentFailure("no HTML structure was retrieved")
        with b.lock:
            b.findings["html"] = out
        n = sum(len(v["facts"]) for v in out.values())
        return f"Identified {n} facts about the page's content and behaviour"

    # ------------------------------------------------------------------ comparison
    def op_compare(self, task, b: Board, tb) -> str:
        fids = task.args["file_ids"][:2]
        r = tb.call("analysis", "compare_documents", file_ids=fids, summarize=lambda d: f"similarity {d['comparison']['similarity']:.0%}")
        if not r.ok:
            raise AgentFailure("comparison failed")
        aspect = task.args.get("aspect")
        per_doc = {}
        for fid in fids:
            evs = b.passages_read.get(fid, [])
            if aspect is not None and aspect.stems:
                ranked = R.best_sentences(aspect, evs, 3)
                ranked = [c for c in ranked if c["cov"] >= R.need_threshold(len(aspect.stems)) * 0.75]
            else:
                inf = info_for(b, [fid])[fid]
                s = qa.summarize_text(text_of(b, fid), inf["meta"], inf["file_type"], 3)
                ranked = [{"sentence": x["sentence"], "start": x["start"], "end": x["end"], "ev": qa._ev_from(x["p"], fid, text_of(b, fid)), "cov": 1, "type_ok": True} for x in s["opening"][:1] + s["points"][:3]]
            per_doc[fid] = ranked
        with b.lock:
            b.findings["comparison"] = {"stats": r.data["comparison"], "per_doc": per_doc, "aspect": aspect, "files": fids}
        return f"Compared {len(fids)} documents; {sum(len(v) for v in per_doc.values())} aspect statement(s)"

    # ------------------------------------------------------------------ spreadsheets
    def op_calculate(self, task, b: Board, tb) -> str:
        last, tool_errors = "", 0
        for fid in task.args["file_ids"]:
            r = tb.call("analysis", "analyze_table", file_id=fid, question=b.question, summarize=lambda d: (d.get("sentence") or d.get("reason", ""))[:100])
            if not r.ok:
                last, tool_errors = r.error, tool_errors + 1
                continue
            if r.data.get("ok"):
                with b.lock:
                    b.findings["calc"] = {**r.data, "file_id": fid}
                return f"Calculated: {r.data['sentence'][:110]}"
            last = r.data.get("reason", "")
        b.findings["calc_fail"] = last
        raise AgentFailure("could not turn the question into a calculation: " + last[:120], retryable=bool(tool_errors))

    # ------------------------------------------------------------------ projects / workflows
    def op_trace(self, task, b: Board, tb) -> str:
        q = b.plan_q
        files = [c for c in b.candidates if c["filename"].rsplit(".", 1)[-1] in qa.CODE_TYPES]
        focus = [t for t in q.all_terms if t not in GENERIC_PROJECT]
        stems = list(dict.fromkeys(T._stem(t) for t in focus))
        kws = list(dict.fromkeys(x for s in stems for x in synonym_stems(s)))
        r = tb.call("analysis", "trace_workflow", file_ids=[c["file_id"] for c in files][:40], keywords=kws,
                    summarize=lambda d: f"{len(d['chain'])} step(s) from {', '.join(d['entry']) or 'no entry'}" + (f" — {d['notes'][0]}" if d["notes"] else ""))
        if not r.ok:
            raise AgentFailure("workflow tracing failed")
        cfg = {}
        for fid, st in b.structures.items():
            for k, v in (st.get("config") or {}).items():
                if re.search(r"learning|lr|epoch|batch|optim|loss|seed|layer|hidden|dropout|model|split|metric|dataset|path|n_", k, re.I):
                    cfg[f"{info_for(b, [fid])[fid]['filename']}: {k}"] = v
        if not r.data["chain"]:
            raise AgentFailure("no code matches the asked-for behaviour", retryable=False)
        with b.lock:
            b.findings["trace"] = {**r.data, "focus": focus, "config": dict(list(cfg.items())[:12])}
        return f"Traced {len(r.data['chain'])} step(s) across {len({c['file_id'] for c in r.data['chain']})} file(s)"

    def op_explain_project(self, task, b: Board, tb) -> str:
        inf = info_for(b, [c["file_id"] for c in b.candidates])
        readme = next(((fid, st["doc"]) for fid, st in b.structures.items() if "doc" in st and re.match(r"readme", inf.get(fid, {}).get("filename", ""), re.I)), None)
        mods = []
        for fid, st in b.structures.items():
            o = st.get("outline")
            f = inf.get(fid)
            if o and f and (o["doc"] or o["entry_point"] or o["items"]):
                mods.append({"file_id": fid, "rel_path": f["file_path"], "filename": f["filename"], "doc": o["doc"], "entry": o["entry_point"], "imports": o["imports"],
                             "names": [it["name"] for it in o["items"] if it["kind"] in ("class", "function")][:4], "lang": o["language"] or f["file_type"]})
        mods.sort(key=lambda m: (-(1 if m["entry"] else 0), -(1 if m["doc"] else 0), -len(m["names"])))
        manifests = [(fid, f["filename"]) for fid, f in inf.items() if f["filename"].lower() in ("package.json", "pyproject.toml", "requirements.txt", "setup.py", "cargo.toml", "go.mod")]
        purpose = codeqa.readme_blurb(readme[1]) if readme else ""
        with b.lock:
            b.findings["project"] = {"name": b.discovery_meta.get("project", "the project"), "readme": readme, "purpose": purpose, "modules": mods, "manifests": manifests,
                                     "n_files": len(b.candidates), "langs": sorted({m["lang"] for m in mods})}
        if not mods and not readme:
            raise AgentFailure("no README or source structure was retrieved", retryable=False)
        return f"Project overview: {len(mods)} module(s), README {'found' if readme else 'missing'}"

    # ------------------------------------------------------------------ résumé / recommendations
    def op_review_resume(self, task, b: Board, tb) -> str:
        fid = task.args["file_id"]
        r = tb.call("analysis", "review_resume", file_id=fid, summarize=lambda d: f"score {d['review']['score']}/100, {len(d['review']['suggestions'])} improvement area(s)")
        if not r.ok:
            raise AgentFailure("résumé review failed")
        with b.lock:
            b.findings["resume"] = r.data["review"]
        return f"Résumé score {r.data['review']['score']}/100"
