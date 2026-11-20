"""Response Agent -- turns VERIFIED findings into a clear answer with sources.

  respond   writes a draft answer: an extractive, plain-English version built straight from the verified findings, and (if a model is
            configured, permitted and the task benefits) one rewrite by the language model. The draft goes back to the Verification
            Agent; if a model rewrite is rejected the extractive version is used.
  finalize  picks the accepted text, attaches numbered sources and excerpts, reports limitations (unreadable files, partial indexing)
No searching or analysis happens here, and nothing is stated that is not in the findings.
"""
import re
from typing import Dict, List, Optional

from app.agents.core import AgentFailure, Board, Task
from app.agents.tools import info_for, text_of
from app.analysis import safety
from app.analysis import text as T
from app.assistant import qa
from app.assistant import retrieval as R
from app.assistant.qa import SourceBook, _clip, _lead, _strip_bullet, short_loc, title_of
from app.llm.provider import LLMUnavailableError, get_default_provider, llm_status


class ResponseAgent:
    name = "response"

    def run(self, task: Task, b: Board, tb) -> str:
        op = getattr(self, "op_" + task.op, None)
        if op is None:
            raise AgentFailure(f"unknown response operation {task.op}", retryable=False)
        return op(task, b, tb)

    # ------------------------------------------------------------------ draft
    def op_respond(self, task, b: Board, tb) -> str:
        q = b.plan_q
        info = info_for(b, self._all_file_ids(b))
        book = SourceBook(b.owner_id, info)
        d = {"llm": False, "min_support": 0.55, "extra": "", "status": "ok", "kind": q.kind, "rel": {}, "reasons": {}, "blocks": [], "mode": "qa"}
        kind = q.kind
        if kind in ("fact", "multi_doc", "code_explain"):
            self._fact(b, book, info, d)
        elif kind == "summary":
            self._summary(b, book, info, d)
        elif kind == "compare":
            self._compare(b, book, info, d)
        elif kind == "data_calc":
            self._calc(b, book, info, d)
        elif kind == "project_explain":
            self._project(b, book, info, d)
        elif kind in ("file_search", "resume_improve"):
            self._find(b, book, info, d)
        else:
            raise AgentFailure("unsupported task kind")
        d["book"] = book
        d["fallback"] = d["text"]
        d["sources"] = {n: list(book.texts[n]) for n in book.texts}
        # one optional model rewrite (never for unanswerable questions or computed numbers: nothing to phrase / nothing it may change)
        if d["status"] in ("ok", "partial") and d["blocks"] and kind not in ("data_calc",) and d["mode"] == "qa":
            text, meta = self._llm(b, d, q)
            if text:
                d["text"], d["llm"], d["llm_meta"] = text, True, meta
        if kind == "resume_improve" and d["mode"] == "find" and d["status"] == "ok":
            text, meta = self._llm_find(b, d, q)
            if text:
                d["text"], d["llm"], d["llm_meta"] = text, True, meta
        b.out["draft"] = d
        return "Wrote the answer with a language model" if d["llm"] else "Wrote the answer from the verified findings (no model needed)" if d["status"] in ("ok", "partial") else "Explained what could not be established"

    @staticmethod
    def _all_file_ids(b: Board) -> List[int]:
        ids = set(b.target_ids) | set(b.structures) | set(b.passages_read)
        for r in b.results:
            ids |= {e.file_id for e in r["evs"]}
        for s in b.findings.get("summaries", {}):
            ids.add(s)
        calc = b.findings.get("calc")
        if calc:
            ids.add(calc["file_id"])
        for v in (b.findings.get("find_refs") or []):
            ids.add(v["file_id"])
        return [i for i in ids if isinstance(i, int)]

    # ------------------------------------------------------------------ model rewrite (single call)
    def _llm(self, b: Board, d: dict, q):
        st = llm_status()
        bus = b.bus
        if not st["active"]:
            bus.step("llm", "Wrote the answer without a language model", st["reason"], status="skipped")
            return None, {"used": False, "reason": st["reason"]}
        book: SourceBook = d["book"]
        used, parts = 0, []
        for n, header, body in d["blocks"]:
            body, _ = safety.sanitize_for_llm(body)
            chunk = f"[{n}] {header}\n<document ref=\"{n}\">\n{body}\n</document>"
            if used + len(chunk) > d.get("budget", 7000):
                break
            used += len(chunk)
            parts.append(chunk)
        user = (f"Question: {q.question}\n" + (f"Context from earlier in this conversation: {b.conv_hint}\n" if b.conv_hint else "") +
                (f"\nVerified findings (exact, produced by analysis tools):\n{d['extra']}\n" if d["extra"] else "") + "\nSources:\n" + "\n\n".join(parts))
        b.metrics.inc("llm_calls")
        bus.step("llm", f"Writing with {st['provider']} ({st['model'] or 'default'})", f"{len(parts)} source(s), ~{used:,} characters sent", status="running", tool="llm")
        try:
            text = get_default_provider().complete(qa.QA_SYSTEM, user)
        except LLMUnavailableError as exc:
            bus.step("llm", "Model call failed; using the extractive answer", str(exc)[:140], status="error")
            return None, {"used": False, "reason": "model call failed"}
        d["min_support"] = 0.55
        return text, {"used": True, "provider": st["provider"], "model": st["model"], "external": st["external"]}

    def _llm_find(self, b: Board, d: dict, q):
        from app.assistant import engine as E
        st = llm_status()
        if not st["active"]:
            return None, {"used": False}
        refs = d["refs"]
        b.metrics.inc("llm_calls")
        run = b.bus
        plan = d["find_plan"]
        text, meta = E._llm_answer(run, plan, q.question, refs, d["analyses"], b.settings["assistant"])
        return text, meta

    # ------------------------------------------------------------------ fact / code
    def _fact(self, b: Board, book: SourceBook, info, d):
        q = b.plan_q
        results = b.results
        answered = [r for r in results if r["ok"]]
        if not answered:
            out = {}
            qa._insufficient(b.bus, b.owner_id, q, results, info, book, out, b.health)
            d.update(text=out["answer"], status="insufficient", refs=out["references"], rel={}, reasons={}, preset_refs=out["references"], verification=out["verification"])
            return
        code_items = b.findings.get("code_items")
        if b.findings.get("code_mode") and code_items:
            return self._code_items(b, book, info, d, code_items)
        text, rel, reasons = qa._compose_fact(q, results, book, info, b.conflicts)
        d.update(text=text, rel=rel, reasons=reasons, status="ok" if len(answered) == len(results) else "partial")
        d["blocks"] = qa._blocks_for_llm(answered, info, book, 2400 if q.kind != "multi_doc" else 1800)
        d["budget"] = 6500 if q.kind != "multi_doc" else 9000
        if b.conflicts:
            d["blocks"] = []                                # disagreement is reported verbatim, never smoothed over by a model

    def _code_items(self, b: Board, book: SourceBook, info, d, items):
        rel, reasons, lines_out, facts = {}, {}, [], []
        for c in items:
            f, it, body, ev = c["file"], c["item"], c["body"], c["ev"]
            start = sum(len(x) + 1 for x in text_of(b, ev.file_id).split("\n")[: it["line_start"] - 1])
            text = body[:700]
            pe = R.Ev(file_id=ev.file_id, idx=ev.idx, start=start, end=start + len(text), text=text, line_start=it["line_start"], line_end=it["line_end"], page_label="Line")
            n = book.add(pe, (pe.start, pe.end, text))
            lines_out.append((c["desc"], f, n, pe))
            facts.append(c["desc"])
            book.texts[n].append(body[:1500] + " " + c["desc"] + " " + " ".join(c["imports"]))
            rel[f["file_id"]], reasons[f["file_id"]] = min(1.0, c["score"] / 1.5 + 0.3), "Contains code that matches the question."
        top_desc, top_f, top_n, top_pe = lines_out[0]
        text = f"This is handled in `{top_f['file_path']}`: {top_desc} {book.mark(top_pe, top_n)}"
        if len(lines_out) > 1:
            text += "\n\nRelated code:\n" + "\n".join(f"- {dd} (in `{f['file_path']}`) {book.mark(pe, n)}" for dd, f, n, pe in lines_out[1:])
        text += "\n\nThis comes from reading the code, not from running it."
        d.update(text=text, rel=rel, reasons=reasons, status="ok", extra="\n".join(facts), blocks=[(n, f["file_path"], pe.text) for dd, f, n, pe in lines_out], budget=8000)

    # ------------------------------------------------------------------ summary / html
    def _summary(self, b: Board, book: SourceBook, info, d):
        if b.findings.get("html"):
            return self._html(b, book, info, d)
        sums = b.findings.get("summaries", {})
        focus = b.findings.get("summary_focus")
        paragraphs, rel, reasons, blocks = [], {}, {}, []
        targets = list(sums)
        for fid, s in sums.items():
            f = info.get(fid)
            if not f:
                continue
            text = text_of(b, fid)
            title = title_of(f["filename"])
            n = book.num(fid)
            evs, head, lines = [], [], []
            for o in s["opening"]:
                ev = qa._ev_from(o["p"], fid, text)
                book.add(ev, (o["start"], o["end"], o["sentence"]))
                head.append(_strip_bullet(o["sentence"]))
                evs.append(ev)
            for p in s["points"]:
                ev = qa._ev_from(p["p"], fid, text)
                book.add(ev, (p["start"], p["end"], p["sentence"]))
                lines.append(f"- {_clip(_strip_bullet(p['sentence']), 260)} {book.mark(ev, n)}")
                evs.append(ev)
            if not evs:
                continue
            intro = (_clip(" ".join(head), 420) + f" {book.mark(qa._ev_from(s['opening'][0]['p'], fid, text), n)}") if head else f"{title}:"
            block = (f"**{title}**\n\n" if len(targets) > 1 else "") + intro + ("\n\nThe main points:\n" + "\n".join(lines) if lines else "")
            if focus and focus.stems:
                block = (f"**{title}**\n\n" if len(targets) > 1 else "") + (f"Here is what it says about {', '.join(focus.terms[:3])}:\n" + "\n".join(lines) if lines else intro)
            paragraphs.append(block)
            rel[fid], reasons[fid] = 0.95, "The document being summarized."
            blocks.append((n, f["filename"], "\n---\n".join(f"({e.location()}) {e.text}" for e in evs[:8])))
            for e in evs:
                book.texts[n].append(e.text)
            if f["meta"].get("truncated"):
                b.warnings.append(f"{f['filename']} is very large; the summary reflects the part that was indexed.")
        if not paragraphs:
            d.update(text="I couldn't read any text in that file, so I can't summarize it.", status="insufficient")
            return
        d.update(text="\n\n".join(paragraphs), rel=rel, reasons=reasons, blocks=blocks, budget=9000)

    def _html(self, b: Board, book: SourceBook, info, d):
        html = b.findings["html"]
        paragraphs, rel, reasons, blocks = [], {}, {}, []
        for fid, h in html.items():
            f = info.get(fid)
            full = text_of(b, fid)
            n = book.num(fid)
            st = h["structure"]
            explicit = [x for x in h["facts"] if x["kind"] == "explicit"]
            inferred = [x for x in h["facts"] if x["kind"] == "inferred"]
            title = st["title"] or title_of(f["filename"])
            lead = f"**{f['filename']}** is a web page" + (f" titled “{st['title']}”" if st["title"] else "") + "."
            if inferred:
                lead += " " + inferred[-1]["text"].replace("Taken together, this looks like", "It looks like") + " (that last part is my interpretation of the markup)"
            body = []
            for x in explicit[:9]:
                if x["text"].startswith("The page is titled"):
                    continue
                body.append("- " + x["text"])
                if x.get("needle") and x["needle"] in full:
                    pos = full.find(x["needle"])
                    a, e = max(0, pos - 80), min(len(full), pos + len(x["needle"]) + 120)
                    ev = R.Ev(file_id=fid, idx=0, start=a, end=e, text=full[a:e], line_start=full[:a].count("\n") + 1, line_end=full[:e].count("\n") + 1, page_label="Line")
                    book.add(ev, (pos, pos + len(x["needle"]), x["needle"]))
                    book.texts[n].append(full[a:e])
            mark = f" [{n}]"
            paragraphs.append(lead + mark + ("\n\nWhat the page contains and does:\n" + "\n".join(b2 + mark for b2 in body) if body else ""))
            rel[fid], reasons[fid] = 0.95, "The page being explained."
            facts = "\n".join(x["text"] for x in h["facts"])
            blocks.append((n, f["filename"], facts + "\n---\n" + st["text_head"]))
            book.texts[n].append(facts + " " + st["text_head"] + " " + full[:2000])
        d.update(text="\n\n".join(paragraphs) + "\n\nThis is based on the page's source markup; I haven't opened it in a browser.", rel=rel, reasons=reasons, blocks=blocks, budget=6000,
                 extra="\n".join(f"{'(inference) ' if x['kind'] == 'inferred' else ''}{x['text']}" for h in html.values() for x in h["facts"]))

    # ------------------------------------------------------------------ compare
    def _compare(self, b: Board, book: SourceBook, info, d):
        comp = b.findings.get("comparison")
        if not comp:
            d.update(text="I couldn't compare the documents.", status="insufficient")
            return
        targets, per_doc, aspect, stats = comp["files"], comp["per_doc"], comp["aspect"], comp["stats"]
        rel, reasons = {}, {}
        for fid in targets:
            n = book.num(fid)
            rel[fid], reasons[fid] = 0.95, "One of the documents being compared."
            for c in per_doc[fid][:3]:
                book.add(c["ev"], (c["start"], c["end"], c["sentence"]))
                book.texts[n].append(c["ev"].text)
        t1, t2 = (title_of(info[targets[0]]["filename"]), title_of(info[targets[1]]["filename"]))
        lines = []
        if aspect is not None and aspect.stems:
            for fid in targets:
                title, n = title_of(info[fid]["filename"]), book.num(fid)
                if per_doc[fid]:
                    c = per_doc[fid][0]
                    lines.append(f"- **{title}** says: {_clip(_strip_bullet(c['sentence']), 280)} {book.mark(c['ev'], n)}")
                else:
                    lines.append(f"- **{title}** doesn't say anything about {', '.join(aspect.terms[:3])}.")
            vals = [qa._values(per_doc[f][0]["sentence"]) if per_doc[f] else set() for f in targets]
            if all(vals) and not (vals[0] & vals[1]):
                head = f"They give different figures for {', '.join(aspect.terms[:3])}."
            elif all(per_doc[f] for f in targets):
                head = f"Here is how the documents treat {', '.join(aspect.terms[:3])}."
            else:
                head = f"Only part of what you asked can be compared, because not every document covers {', '.join(aspect.terms[:3])}."
            d["status"] = "ok" if all(per_doc[f] for f in targets) else "partial"
        else:
            sim = stats["similarity"]
            head = (f"{t1} and {t2} cover " + ("largely the same ground" if sim > 0.6 else "some of the same ground" if sim > 0.3 else "mostly different ground") + f" (about {round(sim * 100)}% overlap in wording).")
            for fid in targets[:2]:
                title, n = title_of(info[fid]["filename"]), book.num(fid)
                pts = per_doc[fid][:2]
                if pts:
                    lines.append(f"- **{title}**: " + " ".join(_clip(_strip_bullet(c['sentence']), 220) + " " + book.mark(c["ev"], n) for c in pts))
            w1 = [w for w in stats["only_in_first"] if w.isalpha() and len(w) >= 4][:6]
            w2 = [w for w in stats["only_in_second"] if w.isalpha() and len(w) >= 4][:6]
            if w1 or w2:
                lines.append(f"- Topics mostly in {t1}: {', '.join(w1) or '—'}.")
                lines.append(f"- Topics mostly in {t2}: {', '.join(w2) or '—'}.")
            top1, top2 = (per_doc[targets[0]] or [None])[0], (per_doc[targets[1]] or [None])[0]
            v1 = v2 = set()
            if top1 and top2:
                a_s, b_s = set(T.stems(top1["sentence"])), set(T.stems(top2["sentence"]))
                v1, v2 = qa._values(top1["sentence"]), qa._values(top2["sentence"])
                if len(a_s & b_s) / max(len(a_s | b_s), 1) >= 0.5 and v1 and v2 and not (v1 & v2):
                    head += f" Where they overlap, the figure differs: {t1} says {', '.join(sorted(v1))}, {t2} says {', '.join(sorted(v2))}."
            if (stats["numbers_only_in_first"] or stats["numbers_only_in_second"]) and not (top1 and top2 and v1 and v2 and not (v1 & v2)):
                lines.append(f"- Figures only in {t1}: {', '.join(stats['numbers_only_in_first'][:6]) or '—'}; only in {t2}: {', '.join(stats['numbers_only_in_second'][:6]) or '—'}.")
        blocks = [(book.num(fid), info[fid]["filename"], "\n---\n".join(f"({c['ev'].location()}) {c['ev'].text}" for c in per_doc[fid][:3])) for fid in targets if per_doc[fid]]
        extra = f"similarity {stats['similarity']}; only in first: {', '.join(stats['only_in_first'][:8])}; only in second: {', '.join(stats['only_in_second'][:8])}"
        d.update(text=head + "\n\n" + "\n".join(lines), rel=rel, reasons=reasons, blocks=blocks, extra=extra, budget=8000)
        d.setdefault("status", "ok")
        b.out["analyses"] = {"comparison": {**{k: stats[k] for k in ("similarity", "shared_terms", "only_in_first", "only_in_second")}, "files": [info[f]["filename"] for f in targets[:2]]}}

    # ------------------------------------------------------------------ spreadsheet
    def _calc(self, b: Board, book: SourceBook, info, d):
        res = b.findings.get("calc")
        if not res:
            d.update(text="I looked at the spreadsheets but couldn't turn the question into a calculation: " + (b.findings.get("calc_fail") or "no matching columns") +
                     " Try naming the column you mean and what to calculate (for example, “total revenue by region”).", status="insufficient")
            return
        fid = res["file_id"]
        f = info[fid]
        n = book.num(fid)
        sample_txt = "\n".join(" | ".join(r) for r in res["source_rows"])
        book.excerpts[n].append({"text": sample_txt, "start": 0, "end": len(sample_txt), "verified": True, "location": f"sheet “{res['sheet']}”, first rows used in the calculation", "short": res["sheet"], "highlight": None})
        book.flag(n, computed=True)
        book.texts[n].append(res["sentence"] + " " + " ".join(" ".join(map(str, r)) for r in res["table"]))
        how = {"sum": "adding up", "avg": "averaging", "median": "taking the median of", "min": "finding the lowest value of", "max": "finding the highest value of",
               "count": "counting rows in", "share": "dividing part by whole for", "change": "comparing start and end of", "std": "computing the spread of"}.get(res["operation"], "calculating")
        if res["operation"] == "count" and not res["column"]:
            method = f"I got this by counting the rows of {f['filename']}" + (f" that match your conditions ({res['rows_used']:,} of {res['total_rows']:,})." if res["rows_used"] != res["total_rows"] else ".")
        else:
            method = f"I got this by {how} “{res['column'] or 'the rows'}” across {res['rows_used']:,} of {res['total_rows']:,} rows in {f['filename']}."
        rc = b.verdict.get("recheck") or {}
        text = f"{res['sentence']} [{n}]\n\n{method}" + (" I then re-computed it separately from the raw cells and got the same result." if rc.get("checked") and rc.get("match") else "")
        if res["notes"]:
            text += "\n\nA data note: " + " ".join(res["notes"])
        b.out["analyses"] = {"calc": {"file": f["filename"], "sheet": res["sheet"], "operation": res["operation"], "table": res["table"], "rows_used": res["rows_used"], "notes": res["notes"], "columns": res["columns"]}}
        d.update(text=text, rel={fid: 1.0}, reasons={fid: f"Calculated from the actual rows ({res['rows_used']:,} used)."}, computed=True)
        b.out["coverage_extra"] = {"rows_used": res["rows_used"]}

    # ------------------------------------------------------------------ project / workflow
    def _project(self, b: Board, book: SourceBook, info, d):
        pr, tr = b.findings.get("project"), b.findings.get("trace")
        if not pr and not tr:
            why = "; ".join(f"{t}" for t in b.findings.get("notes", [])) or ""
            d.update(text="I couldn't establish that from your files: no README or source structure related to your question could be read." + why, status="insufficient")
            return
        rel, reasons, lines, blocks, facts = {}, {}, [], [], []
        if pr:
            if pr["readme"] and pr["purpose"]:
                fid, rtext = pr["readme"]
                pos = rtext.find(pr["purpose"][:40])
                s0 = pos if pos >= 0 else 0
                ev = R.Ev(file_id=fid, idx=0, start=s0, end=s0 + min(len(pr["purpose"]), 400), text=rtext[s0:s0 + min(len(pr["purpose"]), 400)])
                full = text_of(b, fid)
                s1 = full.find(ev.text[:40])
                ev.start, ev.end = (s1, s1 + len(ev.text)) if s1 >= 0 else (0, min(len(full), 400))
                ev.text = full[ev.start:ev.end]
                n = book.add(ev, (ev.start, ev.end, ev.text))
                rel[fid], reasons[fid] = 1.0, "The project's own description."
                lines.append(f"{_clip(pr['purpose'], 520)} [{n}]")
                blocks.append((n, info[fid]["filename"], rtext[:3500]))
                book.texts[n].append(rtext[:6000])
            elif not tr:
                lines.append(f"“{pr['name']}” has no README, so I'm describing it from its code only.")
            if not tr:
                lines.append(f"It contains {pr['n_files']} file(s), mostly {', '.join(pr['langs'][:4]) or 'code'}.")
                mod_lines = []
                for m in pr["modules"][:6]:
                    full = text_of(b, m["file_id"])
                    ev = R.Ev(file_id=m["file_id"], idx=0, start=0, end=min(len(full), 300), text=full[:300], line_start=1, line_end=full[:300].count("\n") + 1, page_label="Line")
                    n = book.add(ev, (0, ev.end, ev.text))
                    what = m["doc"] or ("starts the program" if m["entry"] else "")
                    mod_lines.append(f"- `{m['rel_path']}`" + (f": {what}" if what else "") + (f" (defines {', '.join(f'`{x}`' for x in m['names'])})" if m["names"] else "") + f" [{n}]")
                    rel[m["file_id"]], reasons[m["file_id"]] = 0.7, "Part of the project's code."
                    blocks.append((n, m["rel_path"], f"{what}\nDefines: {', '.join(m['names'])}\nImports: {', '.join(m['imports'][:8])}\n{full[:600]}"))
                    book.texts[n].append(f"{what} {' '.join(m['names'])} {' '.join(m['imports'])} {full[:600]}")
                    facts.append(f"{m['rel_path']}: {what}; defines {', '.join(m['names'])}")
                if mod_lines:
                    lines.append("The main parts are:\n" + "\n".join(mod_lines))
                for fid, name in pr["manifests"][:2]:
                    n = book.num(fid)
                    full = text_of(b, fid)
                    book.add(R.Ev(file_id=fid, idx=0, start=0, end=min(len(full), 300), text=full[:300]), (0, min(len(full), 300), full[:300]))
                    book.texts[n].append(full[:3000])
                    rel[fid], reasons[fid] = 0.5, "Lists the project's dependencies."
        if tr:
            chain = tr["chain"]
            focus = ", ".join(tr["focus"][:3])
            lines.append("Here is how it works, step by step, following the code from its entry point" + (f" (I looked for code related to: {focus})" if focus else "") + ":")
            steps = []
            for k, st in enumerate(chain):
                fid = st["file_id"]
                full = text_of(b, fid)
                ln = full.split("\n")
                start = sum(len(x) + 1 for x in ln[: st["line_start"] - 1])
                text = "\n".join(ln[st["line_start"] - 1: st["line_end"]])[:700]
                ev = R.Ev(file_id=fid, idx=k, start=start, end=start + len(text), text=text, line_start=st["line_start"], line_end=st["line_end"], page_label="Line")
                n = book.add(ev, (ev.start, ev.end, text))
                sig = f"`{st['name']}({', '.join(a for a in st['args'] if a != 'self')})`"
                desc = f"{'   ' * min(st['depth'], 3)}{k + 1}. {sig} in `{st['rel_path']}`"
                if st["doc"]:
                    desc += f" — {codeqa_one(st['doc']).rstrip('.')}"
                if st["calls_here"]:
                    desc += f". It then calls {', '.join(f'`{c}`' for c in st['calls_here'][:4])}"
                if st["library_calls"]:
                    desc += f"; library calls include {', '.join(f'`{c}`' for c in st['library_calls'][:4])}"
                steps.append(desc + f" {book.mark(ev, n)}")
                book.texts[n].append(text + " " + " ".join(st["calls_here"] + st["library_calls"]) + " " + st["doc"])
                rel[fid], reasons[fid] = 0.8, "Part of the traced workflow."
                blocks.append((n, st["rel_path"], text))
                facts.append(f"{st['name']} ({st['rel_path']} lines {st['line_start']}-{st['line_end']}) calls {', '.join(st['calls_here'])}; library: {', '.join(st['library_calls'])}")
            lines.append("\n".join(steps))
            if tr["config"]:
                lines.append("Settings found in the project's config files: " + "; ".join(f"{k} = {v}" for k, v in list(tr["config"].items())[:6]) + ".")
                facts.append("config: " + "; ".join(f"{k}={v}" for k, v in tr["config"].items()))
                for fid, st in b.structures.items():
                    if st.get("config"):
                        book.texts[book.num(fid)].append(" ".join(f"{k} {v}" for k, v in st["config"].items()))
                        rel.setdefault(fid, 0.5)
                        reasons.setdefault(fid, "Holds settings used by the workflow.")
                        book.add(R.Ev(file_id=fid, idx=0, start=0, end=min(len(text_of(b, fid)), 300), text=text_of(b, fid)[:300]), (0, min(len(text_of(b, fid)), 300), text_of(b, fid)[:300]))
            libs = [i for i in tr["imports"] if not i.startswith(".")][:8]
            if libs:
                lines.append("Libraries it relies on: " + ", ".join(libs) + ".")
        lines.append("This is based on reading the documentation and code, not on running the program.")
        d.update(text="\n\n".join(lines), rel=rel, reasons=reasons, blocks=blocks, extra="\n".join(facts), budget=9000)
        b.out["analyses"] = {"project": {"name": (pr or {}).get("name"), "files": (pr or {}).get("n_files", 0), "modules": [m["rel_path"] for m in (pr or {}).get("modules", [])[:6]]}}
        if tr:
            b.out["analyses"]["workflow"] = {"entry": tr["entry"], "steps": [{"name": s["name"], "file": s["rel_path"], "depth": s["depth"]} for s in tr["chain"]]}

    # ------------------------------------------------------------------ recommendations / résumé review
    def _find(self, b: Board, book: SourceBook, info, d):
        from app.assistant import engine as E
        refs = b.findings.get("find_refs", [])
        plan = b.plan_q.find_plan
        analyses = {}
        if b.findings.get("resume"):
            analyses["resume"] = b.findings["resume"]
        warnings = list(b.warnings)
        text = E._compose(plan, refs, analyses, warnings)
        d.update(text=text, mode="find", refs=refs, analyses=analyses, find_plan=plan, status="ok" if refs else "no_results")
        d["blocks"] = []

    # ------------------------------------------------------------------ finalize
    def op_finalize(self, task, b: Board, tb) -> str:
        d = b.out["draft"]
        book: SourceBook = d["book"]
        used_llm = bool(d["llm"] and d.get("accepted", False))
        text = d.get("verified_text") if used_llm else d["fallback"]
        if d["llm"] and not d.get("accepted"):
            b.bus.step("verify", "Using the extractive answer instead of the model's", "", status="warning")
        out = b.out
        out["answer"] = text
        out["mode"] = d["mode"]
        out["status"] = d["status"]
        out["llm"] = d.get("llm_meta", {"used": False}) if used_llm else {"used": False}
        if d["mode"] == "find":
            out["references"] = d["refs"]
            out["analyses"] = d["analyses"]
        elif d.get("preset_refs") is not None:
            out["references"] = d["preset_refs"]
        else:
            out["references"] = book.references(d["rel"], d["reasons"])
        rep = d.get("report") or {}
        out["verification"] = (d.get("verification") or {}) if d["status"] == "insufficient" else {**rep, "removed": rep.get("removed", []) if used_llm else [],
                                                                                                    **({"note": "extractive answer: every statement is taken from the cited passages"} if not used_llm else {})}
        out["conflicts"] = [{"need": c["need"], "claims": [{k: v for k, v in cl.items() if k not in ("ev", "span")} for cl in c["claims"]]} for c in b.conflicts]
        ev_files = [r["filename"] for r in out["references"]]
        notes = qa.limitation_notes(b.health, d["status"] in ("ok", "partial"), ev_files) if b.health else []
        out["warnings"] = list(dict.fromkeys(b.warnings + notes))
        for n in notes:
            b.bus.step("verify", "Index limitation noted", n, status="warning")
        focus = []
        if d["kind"] in ("summary", "compare", "data_calc"):
            focus = [r["file_id"] for r in out["references"]][:3]
        out["_focus"] = focus or [r["file_id"] for r in out["references"] if not r.get("closest_only")][:3]
        if d.get("clarify"):
            out["clarify_options"] = d["clarify"]
        return f"{len(out['references'])} source(s) attached"


def codeqa_one(doc: str, n: int = 160) -> str:
    s = re.sub(r"\s+", " ", doc).strip()
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"
