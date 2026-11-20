"""File Discovery Agent -- decides WHICH files matter (not what they say).

Strategy per task (chosen from the request, not fixed):
  locate_named   a file name was given            -> search_files (exact / phrase match)
  find_by_topic  a topic or concept               -> search_contents (what the files actually say), file-level ranking
  find_pair      comparison                       -> one best file per side (name first, then content)
  find_tabular   numbers / spreadsheet question   -> match_tables (column names vs question)
  find_code      code or project question         -> search_files(code, config, docs) + search_contents + find_related
  recommend      "which files ..." requests       -> the LangGraph search agent, then relevance filtering and de-duplication
Returns candidates with a reason each; filters irrelevant / duplicate / unreadable files.
"""
import re
from typing import List

from app.agents.core import AgentFailure, Board, Task
from app.analysis import text as T
from app.assistant import qa
from app.assistant.intent import _SPLIT_COMPARE, clean_query
from app.assistant.understand import _make_need, detect_file_mentions


class DiscoveryAgent:
    name = "discovery"

    def run(self, task: Task, b: Board, tb) -> str:
        op = getattr(self, "op_" + task.op, None)
        if op is None:
            raise AgentFailure(f"unknown discovery operation {task.op}", retryable=False)
        return op(task, b, tb)

    # ------------------------------------------------------------------ helpers
    def _set(self, b: Board, cands: List[dict]) -> None:
        with b.lock:
            seen, out = set(), []
            for c in cands:
                if c["file_id"] not in seen:
                    seen.add(c["file_id"])
                    out.append(c)
            b.candidates = out
            b.target_ids = [c["file_id"] for c in out]

    # ------------------------------------------------------------------ operations
    def op_locate_named(self, task, b, tb) -> str:
        names = task.args.get("names") or b.plan_q.target_names
        found = []
        for n in names:
            r = tb.call("discovery", "search_files", query=n, limit=4, summarize=lambda d: f"{len(d['matches'])} match(es)")
            if r.ok:
                found += [m for m in r.data["matches"] if m["score"] >= 1.5][:2]
        if not found and b.plan_q.target_file_ids:      # the planner already resolved ids from the question
            md = tb.call("discovery", "get_metadata", file_ids=b.plan_q.target_file_ids, summarize=lambda d: f"{len(d['files'])} file(s)")
            found = [{"file_id": f["file_id"], "filename": f["filename"], "why": "named in the question", "score": 3.0} for f in (md.data["files"] if md.ok else [])]
        if not found:
            raise AgentFailure("no file with that name is indexed", retryable=False)
        self._set(b, [{"file_id": m["file_id"], "filename": m["filename"], "why": m["why"], "score": m["score"], "via": ["search_files"]} for m in found])
        return "Located " + ", ".join(c["filename"] for c in b.candidates)

    def op_reuse_context(self, task, b, tb) -> str:
        md = tb.call("discovery", "get_metadata", file_ids=b.plan_q.target_file_ids, summarize=lambda d: f"{len(d['files'])} file(s) from the previous answer")
        files = md.data["files"] if md.ok else []
        self._set(b, [{"file_id": f["file_id"], "filename": f["filename"], "why": "the file being discussed", "score": 3.0, "via": ["conversation"]} for f in files])
        return "Continuing with " + ", ".join(c["filename"] for c in b.candidates)

    def op_find_by_topic(self, task, b, tb) -> str:
        q = b.plan_q
        need = q.needs[0] if q.needs else _make_need(q.question, set(), [])
        r = tb.call("discovery", "search_contents", need=need, need_no=0, top_n=12, per_file_cap=3, pool=40,
                    summarize=lambda d: f"{len(d['evidence'])} passage(s) in {len(d['file_scores'])} file(s) of {d['searched_files']} searched")
        cands = []
        if r.ok:
            info = {}
            from app.agents.tools import info_for
            info = info_for(b, [f for f, _ in r.data["file_scores"]])
            for fid, sc in r.data["file_scores"]:
                if fid in info:
                    cands.append({"file_id": fid, "filename": info[fid]["filename"], "why": f"its content mentions {', '.join(need.terms[:3])}", "score": round(sc, 3), "via": ["search_contents"]})
            b.discovery_meta.update(searched_files=r.data["searched_files"], searched_passages=r.data["searched_passages"])
        nm = tb.call("discovery", "search_files", query=q.question, limit=5, summarize=lambda d: f"{len(d['matches'])} name hint(s)")
        if nm.ok:                                        # a file NAME is only a lead: it can add a candidate but never counts as evidence
            for m in nm.data["matches"]:
                if m["score"] >= 0.5 and all(c["file_id"] != m["file_id"] for c in cands):
                    cands.append({"file_id": m["file_id"], "filename": m["filename"], "why": m["why"] + " (name only — content must confirm)", "score": 0.1, "via": ["search_files"]})
        top = cands[: task.args.get("limit", 8)]
        if not top:
            b.discovery_meta["empty"] = True
        self._set(b, top)
        return f"{len(top)} candidate file(s)" if top else "No file looks relevant by content"

    def op_find_pair(self, task, b, tb) -> str:
        q = b.plan_q
        targets = list(q.target_file_ids)
        if len(targets) < 2:
            parts = [p.strip() for p in _SPLIT_COMPARE.split(clean_query(q.question)) if p.strip()]
            for part in parts[:3]:
                m = detect_file_mentions(part, b.owner_id)
                if m and m[0]["file_id"] not in targets:
                    targets.append(m[0]["file_id"])
                    continue
                sub = _make_need(part, set(), [])
                if sub.stems:
                    r = tb.call("discovery", "search_contents", need=sub, need_no=0, top_n=3, per_file_cap=1, summarize=lambda d: f"{len(d['evidence'])} passage(s)")
                    if r.ok:
                        for e in r.data["evidence"]:
                            if e.file_id not in targets and e.coverage >= 0.5:
                                targets.append(e.file_id)
                                break
        targets = targets[:3]
        if len(targets) < 2:
            b.discovery_meta["needs_clarification"] = "two documents to compare"
            from app.agents.tools import info_for
            have = info_for(b, targets)
            self._set(b, [{"file_id": f, "filename": have[f]["filename"], "why": "found so far", "score": 1.0, "via": ["search_files"]} for f in targets if f in have])
            return "Only found %d document(s)" % len(targets)
        from app.agents.tools import info_for
        info = info_for(b, targets)
        self._set(b, [{"file_id": f, "filename": info[f]["filename"], "why": "one of the documents to compare", "score": 1.0, "via": ["search_files", "search_contents"]} for f in targets if f in info])
        return "Found " + " and ".join(c["filename"] for c in b.candidates)

    def op_find_tabular(self, task, b, tb) -> str:
        q = b.plan_q
        if q.target_file_ids:
            r = tb.call("discovery", "get_metadata", file_ids=q.target_file_ids, summarize=lambda d: f"{len(d['files'])} file(s)")
            files = [f for f in (r.data["files"] if r.ok else []) if f["file_type"] in qa.TABULAR]
            if files:
                self._set(b, [{"file_id": f["file_id"], "filename": f["filename"], "why": "the spreadsheet you named", "score": 3.0, "via": ["get_metadata"]} for f in files])
                return "Using " + ", ".join(f["filename"] for f in files)
        r = tb.call("discovery", "match_tables", question=q.question, summarize=lambda d: f"{len(d['tables'])} of {d['all']} spreadsheet(s) fit the question")
        tabs = r.data["tables"] if r.ok else []
        if not tabs:
            b.discovery_meta["no_tables"] = True
            raise AgentFailure("no spreadsheet matches the question", retryable=False)
        # guard: "about a table" only when the table's columns explain most of the question (otherwise it is a text question)
        from app.assistant.qa import dataqa
        calc_words = {"total", "sum", "average", "avg", "mean", "median", "minimum", "maximum", "highest", "lowest", "count", "percentage", "percent", "growth", "rows", "row",
                      "records", "number", "much", "many", "calculate", "compute", "overall", "trend", "trends", "data", "each", "per", "top", "analyze", "analyse", "identify", "csv", "xlsx", "excel",
                      "spreadsheet", "spreadsheets", "sheet", "insight", "insights", "pattern", "patterns", "show", "list", "summary", "growth", "change"}
        key = [t for t in q.all_terms if t not in calc_words]
        top = tabs[0]
        names = set()
        for h in list(top.get("columns") or []) + [x for sh in (top.get("sheets") or []) for x in (sh.get("columns") or []) + [sh.get("name", "")]]:
            names |= set(dataqa._norm(h))
        names |= set(dataqa._norm(top["filename"]))
        for h in list(names):
            names.update(dataqa._SYN_STEMMED.get(h, []))
        has_date = bool(names & {"date", "day", "week", "month", "year", "period", "time", "quarter"})
        months = set(dataqa.MONTHS) | set(dataqa.MONTH_ABBR)

        def matched(t):
            return T._stem(t) in names or (has_date and (t in months or re.fullmatch(r"(19|20)\d{2}|q[1-4]|quarter|month|year|week|monthly|yearly", t) is not None))
        fit = (sum(1 for t in key if matched(t)) / len(key)) if key else 1.0
        b.discovery_meta["table_fit"] = round(fit, 2)
        if fit < 0.6:
            b.discovery_meta["no_tables"] = True
            raise AgentFailure(f"only {round(fit * 100)}% of the question matches spreadsheet columns, so it is not a spreadsheet question", retryable=False)
        self._set(b, [{"file_id": t["file_id"], "filename": t["filename"], "why": f"its columns match the question ({round(fit * 100)}%)", "score": round(fit, 2), "via": ["match_tables"]} for t in tabs[:3]])
        return "Spreadsheet: " + ", ".join(c["filename"] for c in b.candidates[:2])

    def op_find_code(self, task, b, tb) -> str:
        q = b.plan_q
        cands = []
        if q.target_file_ids:
            r = tb.call("discovery", "get_metadata", file_ids=q.target_file_ids, summarize=lambda d: f"{len(d['files'])} named file(s)")
            for f in (r.data["files"] if r.ok else []):
                cands.append({"file_id": f["file_id"], "filename": f["filename"], "why": "named in the question", "score": 3.0, "via": ["get_metadata"]})
        if task.args.get("project"):
            cands += self._project_files(b, tb, q)
        need = q.needs[0] if q.needs else _make_need(q.question, set(), [])
        if need.stems and not task.args.get("project_only"):
            r = tb.call("discovery", "search_contents", need=need, need_no=0, top_n=10, per_file_cap=2, pool=40,
                        summarize=lambda d: f"{len(d['evidence'])} passage(s) in {len(d['file_scores'])} file(s)")
            if r.ok:
                from app.agents.tools import info_for
                info = info_for(b, [f for f, _ in r.data["file_scores"]])
                for fid, sc in r.data["file_scores"]:
                    f = info.get(fid)
                    if f and f["file_type"] in set(qa.CODE_TYPES) | {"md", "txt", "yaml", "yml", "json", "toml", "ini", "cfg"}:
                        cands.append({"file_id": fid, "filename": f["filename"], "why": f"mentions {', '.join(need.terms[:3])}", "score": round(sc, 3), "via": ["search_contents"]})
        # related files for the best code hit: its imports / importers / siblings
        code = [c for c in cands if c["filename"].rsplit(".", 1)[-1] in qa.CODE_TYPES]
        if code and len(cands) < 30:
            rel = tb.call("discovery", "find_related", file_id=code[0]["file_id"], limit=5, summarize=lambda d: f"{len(d['related'])} related file(s)")
            for x in (rel.data["related"] if rel.ok else []):
                cands.append({"file_id": x["file_id"], "filename": x["filename"], "why": x["why"], "score": x["score"], "via": ["find_related"]})
        if not cands:
            b.discovery_meta["empty"] = True
        cands = self._rank_code(cands)
        self._set(b, cands[: task.args.get("limit", 40)])
        return f"{len(b.candidates)} code/config/doc file(s)" if b.candidates else "No code files found"

    @staticmethod
    def _rank_code(cands):
        return sorted(cands, key=lambda c: -c["score"])

    def _project_files(self, b, tb, q) -> List[dict]:
        r = tb.call("discovery", "list_project", question=q.question, summarize=lambda d: (f"project “{d['project']}”: {len(d['files'])} relevant file(s)" if d["files"] else "no project found"))
        if not r.ok or not r.data["files"]:
            return []
        b.discovery_meta["project"] = r.data["project"]
        return r.data["files"]

    def op_recommend(self, task, b, tb) -> str:
        from app.assistant import engine as E
        q = b.plan_q
        plan = q.find_plan
        terms: list = []
        r = tb.call("discovery", "agent_search", query=plan["retrieval_query"], plan_find=plan, terms_out=terms,
                    summarize=lambda d: f"{len(d['cands'])} ranked candidate(s) from the search agent")
        if not r.ok:
            raise AgentFailure("search agent failed: " + r.error[:80])
        cands = r.data["cands"]
        b.discovery_meta["goal_terms"] = r.data["goal_terms"]
        if plan["type_group"] or plan["intent"] == "project_files":
            from app.tools import build_registry
            from app.scope import resolve_scope
            reg, scope = build_registry(), resolve_scope(b.owner_id)
            extra = reg.execute("hybrid_search", {"query": plan["retrieval_query"], "top_k": 60}, scope)
            top = max([c["score"] for c in extra.candidates] or [1.0])
            added = 0
            for i, c in enumerate(extra.candidates):
                if c["file_id"] not in cands:
                    md = c["metadata"] or {}
                    cands[c["file_id"]] = {"file_id": c["file_id"], "score": 0.6 * c["score"] / top, "rank": 100 + i, "sources": ["hybrid_search"], "sem": md.get("semantic_score"),
                                           "snippet": md.get("semantic_matched_chunk") or md.get("bm25_snippet")}
                    added += 1
            b.bus.step("tool", "Used hybrid_search (wider pass)", f"{len(extra.candidates)} candidate(s), {added} new", tool="hybrid_search")
        terms_all = T.query_terms(plan["retrieval_query"], r.data["goal_terms"])
        if plan["intent"] == "resume_improve":
            terms_all = [t for t in terms_all if t not in ("cv", "curriculum", "vitae")] or ["resume"]
        files = E._load_files(b.owner_id, list(cands))
        kept = E._dedupe(E._rank_and_filter(b.bus, b.owner_id, plan, cands, terms_all, files), files)[: plan["max_results"]]
        b.discovery_meta.update(find_terms=terms_all, find_files=files, find_kept=kept)
        self._set(b, [{"file_id": c["file_id"], "filename": files[c["file_id"]]["filename"], "why": E._reason(c, files[c["file_id"]], terms_all), "score": round(c["score"], 3), "via": c.get("sources", [])} for c in kept])
        return f"{len(b.candidates)} relevant file(s) after filtering" if b.candidates else "No relevant files"
