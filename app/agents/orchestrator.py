"""
Orchestrator Agent + LangGraph control flow.

The Orchestrator owns the task graph. It does not read files or write answers; it
  * identifies the kind of task (from the QuestionPlan produced by understand.plan_question) and picks the agents it needs,
  * plans in STAGES: first discovery; then -- from what discovery actually found -- it creates the retrieval / analysis tasks
    (one retrieval task per information need or per file, run in parallel when independent),
  * routes on results (conditional edges): a failed discovery changes the plan, a verification verdict of "needs more evidence"
    sends the Retrieval Agent back out (bounded to MAX_ROUNDS), an unreliable model draft falls back to the extractive answer,
  * skips agents that are not needed (no analysis for a plain recommendation, no response drafting when it must ask for clarification),
  * hands structured results from agent to agent through the Board and merges the outcome.

LangGraph (already a dependency of this project's search agent) provides the graph: nodes are stages, edges are conditional.
"""
import re
import threading
from typing import Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.analysis_agent import AnalysisAgent
from app.agents.core import MAX_ROUNDS, Board, EventBus, Plan, Task, ToolBox, agent_label, run_plan
from app.agents.discovery import DiscoveryAgent
from app.agents.response_agent import ResponseAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.tools import REGISTRY, info_for
from app.agents.verification_agent import VerificationAgent
from app.assistant import qa
from app.assistant.understand import QuestionPlan, _make_need

AGENTS = {"discovery": DiscoveryAgent(), "retrieval": RetrievalAgent(), "analysis": AnalysisAgent(), "verification": VerificationAgent(), "response": ResponseAgent()}


class GState(TypedDict, total=False):
    route: str


KIND_LABEL = {"fact": "a specific factual question", "multi_doc": "a question that spans several files", "summary": "a summarization request", "compare": "a multi-document comparison",
              "data_calc": "a spreadsheet calculation", "code_explain": "a question about source code", "project_explain": "a project / code-workflow analysis",
              "file_search": "a file recommendation request", "resume_improve": "a résumé review"}


class Orchestrator:
    def __init__(self, board: Board):
        self.b = board
        self.plan = Plan()
        self.tb = ToolBox(REGISTRY, board)
        self.q: QuestionPlan = board.plan_q
        self.fallback_used = False
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ small helpers
    def _t(self, id, agent, op, goal, deps=(), optional=False, **args) -> Task:
        optional = optional or id == "d1"      # discovery failing must not block planning: the orchestrator decides what to do
        return self.plan.add(Task(id=id, agent=agent, op=op, goal=goal, args=args, deps=list(deps), optional=optional))

    def _say(self, title, detail="", status="done", **extra):
        self.b.bus.set_context("Orchestrator", None)
        self.b.bus.step("plan", title, detail, status=status, agent="Orchestrator", **extra)

    def _run(self, ids):
        run_plan(self.plan, self.b, AGENTS, self.tb, only=set(ids))

    def _failed(self, tid) -> bool:
        return self.plan.tasks[tid].status == "failed"

    # ------------------------------------------------------------------ graph
    def _build_graph(self):
        g = StateGraph(GState)
        for name in ("preflight", "discover", "expand", "gather", "verify", "more", "respond", "check_draft", "finalize", "clarify"):
            g.add_node(name, getattr(self, "n_" + name))
        g.add_edge(START, "preflight")
        g.add_edge("preflight", "discover")
        g.add_conditional_edges("discover", lambda s: s["route"], {"expand": "expand", "clarify": "clarify", "replan": "discover"})
        g.add_edge("expand", "gather")
        g.add_edge("gather", "verify")
        g.add_conditional_edges("verify", lambda s: s["route"], {"respond": "respond", "more": "more"})
        g.add_edge("more", "verify")
        g.add_edge("respond", "check_draft")
        g.add_edge("check_draft", "finalize")
        g.add_edge("finalize", END)
        g.add_edge("clarify", END)
        return g.compile()

    def run(self) -> dict:
        self.graph.invoke({"route": ""}, config={"recursion_limit": 40})
        return self.b.out

    # ------------------------------------------------------------------ nodes
    def n_preflight(self, s):
        b, q = self.b, self.q
        b.bus.set_context("Orchestrator", None)
        b.metrics.agent("Orchestrator")
        scope = q.target_file_ids or None
        b.health = qa.index_health(b.owner_id, scope, q.all_terms)
        h = b.health
        self._say(f"Identified {KIND_LABEL[q.kind]}", ", ".join(q.notes) if q.notes else
                  f"{h['indexed']:,} indexed file(s) available" + (f"; {len(h['failed'])} could not be read" if h["failed"] else ""), kind_name=q.kind)
        # first stage plan: which agent finds the files?
        if q.kind in ("file_search", "resume_improve"):
            self._t("d1", "discovery", "recommend", "Find and rank the relevant files")
        elif q.kind == "compare":
            self._t("d1", "discovery", "find_pair", "Find the documents to compare")
        elif q.kind == "data_calc":
            self._t("d1", "discovery", "find_tabular", "Find the spreadsheet that fits the question")
        elif q.kind == "project_explain" or (q.kind == "code_explain" and not q.explicit_target):
            self._t("d1", "discovery", "find_code", "Find the project's code, configuration and documentation", project=q.kind == "project_explain", limit=60)
        elif q.inherited:
            self._t("d1", "discovery", "reuse_context", "Reuse the files from the previous answer")
        elif q.explicit_target:
            self._t("d1", "discovery", "locate_named", "Locate " + ", ".join(q.target_names[:3]), names=q.target_names)
        else:
            self._t("d1", "discovery", "find_by_topic", "Find the files that talk about " + (", ".join(q.all_terms[:4]) or "this topic"), limit=3 if q.kind == "summary" else 8)
        self._say("Plan: start with the File Discovery Agent", "later stages are planned from what it finds", plan=self.plan.describe())
        return {"route": ""}

    def n_discover(self, s):
        b, q = self.b, self.q
        self._run(["d1"])
        d = self.plan.tasks["d1"]
        if d.status == "failed":
            # conditional replanning: a "spreadsheet" question with no fitting table is really a text question
            if q.kind == "data_calc" and b.discovery_meta.get("no_tables") and not self.fallback_used:
                self.fallback_used = True
                self._say("Re-planning: not a spreadsheet question", "treating it as a question about document text", status="warning")
                q.kind = "fact"
                q.type_group = None
                self.plan = Plan()
                b.discovery_meta.pop("no_tables", None)
                self._t("d1", "discovery", "find_by_topic", "Find the files that talk about " + (", ".join(q.all_terms[:4]) or "this topic"), limit=8)
                return {"route": "replan"}
            if q.kind in ("compare",) or (q.kind == "summary"):
                return {"route": "clarify"}
        if q.kind == "compare" and b.discovery_meta.get("needs_clarification"):
            return {"route": "clarify"}
        if q.kind == "summary" and not b.target_ids:
            return {"route": "clarify"}
        return {"route": "expand"}

    def n_clarify(self, s):
        b, q = self.b, self.q
        if q.kind == "summary":
            recent = qa._recent_files(b.owner_id, 5)
            ans = "Which file would you like me to summarize? " + ("For example: " + ", ".join(f"“{r['filename']}”" for r in recent) + "." if recent else "")
            opts = [r["filename"] for r in recent]
        else:
            have = f" I found one so far: {qa.title_of(b.candidates[0]['filename'])}." if b.candidates else ""
            ans, opts = "I need two documents to compare. Which ones do you mean?" + have, []
        self._say("Needs a clarification — skipping retrieval, analysis and verification", "the request does not identify enough files", status="warning")
        b.out.update(status="clarify", answer=ans, references=[], analyses={}, warnings=[], conflicts=[], llm={"used": False}, verification={"sentences_checked": 0, "supported": 0, "removed": []},
                     clarify_options=opts, mode="qa", _focus=[])
        return {}

    def n_expand(self, s):
        """Second planning stage: create the remaining tasks from what discovery found."""
        b, q, P = self.b, self.q, self.plan
        cands = b.candidates
        ids = [c["file_id"] for c in cands]
        inf = info_for(b, ids)
        ftype = lambda fid: inf.get(fid, {}).get("file_type", "")
        retr, ana = [], []
        kind = q.kind
        if kind in ("fact", "multi_doc", "code_explain"):
            for i, need in enumerate(q.needs):
                retr.append(self._t(f"r{i + 1}", "retrieval", "retrieve", f"Retrieve passages for “{', '.join(need.terms[:4]) or need.text[:30]}”", deps=["d1"], optional=True, need_no=i).id)
            ana.append(self._t("a1", "analysis", "answer_facts", "Pick the sentences that answer each question", deps=retr).id)
            ver = self._t("v1", "verification", "verify_facts", "Check the evidence covers the question and agrees across sources", deps=ana)
        elif kind == "summary":
            targets = ids[:3] if (q.explicit_target or q.inherited) else ids[:1]
            html_t = [f for f in targets if ftype(f) in ("html", "htm")]
            doc_t = [f for f in targets if f not in html_t]
            rid = []
            for k, fid in enumerate(targets):
                rid.append(self._t(f"r{k + 1}", "retrieval", "read_document", f"Read {inf[fid]['filename']}" if fid in inf else "Read the document", deps=["d1"], optional=True, file_id=fid).id)
            vdeps = []
            if html_t:
                a = self._t("a_html", "analysis", "explain_html", "Explain what the web page is and does", deps=[r for r, f in zip(rid, targets) if f in html_t], file_ids=html_t)
                vdeps.append(self._t("v_html", "verification", "verify_html", "Check each statement about the page against its markup", deps=[a.id]).id)
                ana.append(a.id)
            if doc_t:
                a = self._t("a_sum", "analysis", "summarize", "Select the representative content and key points", deps=[r for r, f in zip(rid, targets) if f in doc_t], file_ids=doc_t)
                vdeps.append(self._t("v_sum", "verification", "verify_summary", "Check every summary sentence exists at its cited location", deps=[a.id]).id)
                ana.append(a.id)
            ver = None
        elif kind == "compare":
            fids = ids[:2]
            names = {w for f in fids for w in re.split(r"[^a-z0-9]+", inf[f]["filename"].lower())} if inf else set()
            need = q.needs[0] if q.needs else None
            aspect_terms = [t for t in (need.terms if need else []) if t not in names and t not in ("difference", "differences", "differ", "compare", "comparison", "similarities", "contrast", "versus")]
            aspect = _make_need(" ".join(aspect_terms), set(), []) if aspect_terms else None
            rid = [self._t(f"r{k + 1}", "retrieval", "read_aspect", f"Read {inf[f]['filename']}" + (f" for “{', '.join(aspect_terms[:3])}”" if aspect_terms else ""), deps=["d1"], file_id=f, need=aspect).id
                   for k, f in enumerate(fids) if f in inf]
            a = self._t("a1", "analysis", "compare", "Compare the two documents", deps=rid, file_ids=fids, aspect=aspect)
            ana.append(a.id)
            ver = self._t("v1", "verification", "verify_comparison", "Check both documents are represented and every statement is real", deps=[a.id])
        elif kind == "data_calc":
            r = self._t("r1", "retrieval", "read_table", "Read the spreadsheet's structure and rows", deps=["d1"], file_ids=ids[:3])
            a = self._t("a1", "analysis", "calculate", "Calculate the answer from the actual rows", deps=[r.id], file_ids=ids[:3])
            ana.append(a.id)
            ver = self._t("v1", "verification", "verify_calc", "Re-compute the result independently from the raw cells", deps=[a.id])
        elif kind == "project_explain":
            docs = [c["file_id"] for c in cands if ftype(c["file_id"]) in ("md", "txt", "rst") or inf.get(c["file_id"], {}).get("filename", "").lower() in ("package.json", "pyproject.toml", "requirements.txt", "setup.py", "cargo.toml", "go.mod")]
            cfg = [c["file_id"] for c in cands if ftype(c["file_id"]) in ("json", "yaml", "yml", "toml", "ini", "cfg") and c["file_id"] not in docs]
            src = [c["file_id"] for c in cands if ftype(c["file_id"]) in qa.CODE_TYPES]
            rid = []
            if docs:
                rid.append(self._t("r_docs", "retrieval", "read_code", "Read the README and manifests", deps=["d1"], optional=True, file_ids=docs[:6], role="doc").id)
            if cfg:
                rid.append(self._t("r_cfg", "retrieval", "read_code", "Read configuration values", deps=["d1"], optional=True, file_ids=cfg[:8], role="config").id)
            for k in range(0, min(len(src), 48), 12):
                rid.append(self._t(f"r_src{k // 12 + 1}", "retrieval", "read_code", f"Read the structure of source files {k + 1}–{min(k + 12, len(src))}", deps=["d1"], optional=True, file_ids=src[k:k + 12], role="source").id)
            focus = [t for t in q.all_terms if t not in __import__("app.agents.analysis_agent", fromlist=["GENERIC_PROJECT"]).GENERIC_PROJECT]
            deps_a = list(rid)
            a1 = self._t("a_proj", "analysis", "explain_project", "Work out what the project is for and how it is organised", deps=deps_a, optional=True)
            ana.append(a1.id)
            if focus:
                a2 = self._t("a_trace", "analysis", "trace", f"Trace how {', '.join(focus[:3])} works across the files", deps=deps_a, optional=True)
                ana.append(a2.id)
            ver = self._t("v1", "verification", "verify_trace", "Check that every function, setting and README claim exists in the files", deps=ana)
        elif kind in ("file_search", "resume_improve"):
            r = self._t("r1", "retrieval", "excerpts", "Pull verbatim excerpts from each recommended file", deps=["d1"])
            ana_deps = [r.id]
            if kind == "resume_improve" and ids:
                a = self._t("a1", "analysis", "review_resume", "Review the résumé against a checklist, quoting evidence", deps=[r.id], file_id=ids[0])
                ana.append(a.id)
            ver = self._t("v1", "verification", "verify_find", "Check excerpts match the files, files are unchanged, and nothing instruction-like slipped in", deps=[r.id] + ana)
        else:
            ver = None
        if kind == "summary":
            vlist = vdeps
        else:
            vlist = [ver.id]
        self._t("s1", "response", "respond", "Write the answer from the verified findings", deps=vlist)
        self._t("v2", "verification", "verify_draft", "Check every sentence of the draft against its sources", deps=["s1"])
        self._t("s2", "response", "finalize", "Attach sources and limitations", deps=["v2"])
        skipped = []
        if kind in ("file_search",):
            skipped.append("Analysis Agent (a plain recommendation needs no analysis)")
        if kind in ("fact", "multi_doc") and not q.explicit_target:
            pass
        self._say("Plan: " + " → ".join(dict.fromkeys(agent_label(t.agent) for t in P.tasks.values())), ("skipped: " + "; ".join(skipped)) if skipped else
                  f"{len(P.tasks)} tasks, {len([t for t in P.tasks.values() if t.agent == 'retrieval'])} retrieval task(s) can run in parallel", plan=P.describe())
        if skipped:
            b.metrics.inc("tasks_skipped", len(skipped))
        self._stage_gather = [t.id for t in P.tasks.values() if t.agent in ("retrieval", "analysis")]
        self._stage_verify = [t.id for t in P.tasks.values() if t.agent == "verification" and t.id not in ("v2",)]
        return {}

    def n_gather(self, s):
        self._run(self._stage_gather)
        b, q = self.b, self.q
        # dynamic follow-up: code evidence was found -> read definitions that implement the behaviour (not planned in advance)
        if q.kind in ("fact", "multi_doc", "code_explain") and b.findings.get("code_mode") and "a_code" not in self.plan.tasks and self.plan.tasks.get("a1") and self.plan.tasks["a1"].status == "done":
            self._say("Code evidence found — adding a code-analysis task", "", status="done")
            self._t("a_code", "analysis", "explain_code", "Find the definitions that implement this behaviour", deps=["a1"], optional=True)
            self.plan.tasks["v1"].deps.append("a_code")
            self._run(["a_code"])
        return {}

    def n_verify(self, s):
        b = self.b
        ids = self._stage_verify
        self._run(ids)
        v = b.verdict or {}
        more = v.get("request_more") or {}
        if more and b.rounds < MAX_ROUNDS:
            return {"route": "more"}
        return {"route": "respond"}

    def n_more(self, s):
        b = self.b
        more = b.verdict["request_more"]
        b.rounds += 1
        b.metrics.inc("rounds")
        self._say(f"Verification asked for more evidence on {len(more)} question(s) — round {b.rounds}", "; ".join(f"missing {', '.join(h.get('missing', [])[:3]) or 'a direct statement'}" for h in more.values()), status="warning")
        b.retrieval_hints = dict(more)
        ids = []
        for i in more:
            tid = f"r{i + 1}_x{b.rounds}"
            self._t(tid, "retrieval", "retrieve", f"Search again for question {i + 1} with looser matching across all files", need_no=i, optional=True)
            ids.append(tid)
        aid = f"a1_x{b.rounds}"
        self._t(aid, "analysis", "answer_facts", "Re-pick the answering sentences", deps=ids)
        vid = f"v1_x{b.rounds}"
        self._t(vid, "verification", "verify_facts", "Re-check the evidence", deps=[aid])
        self._run(ids + [aid, vid])
        self._stage_verify = []                 # verification already re-ran inside this node
        b.verdict.setdefault("request_more", {})
        if b.rounds >= MAX_ROUNDS:
            b.verdict["request_more"] = {}
        return {}

    def n_respond(self, s):
        self._run(["s1"])
        return {}

    def n_check_draft(self, s):
        self._run(["v2"])
        return {}

    def n_finalize(self, s):
        self._run(["s2"])
        b = self.b
        if not b.out.get("answer"):
            b.out.update(status="error", answer="I couldn't complete that request: " + "; ".join(f"{t.op}: {t.error}" for t in self.plan.tasks.values() if t.status == "failed" and t.error)[:300], references=[])
        return {}


def run_agents(owner_id: str, question: str, qplan: QuestionPlan, settings: dict, conv_hint: str, bus: EventBus) -> dict:
    b = Board(owner_id=owner_id, question=question, plan_q=qplan, settings=settings, conv_hint=conv_hint, bus=bus)
    orch = Orchestrator(b)
    out = orch.run()
    out["agents"] = {**b.metrics.snapshot(), "tasks": orch.plan.describe(), "wall_s": round(__import__("time").perf_counter() - bus.t0, 2)}
    out["coverage"] = _coverage(b)
    out["_board"] = b
    return out


def _coverage(b: Board) -> dict:
    h = b.health or {}
    meta = [m["ret"] for m in b.retrieval_meta.values() if m.get("ret")]
    c = {"files_indexed": h.get("indexed", 0), "files_failed": len(h.get("failed", [])), "files_unsupported": len(h.get("unsupported", [])), "files_pending": h.get("pending", 0)}
    if meta:
        c.update(passages_searched=max(m.searched_passages for m in meta), files_searched=max(m.searched_files for m in meta))
    elif b.target_ids:
        c["files_searched"] = len(b.target_ids)
    c.update(b.out.get("coverage_extra", {}))
    return c
