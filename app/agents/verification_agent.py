"""Verification Agent -- decides whether the analysis may be trusted, and says what is missing.

It does not write anything for the user. For each kind of finding it re-checks the work against the source:
  verify_facts     is every needed fact covered? does the cited span really contain the claim and its figures? do sources disagree?
  verify_summary   is every summary sentence literally present at its cited span?
  verify_html      does each explicit statement about a page match markup actually present in the file?
  verify_comparison both documents represented and every compared statement verified
  verify_calc      the calculation is RE-COMPUTED from the raw cells by separate code and must match
  verify_trace     every function / config value / README claim exists in the file at the cited place
  verify_find      excerpts match the stored text, files unchanged on disk, instruction-like text flagged
  verify_draft     sentence-level check of the Response Agent's draft (figures, quotes, citations)
When evidence is missing and the round budget allows, it sets retrieval hints; the Orchestrator then sends the Retrieval Agent back out.
"""
import re
from typing import List

from app.agents.core import MAX_ROUNDS, AgentFailure, Board, Task
from app.agents.tools import info_for, text_of
from app.analysis import safety
from app.assistant import qa


def _claim(cid, text, fid, start, end):
    return {"id": cid, "text": text, "file_id": fid, "start": start, "end": end}


def _merge(b: Board, status: str, **kw) -> None:
    """Several verification tasks can run for one request (e.g. an HTML page and a PDF): combine their verdicts."""
    with b.lock:
        sts = b.verdict.get("statuses", []) + [status]
        overall = "ok" if all(x == "ok" for x in sts) else "partial" if any(x == "ok" for x in sts) else "insufficient"
        b.verdict = {**b.verdict, **kw, "statuses": sts, "status": overall}


class VerificationAgent:
    name = "verification"

    def run(self, task: Task, b: Board, tb) -> str:
        op = getattr(self, "op_" + task.op, None)
        if op is None:
            raise AgentFailure(f"unknown verification operation {task.op}", retryable=False)
        return op(task, b, tb)

    # ------------------------------------------------------------------ facts
    def op_verify_facts(self, task, b: Board, tb) -> str:
        results = b.results
        failed_claims, conflicts, hints = 0, [], {}
        for i, r in enumerate(results):
            need = r["need"]
            if not need.stems:
                continue
            suff = tb.call("verification", "check_sufficiency", need_no=i,
                           summarize=lambda d: ("evidence covers the question" if d["ok"] else f"evidence does NOT cover it ({d['reason']}; missing: {', '.join(d['missing'][:4]) or 'a direct statement'})"))
            if r["ok"] and r["assess"].get("best") and r["assess"]["best"].get("sentence"):
                c = r["assess"]["best"]
                vc = tb.call("verification", "verify_claims", claims=[_claim(f"n{i}", c["sentence"], c["ev"].file_id, c["start"], c["end"])],
                             summarize=lambda d: f"{sum(1 for x in d['claims'] if x['supported'])}/{len(d['claims'])} claim(s) found verbatim in the file")
                if vc.ok and not vc.data["claims"][0]["supported"]:
                    r["ok"] = False
                    failed_claims += 1
                    b.bus.step("verify", "A claimed sentence could not be confirmed in the file", c["sentence"][:100], status="warning")
            if r["ok"] and need.answer_type in ("number", "date"):
                dc = tb.call("verification", "detect_conflicts", need_no=i, summarize=lambda d: "sources DISAGREE" if d["conflict"] else "no disagreement between sources")
                if dc.ok and dc.data["conflict"]:
                    conflicts.append(dc.data["conflict"])
            if not r["ok"] and b.rounds < MAX_ROUNDS:
                hints[i] = {"global": True, "relaxed": True, "missing": suff.data.get("missing", []) if suff.ok else []}
        ok_n = sum(1 for r in results if r["ok"])
        status = "ok" if results and ok_n == len(results) else "partial" if ok_n else "insufficient"
        with b.lock:
            b.conflicts = conflicts
            b.verdict = {"status": status, "needs_ok": ok_n, "needs": len(results), "claims_failed": failed_claims, "conflicts": len(conflicts), "request_more": hints}
        if conflicts:
            b.bus.step("verify", "Sources disagree", " vs ".join(f"{c['filename']}: {', '.join(c['values'][:2])}" for c in conflicts[0]["claims"][:3]), status="warning")
        return {"ok": "Evidence verified", "partial": "Evidence covers only part of the question", "insufficient": "No sufficient evidence in the files"}[status] + (f" — requesting {len(hints)} more retrieval(s)" if hints else "")

    # ------------------------------------------------------------------ summaries
    def op_verify_summary(self, task, b: Board, tb) -> str:
        sums = b.findings.get("summaries", {})
        claims = []
        for fid, s in sums.items():
            for k, o in enumerate(s["opening"] + s["points"]):
                claims.append(_claim(f"{fid}:{k}", o["sentence"], fid, o["start"], o["end"]))
        r = tb.call("verification", "verify_claims", claims=claims, summarize=lambda d: f"{sum(1 for x in d['claims'] if x['supported'])}/{len(d['claims'])} summary sentence(s) found at their cited location")
        if not r.ok:
            raise AgentFailure("verification tool failed")
        bad = {c["id"] for c in r.data["claims"] if not c["supported"]}
        for fid, s in sums.items():
            n_open = len(s["opening"])
            s["opening"] = [o for k, o in enumerate(s["opening"]) if f"{fid}:{k}" not in bad]
            s["points"] = [o for k, o in enumerate(s["points"]) if f"{fid}:{n_open + k}" not in bad]
        left = sum(len(s["opening"]) + len(s["points"]) for s in sums.values())
        _merge(b, "ok" if left else "insufficient", claims_failed=len(bad))
        if bad:
            b.bus.step("verify", f"Dropped {len(bad)} unconfirmed summary sentence(s)", "", status="warning")
        return f"{len(claims) - len(bad)}/{len(claims)} summary sentences verified"

    # ------------------------------------------------------------------ html
    def op_verify_html(self, task, b: Board, tb) -> str:
        html = b.findings.get("html", {})
        claims, owner = [], {}
        for fid, h in html.items():
            full_len = len(text_of(b, fid))
            for k, f in enumerate(h["facts"]):
                if f.get("needle"):
                    cid = f"{fid}:{k}"
                    claims.append(_claim(cid, f["needle"], fid, 0, full_len))
                    owner[cid] = f
        r = tb.call("verification", "verify_claims", claims=claims, summarize=lambda d: f"{sum(1 for x in d['claims'] if x['supported'])}/{len(d['claims'])} statements match markup in the file")
        if not r.ok:
            raise AgentFailure("verification tool failed")
        bad = {c["id"] for c in r.data["claims"] if not c["supported"]}
        for fid, h in html.items():
            h["facts"] = [f for k, f in enumerate(h["facts"]) if f"{fid}:{k}" not in bad]
        inferred = sum(1 for h in html.values() for f in h["facts"] if f["kind"] == "inferred")
        _merge(b, "ok", claims_failed=len(bad), inferred=inferred)
        return f"{len(claims) - len(bad)} explicit statement(s) confirmed; {inferred} inference(s) labelled as such"

    # ------------------------------------------------------------------ comparison
    def op_verify_comparison(self, task, b: Board, tb) -> str:
        comp = b.findings.get("comparison")
        if not comp:
            raise AgentFailure("nothing to verify")
        claims = []
        for fid, ranked in comp["per_doc"].items():
            for k, c in enumerate(ranked):
                claims.append(_claim(f"{fid}:{k}", c["sentence"], fid, c["start"], c["end"]))
        r = tb.call("verification", "verify_claims", claims=claims, summarize=lambda d: f"{sum(1 for x in d['claims'] if x['supported'])}/{len(d['claims'])} compared statement(s) verified")
        bad = {c["id"] for c in r.data["claims"] if not c["supported"]} if r.ok else set()
        for fid in list(comp["per_doc"]):
            comp["per_doc"][fid] = [c for k, c in enumerate(comp["per_doc"][fid]) if f"{fid}:{k}" not in bad]
        missing = [fid for fid, v in comp["per_doc"].items() if not v and comp["aspect"] is not None and comp["aspect"].stems]
        with b.lock:
            b.verdict = {"status": "partial" if missing else "ok", "claims_failed": len(bad), "missing_for": missing}
        return "Both documents represented" if not missing else f"{len(missing)} document(s) say nothing about the compared aspect"

    # ------------------------------------------------------------------ calculations
    def op_verify_calc(self, task, b: Board, tb) -> str:
        calc = b.findings.get("calc")
        if not calc:
            raise AgentFailure("no calculation to verify")
        r = tb.call("verification", "recompute_calc", calc=calc, summarize=lambda d: (f"re-computed {d['recomputed']:,.2f} from {d['rows']:,} raw cells — {'matches' if d['match'] else 'MISMATCH'}" if d.get("checked") else d.get("reason", "")))
        if not r.ok:
            raise AgentFailure("recomputation failed")
        res = r.data
        with b.lock:
            b.verdict = {"status": "ok", "recheck": res}
            if res.get("checked") and not res["match"]:
                b.verdict["status"] = "insufficient"
        if res.get("checked") and not res["match"]:
            b.bus.step("verify", "Calculation mismatch — result withheld", f"reported {res['reported']}, recomputed {res['recomputed']}", status="error")
            raise AgentFailure("independent recomputation disagrees", retryable=False)
        return "Calculation independently re-checked" if res.get("checked") else "Result is derived directly from the rows (no simple aggregate to re-check)"

    # ------------------------------------------------------------------ code / project
    def op_verify_trace(self, task, b: Board, tb) -> str:
        tr, pr, ci = b.findings.get("trace"), b.findings.get("project"), b.findings.get("code_items")
        claims = []
        if tr:
            for k, st in enumerate(tr["chain"]):
                full = text_of(b, st["file_id"])
                lines = full.split("\n")
                start = sum(len(x) + 1 for x in lines[: st["line_start"] - 1])
                end = start + sum(len(x) + 1 for x in lines[st["line_start"] - 1: st["line_end"]])
                claims.append(_claim(f"t{k}", st["name"], st["file_id"], start, end))
        if pr:
            for k, m in enumerate(pr["modules"][:6]):
                full = text_of(b, m["file_id"])
                for nm in m["names"][:1]:
                    claims.append(_claim(f"m{k}", nm, m["file_id"], 0, len(full)))
        if ci:
            for k, c in enumerate(ci):
                full = text_of(b, c["file"]["file_id"])
                claims.append(_claim(f"c{k}", c["item"]["name"], c["file"]["file_id"], 0, len(full)))
        r = tb.call("verification", "verify_claims", claims=claims, summarize=lambda d: f"{sum(1 for x in d['claims'] if x['supported'])}/{len(d['claims'])} code reference(s) exist at the cited location")
        bad = {c["id"] for c in r.data["claims"] if not c["supported"]} if r.ok else set()
        if tr:
            tr["chain"] = [st for k, st in enumerate(tr["chain"]) if f"t{k}" not in bad]
        with b.lock:
            b.verdict = {"status": "ok" if claims and len(bad) < len(claims) else "insufficient", "claims_failed": len(bad)}
        if not claims:
            b.bus.step("verify", "Nothing to verify: the analysis produced no findings", "", status="warning")
        return f"{len(claims) - len(bad)}/{len(claims)} code references confirmed; behaviour is read from the code, not executed"

    # ------------------------------------------------------------------ recommendations
    def op_verify_find(self, task, b: Board, tb) -> str:
        from app.assistant import engine as E
        refs = b.findings.get("find_refs", [])
        files = b.discovery_meta.get("find_files", {})
        bad = stale = inj = 0
        for r in refs:
            f = files.get(r["file_id"])
            if f:
                s = E._staleness(b.owner_id, f)
                if s:
                    r["stale"] = s
                    b.warnings.append(f"{r['filename']} {s}.")
                    stale += 1
            hits = safety.scan(" ".join(x["text"] for x in r["excerpts"]))
            if hits:
                r["injection_suspected"] = hits[:3]
                b.warnings.append(f"{r['filename']} contains text that looks like instructions to an AI system ({len(hits)} line(s)); it was treated as data and ignored.")
                b.bus.step("verify", "Possible prompt injection in a document", f"{r['filename']}: {hits[0]['snippet'][:100]}", status="warning")
                inj += 1
            bad += sum(1 for x in r["excerpts"] if x.get("verified") is False)
        with b.lock:
            b.verdict = {"status": "ok" if refs else "insufficient", "excerpts_failed": bad}
        return f"{sum(len(r['excerpts']) for r in refs) - bad} excerpt(s) match the files" + (f"; {stale} changed on disk" if stale else "") + (f"; {inj} with instruction-like text" if inj else "")

    # ------------------------------------------------------------------ the Response Agent's draft
    def op_verify_draft(self, task, b: Board, tb) -> str:
        d = b.out.get("draft")
        if not d:
            raise AgentFailure("no draft to verify", retryable=False)
        if d["mode"] == "find":                     # a recommendation list is assembled from already-verified excerpts; there is no free text to re-check
            with b.lock:
                d["verified_text"], d["report"], d["accepted"] = d["text"], {}, True
            return "Recommendation list built from verified excerpts (no free-form claims to re-check)"
        r = tb.call("verification", "verify_text", text=d["text"], sources=d["sources"], extra=d.get("extra", ""), min_support=d.get("min_support", 0.55),
                    summarize=lambda x: f"{x['report']['supported']}/{x['report']['sentences_checked']} claim(s) supported" + (f"; removed {len(x['report']['removed'])}" if x["report"]["removed"] else ""))
        if not r.ok:
            raise AgentFailure("draft verification failed")
        rep = r.data["report"]
        with b.lock:
            d["verified_text"], d["report"] = r.data["text"], rep
            rejected = d["llm"] and (not r.data["text"].strip() or (rep["removed"] and len(rep["removed"]) >= max(1, rep["sentences_checked"] // 2)))
            d["accepted"] = not rejected
        if rejected:
            b.bus.step("verify", "Model-written answer rejected", f"{len(rep['removed'])} unsupported claim(s): " + "; ".join(rep["removed"][:2]), status="warning")
        elif rep["removed"] and d["llm"]:
            b.bus.step("verify", "Removed unsupported sentence(s) from the model's answer", "; ".join(rep["removed"][:2]), status="warning")
        return "Draft accepted" if d["accepted"] else "Draft rejected — falling back to the extractive answer"
