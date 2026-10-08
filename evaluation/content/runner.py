"""
Phase 11 real-content evaluation runner. SEPARATE from the Phase 7/8/9 regression
benchmark (evaluation/runner.py is untouched): this one ingests the Phase 11
corpus through the real ingestion pipeline into an isolated in-memory Qdrant and
the live PostgreSQL, then drives the real run_query() agent.

Everything is read from OUTSIDE the production path (benchmark -> run_query() ->
inspect result / trace / candidate_pool). Stage rankings are reconstructed from the
per-source scores the agent already stores in its candidate pool; the "without
reranker" ranking is a counterfactual computed by calling the production pure
function final_rank() on a copy of that same pool with the rerank scores removed.
"""
import copy
import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

from app.db.connection import get_cursor
from app.final_ranking import PERSONALIZATION_SOURCE, final_rank
from app.llm.provider import NullProvider
from app.run import run_query

from evaluation.content.corpus_spec import CORPUS
from evaluation.content.queries import DIAGNOSTIC_QUERIES, QUERIES, TELEMETRY, ContentQuery, validate
from evaluation.metrics import judge_success, retrieval_metrics
from evaluation.providers import STAGES, CountingProvider
from evaluation.report import behavior_summary, retrieval_summary

CORPUS_DIR = Path(__file__).parent / "corpus"
OUT_DIR = Path(__file__).parent / "output"
USERS = {"alice": "p11_alice", "bob": "p11_bob"}
SCHEMA_VERSION = 1
STAGE_SOURCES = ("metadata_search", "bm25_search", "semantic_search", "hybrid_search", "rerank_candidates")
RETRIEVAL_SOURCES = ("metadata_search", "bm25_search", "semantic_search", "hybrid_search")
RETRIEVAL_TOOLS = RETRIEVAL_SOURCES
TRACKED_TOOLS = (*RETRIEVAL_TOOLS, "rerank_candidates", "get_user_activity")


class ContentEvalSetupError(RuntimeError):
    pass


# --------------------------------------------------------------------------- environment
def _purge():
    owners = list(USERS.values())
    with get_cursor(commit=True) as cur:
        cur.execute("DELETE FROM user_file_telemetry WHERE user_id = ANY(%s) OR file_id IN (SELECT file_id FROM files WHERE owner_id = ANY(%s));", (owners, owners))
        cur.execute("DELETE FROM user_profiles WHERE user_id = ANY(%s);", (owners,))
        cur.execute("DELETE FROM files WHERE owner_id = ANY(%s);", (owners,))
        cur.execute("DELETE FROM users WHERE user_id = ANY(%s);", (owners,))


@contextmanager
def content_environment(rebuild_qdrant: bool = True):
    """Users + isolated in-memory Qdrant + real ingestion + telemetry fixture; removes everything on exit
    (the shared Qdrant collection and the Phase 7 seed corpus are never touched)."""
    import app.tools.semantic_search as sem
    import app.vectorstore as vs
    from qdrant_client import QdrantClient

    from app.ingestion import ingest_directory

    problems = validate(CORPUS)
    if problems:
        raise ContentEvalSetupError("invalid query set: " + "; ".join(problems))
    _purge()
    old_client, old_flag = vs._client, sem._build_attempted
    vs._client, sem._build_attempted = QdrantClient(":memory:"), True
    try:
        with get_cursor(commit=True) as cur:
            cur.executemany("INSERT INTO users (user_id, display_name) VALUES (%s, %s);", [(u, u) for u in USERS.values()])
        reports = {}
        for owner, uid in USERS.items():
            reports[owner] = ingest_directory(CORPUS_DIR / owner, uid)
        with get_cursor() as cur:
            cur.execute("SELECT f.file_id, f.owner_id, f.file_path, f.file_type FROM files f WHERE f.owner_id = ANY(%s);", (list(USERS.values()),))
            rows = [dict(r) for r in cur.fetchall()]
        label_of = {r["file_id"]: f"{_owner_of(r['owner_id'])}/{r['file_path']}" for r in rows}
        type_of = {label_of[r["file_id"]]: r["file_type"] for r in rows}
        id_of = {v: k for k, v in label_of.items()}
        _insert_telemetry(id_of)
        yield {"label_of": label_of, "id_of": id_of, "type_of": type_of, "ingestion": {k: v.summary() for k, v in reports.items()}}
    finally:
        vs._client, sem._build_attempted = old_client, old_flag
        _purge()


def _owner_of(user_id: str) -> str:
    return next(k for k, v in USERS.items() if v == user_id)


def _insert_telemetry(id_of: Dict[str, int]) -> None:
    with get_cursor(commit=True) as cur:
        for owner, spec in TELEMETRY.items():
            uid = USERS[owner]
            cur.execute("INSERT INTO user_profiles (user_id, preferred_file_types, frequent_topics) VALUES (%s, %s, %s::jsonb);",
                        (uid, spec["profile"]["preferred_file_types"], json.dumps(spec["profile"]["frequent_topics"])))
            for rel, events in spec["events"].items():
                for action, seconds, days_ago in events:
                    cur.execute("INSERT INTO user_file_telemetry (user_id, file_id, action_type, time_spent_seconds, timestamp) "
                                "VALUES (%s, %s, %s, %s, now() - make_interval(days => %s));",
                                (uid, id_of[f"{owner}/{rel}"], action, seconds, days_ago))


# --------------------------------------------------------------------------- stage analysis (pure helpers)
def stable_reason(text, label_of: dict):
    """The agent's free-text reason names files by database id ("file 17"), and serial ids differ between runs;
    translate them to the stable owner/path labels so records are reproducible (and more readable)."""
    if not text:
        return text
    return re.sub(r"\bfile (\d+)\b", lambda m: f"file {label_of.get(int(m.group(1)), m.group(1))}", text)


def _rounded(x, n=3):
    return None if x is None else round(float(x), n)


def stage_rankings(pool: dict, label_of: dict, k: int = 5) -> dict:
    """Per retrieval/rerank source: that source's own ordering of the candidates it scored (ties by label)."""
    out = {}
    for src in STAGE_SOURCES:
        items = [(label_of[fid], e["scores"][src]) for fid, e in pool.items() if src in e["scores"]]
        items.sort(key=lambda t: (-t[1], t[0]))
        out[src] = [[lab, _rounded(sc)] for lab, sc in items[:k]]
    return out


def counterfactual_rankings(pool: dict, label_of: dict, k: int = 5) -> dict:
    """final_rank() (production pure function) over copies of the SAME pool with evidence sources removed."""
    def ranked(drop):
        p = copy.deepcopy(pool)
        for e in p.values():
            for s in drop:
                e["scores"].pop(s, None)
        p = {fid: e for fid, e in p.items() if e["scores"]}
        return [label_of[e["file_id"]] for e in final_rank(p)][:k]
    return {
        "final": ranked(()),
        "without_rerank": ranked(("rerank_candidates",)),
        "without_personalization": ranked((PERSONALIZATION_SOURCE,)),
        "retrieval_only": ranked(("rerank_candidates", PERSONALIZATION_SOURCE)),
        "without_hybrid": ranked(("hybrid_search",)),
    }


def first_rank(labels: List[str], relevant: set) -> Optional[int]:
    for i, lab in enumerate(labels, start=1):
        if lab in relevant:
            return i
    return None


def attribute(rec: dict) -> dict:
    """Where does a correct answer become incorrect? PURE function of one run record. Returns
    {"verdict": "ok" | <failure kind>, "stage": <pipeline stage or None>}.

    Failure kinds, in the order they are checked:
      isolation            a forbidden/foreign file was returned or pooled
      false_positive       a no_result query returned files (stage = strongest evidence source of the top one)
      goal                 the goal's file_type filter excludes every relevant file
      retrieval_miss       no relevant file is in the candidate pool at all (stage = expected retrieval path)
      stopping             the final ranking WOULD put a relevant file first, but the agent clarified/returned nothing
      retrieval_ordering   a relevant file is in the pool, but retrieval evidence alone already ranks a wrong file first
      reranker             retrieval evidence ranks a relevant file first; adding rerank evidence breaks it
      personalization      ... only personalization breaks it
      final_ranking        ... neither reranker nor personalization explains it
      post_ranking         the pool's final ranking is right but the returned result differs (5C/5D)
    """
    if rec["success"]:
        return {"verdict": "ok", "stage": None}
    if rec["scope_violations"]:
        return {"verdict": "isolation", "stage": "scope"}
    relevant = set(rec["expected_files"])
    cf = rec["counterfactual"]
    if rec["expected_outcome"] == "no_result":
        top = rec["returned_files"][0] if rec["returned_files"] else None
        src = rec["top_evidence_source"].get(top) if top else None
        return {"verdict": "false_positive", "stage": src}
    if rec["goal_blocks_relevant"]:
        return {"verdict": "goal", "stage": "goal"}
    if not (relevant & set(rec["pool_files"])):
        path = "bm25+semantic" if rec["expect_bm25"] and rec["expect_semantic"] else "semantic" if rec["expect_semantic"] else "bm25" if rec["expect_bm25"] else "metadata"
        return {"verdict": "retrieval_miss", "stage": path}
    final_top1_ok = bool(cf["final"]) and cf["final"][0] in relevant
    if final_top1_ok and not rec["returned_files"]:
        return {"verdict": "stopping", "stage": "reflect/plan"}
    if not (bool(cf["retrieval_only"]) and cf["retrieval_only"][0] in relevant):
        return {"verdict": "retrieval_ordering", "stage": rec["top_evidence_source"].get(cf["retrieval_only"][0]) if cf["retrieval_only"] else None}
    if not final_top1_ok:
        if cf["without_rerank"] and cf["without_rerank"][0] in relevant:
            return {"verdict": "reranker", "stage": "rerank_candidates"}
        if cf["without_personalization"] and cf["without_personalization"][0] in relevant:
            return {"verdict": "personalization", "stage": "get_user_activity"}
        return {"verdict": "final_ranking", "stage": "final_rank"}
    return {"verdict": "post_ranking", "stage": "compare/explain"}


def standalone_probes(query: str, uid: str, label_of: dict, relevant: set, k: int = 5) -> dict:
    """What EACH retrieval stage would return for this query on its own -- the same scope and the same arguments
    the planner builds for it -- regardless of whether the agent ever consulted it. This is what separates
    "the provider failed" (semantic alone is wrong) from "the agent never asked / discarded it".
    Outside production: it only calls the existing tools."""
    from app.goal import understand_goal
    from app.planning import _bm25_args, _hybrid_args, _semantic_args
    from app.scope import resolve_scope
    from app.tools.bm25_search import bm25_search
    from app.tools.hybrid_search import hybrid_search
    from app.tools.rerank_candidates import rerank_candidates
    from app.tools.semantic_search import semantic_search

    goal = understand_goal(query).to_dict()
    goal["expanded_terms"] = []
    scope = resolve_scope(uid)
    out, first = {}, {}
    sem_cands = []
    for name, build, tool in (("bm25", _bm25_args, bm25_search), ("semantic", _semantic_args, semantic_search),
                              ("hybrid", _hybrid_args, hybrid_search)):
        args = build(goal)
        cands = tool(args, scope).candidates if args else []
        if name == "semantic":
            sem_cands = cands
        out[name] = [[label_of[c["file_id"]], _rounded(c["score"], 4)] for c in cands[:k]]
        first[name] = first_rank([label_of[c["file_id"]] for c in cands], relevant) if relevant else None
        if name == "bm25":
            out["bm25_top_coverage"] = cands[0].get("coverage") if cands else None
    rr = rerank_candidates({"query": query, "candidates": sem_cands}, scope).candidates if len(sem_cands) > 1 else []
    out["rerank_of_semantic"] = [[label_of[c["file_id"]], _rounded(c["score"], 4)] for c in rr[:k]]
    first["rerank_of_semantic"] = first_rank([label_of[c["file_id"]] for c in rr], relevant) if relevant else None
    return {"top": out, "first_relevant_rank": first}


def attribute_layer(rec: dict) -> dict:
    """Provider-vs-agent attribution. PURE function of one record. `layer` is one of:
      ok | semantic_provider | lexical_retrieval | agent_decision | hybrid | reranker | personalization |
      final_ranking | stopping | post_ranking | goal | isolation | no_relevance_floor | lexical_false_positive | other
    "semantic_provider": the semantic stage ALONE did not rank a relevant file first -- the embedding model, not the
    agent, failed. It is checked BEFORE any agent layer, so an agent is never blamed for a provider failure, and a
    provider is never blamed when its correct candidate was retrieved and later lost."""
    if rec["success"]:
        return {"layer": "ok"}
    if rec["scope_violations"]:
        return {"layer": "isolation"}
    if rec["expected_outcome"] == "no_result":
        consulted = rec["tool_counts"]["semantic_search"] > 0
        return {"layer": "no_relevance_floor" if consulted else "lexical_false_positive"}
    if rec["goal_blocks_relevant"]:
        return {"layer": "goal"}
    rel = set(rec["expected_files"])
    path = "semantic" if rec["expect_semantic"] else "bm25" if rec["expect_bm25"] else None
    probe = rec.get("probes") or {}
    if path in ("semantic", "bm25"):
        top = (probe.get("top") or {}).get(path) or []
        if not top or top[0][0] not in rel:
            return {"layer": "semantic_provider" if path == "semantic" else "lexical_retrieval",
                    "path_first_relevant_rank": (probe.get("first_relevant_rank") or {}).get(path)}
        tool = "semantic_search" if path == "semantic" else "bm25_search"
        if rec["tool_counts"][tool] == 0:
            return {"layer": "agent_decision", "detail": f"{tool} was never consulted although it ranks a relevant file first"}
    cf = rec["counterfactual"]
    if rec["tool_counts"]["hybrid_search"] and cf["without_hybrid"] and cf["without_hybrid"][0] in rel and not (cf["final"] and cf["final"][0] in rel):
        return {"layer": "hybrid"}
    verdict = attribute(rec)["verdict"]
    return {"layer": verdict if verdict in ("reranker", "personalization", "final_ranking", "stopping", "post_ranking") else "other"}


def coverage_sensitivity(records: List[dict], grid=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8)) -> dict:
    """For each candidate 'minority' boundary: how many CORRECT vs WRONG standalone-BM25 top hits would be treated
    as weak lexical evidence. Descriptive only -- shows how much room the boundary has; it does not choose it."""
    pts = []
    for r in records:
        pr = r.get("probes") or {}
        cov = (pr.get("top") or {}).get("bm25_top_coverage")
        top = (pr.get("top") or {}).get("bm25") or []
        if cov is None or not top:
            continue
        pts.append({"qid": r["qid"], "coverage": cov, "correct": top[0][0] in set(r["expected_files"])})
    return {"points": pts, "by_boundary": {str(t): {"correct_flagged_weak": sum(1 for p in pts if p["correct"] and p["coverage"] < t),
                                                    "wrong_flagged_weak": sum(1 for p in pts if not p["correct"] and p["coverage"] < t),
                                                    "n_correct": sum(1 for p in pts if p["correct"]),
                                                    "n_wrong": sum(1 for p in pts if not p["correct"])} for t in grid}}


def semantic_floor_evidence(records: List[dict]) -> dict:
    """Is an absolute semantic-similarity floor supported by THIS provider's data? Compares the raw top-1 cosine of
    queries that have NO relevant file against queries whose semantic top-1 IS relevant. A floor is only defensible
    if the two ranges are disjoint (there is a value that keeps every correct hit and rejects every false one).
    Descriptive evidence only -- it never sets a threshold."""
    no_rel, correct, wrong = [], [], []
    for r in records:
        top = r.get("raw_semantic_top") or []
        if not top:
            continue
        raw = top[0][1]
        if r["expected_outcome"] == "no_result":
            no_rel.append(raw)
        elif top[0][0] in set(r["expected_files"]):
            correct.append(raw)
        else:
            wrong.append(raw)
    rng = lambda xs: [min(xs), max(xs)] if xs else None  # noqa: E731
    overlap = bool(no_rel and correct) and max(no_rel) >= min(correct)
    return {"no_relevant_top1_raw_cosine": rng(no_rel), "correct_top1_raw_cosine": rng(correct), "wrong_top1_raw_cosine": rng(wrong),
            "n": {"no_relevant": len(no_rel), "correct": len(correct), "wrong": len(wrong)},
            "ranges_overlap": overlap, "floor_supported": bool(no_rel and correct) and not overlap}


def raw_semantic_diagnostic(query: str, owner_id: str, label_of: dict) -> dict:
    """DIAGNOSTIC (outside production): the raw cosine of the best chunk per file, which semantic_search
    hides by max-normalising. Same embedding + same owner filter semantic_search uses."""
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    import app.vectorstore as vs
    from app.config import QDRANT_COLLECTION
    from app.embeddings import embed_texts

    vec = embed_texts([query])[0]
    hits = vs.get_client().query_points(QDRANT_COLLECTION, query=vec, limit=200, with_payload=True,
                                        query_filter=Filter(must=[FieldCondition(key="owner_id", match=MatchAny(any=[owner_id]))])).points
    best: Dict[int, float] = {}
    for h in hits:
        fid = h.payload["file_id"]
        best[fid] = max(best.get(fid, -1.0), h.score)
    ranked = sorted(best.items(), key=lambda t: -t[1])
    return {"top": [[label_of.get(f, str(f)), _rounded(s, 4)] for f, s in ranked[:3]]}


# --------------------------------------------------------------------------- one query
def run_one(q: ContentQuery, env: dict, with_raw_semantic: bool = True) -> dict:
    label_of, id_of = env["label_of"], env["id_of"]
    uid = USERS[q.user]
    relevance = {f"{q.user}/{rel}": g for rel, g in q.relevant.items()}
    forbidden = {f"{o}/{rel}" for o, rel in q.forbidden}
    provider = CountingProvider(NullProvider())
    result = run_query(uid, q.query, llm_provider=provider)

    ranking = result.get("final_ranking") or []
    returned = [label_of[r["file_id"]] for r in ranking]
    pool = result.get("candidate_pool") or {}
    pool_files = sorted(label_of[f] for f in pool)
    actions = result.get("committed_actions") or []
    seq = [a["tool"] for a in actions]
    clarified = result.get("status") == "clarify"

    violations = sorted({lab for lab in (set(returned) | set(pool_files))
                         if lab in forbidden or not lab.startswith(q.user + "/")})
    goal = next((t for t in result.get("trace", []) if t.get("event") == "understand_goal"), {}).get("goal") or {}
    ftype = (goal.get("filters") or {}).get("file_type")
    blocks = bool(ftype) and bool(relevance) and all(env["type_of"].get(lab) != ftype for lab in relevance)

    stages = stage_rankings(pool, label_of)
    cf = counterfactual_rankings(pool, label_of)
    # Which RETRIEVAL source scores each candidate highest. The reranker never introduces candidates and its
    # min-max normalisation always hands its own best one 1.0, so it is excluded here (an earlier draft of this
    # analysis included it and mis-attributed false positives to the reranker; corrected before any fix).
    top_src = {}
    for fid, e in pool.items():
        scored = {s: v for s, v in e["scores"].items() if s in RETRIEVAL_SOURCES}
        top_src[label_of[fid]] = max(scored, key=scored.get) if scored else None
    raw_rr = [e["metadata"].get("rerank_raw_score") for e in pool.values() if e["metadata"].get("rerank_raw_score") is not None]

    rel_set = set(relevance)
    m = retrieval_metrics(returned, relevance)
    pm = retrieval_metrics(cf["final"], relevance)
    hr = {lab for lab, g in relevance.items() if g >= 2}
    scope_n = len(violations)
    rec = {
        "qid": q.qid, "user": q.user, "category": q.category, "query": q.query, "variant": "content_eval",
        "expected_outcome": q.outcome, "expect_bm25": q.expect_bm25, "expect_semantic": q.expect_semantic,
        "personalization_relevant": q.personalization,
        "expected_files": sorted(rel_set), "expected_file_ids": sorted(rel_set), "forbidden_files": sorted(forbidden),
        "returned_files": returned, "returned_file_ids": returned, "pool_files": pool_files,
        **{k: (_rounded(v, 4) if k != "first_relevant_rank" else v) for k, v in m.items()},
        "success": judge_success(q.outcome, returned, clarified, relevance, scope_n),
        "status": result.get("status"), "clarified": clarified, "no_result": not returned,
        "degraded_mode": bool(result.get("degraded_mode")), "hard_failure": result.get("status") == "hard_failure",
        "scope_violations": scope_n, "scope_violation_files": violations,
        "decision_basis": result.get("decision_basis"),
        "goal": {"file_type": ftype, "filename_candidate": (goal.get("filters") or {}).get("filename_candidate"),
                 "topic_tokens": goal.get("topic_tokens"), "ambiguity_score": goal.get("ambiguity_score")},
        "goal_blocks_relevant": blocks,
        "expansion_status": (result.get("query_expansion_metadata") or {}).get("expansion_status"),
        "tool_sequence": seq, "iterations": result.get("iteration_count", 0), "tool_calls": len(actions),
        "retrieval_actions": sum(1 for t in seq if t in RETRIEVAL_TOOLS),
        "tool_counts": {t: seq.count(t) for t in TRACKED_TOOLS},
        "llm_calls": provider.total_calls, "llm_attempts": provider.total_attempts,
        "llm_calls_by_stage": {s: provider.calls[s] for s in STAGES},
        "stages": stages,
        "stage_first_relevant_rank": {s: first_rank([lab for lab, _ in v], rel_set) if rel_set else None for s, v in stages.items()},
        "counterfactual": cf,
        "top_evidence_source": top_src,
        "rerank_raw_scores": sorted(_rounded(x, 4) for x in raw_rr),
        "rerank_zero_spread": bool(raw_rr) and len(raw_rr) > 1 and max(raw_rr) == min(raw_rr),
        "pool_ranking_reciprocal_rank": _rounded(pm["reciprocal_rank"], 4),
        "highly_relevant_at_1": (float(bool(returned) and returned[0] in hr)) if hr else None,
        "pool_ranking_highly_relevant_at_1": (float(bool(cf["final"]) and cf["final"][0] in hr)) if hr else None,
        "reason": stable_reason(result.get("reason"), label_of),
    }
    if with_raw_semantic:
        rec["raw_semantic_top"] = raw_semantic_diagnostic(q.query, uid, label_of)["top"]
    rec["probes"] = standalone_probes(q.query, uid, label_of, rel_set)
    rec["attribution"] = attribute(rec)
    rec["layer"] = attribute_layer(rec)
    return rec


# --------------------------------------------------------------------------- whole evaluation
def _group(recs, key):
    out: Dict[str, list] = {}
    for r in recs:
        out.setdefault(r[key], []).append(r)
    return out


def summarize(records: List[dict]) -> dict:
    from collections import Counter

    def attribution_counts(rs):
        return dict(sorted(Counter(r["attribution"]["verdict"] if r["attribution"]["verdict"] == "ok"
                                   else f"{r['attribution']['verdict']}:{r['attribution']['stage']}" for r in rs).items()))

    def layer_counts(rs):
        return dict(sorted(Counter(r["layer"]["layer"] for r in rs).items()))

    return {
        "overall": {"retrieval": retrieval_summary(records), "behavior": behavior_summary(records),
                    "attribution": attribution_counts(records), "layers": layer_counts(records)},
        "by_category": {c: {"retrieval": retrieval_summary(rs), "behavior": behavior_summary(rs),
                            "attribution": attribution_counts(rs), "layers": layer_counts(rs)} for c, rs in sorted(_group(records, "category").items())},
        "coverage_sensitivity": coverage_sensitivity(records),
        "semantic_floor_evidence": semantic_floor_evidence(records),
    }


def provider_info() -> dict:
    info = {}
    try:
        from app.embeddings import get_provider_name
        info["embedding_provider"] = get_provider_name()
    except Exception as exc:  # noqa: BLE001
        info["embedding_provider"] = f"unavailable ({type(exc).__name__})"
    try:
        from app.reranker import get_reranker_provider_name
        info["reranker_provider"] = get_reranker_provider_name()
    except Exception as exc:  # noqa: BLE001
        info["reranker_provider"] = f"unavailable ({type(exc).__name__})"
    info["production_models"] = production_model_status(info)
    return info


def production_model_status(info: dict) -> dict:
    """Is the INTENDED production embedding model / reranker what this run actually used? Derived from the providers the
    environment resolved (no network access, no guessing). When either fell back, production-model quality is
    UNMEASURED here -- nothing in this report may be read as BGE / CrossEncoder performance."""
    from app.embeddings import PREFERRED_MODEL_NAME as EMB
    from app.reranker import PREFERRED_MODEL_NAME as RR

    emb_used, rr_used = str(info.get("embedding_provider")), str(info.get("reranker_provider"))
    emb_ok = emb_used.startswith("sentence-transformers") and EMB in emb_used
    rr_ok = rr_used.startswith("cross") and RR in rr_used
    return {
        "embedding": {"intended": EMB, "used": emb_used, "intended_model_used": emb_ok},
        "reranker": {"intended": RR, "used": rr_used, "intended_model_used": rr_ok},
        "production_model_evaluation": "available" if (emb_ok and rr_ok) else "BLOCKED (environment): results reflect the fallback providers only",
    }


def run_content_eval(queries=None, with_raw_semantic: bool = True) -> dict:
    queries = list(queries if queries is not None else QUERIES)
    with content_environment() as env:
        records = [run_one(q, env, with_raw_semantic) for q in queries]
        diagnostics = [run_one(q, env, with_raw_semantic) for q in DIAGNOSTIC_QUERIES] if queries is QUERIES or len(queries) == len(QUERIES) else []
        ingestion = env["ingestion"]
    return {
        "metadata": {"schema_version": SCHEMA_VERSION, "evaluation": "phase11_real_content (SEPARATE from the Phase 7/8/9 regression benchmark)",
                     "queries": len(queries), "corpus_files": {o: len(f) for o, f in CORPUS.items()}, "ingestion": ingestion,
                     **provider_info(),
                     "provider_note": "a 'lexical_overlap_fallback' reranker is NOT a CrossEncoder and 'spacy:' embeddings are the sandbox fallback; "
                                      "results do not represent production-model quality",
                     "llm": "NullProvider (no LLM); llm_attempts counts stage attempts"},
        "results": records,
        "summary": summarize(records),
        "diagnostics": {"note": "pre-registered, reported separately; NOT part of the 31 frozen queries or any overall metric",
                        "results": diagnostics},
    }


def write_results(output: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
