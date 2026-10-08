"""
Phase 11 -- tests for the real-content evaluation (evaluation/content/), kept SEPARATE from the Phase 7/8/9
regression benchmark. Retrieval QUALITY numbers are deliberately not asserted (they depend on the embedding
model and reranker the environment resolves); what is asserted is that the evaluation is honest, deterministic,
isolated, correctly attributes failures, and that the proven fix (goal) holds on real content.
"""
import copy
import json
import re
from pathlib import Path

import pytest

from app.final_ranking import PERSONALIZATION_SOURCE
from app.goal import FILE_TYPE_KEYWORDS
from app.ingestion import ContentExtractor
from app.tools.registry import ToolResult

from evaluation.content import runner
from evaluation.content.corpus_spec import ALICE, BOB, CORPUS
from evaluation.content.queries import CATEGORIES, QUERIES, TELEMETRY, validate
from evaluation.content.report import render_markdown

CORPUS_DIR = Path(runner.CORPUS_DIR)
tok = lambda s: set(re.findall(r"[a-z0-9]+", s.lower()))  # noqa: E731
STOP = {"the", "a", "an", "of", "in", "on", "at", "for", "and", "to", "my", "is", "are", "from", "with", "when", "about", "how"}
Q = {q.qid: q for q in QUERIES}


# ============================================================ static: corpus + ground truth integrity
def test_query_set_is_valid_and_covers_every_category():
    assert validate(CORPUS) == []
    assert len(QUERIES) == 31 and {q.category for q in QUERIES} == set(CATEGORIES)
    assert len({q.qid for q in QUERIES}) == 31


def test_committed_corpus_files_are_exactly_the_spec():
    for owner, files in CORPUS.items():
        on_disk = sorted(p.name for p in (CORPUS_DIR / owner).iterdir())
        assert on_disk == sorted(files), owner                                   # no stray or missing files
        for name, text in files.items():
            ext = name.rsplit(".", 1)[1]
            got = ContentExtractor().extract(ext, (CORPUS_DIR / owner / name).read_bytes())
            assert got.status == "ok", name
            assert " ".join(got.text.split()) == " ".join(text.replace("# ", "# ").split()), name


def test_ground_truth_comes_only_from_fixture_files_of_the_querying_user():
    for q in QUERIES:
        assert all(rel in CORPUS[q.user] for rel in q.relevant)
        assert all(owner != q.user and rel in CORPUS[owner] for owner, rel in q.forbidden)
    assert set(TELEMETRY["alice"]["events"]) <= set(ALICE)


# ============================================================ static: the DESIGN INTENT of each category
def test_semantic_queries_were_designed_without_lexical_overlap_with_their_target():
    for qid in ("S1", "S2", "S3", "S4", "S5", "S6"):
        q = Q[qid]
        target = next(iter(q.relevant))
        assert not ((tok(q.query) - STOP) & (tok(ALICE[target]) | tok(target))), qid
    for qid in ("S1", "S2", "S3", "S4", "S5"):   # S6 deliberately shares `sunlight` with a distractor (a lexical trap)
        assert not any((tok(Q[qid].query) - STOP) & (tok(t) | tok(f)) for f, t in ALICE.items()), qid
    assert (tok(Q["S6"].query) - STOP) & tok(ALICE["plant_biology_primer.md"]) == {"sunlight"}


def test_content_queries_overlap_the_body_but_never_the_filename():
    for qid in ("C1", "C2", "C3", "C4"):
        q = Q[qid]
        target = next(iter(q.relevant))
        assert (tok(q.query) - STOP) & tok(ALICE[target]) and not ((tok(q.query) - STOP) & tok(target)), qid


def test_goal_probes_each_contain_a_bare_substring_keyword_but_target_a_different_type():
    for qid in ("G1", "G2", "G3", "G4", "G5"):
        q = Q[qid]
        triggered = {ft for ft, kws in FILE_TYPE_KEYWORDS.items() if any(kw in q.query.lower() for kw in kws)}
        target_ext = next(iter(q.relevant)).rsplit(".", 1)[1]
        assert triggered and target_ext not in triggered, (qid, triggered, target_ext)


def test_no_result_queries_have_no_lexical_overlap_with_any_document():
    for qid in ("N1", "N2", "N3", "X1"):
        qt = tok(Q[qid].query) - STOP
        assert not any(qt & (tok(t) | tok(f)) for f, t in ALICE.items()), qid


# ============================================================ pure helpers: stages, counterfactuals, attribution
def _entry(fid, **scores):
    return {"file_id": fid, "scores": dict(scores), "metadata": {"filename": f"f{fid}"}, "evidence_sources": list(scores)}


LABELS = {1: "alice/a.txt", 2: "alice/b.txt", 3: "alice/c.txt"}


def test_stage_rankings_order_each_source_independently_with_label_tiebreak():
    pool = {1: _entry(1, bm25_search=0.4, semantic_search=0.9), 2: _entry(2, bm25_search=0.8, semantic_search=0.9),
            3: _entry(3, semantic_search=0.5)}
    st = runner.stage_rankings(pool, LABELS)
    assert [l for l, _ in st["bm25_search"]] == ["alice/b.txt", "alice/a.txt"]            # candidate 3 has no bm25 score
    assert [l for l, _ in st["semantic_search"]] == ["alice/a.txt", "alice/b.txt", "alice/c.txt"]  # tie -> label order
    assert st["hybrid_search"] == [] and st["rerank_candidates"] == []


def test_counterfactual_removes_evidence_from_a_copy_and_never_mutates_the_pool():
    # reranker (zero-spread style: everything 1.0) erases the semantic ordering; without it the ordering is intact
    pool = {1: _entry(1, semantic_search=0.6, rerank_candidates=1.0), 2: _entry(2, semantic_search=1.0, rerank_candidates=1.0),
            3: _entry(3, semantic_search=0.8, rerank_candidates=1.0)}
    snapshot = copy.deepcopy(pool)
    cf = runner.counterfactual_rankings(pool, LABELS)
    assert pool == snapshot
    assert cf["without_rerank"] == ["alice/b.txt", "alice/c.txt", "alice/a.txt"]            # semantic order
    assert cf["final"] == ["alice/a.txt", "alice/b.txt", "alice/c.txt"]                     # all tied at 1.0 -> file_id order
    assert cf["retrieval_only"] == cf["without_rerank"]


def test_counterfactual_personalization_removal():
    pool = {1: _entry(1, bm25_search=0.5, **{PERSONALIZATION_SOURCE: 0.0}), 2: _entry(2, bm25_search=0.5, **{PERSONALIZATION_SOURCE: 0.9})}
    cf = runner.counterfactual_rankings(pool, LABELS)
    assert cf["final"][0] == "alice/b.txt" and cf["without_personalization"][0] == "alice/a.txt"


def _rec(**kw):
    base = {"success": False, "scope_violations": 0, "expected_files": ["alice/a.txt"], "expected_outcome": "results",
            "returned_files": ["alice/c.txt"], "pool_files": ["alice/a.txt", "alice/c.txt"], "goal_blocks_relevant": False,
            "expect_bm25": False, "expect_semantic": True, "top_evidence_source": {"alice/c.txt": "semantic_search", "alice/a.txt": "semantic_search"},
            "counterfactual": {"final": ["alice/c.txt", "alice/a.txt"], "retrieval_only": ["alice/c.txt", "alice/a.txt"],
                               "without_rerank": ["alice/c.txt", "alice/a.txt"], "without_personalization": ["alice/c.txt", "alice/a.txt"]}}
    base.update(kw)
    return base


def test_attribution_distinguishes_retrieval_from_ranking_failures():
    A = runner.attribute
    assert A(_rec(success=True)) == {"verdict": "ok", "stage": None}
    assert A(_rec(scope_violations=1))["verdict"] == "isolation"
    assert A(_rec(goal_blocks_relevant=True)) == {"verdict": "goal", "stage": "goal"}
    assert A(_rec(pool_files=["alice/c.txt"])) == {"verdict": "retrieval_miss", "stage": "semantic"}
    assert A(_rec(pool_files=["alice/c.txt"], expect_bm25=True, expect_semantic=False))["stage"] == "bm25"
    assert A(_rec(pool_files=["alice/c.txt"], expect_bm25=False, expect_semantic=False))["stage"] == "metadata"
    # semantic -> wrong : a RETRIEVAL ordering failure
    assert A(_rec()) == {"verdict": "retrieval_ordering", "stage": "semantic_search"}


def test_attribution_reranker_vs_personalization_vs_final_ranking_vs_stopping():
    A = runner.attribute
    right = ["alice/a.txt", "alice/c.txt"]
    wrong = ["alice/c.txt", "alice/a.txt"]
    # retrieval correct, adding rerank evidence breaks it -> reranker
    assert A(_rec(counterfactual={"final": wrong, "retrieval_only": right, "without_rerank": right, "without_personalization": wrong}))["verdict"] == "reranker"
    # only personalization breaks it
    assert A(_rec(counterfactual={"final": wrong, "retrieval_only": right, "without_rerank": wrong, "without_personalization": right}))["verdict"] == "personalization"
    # neither explains it
    assert A(_rec(counterfactual={"final": wrong, "retrieval_only": right, "without_rerank": wrong, "without_personalization": wrong}))["verdict"] == "final_ranking"
    # the final ranking WOULD be right, but the agent returned nothing (clarified)
    assert A(_rec(returned_files=[], counterfactual={"final": right, "retrieval_only": right, "without_rerank": right, "without_personalization": right}))["verdict"] == "stopping"
    # ranking right, returned result differs (5C/5D)
    assert A(_rec(counterfactual={"final": right, "retrieval_only": right, "without_rerank": right, "without_personalization": right}))["verdict"] == "post_ranking"


def test_attribution_of_false_positives_names_the_retrieval_source_not_the_reranker():
    rec = _rec(expected_outcome="no_result", expected_files=[], returned_files=["alice/c.txt"],
               top_evidence_source={"alice/c.txt": "semantic_search"})
    assert runner.attribute(rec) == {"verdict": "false_positive", "stage": "semantic_search"}


# ============================================================ end-to-end evaluation run (real DB, real ingestion, real agent)
@pytest.fixture(scope="module")
def evaluation_output():
    return runner.run_content_eval()


def by_id(out):
    return {r["qid"]: r for r in out["results"]}


def test_evaluation_completes_with_the_documented_schema_and_providers(evaluation_output):
    out = evaluation_output
    assert out["metadata"]["queries"] == 31 and len(out["results"]) == 31
    assert out["metadata"]["corpus_files"] == {"alice": len(ALICE), "bob": len(BOB)}
    assert "NOT a CrossEncoder" in out["metadata"]["provider_note"] or "lexical_overlap_fallback" in out["metadata"]["provider_note"]
    assert out["metadata"]["embedding_provider"] and out["metadata"]["reranker_provider"]
    required = {"qid", "query", "category", "user", "expected_files", "returned_files", "first_relevant_rank", "precision_at_1", "precision_at_3",
                "recall_at_3", "reciprocal_rank", "success", "clarified", "no_result", "tool_sequence", "iterations", "tool_counts",
                "llm_calls", "goal", "goal_blocks_relevant", "expansion_status", "stages", "stage_first_relevant_rank",
                "counterfactual", "raw_semantic_top", "attribution"}
    for r in out["results"]:
        assert required <= set(r), required - set(r)
        assert set(r["stages"]) == set(runner.STAGE_SOURCES) and set(r["counterfactual"]) == {"final", "without_rerank", "without_personalization", "retrieval_only", "without_hybrid"}
    assert {"overall", "by_category"} <= set(out["summary"]) and set(out["summary"]["by_category"]) == set(CATEGORIES)
    md = render_markdown(out)
    assert "stage attribution" in md and "NOT a CrossEncoder" in md
    json.dumps(out, default=str)


def test_all_ingested_files_were_indexed_through_the_real_pipeline(evaluation_output):
    ing = evaluation_output["metadata"]["ingestion"]
    assert ing["alice"]["failed"] == 0 and ing["alice"]["new"] == len(ALICE) and ing["alice"]["indexed"] == len(ALICE)
    assert ing["bob"]["failed"] == 0 and ing["bob"]["indexed"] == len(BOB)


def test_cross_user_isolation_holds_for_every_query(evaluation_output):
    for r in evaluation_output["results"]:
        assert r["scope_violations"] == 0, (r["qid"], r["scope_violation_files"])
        assert all(f.startswith(r["user"] + "/") for f in r["returned_files"] + r["pool_files"]), r["qid"]
        assert not set(r["returned_files"]) & set(r["forbidden_files"]), r["qid"]
    x = by_id(evaluation_output)
    assert x["X2"]["returned_files"] == ["alice/garden_journal.md"]                         # bob's near-identical diary never appears
    assert "bob/bob_garden_diary.txt" not in x["X2"]["pool_files"] and "alice/garden_journal.md" not in x["X3"]["pool_files"]
    assert x["X4"]["returned_files"] == ["bob/quarterly_budget_review.txt"]                  # same filename, own file only


def test_lexical_paths_work_on_real_content(evaluation_output):
    x = by_id(evaluation_output)
    for qid in ("L1", "L2", "L3", "L4", "C1", "C2", "C3", "C4"):
        assert x[qid]["success"], qid
        assert x[qid]["returned_files"][0] == "alice/" + next(iter(Q[qid].relevant)), qid
    for qid in ("C1", "C2", "C3", "C4"):                                                     # content-only -> BM25 found it, metadata could not
        assert x[qid]["tool_sequence"] == ["metadata_search", "bm25_search"]
        assert x[qid]["stage_first_relevant_rank"]["bm25_search"] == 1


def test_goal_probes_all_succeed_after_the_whole_token_fix(evaluation_output):
    x = by_id(evaluation_output)
    for qid in ("G1", "G2", "G3", "G4", "G5"):
        assert x[qid]["success"], qid
        assert x[qid]["goal"]["file_type"] is None and not x[qid]["goal_blocks_relevant"], qid
    assert x["G4"]["returned_files"] == ["alice/embedding_primer.md"]                        # the Phase 10 'representations' case


def test_every_attribution_is_consistent_with_the_outcome(evaluation_output):
    for r in evaluation_output["results"]:
        verdict = r["attribution"]["verdict"]
        assert (verdict == "ok") == r["success"], r["qid"]
        if verdict == "goal":
            assert r["goal_blocks_relevant"]
    stages = {r["attribution"]["stage"] for r in evaluation_output["results"] if r["attribution"]["verdict"] == "false_positive"}
    assert "rerank_candidates" not in stages                                                  # the reranker never introduces candidates


def test_no_hard_failures_no_llm_calls_and_nothing_left_behind(evaluation_output):
    assert not any(r["hard_failure"] for r in evaluation_output["results"])
    assert all(r["llm_calls"] == 0 for r in evaluation_output["results"])                    # NullProvider: offline, deterministic
    from app.db.connection import get_cursor
    with get_cursor() as cur:
        cur.execute("SELECT (SELECT count(*) FROM files) f, (SELECT count(*) FROM file_ingestion_state) s, "
                    "(SELECT count(*) FROM users WHERE user_id LIKE 'p11_%%') u, (SELECT count(*) FROM user_file_telemetry) t;")
        assert dict(cur.fetchone()) == {"f": 13, "s": 0, "u": 0, "t": 0}                     # only the Phase 7 seed corpus remains


def test_evaluation_is_deterministic(evaluation_output):
    again = runner.run_content_eval()
    assert [r for r in again["results"]] == [r for r in evaluation_output["results"]]
    assert again["summary"] == evaluation_output["summary"]


def test_personalization_telemetry_fixture_is_applied_to_the_right_owner_only():
    assert set(TELEMETRY) == {"alice"}                                                        # bob stays cold-start


# ============================================================ documented limitation (kept visible, not hidden)
@pytest.mark.xfail(strict=True, reason="KNOWN, NOT FIXED in Phase 11 (see evaluation/content/output/experiments/): when a reranker scores "
                                       "every candidate identically, rerank_candidates normalises ALL of them to 1.0 and final ranking's max() "
                                       "merge then erases the retrieval ordering. The fix is tiny and tested but changes `confidence` in 3 of "
                                       "the 310 old-benchmark records, so it needs an explicit decision. strict=True: when it is fixed this "
                                       "test XPASSes and must be updated.")
def test_zero_spread_reranker_should_not_lift_every_candidate_to_full_score(monkeypatch):
    from app.tools import rerank_candidates as rc
    cands = [{"file_id": i, "score": 0.5, "source": "semantic_search", "metadata": {"filename": f"f{i}", "chunk_text": "x"}} for i in (1, 2, 3)]
    monkeypatch.setattr(rc, "score_pairs", lambda q, texts: [0.0] * len(texts), raising=False)
    res = rc.rerank_candidates({"query": "anything", "candidates": cands}, None)
    assert not all(c["score"] == 1.0 for c in res.candidates)
