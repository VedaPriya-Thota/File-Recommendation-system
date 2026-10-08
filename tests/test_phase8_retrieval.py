"""
Phase 8 -- retrieval-correctness regression tests:
  * exact-filename requests are never relaxed into unrelated same-extension hits
  * very short metadata tokens no longer match by substring coincidence

Planner behaviour is tested directly with synthetic goals (no corpus
dependence); end-to-end behaviour uses the seed corpus generically.
"""
import pytest

from app.config import METADATA_SHORT_TOKEN_MAX_LEN
from app.goal import understand_goal
from app.llm.provider import NullProvider
from app.planning import _candidate_strategies, is_exact_filename_request, plan_next_action
from app.run import run_query
from app.scope import resolve_scope
from app.tools.metadata_search import _score_row, _token_in_text, metadata_search


def goal_for(query):
    g = understand_goal(query).to_dict()
    g["expanded_terms"] = []
    return g


# ===========================================================================
# Exact-filename requests
# ===========================================================================
@pytest.mark.parametrize("query,exact", [
    ("Quarterly_Report_Final.xlsx", True),
    ("find Project-Plan_v2.docx please", True),
    ("open notes.pdf", True),
    ("file named Meeting_Notes_January", False),         # hint, no extension
    ("the vendor budget estimate", False),                # topic words
    ("budget spreadsheet from last week", False),
])
def test_exact_filename_detection_is_generic(query, exact):
    assert is_exact_filename_request(goal_for(query)) is exact


def test_llm_suggested_filename_never_becomes_a_hard_constraint():
    g = goal_for("that budget thing")
    g["filters"]["filename_candidate"] = "Invented_By_LLM.xlsx"  # as a 5A goal merge might supply
    assert is_exact_filename_request(g) is False  # raw query contains no such filename


def test_exact_filename_is_kept_in_every_relaxation_level():
    g = goal_for("Quarterly_Report_Final.xlsx")
    strategies = _candidate_strategies(g)
    assert len(strategies) >= 2  # file_type can still be relaxed
    assert all(s.get("filename_contains") == "Quarterly_Report_Final.xlsx" for s in strategies)


def test_hint_style_filename_is_still_relaxable():
    g = goal_for("file named Quarterly_Report")
    strategies = _candidate_strategies(g)
    assert any("filename_contains" not in s for s in strategies)  # unchanged legacy behaviour


def test_exact_filename_miss_does_not_escalate_to_free_text_search():
    g = goal_for("Quarterly_Report_Final.xlsx")
    committed, failed = [], []
    # walk the whole metadata ladder as zero-results
    while True:
        a = plan_next_action(g, committed, failed, len(committed) + 1, observation_hint="zero_results" if committed else None)
        if a is None:
            break
        assert a["tool"] == "metadata_search", f"escalated to {a['tool']}"
        committed.append(a)
        failed.append(__import__("app.planning", fromlist=["args_signature"]).args_signature(a["tool"], a["args"]))
    assert committed  # it did try


def test_hint_style_filename_miss_still_escalates():
    g = goal_for("file named Quarterly_Report budget")
    committed, failed, seen = [], [], []
    from app.planning import args_signature
    while True:
        a = plan_next_action(g, committed, failed, len(committed) + 1, observation_hint="zero_results" if committed else None)
        if a is None:
            break
        seen.append(a["tool"])
        committed.append(a)
        failed.append(args_signature(a["tool"], a["args"]))
    assert "bm25_search" in seen  # legitimate broader behaviour preserved


def test_exact_filename_miss_end_to_end_returns_nothing_and_never_unrelated_same_extension():
    # A name that exists for no one, and the name of ANOTHER user's file, both with an extension that
    # user_a does own files of (xlsx): previously degraded into "any xlsx whose name shares a token".
    for query in ("Nonexistent_Quarterly_Forecast.xlsx", "Confidential_HR_Salaries.xlsx"):
        r = run_query("user_a", query, llm_provider=NullProvider())
        assert r["status"] == "finish"
        assert r["final_ranking"] == [], (query, [e["metadata"]["filename"] for e in r["final_ranking"]])
        assert r["degraded_mode"] is True
        assert {a["tool"] for a in r["committed_actions"]} == {"metadata_search"}


def test_exact_filename_hit_and_broader_queries_still_work():
    names = lambda r: [e["metadata"]["filename"] for e in r["final_ranking"]]  # noqa: E731
    assert names(run_query("user_a", "Q3_Budget_Review.xlsx", llm_provider=NullProvider())) == ["Q3_Budget_Review.xlsx"]
    assert names(run_query("user_a", "file named Meeting_Notes_January", llm_provider=NullProvider()))[0] == "Meeting_Notes_January.docx"
    assert names(run_query("user_a", "spreadsheet with department spend breakdown", llm_provider=NullProvider()))[0] == "Q3_Budget_Review.xlsx"
    # case-insensitive filename match is preserved
    assert names(run_query("user_a", "q3_budget_review.XLSX", llm_provider=NullProvider())) == ["Q3_Budget_Review.xlsx"]


# ===========================================================================
# Very short tokens
# ===========================================================================
@pytest.mark.parametrize("token,text,expected", [
    ("ai", "marketing plan slide deck for the spring campaign.", False),   # the Phase 7 false positive
    ("ai", "ai_architecture_overview.pptx", True),                         # underscore delimits
    ("ai", "project 5 ai architecture deck", True),
    ("ai", "ai-powered search", True),
    ("ai", "explain it again", False),
    ("hr", "three quarters", False),
    ("hr", "confidential_hr_salaries.xlsx", True),
    ("q3", "q3_budget_review.xlsx", True),
    ("q3", "q33_review", False),
    ("AI", "AI_Architecture", True),                                       # case-insensitive
    ("budget", "q2 budgets", True),                                        # long tokens keep substring matching
    ("budg", "budget", True),
    ("plan", "planning meeting", True),
])
def test_token_matching_semantics(token, text, expected):
    assert _token_in_text(token, text) is expected


def test_short_token_threshold_is_the_documented_default():
    assert METADATA_SHORT_TOKEN_MAX_LEN == 3


def _names(tokens, user, **extra):
    res = metadata_search({"topic_tokens": tokens, **extra}, resolve_scope(user))
    return [c["metadata"]["filename"] for c in res.candidates], res


def test_ai_does_not_match_campaign_but_does_match_real_ai_metadata():
    names_b, _ = _names(["ai"], "user_b")          # user_b owns Marketing_Plan.pptx ("...spring campaign.")
    assert names_b == []
    names_a, _ = _names(["ai"], "user_a")          # user_a owns AI_Architecture_Overview.pptx
    assert names_a == ["AI_Architecture_Overview.pptx"]


def test_multi_token_searches_still_work_with_a_mixed_short_and_long_token_set():
    names, _ = _names(["ai", "marketing"], "user_b")
    assert names == ["Marketing_Plan.pptx"]        # matched via the long token only
    # Match strength must agree with the SQL semantics. Tagless row so only text evidence counts:
    # 'marketing' earns credit, the coincidental 'ai' (inside "campaign") earns none -> 1 of 2 tokens.
    # (Pre-fix substring scoring would have given 1.0 here.)
    row = {"filename": "Marketing_Plan.pptx", "summary": "Marketing plan slide deck for the spring campaign.", "tags": []}
    assert _score_row(row, {"topic_tokens": ["ai", "marketing"]}) == pytest.approx(0.5)
    names, _ = _names(["budget", "q3"], "user_a")
    assert "Q3_Budget_Review.xlsx" in names


def test_score_row_never_credits_a_coincidental_short_token():
    row = {"filename": "Marketing_Plan.pptx", "summary": "Marketing plan slide deck for the spring campaign.", "tags": []}
    assert _score_row(row, {"topic_tokens": ["ai"]}) == 0.0
    assert _score_row(row, {"topic_tokens": ["marketing"]}) == 1.0


def test_short_tokens_remain_scoped_to_the_callers_files():
    names_x, _ = _names(["hr"], "user_x")          # user_x owns Confidential_HR_Salaries.xlsx
    assert names_x == ["Confidential_HR_Salaries.xlsx"]
    names_a, _ = _names(["hr"], "user_a")
    assert names_a == []                            # never leaks user_x's file


def test_partial_filename_search_with_filename_contains_is_unchanged():
    res = metadata_search({"filename_contains": "Budget_Rev"}, resolve_scope("user_a"))
    assert [c["metadata"]["filename"] for c in res.candidates] == ["Q3_Budget_Review.xlsx"]
