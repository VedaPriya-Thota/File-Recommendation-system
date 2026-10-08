"""
Phase 2A tests for bm25_search (architecture_final.md §6, §10).

`metadata_search`'s topic_tokens matcher only checks tags/filename/summary
(see metadata_search.py) -- it never looks at `extracted_text`. So the
cleanest, most honest way to prove bm25_search does genuine *lexical*
retrieval (not a metadata_search restatement) is a file whose only textual
evidence lives in `extracted_text`: metadata_search must be unable to find
it via topic tokens; bm25_search must.
"""
import pytest

from app.candidate_pool import update_candidate_pool
from app.db.connection import get_cursor
from app.run import run_query
from app.scope import UserScope, resolve_scope
from app.tools import build_registry
from app.tools.registry import ToolResult

# --- extra seed-adjacent files (Phase 1's seed never populates
# extracted_text, per the Phase 1.1 report -- these add exactly the
# minimum needed to exercise it) --------------------------------------
EXTRA_FILES = [
    dict(
        filename="Internal_File_042.txt",
        file_type="txt",
        file_path="/files/user_a/Internal_File_042.txt",
        owner_id="user_a",
        tags=["internal"],
        summary="Internal reference material.",
        extracted_text=(
            "This is the kubernetes rollback runbook. It covers kubernetes "
            "cluster rollback procedures, kubernetes deployment rollback "
            "steps, and on-call escalation for kubernetes incidents."
        ),
    ),
    dict(
        # Weaker lexical match than the file above (mentions kubernetes
        # once, in passing) -- used for the ranking-order test.
        filename="Internal_File_043.txt",
        file_type="txt",
        file_path="/files/user_a/Internal_File_043.txt",
        owner_id="user_a",
        tags=["internal"],
        summary="Internal reference material.",
        extracted_text="Unrelated quarterly planning notes that briefly mention kubernetes in passing.",
    ),
    dict(
        # xlsx type + older updated_at, for the file_type/date constraint test.
        filename="Internal_File_044.xlsx",
        file_type="xlsx",
        file_path="/files/user_a/Internal_File_044.xlsx",
        owner_id="user_a",
        tags=["internal"],
        summary="Internal reference material.",
        extracted_text="Older kubernetes rollback runbook spreadsheet, superseded by the current one.",
    ),
    dict(
        # Same topic, owned by user_x -- access-control isolation test.
        filename="Internal_File_099.txt",
        file_type="txt",
        file_path="/files/user_x/Internal_File_099.txt",
        owner_id="user_x",
        tags=["internal"],
        summary="Internal reference material.",
        extracted_text=(
            "Confidential kubernetes rollback runbook belonging to user_x. "
            "Must never be visible to user_a or user_b."
        ),
    ),
]


@pytest.fixture(scope="module", autouse=True)
def extra_bm25_files():
    ids = []
    with get_cursor(commit=True) as cur:
        for f in EXTRA_FILES:
            cur.execute(
                """
                INSERT INTO files
                    (filename, file_type, file_path, owner_id, tags, summary, extracted_text)
                VALUES
                    (%(filename)s, %(file_type)s, %(file_path)s, %(owner_id)s,
                     %(tags)s, %(summary)s, %(extracted_text)s)
                RETURNING file_id;
                """,
                f,
            )
            ids.append(cur.fetchone()["file_id"])
    yield
    with get_cursor(commit=True) as cur:
        cur.execute("DELETE FROM files WHERE file_id = ANY(%s);", (ids,))


QUERY = "kubernetes rollback escalation runbook"


# --- A. Genuine BM25 retrieval -----------------------------------------

def test_A_finds_file_whose_only_evidence_is_extracted_text():
    registry = build_registry()
    scope = resolve_scope("user_a")

    metadata_result = registry.execute(
        "metadata_search", {"topic_tokens": QUERY.split()}, scope
    )
    metadata_filenames = {c["metadata"]["filename"] for c in metadata_result.candidates}
    assert "Internal_File_042.txt" not in metadata_filenames, (
        "metadata_search must NOT find this file -- it never checks "
        "extracted_text, only tags/filename/summary"
    )

    bm25_result = registry.execute("bm25_search", {"query": QUERY}, scope)
    bm25_filenames = {c["metadata"]["filename"] for c in bm25_result.candidates}
    assert "Internal_File_042.txt" in bm25_filenames, (
        "bm25_search must find it via extracted_text -- this is genuine "
        "lexical retrieval, not a metadata_search restatement"
    )


# --- B. Ranking ----------------------------------------------------------

def test_B_multiple_matches_ranked_by_bm25_score():
    registry = build_registry()
    scope = resolve_scope("user_a")
    result = registry.execute("bm25_search", {"query": QUERY}, scope)

    scores = [c["score"] for c in result.candidates]
    assert scores == sorted(scores, reverse=True), "candidates must be in descending score order"

    filenames = [c["metadata"]["filename"] for c in result.candidates]
    strong_idx = filenames.index("Internal_File_042.txt")  # mentions all 3 terms repeatedly
    weak_idx = filenames.index("Internal_File_043.txt")  # mentions "kubernetes" once, in passing
    assert strong_idx < weak_idx, "the stronger lexical match must rank above the weaker one"


# --- C. Access control ---------------------------------------------------

def test_C_never_returns_another_users_file():
    registry = build_registry()
    scope = resolve_scope("user_a")
    result = registry.execute("bm25_search", {"query": QUERY}, scope)

    owners = {c["metadata"]["owner_id"] for c in result.candidates}
    filenames = {c["metadata"]["filename"] for c in result.candidates}
    assert owners <= {"user_a"}
    assert "user_x" not in owners
    assert "Internal_File_099.txt" not in filenames  # user_x's file, same topic


def test_C_multi_owner_scope_only_returns_scoped_owners():
    registry = build_registry()
    scope = UserScope(user_id="user_a", owner_ids=("user_a", "user_b"))
    result = registry.execute("bm25_search", {"query": QUERY}, scope)

    owners = {c["metadata"]["owner_id"] for c in result.candidates}
    assert owners <= {"user_a", "user_b"}
    assert "user_x" not in owners


# --- D. Metadata constraints ---------------------------------------------

def test_D_file_type_constraint_is_respected():
    registry = build_registry()
    scope = resolve_scope("user_a")
    result = registry.execute("bm25_search", {"query": QUERY, "file_type": "xlsx"}, scope)

    file_types = {c["metadata"]["file_type"] for c in result.candidates}
    assert file_types <= {"xlsx"}
    filenames = {c["metadata"]["filename"] for c in result.candidates}
    assert "Internal_File_044.xlsx" in filenames
    assert "Internal_File_042.txt" not in filenames  # matches the text, wrong file_type


# --- E. Zero results -----------------------------------------------------

def test_E_no_lexical_evidence_returns_zero_results_safely():
    registry = build_registry()
    scope = resolve_scope("user_a")
    result = registry.execute(
        "bm25_search", {"query": "zzznonexistentxyz122 pineapplesubmarine987"}, scope
    )
    assert result.candidates == []
    assert result.error is None


def test_E_empty_query_returns_zero_results_safely():
    registry = build_registry()
    scope = resolve_scope("user_a")
    result = registry.execute("bm25_search", {"query": ""}, scope)
    assert result.candidates == []
    assert result.error is None


# --- F. Tool registration -------------------------------------------------

def test_F_bm25_search_is_registered_and_executable():
    registry = build_registry()
    assert "bm25_search" in registry.names()
    tool = registry.get("bm25_search")
    assert tool.name == "bm25_search"

    scope = resolve_scope("user_a")
    result = registry.execute("bm25_search", {"query": QUERY}, scope)
    assert isinstance(result, ToolResult)
    assert result.tool == "bm25_search"


# --- G. Candidate pool integration ----------------------------------------

def test_G_bm25_results_enter_the_existing_candidate_pool():
    registry = build_registry()
    scope = resolve_scope("user_a")
    result = registry.execute("bm25_search", {"query": QUERY}, scope)
    assert result.candidates, "fixture setup should guarantee at least one match"

    pool = update_candidate_pool({}, result, iteration=1)
    target = next(c for c in result.candidates if c["metadata"]["filename"] == "Internal_File_042.txt")
    entry = pool[target["file_id"]]
    assert entry["evidence_sources"] == ["bm25_search"]
    assert entry["scores"]["bm25_search"] == target["score"]
    assert entry["metadata"]["filename"] == "Internal_File_042.txt"

    # Accumulation: a second tool's evidence for the same file_id merges in
    # rather than replacing (architecture_final.md §4/§6).
    fake_metadata_result = ToolResult(
        tool="metadata_search",
        candidates=[
            {
                "file_id": target["file_id"],
                "metadata": target["metadata"],
                "score": 0.4,
                "source": "metadata_search",
            }
        ],
    )
    pool = update_candidate_pool(pool, fake_metadata_result, iteration=2)
    entry = pool[target["file_id"]]
    assert set(entry["evidence_sources"]) == {"bm25_search", "metadata_search"}
    assert entry["scores"]["bm25_search"] == target["score"]
    assert entry["scores"]["metadata_search"] == 0.4


# --- H. Genuine agentic transition (observation -> reflection -> bm25) ---

def test_H_agent_transitions_from_metadata_search_to_bm25_on_zero_results():
    """Full-loop integration test: metadata_search finds nothing (this
    query's only evidence is in extracted_text, which metadata_search
    never checks), reflection observes zero results, plan_next_action
    escalates to bm25_search, and bm25_search retrieves real candidates --
    all through run_query(), not by calling bm25_search directly."""
    result = run_query("user_a", "Find details about the kubernetes rollback and escalation runbook")

    tools_used = [a["tool"] for a in result["committed_actions"]]
    assert "metadata_search" in tools_used
    assert "bm25_search" in tools_used
    assert tools_used.index("metadata_search") < tools_used.index("bm25_search"), (
        "metadata_search must be tried and observed to fail BEFORE bm25_search is chosen"
    )

    filenames = [c["metadata"]["filename"] for c in result["candidate_pool"].values()]
    assert "Internal_File_042.txt" in filenames
    assert result["status"] in ("finish", "clarify")


# --- I. Existing regression suite is covered by running the full `pytest`
# suite (test_agent_loop.py, test_metadata_search.py, test_planning.py,
# test_reflect.py, test_understand_goal.py) -- nothing here duplicates it.
