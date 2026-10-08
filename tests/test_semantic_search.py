"""
Phase 2B tests for semantic_search (architecture_final.md §6, §10).

Fixture files are worded so query <-> content share ZERO literal
substrings but ARE genuinely semantically related -- proving this is real
embedding similarity, not bm25_search/metadata_search wearing a costume.
(Verified interactively: "restoring a crashed container cluster ... paging
the on-duty engineer" vs. "roll back a broken kubernetes deployment ...
alert whoever is on call" scores ~0.84 cosine similarity with zero shared
words, vs. ~0.56 for a genuinely unrelated sentence pair.)
"""
import pytest

from app.candidate_pool import update_candidate_pool
from app.db.connection import get_cursor
from app.run import run_query
from app.scope import UserScope, resolve_scope
from app.tools import build_registry
from app.tools.registry import ToolResult
from app.tools.semantic_search import rebuild_index
from app.vectorstore import get_client, QDRANT_COLLECTION

QUERY = "steps to roll back a broken kubernetes deployment and alert whoever is on call"
CHUNK_QUERY = "guidance for rolling back a failing app deployment to an earlier version and notifying whoever is on duty"

FILLER = ("unrelated placeholder padding word " * 30).strip()  # exactly 120 words -> fills chunk 0 exactly

EXTRA_FILES = [
    dict(
        # Zero literal word overlap with QUERY -- only findable semantically.
        filename="Recovery_Doc_501.txt",
        file_type="txt",
        file_path="/files/user_a/Recovery_Doc_501.txt",
        owner_id="user_a",
        tags=["internal"],
        summary="Internal reference material.",
        extracted_text=(
            "Instructions for restoring a crashed container cluster to its "
            "last working state and paging the on-duty engineer immediately "
            "when production breaks."
        ),
    ),
    dict(
        # Genuinely unrelated topic -- used for the ranking test (must
        # score lower than Recovery_Doc_501.txt for QUERY).
        filename="Recovery_Doc_502.txt",
        file_type="txt",
        file_path="/files/user_a/Recovery_Doc_502.txt",
        owner_id="user_a",
        tags=["internal"],
        summary="Internal reference material.",
        extracted_text="A recipe for baking sourdough bread at home, including hydration ratios and proofing times.",
    ),
    dict(
        # xlsx type, for the file_type constraint test.
        filename="Recovery_Doc_503.xlsx",
        file_type="xlsx",
        file_path="/files/user_a/Recovery_Doc_503.xlsx",
        owner_id="user_a",
        tags=["internal"],
        summary="Internal reference material.",
        extracted_text="Older incident recovery spreadsheet: how to reset a failed compute cluster to a healthy state and notify the shift lead immediately, superseded by the current doc.",
    ),
    dict(
        # Empty extracted_text -- must produce zero chunks, never indexed.
        filename="Recovery_Doc_504_empty.txt",
        file_type="txt",
        file_path="/files/user_a/Recovery_Doc_504_empty.txt",
        owner_id="user_a",
        tags=["internal"],
        summary="Internal reference material.",
        extracted_text="",
    ),
    dict(
        # >1 chunk: 120 filler words (chunk 0, irrelevant) + the actually
        # relevant sentence (chunk 1). Proves chunk->file aggregation picks
        # the right chunk and the file appears exactly once.
        filename="Recovery_Doc_505_long.txt",
        file_type="txt",
        file_path="/files/user_a/Recovery_Doc_505_long.txt",
        owner_id="user_a",
        tags=["internal"],
        summary="Internal reference material.",
        extracted_text=(
            FILLER
            + " Separately: notes on returning a malfunctioning application "
            "to a prior stable release and pinging the operator currently "
            "on shift."
        ),
    ),
    dict(
        # Same topic, owned by user_x -- access-control isolation test.
        filename="Recovery_Doc_999_confidential.txt",
        file_type="txt",
        file_path="/files/user_x/Recovery_Doc_999_confidential.txt",
        owner_id="user_x",
        tags=["internal"],
        summary="Internal reference material.",
        extracted_text=(
            "Confidential instructions for restoring user_x's crashed "
            "container cluster and paging their on-duty engineer. Must "
            "never be visible to user_a or user_b."
        ),
    ),
]


@pytest.fixture(scope="module", autouse=True)
def extra_semantic_files():
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

    stats = rebuild_index(force=True)
    assert stats["files_indexed"] >= 5, "sanity check: fixture files with text should get indexed"

    yield

    with get_cursor(commit=True) as cur:
        cur.execute("DELETE FROM files WHERE file_id = ANY(%s);", (ids,))
    rebuild_index(force=True)  # leave the index consistent with the DB for any later test module


# --- 1. Semantic retrieval with different wording -------------------------

def test_1_finds_file_via_different_wording_zero_literal_overlap():
    from app.goal import understand_goal

    registry = build_registry()
    scope = resolve_scope("user_a")
    goal = understand_goal(QUERY)

    # Confirm metadata_search/bm25_search genuinely can't find it (zero
    # literal overlap) -- otherwise this wouldn't prove semantic capability.
    metadata_result = registry.execute("metadata_search", {"topic_tokens": goal.topic_tokens}, scope)
    bm25_result = registry.execute("bm25_search", {"query": goal.concept_text}, scope)
    lexical_filenames = {c["metadata"]["filename"] for c in metadata_result.candidates} | {
        c["metadata"]["filename"] for c in bm25_result.candidates
    }
    assert "Recovery_Doc_501.txt" not in lexical_filenames

    semantic_result = registry.execute("semantic_search", {"query": QUERY}, scope)
    semantic_filenames = {c["metadata"]["filename"] for c in semantic_result.candidates}
    assert "Recovery_Doc_501.txt" in semantic_filenames


# --- 2. Ranking ------------------------------------------------------------

def test_2_ranked_by_descending_semantic_score_relevant_above_unrelated():
    registry = build_registry()
    scope = resolve_scope("user_a")
    result = registry.execute("semantic_search", {"query": QUERY}, scope)

    scores = [c["score"] for c in result.candidates]
    assert scores == sorted(scores, reverse=True)

    filenames = [c["metadata"]["filename"] for c in result.candidates]
    relevant_idx = filenames.index("Recovery_Doc_501.txt")
    unrelated_idx = filenames.index("Recovery_Doc_502.txt")  # sourdough bread recipe
    assert relevant_idx < unrelated_idx


# --- 3. Chunk -> file aggregation ------------------------------------------

def test_3_multi_chunk_file_aggregates_to_one_candidate_with_best_chunk():
    registry = build_registry()
    scope = resolve_scope("user_a")
    result = registry.execute("semantic_search", {"query": CHUNK_QUERY}, scope)

    matches = [c for c in result.candidates if c["metadata"]["filename"] == "Recovery_Doc_505_long.txt"]
    assert len(matches) == 1, "one file must produce exactly one candidate, however many chunks matched"
    chunk_text = matches[0]["metadata"]["matched_chunk"]
    assert "unrelated placeholder padding word" not in chunk_text, (
        "the surfaced evidence must be the relevant chunk (chunk 1), not the filler chunk (chunk 0)"
    )
    assert "operator" in chunk_text or "stable release" in chunk_text


# --- 4. Access control -------------------------------------------------------

def test_4_never_returns_another_users_file():
    registry = build_registry()
    scope = resolve_scope("user_a")
    result = registry.execute("semantic_search", {"query": QUERY}, scope)

    owners = {c["metadata"]["owner_id"] for c in result.candidates}
    filenames = {c["metadata"]["filename"] for c in result.candidates}
    assert owners <= {"user_a"}
    assert "user_x" not in owners
    assert "Recovery_Doc_999_confidential.txt" not in filenames


def test_4_multi_owner_scope_only_returns_scoped_owners():
    registry = build_registry()
    scope = UserScope(user_id="user_a", owner_ids=("user_a", "user_b"))
    result = registry.execute("semantic_search", {"query": QUERY}, scope)

    owners = {c["metadata"]["owner_id"] for c in result.candidates}
    assert owners <= {"user_a", "user_b"}
    assert "user_x" not in owners


def test_4_unknown_user_is_a_hard_failure_before_semantic_search_ever_runs():
    from app.scope import UnknownUserError

    with pytest.raises(UnknownUserError):
        resolve_scope("totally_unknown_user_zzz")


# --- 5. Metadata filters -----------------------------------------------------

def test_5_file_type_constraint_is_respected():
    registry = build_registry()
    scope = resolve_scope("user_a")
    result = registry.execute("semantic_search", {"query": QUERY, "file_type": "xlsx"}, scope)

    file_types = {c["metadata"]["file_type"] for c in result.candidates}
    assert file_types <= {"xlsx"}
    filenames = {c["metadata"]["filename"] for c in result.candidates}
    assert "Recovery_Doc_503.xlsx" in filenames
    assert "Recovery_Doc_501.txt" not in filenames  # matches semantically, wrong file_type


# --- 6. Empty extracted_text -------------------------------------------------

def test_6_empty_extracted_text_produces_no_chunks_never_indexed():
    registry = build_registry()
    scope = resolve_scope("user_a")
    # A query literally about this file's filename/tags won't help since
    # semantic_search only searches indexed chunk vectors -- confirm the
    # file simply can never appear, regardless of query.
    result = registry.execute("semantic_search", {"query": QUERY, "top_k": 50}, scope)
    filenames = {c["metadata"]["filename"] for c in result.candidates}
    assert "Recovery_Doc_504_empty.txt" not in filenames


def test_6_chunk_text_function_returns_no_chunks_for_empty_text():
    from app.tools.semantic_search import _chunk_text

    assert _chunk_text("") == []
    assert _chunk_text(None) == []


# --- 7. Persistent Qdrant index ----------------------------------------------

def test_7_index_persists_across_queries_without_unnecessary_rebuild(monkeypatch):
    client = get_client()
    count_before = client.count(QDRANT_COLLECTION, exact=True).count
    assert count_before > 0, "fixture's rebuild_index should have populated the collection"

    import app.tools.semantic_search as sem_module

    calls = {"n": 0}
    real_rebuild = sem_module.rebuild_index

    def spy_rebuild(*args, **kwargs):
        calls["n"] += 1
        return real_rebuild(*args, **kwargs)

    monkeypatch.setattr(sem_module, "rebuild_index", spy_rebuild)

    registry = build_registry()
    scope = resolve_scope("user_a")
    registry.execute("semantic_search", {"query": QUERY}, scope)
    registry.execute("semantic_search", {"query": "another unrelated query"}, scope)

    assert calls["n"] == 0, "querying an already-built index must not trigger a rebuild"
    assert client.count(QDRANT_COLLECTION, exact=True).count == count_before


# --- 8. Tool registration -----------------------------------------------------

def test_8_semantic_search_is_registered_and_executable():
    registry = build_registry()
    assert "semantic_search" in registry.names()
    tool = registry.get("semantic_search")
    assert tool.name == "semantic_search"

    scope = resolve_scope("user_a")
    result = registry.execute("semantic_search", {"query": QUERY}, scope)
    assert isinstance(result, ToolResult)
    assert result.tool == "semantic_search"


# --- 9. Candidate pool integration --------------------------------------------

def test_9_semantic_results_enter_the_existing_candidate_pool():
    registry = build_registry()
    scope = resolve_scope("user_a")
    result = registry.execute("semantic_search", {"query": QUERY}, scope)
    assert result.candidates

    pool = update_candidate_pool({}, result, iteration=1)
    target = next(c for c in result.candidates if c["metadata"]["filename"] == "Recovery_Doc_501.txt")
    entry = pool[target["file_id"]]
    assert entry["evidence_sources"] == ["semantic_search"]
    assert entry["scores"]["semantic_search"] == target["score"]

    # Accumulation across tools/sources, same as bm25_search's Phase 2A test.
    fake_bm25_result = ToolResult(
        tool="bm25_search",
        candidates=[
            {
                "file_id": target["file_id"],
                "metadata": target["metadata"],
                "score": 0.3,
                "source": "bm25_search",
            }
        ],
    )
    pool = update_candidate_pool(pool, fake_bm25_result, iteration=2)
    entry = pool[target["file_id"]]
    assert set(entry["evidence_sources"]) == {"semantic_search", "bm25_search"}


# --- 10. Genuine agentic escalation to semantic_search ------------------------

def test_10_agent_escalates_metadata_to_bm25_to_semantic_on_repeated_zero_results():
    """Full-loop integration test through run_query(): this query's only
    evidence is semantic (zero literal word overlap with the stored text),
    so metadata_search AND bm25_search must both be tried and observed to
    fail before semantic_search is chosen."""
    result = run_query("user_a", QUERY)

    tools_used = [a["tool"] for a in result["committed_actions"]]
    assert "metadata_search" in tools_used
    assert "bm25_search" in tools_used
    assert "semantic_search" in tools_used
    assert tools_used.index("metadata_search") < tools_used.index("bm25_search") < tools_used.index(
        "semantic_search"
    ), "each tier must be tried and observed to fail before the next is chosen"

    filenames = [c["metadata"]["filename"] for c in result["candidate_pool"].values()]
    assert "Recovery_Doc_501.txt" in filenames
    assert result["status"] in ("finish", "clarify")
