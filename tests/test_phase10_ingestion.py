"""
Phase 10 -- real-file ingestion into the EXISTING retrieval substrate.

Real PostgreSQL, a real (isolated, in-memory) Qdrant collection, the real
embedding provider the environment resolves, and the real run_query() agent.
Nothing is mocked except where a test deliberately injects a failure.

Isolation: every test owns users `p10_alice` / `p10_bob`, an in-memory Qdrant
(never the shared collection) and removes everything it created, so the Phase 7
seed corpus and benchmark are untouched.
"""
import json
import logging
import os
import shutil
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

import app.tools.semantic_search as sem
import app.vectorstore as vs
from app.config import QDRANT_COLLECTION
from app.db.connection import DatabaseError, get_cursor
from app.embeddings import get_provider_name
from app.ingestion import (ContentExtractor, LocalDirectorySource, SourceUnavailableError, UnknownOwnerError, ingest,
                           ingest_directory)
from app.ingestion import pipeline
from app.llm.provider import NullProvider
from app.run import run_query
from app.scope import resolve_scope
from app.tools.bm25_search import bm25_search
from app.tools.metadata_search import metadata_search
from app.tools.semantic_search import semantic_search

FIXTURES = Path(__file__).parent / "fixtures" / "ingest_corpus"
ALICE, BOB = "p10_alice", "p10_bob"
SEMANTIC_ONLY_QUERIES = {   # zero lexical overlap with the target's text/filename: only embeddings can find them
    "sample_document.pdf": "growing vegetables in the backyard",
    "finance_report.txt": "profitable firm annual results",
    "sample_document.docx": "staffing and workforce expansion",
}


# ---------------------------------------------------------------- fixtures
def _cleanup():
    with get_cursor(commit=True) as cur:
        cur.execute("DELETE FROM user_file_telemetry WHERE file_id IN (SELECT file_id FROM files WHERE owner_id = ANY(%s));", ([ALICE, BOB],))
        cur.execute("DELETE FROM files WHERE owner_id = ANY(%s);", ([ALICE, BOB],))  # file_ingestion_state cascades
        cur.execute("DELETE FROM users WHERE user_id = ANY(%s);", ([ALICE, BOB],))


@pytest.fixture(autouse=True)
def env(monkeypatch):
    _cleanup()
    with get_cursor(commit=True) as cur:
        cur.executemany("INSERT INTO users (user_id, display_name) VALUES (%s, %s);", [(ALICE, "Alice"), (BOB, "Bob")])
    client = QdrantClient(":memory:")
    monkeypatch.setattr(vs, "_client", client)            # isolated vector store
    monkeypatch.setattr(sem, "_build_attempted", True)    # no lazy global rebuild from the whole DB
    yield client
    _cleanup()


@pytest.fixture
def corpus(tmp_path):
    dest = tmp_path / "corpus"
    shutil.copytree(FIXTURES, dest)
    return dest


def rows(owner):
    with get_cursor() as cur:
        cur.execute("SELECT f.file_id, f.filename, f.file_path, f.extracted_text, f.owner_id, s.state, s.extraction_status, "
                    "s.error_code, s.chunk_count, s.fingerprint, s.mtime_ns FROM files f JOIN file_ingestion_state s USING (file_id) "
                    "WHERE f.owner_id = %s ORDER BY f.file_path;", (owner,))
        return {r["file_path"]: dict(r) for r in cur.fetchall()}


def points(client, file_id=None):
    flt = Filter(must=[FieldCondition(key="file_id", match=MatchValue(value=file_id))]) if file_id is not None else None
    if not client.collection_exists(QDRANT_COLLECTION):
        return []
    return client.scroll(QDRANT_COLLECTION, scroll_filter=flt, limit=10_000, with_payload=True)[0]


def touch(path, delta_s=5):
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + delta_s * 10**9))


def semantic_real():
    if "hashing_fallback" in get_provider_name():
        pytest.skip("embedding provider is the hashing stub, which is NOT semantic")


# ------------------------------------------------------ new / lifecycle state
def test_new_files_are_ingested_with_lifecycle_state_and_normalized_text(corpus):
    report = ingest_directory(corpus, ALICE)
    assert (report.discovered, report.new, report.indexed, report.empty, report.unsupported, report.failed) == (7, 7, 5, 1, 1, 0)
    r = rows(ALICE)
    assert set(r) == {"empty.txt", "finance_report.txt", "notes/standup.txt", "retrieval_notes.md",
                      "sample_document.docx", "sample_document.pdf", "unsupported.xyz"}
    assert r["finance_report.txt"]["state"] == "content_indexed" and r["finance_report.txt"]["extraction_status"] == "ok"
    assert r["finance_report.txt"]["chunk_count"] >= 1 and len(r["finance_report.txt"]["fingerprint"]) == 64
    assert "\n\n\n" not in r["retrieval_notes.md"]["extracted_text"] and "  " not in r["finance_report.txt"]["extracted_text"]
    assert (r["empty.txt"]["state"], r["empty.txt"]["extraction_status"], r["empty.txt"]["chunk_count"]) == ("content_indexed", "empty", 0)
    assert (r["unsupported.xyz"]["state"], r["unsupported.xyz"]["extracted_text"]) == ("unsupported", "")
    assert all(x["owner_id"] == ALICE for x in r.values())
    assert all(not x["file_path"].startswith("/") for x in r.values())  # relative path only, never a machine path


def test_state_table_rejects_invalid_lifecycle_values(corpus):
    ingest_directory(corpus, ALICE, index_semantic=False)
    with pytest.raises(DatabaseError):
        with get_cursor(commit=True) as cur:
            cur.execute("UPDATE file_ingestion_state SET state = 'bogus' WHERE owner_id = %s;", (ALICE,))


def test_unknown_owner_is_rejected_and_nothing_is_written(corpus):
    with pytest.raises(UnknownOwnerError):
        ingest_directory(corpus, "nobody_p10")
    with get_cursor() as cur:
        cur.execute("SELECT count(*) AS c FROM file_ingestion_state WHERE owner_id = 'nobody_p10';")
        assert cur.fetchone()["c"] == 0


# ------------------------------------------------------------- incremental
class CountingEmbed:
    def __init__(self, monkeypatch):
        self.batches = []
        real = sem.embed_texts

        def wrapper(texts):
            self.batches.append(list(texts))
            return real(texts)
        monkeypatch.setattr(sem, "embed_texts", wrapper)


def test_unchanged_files_are_skipped_without_reading_or_embedding(corpus, monkeypatch, env):
    ingest_directory(corpus, ALICE)
    n_points = len(points(env))
    counter = CountingEmbed(monkeypatch)

    class NoReadSource(LocalDirectorySource):
        def read_bytes(self, *a, **k):
            raise AssertionError("an unchanged file must not be read")

    report = ingest(NoReadSource(corpus), ALICE)
    assert (report.unchanged, report.new, report.updated, report.deleted) == (7, 0, 0, 0)
    assert counter.batches == [] and len(points(env)) == n_points


def test_touched_file_with_identical_bytes_is_bookkeeping_only(corpus, monkeypatch, env):
    ingest_directory(corpus, ALICE)
    before = rows(ALICE)["finance_report.txt"]
    touch(corpus / "finance_report.txt")
    counter = CountingEmbed(monkeypatch)
    report = ingest_directory(corpus, ALICE)
    after = rows(ALICE)["finance_report.txt"]
    assert report.unchanged == 7 and counter.batches == []          # same bytes: no re-extract, no re-embed
    assert after["mtime_ns"] != before["mtime_ns"] and after["fingerprint"] == before["fingerprint"]


def test_modified_file_is_reextracted_bm25_sees_new_text_and_old_semantic_points_are_replaced(corpus, env, monkeypatch):
    ingest_directory(corpus, ALICE)
    scope = resolve_scope(ALICE)
    assert [c["metadata"]["filename"] for c in bm25_search({"query": "dividend shareholders"}, scope).candidates][:1] == ["finance_report.txt"]
    fid = rows(ALICE)["finance_report.txt"]["file_id"]
    old_points = {p.id for p in points(env, fid)}
    (corpus / "finance_report.txt").write_text("Zeppelin maintenance schedule and hangar inspection checklist.\n")
    touch(corpus / "finance_report.txt")
    counter = CountingEmbed(monkeypatch)
    report = ingest_directory(corpus, ALICE)
    assert (report.updated, report.unchanged) == (1, 6)
    assert counter.batches == [["Zeppelin maintenance schedule and hangar inspection checklist."]]  # only the changed file
    assert "dividend" not in rows(ALICE)["finance_report.txt"]["extracted_text"]
    # BM25 (single existing implementation, cache keyed by content signal) sees the new text and not the old
    assert [c["metadata"]["filename"] for c in bm25_search({"query": "zeppelin hangar"}, scope).candidates] == ["finance_report.txt"]
    assert bm25_search({"query": "dividend shareholders"}, scope).candidates == []
    new = points(env, fid)
    assert [p.payload["content_text"] for p in new] == ["Zeppelin maintenance schedule and hangar inspection checklist."]
    assert not any("dividend" in (p.payload["content_text"]) for p in points(env))  # no stale point anywhere
    assert {p.id for p in new} <= old_points | {fid * 100000}                       # deterministic ids reused


def test_bm25_cache_is_not_stale_when_an_older_file_is_edited_without_becoming_the_newest(corpus):
    """The scenario the content-aware BM25 cache key exists for: the scope's max(updated_at) and file
    count do NOT change (the edited file keeps an older mtime than another file), yet the text did."""
    touch(corpus / "retrieval_notes.md", 1000)                      # make a DIFFERENT file the newest
    ingest_directory(corpus, ALICE, index_semantic=False)
    scope = resolve_scope(ALICE)
    assert bm25_search({"query": "dividend shareholders"}, scope).candidates                       # populates the BM25 cache
    path = corpus / "finance_report.txt"
    original_mtime = path.stat().st_mtime_ns
    path.write_text("Hangar zeppelin inspection checklist, revised and extended with extra notes.")
    # write_text() stamps "now"; pin an ABSOLUTE mtime (independent of the wall clock) that differs from the
    # original yet stays well below the newest file's, so max(updated_at) cannot move.
    newest = (corpus / "retrieval_notes.md").stat().st_mtime_ns
    pinned = newest - 500 * 10**9
    assert pinned != original_mtime
    os.utime(path, ns=(pinned, pinned))
    before = max(r["mtime_ns"] for r in rows(ALICE).values())
    assert before == newest
    ingest_directory(corpus, ALICE, index_semantic=False)
    assert max(r["mtime_ns"] for r in rows(ALICE).values()) == before   # confirms max(updated_at) did not move
    assert rows(ALICE)["finance_report.txt"]["mtime_ns"] == pinned       # ...while the edited file was reprocessed
    assert [c["metadata"]["filename"] for c in bm25_search({"query": "zeppelin hangar"}, scope).candidates] == ["finance_report.txt"]
    assert bm25_search({"query": "dividend shareholders"}, scope).candidates == []


def test_shrinking_a_file_removes_its_now_surplus_chunk_points(corpus, env):
    long_text = " ".join(f"word{i}" for i in range(300))                 # 300 words -> 3 chunks at 120 words/chunk
    (corpus / "long.txt").write_text(long_text)
    ingest_directory(corpus, ALICE)
    fid = rows(ALICE)["long.txt"]["file_id"]
    assert sorted(p.id for p in points(env, fid)) == [fid * 100000 + i for i in range(3)]
    (corpus / "long.txt").write_text("just ten words now " * 2)
    touch(corpus / "long.txt")
    ingest_directory(corpus, ALICE)
    assert sorted(p.id for p in points(env, fid)) == [fid * 100000]     # chunks 1 and 2 are gone, not stale


def test_deleted_file_is_removed_from_postgres_bm25_and_qdrant(corpus, env):
    ingest_directory(corpus, ALICE)
    scope = resolve_scope(ALICE)
    fid = rows(ALICE)["sample_document.pdf"]["file_id"]
    with get_cursor(commit=True) as cur:   # telemetry (FK to files) must not block cleanup
        cur.execute("INSERT INTO user_file_telemetry (user_id, file_id, action_type) VALUES (%s, %s, 'open');", (ALICE, fid))
    assert bm25_search({"query": "tomato seedlings"}, scope).candidates
    (corpus / "sample_document.pdf").unlink()
    report = ingest_directory(corpus, ALICE)
    assert (report.deleted, report.unchanged) == (1, 6)
    assert "sample_document.pdf" not in rows(ALICE)
    assert points(env, fid) == []
    assert bm25_search({"query": "tomato seedlings"}, scope).candidates == []
    assert metadata_search({"filename_contains": "sample_document.pdf"}, scope).candidates == []
    assert sem.semantic_search({"query": SEMANTIC_ONLY_QUERIES["sample_document.pdf"]}, scope).candidates  # other files remain


def test_deletion_is_never_triggered_by_an_unavailable_or_incomplete_source(corpus, tmp_path):
    ingest_directory(corpus, ALICE, index_semantic=False)
    before = rows(ALICE)
    with pytest.raises(SourceUnavailableError):                           # e.g. an unmounted drive
        ingest_directory(tmp_path / "gone", ALICE, index_semantic=False)
    assert rows(ALICE).keys() == before.keys()

    class Incomplete(LocalDirectorySource):
        def discover(self):
            found = [f for f in super().discover() if f.rel_path != "finance_report.txt"]
            self.last_errors = 1                                           # an entry could not be inspected
            return found

    report = ingest(Incomplete(corpus), ALICE, index_semantic=False)
    assert report.deletion_skipped is True and report.deleted == 0
    assert "finance_report.txt" in rows(ALICE)


def test_only_ingestion_managed_files_are_ever_modified_or_deleted(corpus):
    with get_cursor(commit=True) as cur:   # a pre-existing file of the same owner with NO ingestion state
        cur.execute("INSERT INTO files (filename, file_path, file_type, owner_id, summary) VALUES "
                    "('Manual.docx', '/manual/Manual.docx', 'docx', %s, 'hand curated') RETURNING file_id;", (ALICE,))
        manual = cur.fetchone()["file_id"]
    with get_cursor() as cur:
        cur.execute("SELECT count(*) AS c FROM files WHERE owner_id IN ('user_a','user_b','user_c','user_x');")
        seed_before = cur.fetchone()["c"]
    ingest_directory(corpus, ALICE)
    for f in corpus.iterdir():
        if f.is_file():
            f.unlink()
    ingest_directory(corpus, ALICE)                                         # every ingested file is gone now
    with get_cursor() as cur:
        cur.execute("SELECT summary FROM files WHERE file_id = %s;", (manual,))
        assert cur.fetchone()["summary"] == "hand curated"                  # untouched
        cur.execute("SELECT count(*) AS c FROM files WHERE owner_id IN ('user_a','user_b','user_c','user_x');")
        assert cur.fetchone()["c"] == seed_before                           # seed corpus untouched


def test_update_preserves_user_curated_tags_and_summary(corpus):
    ingest_directory(corpus, ALICE, index_semantic=False)
    fid = rows(ALICE)["finance_report.txt"]["file_id"]
    with get_cursor(commit=True) as cur:
        cur.execute("UPDATE files SET tags = ARRAY['curated'], summary = 'mine' WHERE file_id = %s;", (fid,))
    (corpus / "finance_report.txt").write_text("changed content here")
    touch(corpus / "finance_report.txt")
    ingest_directory(corpus, ALICE, index_semantic=False)
    with get_cursor() as cur:
        cur.execute("SELECT tags, summary FROM files WHERE file_id = %s;", (fid,))
        assert dict(cur.fetchone()) == {"tags": ["curated"], "summary": "mine"}


# -------------------------------------------------------------------- BM25
def test_extracted_content_participates_in_the_existing_bm25(corpus):
    ingest_directory(corpus, ALICE, index_semantic=False)                    # BM25/PostgreSQL only
    scope = resolve_scope(ALICE)
    hit = bm25_search({"query": "tomato seedlings watering schedule"}, scope).candidates
    assert [c["metadata"]["filename"] for c in hit] == ["sample_document.pdf"]
    # content is NOT visible to metadata_search (filename/tags/summary only) -- BM25 is the content path
    assert metadata_search({"topic_tokens": ["tomato", "seedlings"]}, scope).candidates == []


def test_existing_bm25_filename_weighting_still_outranks_content_only_matches(corpus):
    (corpus / "zebra_notes.txt").write_text("completely unrelated body text about weather")
    (corpus / "other.txt").write_text("a long note that mentions zebra once among many other words here")
    ingest_directory(corpus, ALICE, index_semantic=False)
    hits = bm25_search({"query": "zebra"}, resolve_scope(ALICE)).candidates
    assert [c["metadata"]["filename"] for c in hits][:2] == ["zebra_notes.txt", "other.txt"]


# ---------------------------------------------------------------- semantic
def test_content_becomes_chunks_embeddings_and_deterministic_points(corpus, env, monkeypatch):
    counter = CountingEmbed(monkeypatch)
    ingest_directory(corpus, ALICE)
    r = rows(ALICE)
    expected = {name: sem._chunk_text(r[name]["extracted_text"]) for name in r if r[name]["extracted_text"]}
    embedded = [t for batch in counter.batches for t in batch]
    assert sorted(embedded) == sorted(c for chunks in expected.values() for c in chunks)   # embeddings got exactly the normalized chunks
    for name, chunks in expected.items():
        pts = sorted(points(env, r[name]["file_id"]), key=lambda p: p.payload["chunk_id"])
        assert [p.payload["content_text"] for p in pts] == chunks
        assert [p.id for p in pts] == [r[name]["file_id"] * 100000 + i for i in range(len(chunks))]
        assert all(p.payload["owner_id"] == ALICE and p.payload["file_id"] == r[name]["file_id"] for p in pts)
        assert r[name]["chunk_count"] == len(chunks)
    for name in ("empty.txt", "unsupported.xyz"):                                          # never embedded, no points
        assert points(env, r[name]["file_id"]) == []


@pytest.mark.parametrize("target", list(SEMANTIC_ONLY_QUERIES))
def test_semantic_query_retrieves_the_right_file_where_lexical_search_finds_nothing(corpus, target):
    semantic_real()
    ingest_directory(corpus, ALICE)
    scope, query = resolve_scope(ALICE), SEMANTIC_ONLY_QUERIES[target]
    assert bm25_search({"query": query}, scope).candidates == []                           # no lexical path
    assert metadata_search({"topic_tokens": query.split()}, scope).candidates == []        # no metadata path
    cands = semantic_search({"query": query}, scope).candidates
    assert cands[0]["metadata"]["filename"] == target
    assert len({c["file_id"] for c in cands}) == len(cands)                                # one candidate per file
    assert cands[0]["evidence"] if "evidence" in cands[0] else True


def test_semantic_aggregation_keeps_the_strongest_chunk_per_file(corpus):
    semantic_real()
    (corpus / "mixed.txt").write_text(("filler words about weather and traffic " * 40) + " tomato seedlings need sunny soil and watering")
    ingest_directory(corpus, ALICE)
    cands = semantic_search({"query": "growing vegetables in the backyard"}, resolve_scope(ALICE)).candidates
    mixed = [c for c in cands if c["metadata"]["filename"] == "mixed.txt"]
    assert len(mixed) == 1                                                                 # multi-chunk file -> one candidate


# --------------------------------------------------------------- isolation
def test_one_users_ingestion_never_makes_another_users_files_searchable(corpus, tmp_path, env):
    semantic_real()
    bob_root = tmp_path / "bob"
    bob_root.mkdir()
    (bob_root / "bobs_note.txt").write_text("Quarterly kayak rental pricing and paddle inventory.")
    alice_only = corpus / "alice_secret.txt"
    alice_only.write_text("Zeppelin launch codes are stored in the hangar vault.")
    ingest_directory(corpus, ALICE)
    ingest_directory(bob_root, BOB)

    a_ids, b_ids = {r["file_id"] for r in rows(ALICE).values()}, {r["file_id"] for r in rows(BOB).values()}
    assert a_ids.isdisjoint(b_ids) and all(r["owner_id"] == BOB for r in rows(BOB).values())
    scope_b = resolve_scope(BOB)
    for query in ("zeppelin hangar vault", "tomato seedlings", "dividend shareholders"):
        assert bm25_search({"query": query}, scope_b).candidates == []                     # PostgreSQL/BM25 layer
    assert semantic_search({"query": "zeppelin launch codes hangar vault"}, scope_b).candidates[0]["metadata"]["filename"] == "bobs_note.txt"
    assert {c["file_id"] for c in semantic_search({"query": "growing vegetables"}, scope_b).candidates} <= b_ids

    # Qdrant layer, raw: a query filtered to Bob can NEVER return Alice's points, whatever the vector
    q = sem.embed_texts(["Zeppelin launch codes are stored in the hangar vault."])[0]
    raw = env.query_points(QDRANT_COLLECTION, query=q, limit=100,
                           query_filter=Filter(must=[FieldCondition(key="owner_id", match=MatchAny(any=[BOB]))])).points
    assert raw and {p.payload["owner_id"] for p in raw} == {BOB} and {p.payload["file_id"] for p in raw} <= b_ids

    # final agent result
    for query in ("zeppelin hangar vault", "growing vegetables in the backyard"):
        r = run_query(BOB, query, llm_provider=NullProvider())
        owners = {e["metadata"]["owner_id"] for e in r["final_ranking"]} | {e["metadata"]["owner_id"] for e in r["candidate_pool"].values()}
        assert owners <= {BOB}


def test_identical_relative_paths_for_two_owners_are_independent(corpus, tmp_path):
    bob_root = tmp_path / "bob_copy"
    shutil.copytree(FIXTURES, bob_root)
    ingest_directory(corpus, ALICE, index_semantic=False)
    ingest_directory(bob_root, BOB, index_semantic=False)
    assert rows(ALICE).keys() == rows(BOB).keys()
    assert {r["file_id"] for r in rows(ALICE).values()}.isdisjoint({r["file_id"] for r in rows(BOB).values()})
    (corpus / "finance_report.txt").unlink()
    ingest_directory(corpus, ALICE, index_semantic=False)
    assert "finance_report.txt" not in rows(ALICE) and "finance_report.txt" in rows(BOB)   # Alice's deletion never touches Bob


# -------------------------------------------------------- failure recovery
def test_one_bad_file_does_not_stop_the_run_and_good_files_are_indexed(corpus, env):
    (corpus / "broken.pdf").write_bytes(b"%PDF-1.4 SECRET-DOC-BODY not really a pdf")
    (corpus / "broken.docx").write_bytes(b"not a zip at all")
    report = ingest_directory(corpus, ALICE)
    assert (report.failed, report.indexed, report.new) == (2, 5, 9)
    r = rows(ALICE)
    assert (r["broken.pdf"]["state"], r["broken.pdf"]["error_code"]) == ("failed", "malformed_pdf")
    assert (r["broken.docx"]["state"], r["broken.docx"]["error_code"]) == ("failed", "malformed_docx")
    assert r["finance_report.txt"]["state"] == "content_indexed" and points(env, r["finance_report.txt"]["file_id"])
    dumped = json.dumps(report.to_dict())
    assert "SECRET-DOC-BODY" not in dumped and str(corpus) not in dumped        # no content, no absolute path


def test_unreadable_oversized_and_unexpected_failures_are_per_file(corpus, monkeypatch):
    class Flaky(LocalDirectorySource):
        def read_bytes(self, rel_path, max_bytes):
            if rel_path == "retrieval_notes.md":
                raise PermissionError("denied")
            return super().read_bytes(rel_path, max_bytes)

    class Exploding(ContentExtractor):
        def extract(self, ext, data):
            if b"standup" in data:
                raise RuntimeError("boom")
            return super().extract(ext, data)

    monkeypatch.setattr(pipeline, "INGEST_MAX_FILE_BYTES", 5_000)   # the DOCX (~36 KB) is now oversized
    report = ingest(Flaky(corpus), ALICE, extractor=Exploding(), index_semantic=False)
    r = rows(ALICE)
    assert r["retrieval_notes.md"]["error_code"] == "unreadable"
    assert r["sample_document.docx"]["error_code"] == "too_large"
    assert (r["notes/standup.txt"]["state"], r["notes/standup.txt"]["error_code"]) == ("failed", "unexpected_error")
    assert r["notes/standup.txt"]["extracted_text"] == ""        # persisted, not just reported
    assert r["finance_report.txt"]["state"] == "content_indexed" and report.failed == 3


def test_a_file_that_becomes_unreadable_no_longer_leaves_its_old_content_searchable(corpus, env):
    ingest_directory(corpus, ALICE)
    scope, fid = resolve_scope(ALICE), rows(ALICE)["sample_document.pdf"]["file_id"]
    assert bm25_search({"query": "tomato seedlings"}, scope).candidates and points(env, fid)
    (corpus / "sample_document.pdf").write_bytes(b"corrupted now")
    touch(corpus / "sample_document.pdf")
    ingest_directory(corpus, ALICE)
    assert rows(ALICE)["sample_document.pdf"]["state"] == "failed" and rows(ALICE)["sample_document.pdf"]["extracted_text"] == ""
    assert bm25_search({"query": "tomato seedlings"}, scope).candidates == [] and points(env, fid) == []


def test_embedding_failure_keeps_text_and_bm25_and_is_retried_next_run(corpus, env, monkeypatch):
    real = sem.embed_texts
    monkeypatch.setattr(sem, "embed_texts", lambda texts: (_ for _ in ()).throw(RuntimeError("embedding service down")))
    report = ingest_directory(corpus, ALICE)
    r = rows(ALICE)
    assert report.semantic_failed == 5 and report.indexed == 0 and report.failed == 0
    assert (r["finance_report.txt"]["state"], r["finance_report.txt"]["error_code"]) == ("content_extracted", "semantic_index_failed")
    assert points(env) == []                                                               # nothing half-written / stale
    assert bm25_search({"query": "dividend shareholders"}, resolve_scope(ALICE)).candidates  # lexical retrieval still works
    monkeypatch.setattr(sem, "embed_texts", real)                                          # service recovers
    report = ingest_directory(corpus, ALICE)
    assert (report.indexed, report.semantic_failed, report.unchanged) == (5, 0, 2)         # retried; empty/unsupported skipped
    assert rows(ALICE)["finance_report.txt"]["state"] == "content_indexed" and points(env)


def test_qdrant_failure_degrades_the_same_way(corpus, monkeypatch):
    monkeypatch.setattr(sem, "get_client", lambda: (_ for _ in ()).throw(ConnectionError("qdrant down")))
    monkeypatch.setattr(sem, "ensure_collection", lambda: (_ for _ in ()).throw(ConnectionError("qdrant down")))
    report = ingest_directory(corpus, ALICE)
    assert report.semantic_failed == 5 and report.failed == 0
    assert rows(ALICE)["sample_document.pdf"]["state"] == "content_extracted"


def test_a_database_failure_is_a_hard_failure_not_a_silent_skip(corpus, monkeypatch):
    monkeypatch.setattr(pipeline, "_load_state", lambda owner: (_ for _ in ()).throw(DatabaseError("db down")))
    with pytest.raises(DatabaseError):
        ingest_directory(corpus, ALICE)


# ------------------------------------------------- observability + CLI + config
def test_run_report_and_log_are_structured_counts_without_content_or_absolute_paths(corpus, caplog):
    with caplog.at_level(logging.INFO, logger="file_discovery_agent.ingestion"):
        report = ingest_directory(corpus, ALICE)
    s = report.summary()
    assert {"discovered", "new", "updated", "unchanged", "deleted", "indexed", "empty", "unsupported", "failed",
            "semantic_failed", "duration_ms"} <= set(s) and s["duration_ms"] >= 0
    assert all(set(f) == {"rel_path", "action", "state", "extraction_status", "error_code"} for f in report.files)
    line = next(r.getMessage() for r in caplog.records if "ingestion_run" in r.getMessage())
    assert json.loads(line.split("ingestion_run ", 1)[1])["indexed"] == 5
    assert "earnings" not in caplog.text and "Tomato" not in caplog.text and str(corpus) not in caplog.text


def test_cli_requires_explicit_root_and_owner_and_prints_a_report(corpus, capsys):
    from app.ingestion.__main__ import main
    assert main([]) == 2
    assert main(["--root", str(corpus), "--owner", ALICE, "--no-semantic"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["discovered"] == 7 and out["unsupported"] == 1
    assert main(["--root", str(corpus / "missing"), "--owner", ALICE]) == 2


def test_ingestion_configuration_is_environment_driven_with_safe_defaults():
    from app import config
    assert config.INGEST_ROOT == os.environ.get("DISCOVERY_INGEST_ROOT", "")
    assert config.INGEST_OWNER_ID == os.environ.get("DISCOVERY_INGEST_OWNER_ID", "")
    assert config.INGEST_MAX_FILE_BYTES == int(os.environ.get("DISCOVERY_INGEST_MAX_FILE_BYTES", 25 * 1024 * 1024))
    assert not Path(config.INGEST_ROOT or "x").is_absolute() or os.environ.get("DISCOVERY_INGEST_ROOT")  # no baked-in machine path


# ------------------------------------------------- end to end: the EXISTING agent
def test_e2e_agent_finds_a_file_by_its_extracted_content_via_bm25(corpus):
    """local files -> ingestion -> PostgreSQL -> BM25 -> existing run_query() -> correct file first."""
    ingest_directory(corpus, ALICE)
    r = run_query(ALICE, "tomato seedlings watering schedule", llm_provider=NullProvider())
    assert r["status"] == "finish"
    assert [a["tool"] for a in r["committed_actions"]] == ["metadata_search", "bm25_search"]  # content is only visible to BM25
    assert [e["metadata"]["filename"] for e in r["final_ranking"]] == ["sample_document.pdf"]
    assert r["decision_basis"] == "retrieval_dominance"
    assert {e["metadata"]["owner_id"] for e in r["candidate_pool"].values()} == {ALICE}


@pytest.mark.parametrize("target", list(SEMANTIC_ONLY_QUERIES))
def test_e2e_agent_reaches_semantic_search_and_the_ingested_file_is_its_strongest_evidence(corpus, target):
    """local files -> ingestion -> chunks/embeddings/Qdrant -> existing run_query().

    The agent must escalate metadata -> bm25 -> semantic (no lexical path exists) and the correct
    ingested file must be the STRONGEST semantic candidate in its candidate pool. The final ORDER
    is deliberately not asserted: in this environment rerank_candidates uses its lexical-overlap
    fallback (no CrossEncoder), which min-max-stretches stop-word noise on queries with no lexical
    overlap; that known limitation, not ingestion, decides ties there."""
    semantic_real()
    ingest_directory(corpus, ALICE)
    r = run_query(ALICE, SEMANTIC_ONLY_QUERIES[target], llm_provider=NullProvider())
    tools = [a["tool"] for a in r["committed_actions"]]
    assert tools[:3] == ["metadata_search", "bm25_search", "semantic_search"]
    pool = r["candidate_pool"]
    semantic_scores = {e["metadata"]["filename"]: e["scores"]["semantic_search"] for e in pool.values() if "semantic_search" in e["scores"]}
    assert max(semantic_scores, key=semantic_scores.get) == target
    assert target in [e["metadata"]["filename"] for e in r["final_ranking"]]
    assert all(e["metadata"]["owner_id"] == ALICE for e in pool.values())


def test_e2e_agent_sees_updates_and_deletions_after_reingestion(corpus):
    ingest_directory(corpus, ALICE)
    assert run_query(ALICE, "tomato seedlings watering schedule", llm_provider=NullProvider())["final_ranking"]
    (corpus / "sample_document.pdf").unlink()
    (corpus / "finance_report.txt").write_text("Hangar zeppelin inspection checklist for the airship crew.")
    touch(corpus / "finance_report.txt")
    ingest_directory(corpus, ALICE)
    gone = run_query(ALICE, "tomato seedlings watering schedule", llm_provider=NullProvider())
    # semantic_search has no relevance floor (existing behaviour), so OTHER files may come back as weak
    # neighbours -- what must hold is that the deleted file and its content are gone everywhere.
    assert "sample_document.pdf" not in [e["metadata"]["filename"] for e in gone["final_ranking"]]
    assert "sample_document.pdf" not in [e["metadata"]["filename"] for e in gone["candidate_pool"].values()]
    assert bm25_search({"query": "tomato seedlings watering schedule"}, resolve_scope(ALICE)).candidates == []
    new = run_query(ALICE, "hangar zeppelin airship crew", llm_provider=NullProvider())
    assert [e["metadata"]["filename"] for e in new["final_ranking"]] == ["finance_report.txt"]


def test_e2e_phase7_seed_corpus_is_unaffected_by_ingested_files(corpus):
    before = run_query("user_a", "Q3_Budget_Review.xlsx", llm_provider=NullProvider())
    ingest_directory(corpus, ALICE)
    after = run_query("user_a", "Q3_Budget_Review.xlsx", llm_provider=NullProvider())
    assert [e["file_id"] for e in before["final_ranking"]] == [e["file_id"] for e in after["final_ranking"]] == [2]
