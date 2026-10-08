from app.scope import resolve_scope
from app.tools import build_registry


def test_metadata_search_finds_seeded_file_for_owner():
    registry = build_registry()
    scope = resolve_scope("user_a")
    result = registry.execute(
        "metadata_search",
        {"file_type": "pptx", "topic_tokens": ["ai", "architecture"]},
        scope,
    )
    filenames = [c["metadata"]["filename"] for c in result.candidates]
    assert "AI_Architecture_Overview.pptx" in filenames


def test_metadata_search_never_returns_other_owners_files():
    """Critical access-control test: even when topic tokens would match a
    file owned by user_x, a search scoped to user_a must never return it.
    """
    registry = build_registry()
    scope = resolve_scope("user_a")
    result = registry.execute(
        "metadata_search",
        {"topic_tokens": ["confidential", "secret", "roadmap", "hr", "salaries"]},
        scope,
    )
    owners = {c["metadata"]["owner_id"] for c in result.candidates}
    assert owners <= {"user_a"}
    assert "user_x" not in owners


def test_metadata_search_owner_id_predicate_is_always_present():
    """Even with zero args, the query must still scope to owner_id."""
    registry = build_registry()
    scope = resolve_scope("user_b")
    result = registry.execute("metadata_search", {}, scope)
    owners = {c["metadata"]["owner_id"] for c in result.candidates}
    assert owners <= {"user_b"}


def test_metadata_search_zero_results_for_nonmatching_filter():
    registry = build_registry()
    scope = resolve_scope("user_a")
    result = registry.execute(
        "metadata_search", {"file_type": "xlsx", "topic_tokens": ["ai", "architecture"]}, scope
    )
    assert result.candidates == []
    assert result.error is None


def test_metadata_search_multi_owner_scope_only_returns_scoped_owners():
    from app.scope import UserScope

    registry = build_registry()
    scope = UserScope(user_id="user_a", owner_ids=("user_a", "user_b"))
    result = registry.execute("metadata_search", {}, scope)
    owners = {c["metadata"]["owner_id"] for c in result.candidates}
    assert owners <= {"user_a", "user_b"}
    assert "user_x" not in owners
