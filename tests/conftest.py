import pytest

from app.db.init_db import init_db


@pytest.fixture(scope="session", autouse=True)
def seeded_db():
    """Apply schema + seed data once for the whole test session.

    Phase 1 has no write-capable tools (architecture_final.md §15), so
    nothing in the test suite mutates the database -- a session-scoped
    seed is safe and much faster than reseeding per test.
    """
    init_db(with_seed=True)
    yield
