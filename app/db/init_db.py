"""Apply schema.sql to the configured database, then seed it."""
import os

from app.db.connection import get_cursor
from app.db.seed import seed

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


def init_db(with_seed: bool = True):
    with open(SCHEMA_PATH) as f:
        schema_sql = f.read()
    with get_cursor(commit=True) as cur:
        cur.execute(schema_sql)
    if with_seed:
        seed()


if __name__ == "__main__":
    init_db()
    print("Schema applied and seed data loaded.")
