"""
Deterministic seed data for Phase 1 development and tests.

Dates are generated relative to "now" (not hard-coded timestamps) so that
queries like "last week" stay valid however long this environment lives.

Users:
    user_a - primary test user, has an AI-architecture deck and budget files
    user_b - second user, owns unrelated files (used as a neutral bystander)
    user_c - third user, also owns an AI-architecture-topic file (used to
             confirm cross-user isolation even under topic collisions)
    user_x - "victim" user whose files must NEVER be visible to user_a/user_b
             (access-control test, architecture_final.md §11)
"""
from datetime import datetime, timedelta, timezone

from app.db.connection import get_cursor

USERS = [
    ("user_a", "Alice"),
    ("user_b", "Bob"),
    ("user_x", "Xavier (never accessible to others)"),
    ("user_c", "Carla"),
]


def _days_ago(n: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=n)


def build_files():
    """Returns the seed file rows. Built as a function (not a module-level
    constant) so timestamps are always fresh relative to insertion time."""
    return [
        # --- user_a --------------------------------------------------
        dict(
            filename="AI_Architecture_Overview.pptx",
            file_type="pptx",
            file_path="/files/user_a/AI_Architecture_Overview.pptx",
            owner_id="user_a",
            created_at=_days_ago(40),
            updated_at=_days_ago(6),  # "last week"
            tags=["ai", "architecture", "slides"],
            summary="Slide deck about AI architecture for Project 5, covering the agent loop design.",
        ),
        dict(
            filename="Q3_Budget_Review.xlsx",
            file_type="xlsx",
            file_path="/files/user_a/Q3_Budget_Review.xlsx",
            owner_id="user_a",
            created_at=_days_ago(20),
            updated_at=_days_ago(2),
            tags=["budget", "finance", "q3"],
            summary="Quarterly budget review spreadsheet with department spend breakdown.",
        ),
        dict(
            filename="Q2_Budget_Notes.docx",
            file_type="docx",
            file_path="/files/user_a/Q2_Budget_Notes.docx",
            owner_id="user_a",
            created_at=_days_ago(95),
            updated_at=_days_ago(80),
            tags=["budget", "notes"],
            summary="Notes from budget planning discussions in Q2.",
        ),
        dict(
            filename="Team_Offsite_Photo.png",
            file_type="png",
            file_path="/files/user_a/Team_Offsite_Photo.png",
            owner_id="user_a",
            created_at=_days_ago(15),
            updated_at=_days_ago(15),
            tags=["photo", "team"],
            summary="Photo from the team offsite event.",
        ),
        dict(
            filename="Meeting_Notes_January.docx",
            file_type="docx",
            file_path="/files/user_a/Meeting_Notes_January.docx",
            owner_id="user_a",
            created_at=_days_ago(60),
            updated_at=_days_ago(59),
            tags=["meeting", "notes"],
            summary="Notes from the January planning meeting.",
        ),
        dict(
            filename="Vendor_Budget_Estimate.pdf",
            file_type="pdf",
            file_path="/files/user_a/Vendor_Budget_Estimate.pdf",
            owner_id="user_a",
            created_at=_days_ago(10),
            updated_at=_days_ago(3),
            tags=["budget", "vendor"],
            summary="Vendor-submitted budget estimate for the new tooling contract.",
        ),
        dict(
            filename="Budget_Slides_Draft.pptx",
            file_type="pptx",
            file_path="/files/user_a/Budget_Slides_Draft.pptx",
            owner_id="user_a",
            created_at=_days_ago(9),
            updated_at=_days_ago(4),
            tags=["budget", "slides", "draft"],
            summary="Draft slides summarizing the budget for the leadership review.",
        ),
        # --- user_b (bystander, unrelated files) ----------------------
        dict(
            filename="Marketing_Plan.pptx",
            file_type="pptx",
            file_path="/files/user_b/Marketing_Plan.pptx",
            owner_id="user_b",
            created_at=_days_ago(30),
            updated_at=_days_ago(6),
            tags=["marketing", "plan"],
            summary="Marketing plan slide deck for the spring campaign.",
        ),
        dict(
            filename="Recipe_Collection.docx",
            file_type="docx",
            file_path="/files/user_b/Recipe_Collection.docx",
            owner_id="user_b",
            created_at=_days_ago(200),
            updated_at=_days_ago(150),
            tags=["personal"],
            summary="A personal collection of recipes.",
        ),
        # --- user_c (bystander, used for a third-user isolation check) -
        dict(
            filename="Travel_Itinerary.pdf",
            file_type="pdf",
            file_path="/files/user_c/Travel_Itinerary.pdf",
            owner_id="user_c",
            created_at=_days_ago(25),
            updated_at=_days_ago(5),
            tags=["travel", "personal"],
            summary="Travel itinerary for the upcoming conference trip.",
        ),
        dict(
            filename="AI_Architecture_Notes.docx",
            file_type="docx",
            file_path="/files/user_c/AI_Architecture_Notes.docx",
            owner_id="user_c",
            created_at=_days_ago(12),
            updated_at=_days_ago(6),
            tags=["ai", "architecture", "notes"],
            summary="Carla's own notes on AI architecture, unrelated to user_a's deck.",
        ),
        # --- user_x (must never be visible to user_a / user_b) --------
        dict(
            filename="Confidential_HR_Salaries.xlsx",
            file_type="xlsx",
            file_path="/files/user_x/Confidential_HR_Salaries.xlsx",
            owner_id="user_x",
            created_at=_days_ago(50),
            updated_at=_days_ago(5),
            tags=["confidential", "hr"],
            summary="Confidential salary information for HR use only.",
        ),
        dict(
            # Deliberately topic-colliding with user_a's deck to make the
            # access-control test meaningful: same topic tokens, same file
            # type, similar recency -- the only thing that must keep it out
            # of user_a's results is the owner_id scope filter.
            filename="AI_Architecture_Secret_Roadmap.pptx",
            file_type="pptx",
            file_path="/files/user_x/AI_Architecture_Secret_Roadmap.pptx",
            owner_id="user_x",
            created_at=_days_ago(40),
            updated_at=_days_ago(6),
            tags=["ai", "architecture", "slides", "confidential"],
            summary="Confidential slide deck about AI architecture strategy, not for user_a.",
        ),
    ]


def seed(reset: bool = True):
    with get_cursor(commit=True) as cur:
        if reset:
            cur.execute("TRUNCATE files, users RESTART IDENTITY CASCADE;")

        for user_id, display_name in USERS:
            cur.execute(
                """
                INSERT INTO users (user_id, display_name)
                VALUES (%s, %s)
                ON CONFLICT (user_id) DO NOTHING;
                """,
                (user_id, display_name),
            )

        for f in build_files():
            cur.execute(
                """
                INSERT INTO files
                    (filename, file_type, file_path, owner_id, created_at,
                     updated_at, tags, summary)
                VALUES
                    (%(filename)s, %(file_type)s, %(file_path)s, %(owner_id)s,
                     %(created_at)s, %(updated_at)s, %(tags)s, %(summary)s);
                """,
                f,
            )


if __name__ == "__main__":
    seed()
    print("Seed data inserted.")
