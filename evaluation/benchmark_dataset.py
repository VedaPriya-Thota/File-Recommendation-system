"""
Phase 7 benchmark dataset -- explicit, hand-written ground truth.

Every file ID below refers to a file that actually exists in the repository's
seed corpus (app/db/seed.py, inserted in that order into a freshly
TRUNCATE ... RESTART IDENTITY'd `files` table, so IDs 1..13 are stable).
`validate_against_db()` re-checks the ID -> (filename, owner) mapping against
the live database before a benchmark runs, so a drifted/foreign database is
detected instead of silently producing meaningless numbers.

GROUND-TRUTH DISCIPLINE
-----------------------
* Relevance labels are written by hand from the seed file's filename / tags /
  summary / owner. They are NEVER derived from system output.
* Grades:  2 = highly relevant (the intended / strongest answer)
           1 = relevant     (an acceptable answer)
           absent = not relevant
* `forbidden_file_ids` lists files that match the query textually but belong
  to another user. They must never be returned (cross-user isolation).
* `expected_outcome`:
      "results"            -- the system should return the relevant file(s)
      "clarify_or_results" -- genuinely ambiguous; asking, or returning an
                              acceptable candidate first, are both fine
      "no_result"          -- nothing accessible is relevant; the system must
                              not fabricate a result

SUCCESS RULE (single, explicit definition -- see evaluation/metrics.py
`judge_success`):
      results            -> top-1 returned file is relevant (grade >= 1)
      clarify_or_results -> status is `clarify`, OR top-1 returned is relevant
      no_result          -> nothing returned
      (all)              -> AND zero scope violations

CORPUS LIMITATIONS (reported, not papered over -- see evaluation/README.md)
---------------------------------------------------------------------------
* No seed file has `extracted_text`, so the semantic index is EMPTY and
  `semantic_search` / the semantic half of `hybrid_search` cannot return
  anything on this corpus. Category C therefore only exercises
  summary/tag/filename evidence. No content was invented to work around this.
* The seed has no telemetry or profiles. Personalization needs history, so
  `PERSONALIZATION_FIXTURE` below defines explicit, deterministic telemetry
  for user_a (inserted only for the duration of a benchmark run). It touches
  no `files` rows.
* No two users can access the same file, so "same query, different users"
  cannot be tested. Personalization is ablated within-user (history present
  vs. personalization disabled), over an identical corpus and candidate scope.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Tuple

# file_id -> (filename, owner_id). Mirrors app/db/seed.py::build_files() order.
SEED_FILES: Dict[int, Tuple[str, str]] = {
    1: ("AI_Architecture_Overview.pptx", "user_a"),
    2: ("Q3_Budget_Review.xlsx", "user_a"),
    3: ("Q2_Budget_Notes.docx", "user_a"),
    4: ("Team_Offsite_Photo.png", "user_a"),
    5: ("Meeting_Notes_January.docx", "user_a"),
    6: ("Vendor_Budget_Estimate.pdf", "user_a"),
    7: ("Budget_Slides_Draft.pptx", "user_a"),
    8: ("Marketing_Plan.pptx", "user_b"),
    9: ("Recipe_Collection.docx", "user_b"),
    10: ("Travel_Itinerary.pdf", "user_c"),
    11: ("AI_Architecture_Notes.docx", "user_c"),
    12: ("Confidential_HR_Salaries.xlsx", "user_x"),
    13: ("AI_Architecture_Secret_Roadmap.pptx", "user_x"),
}

CATEGORIES = (
    "A_exact",
    "B_filename",
    "C_topic",
    "D_ambiguous",
    "E_personalization",
    "F_no_result",
    "G_cross_user",
    "H_multi_strategy",
)
OUTCOMES = ("results", "clarify_or_results", "no_result")
PERSONALIZATION_CONTEXTS = ("none", "history", "cold_start")


@dataclass(frozen=True)
class BenchmarkQuery:
    query_id: str
    query: str
    user_id: str
    category: str
    relevance: Mapping[int, int]  # file_id -> grade (1 relevant, 2 highly relevant)
    expected_outcome: str
    forbidden_file_ids: Tuple[int, ...] = ()
    personalization_context: str = "none"
    notes: str = ""

    @property
    def relevant_ids(self) -> List[int]:
        return sorted(self.relevance)

    @property
    def highly_relevant_ids(self) -> List[int]:
        return sorted(f for f, g in self.relevance.items() if g >= 2)


def _q(query_id, user_id, query, relevance, outcome="results", forbidden=(), ctx="none", notes=""):
    category = {
        "A": "A_exact", "B": "B_filename", "C": "C_topic", "D": "D_ambiguous",
        "E": "E_personalization", "F": "F_no_result", "G": "G_cross_user", "H": "H_multi_strategy",
    }[query_id[0]]
    return BenchmarkQuery(
        query_id=query_id, query=query, user_id=user_id, category=category,
        relevance=dict(relevance), expected_outcome=outcome,
        forbidden_file_ids=tuple(forbidden), personalization_context=ctx, notes=notes,
    )


QUERIES: Tuple[BenchmarkQuery, ...] = (
    # --- A. exact / near-exact: natural language aimed at one known file --------
    _q("A1", "user_a", "Find the slide deck about AI architecture I edited last week.", {1: 2},
       notes="Only user_a pptx about AI architecture; updated 6 days before seeding. user_x's near-identical deck (13) is inaccessible.",
       forbidden=(13,)),
    _q("A2", "user_a", "the vendor budget estimate pdf", {6: 2},
       notes="Vendor_Budget_Estimate.pdf is the only user_a pdf."),
    _q("A3", "user_a", "the team offsite photo", {4: 2},
       notes="Team_Offsite_Photo.png: filename, tags (photo, team) and summary all name it."),
    _q("A4", "user_b", "Marketing plan slide deck", {8: 2},
       notes="user_b's only pptx."),
    # --- B. filename-oriented: the evidence is the (partial) filename -----------
    _q("B1", "user_a", "file named Meeting_Notes_January", {5: 2}, notes="'named X' filename hint."),
    _q("B2", "user_a", "Q2_Budget_Notes.docx", {3: 2}, notes="Literal filename."),
    _q("B3", "user_b", "Recipe_Collection.docx", {9: 2}, notes="Literal filename; user_b."),
    _q("B4", "user_a", "Budget_Slides_Draft.pptx", {7: 2}, notes="Literal filename; shares the 'budget' token with 3 sibling files."),
    # --- C. topic / content: evidence only in summary / tags --------------------
    _q("C1", "user_a", "spreadsheet with department spend breakdown", {2: 2},
       notes="Summary of Q3_Budget_Review.xlsx: 'department spend breakdown'."),
    _q("C2", "user_a", "notes from the January planning meeting", {5: 2},
       notes="Summary of Meeting_Notes_January.docx: 'January planning meeting'."),
    _q("C3", "user_a", "tooling contract estimate", {6: 2},
       notes="Summary of Vendor_Budget_Estimate.pdf: 'new tooling contract'."),
    _q("C4", "user_b", "spring campaign slides", {8: 2},
       notes="Summary of Marketing_Plan.pptx: 'spring campaign'."),
    _q("C5", "user_c", "conference trip itinerary", {10: 2},
       notes="Summary of Travel_Itinerary.pdf: 'upcoming conference trip'."),
    # --- D. ambiguous: several files legitimately plausible ---------------------
    _q("D1", "user_a", "The budget thing I was looking at.", {2: 1, 3: 1, 6: 1, 7: 1},
       outcome="clarify_or_results",
       notes="Four user_a files are budget-related; nothing disambiguates them."),
    _q("D2", "user_a", "the notes", {3: 1, 5: 1}, outcome="clarify_or_results",
       notes="Two user_a docx notes files (Q2 budget notes, January meeting notes)."),
    _q("D3", "user_a", "the thing from last week", {1: 1, 2: 1, 6: 1, 7: 1}, outcome="clarify_or_results",
       notes="user_a files updated within ~9 days: AI deck (6d), Q3 review (2d), vendor estimate (3d), budget slides (4d)."),
    # --- E. personalization-sensitive -------------------------------------------
    _q("E1", "user_a", "budget file", {2: 2, 3: 1, 6: 1, 7: 1}, ctx="history",
       notes="Four budget files are acceptable; PERSONALIZATION_FIXTURE gives user_a heavy recent activity on Q3_Budget_Review (2), so it is the intended file for this user."),
    _q("E2", "user_a", "my slides", {7: 2, 1: 1}, ctx="history",
       notes="Two user_a pptx files; the fixture gives heavy recent edits on Budget_Slides_Draft (7) and one old view of the AI deck (1)."),
    _q("E3", "user_b", "marketing slides", {8: 2}, ctx="cold_start",
       notes="user_b has no telemetry/profile (cold start); a single obvious file, personalization must not disturb it."),
    _q("E4", "user_c", "ai architecture", {11: 2}, ctx="cold_start", forbidden=(1, 13),
       notes="Cold-start user_c; topic-colliding decks of user_a/user_x must not appear."),
    # --- F. no-result: nothing accessible is relevant ---------------------------
    _q("F1", "user_a", "quantum computing research paper", {}, outcome="no_result",
       notes="No user_a file concerns quantum computing."),
    _q("F2", "user_b", "tax return 2019", {}, outcome="no_result", notes="No user_b file concerns taxes."),
    _q("F3", "user_a", "kubernetes rollback runbook", {}, outcome="no_result", notes="No such file in the seed corpus."),
    # --- G. cross-user isolation: a match exists, for someone else --------------
    _q("G1", "user_a", "Confidential_HR_Salaries.xlsx", {}, outcome="no_result", forbidden=(12,),
       notes="Exact filename of user_x's file."),
    _q("G2", "user_b", "AI architecture slide deck", {}, outcome="no_result", forbidden=(1, 13),
       notes="user_a and user_x own matching decks; user_b owns none. (Short substring tokens such as 'ai' may textually hit user_b's own unrelated file -- that would be a precision finding, not an isolation failure.)"),
    _q("G3", "user_a", "AI architecture secret roadmap", {1: 1}, forbidden=(13,),
       notes="user_x's AI_Architecture_Secret_Roadmap.pptx must not appear; user_a's own AI deck is an acceptable answer."),
    _q("G4", "user_x", "AI architecture", {13: 2}, forbidden=(1, 11),
       notes="Isolation from the other direction: user_x must see only their own deck."),
    _q("G5", "user_b", "salary information hr", {}, outcome="no_result", forbidden=(12,),
       notes="Matches user_x's confidential HR file only."),
    # --- H. multi-strategy: designed so the loop's strategy choice is observable -
    _q("H1", "user_a", "budget estimate from vendor", {6: 2},
       notes="Three tokens all hit one file; expected to resolve with a single dominant metadata_search."),
    _q("H2", "user_a", "documents about planning", {3: 1, 5: 1},
       notes="Two docx files mention planning (Q2 budget planning, January planning meeting): flat evidence, expected to trigger the narrowing chain."),
    _q("H3", "user_a", "financial overview", {2: 2},
       notes="Concept query with no exact lexical anchor: 'finance' is only a tag of Q3_Budget_Review. 'overview' also textually hits AI_Architecture_Overview (a lexical distractor)."),
)


# Explicit telemetry for the personalization ablation. file_id -> events.
# (action_type, time_spent_seconds, days_ago). Only user_a has history; user_b
# and user_c are deliberately left cold-start.
PERSONALIZATION_FIXTURE = {
    "user_a": {
        "profile": {
            "preferred_file_types": ["xlsx", "pptx"],
            "frequent_topics": {"budget": 0.6, "finance": 0.4},
        },
        "events": {
            2: [("open", 300, 1), ("open", 240, 2), ("open", 200, 3), ("open", 180, 5),
                ("edit", 600, 1), ("edit", 500, 2), ("edit", 420, 4)],
            7: [("open", 200, 2), ("open", 150, 3), ("open", 120, 6),
                ("edit", 400, 2), ("edit", 380, 4)],
            6: [("view", 60, 20)],
            1: [("view", 45, 30)],
        },
    }
}


def query_by_id(query_id: str) -> BenchmarkQuery:
    for q in QUERIES:
        if q.query_id == query_id:
            return q
    raise KeyError(query_id)


def validate_dataset(queries=QUERIES, seed_files=SEED_FILES) -> List[str]:
    """Static (DB-free) validation. Returns a list of problems (empty = valid)."""
    problems: List[str] = []
    seen_ids, seen_pairs = set(), set()
    for q in queries:
        tag = f"[{q.query_id}]"
        if q.query_id in seen_ids:
            problems.append(f"{tag} duplicate query_id")
        seen_ids.add(q.query_id)
        pair = (q.user_id, " ".join(q.query.lower().split()))
        if pair in seen_pairs:
            problems.append(f"{tag} duplicate (user_id, query)")
        seen_pairs.add(pair)
        if q.category not in CATEGORIES:
            problems.append(f"{tag} unknown category {q.category!r}")
        if q.expected_outcome not in OUTCOMES:
            problems.append(f"{tag} unknown expected_outcome {q.expected_outcome!r}")
        if q.personalization_context not in PERSONALIZATION_CONTEXTS:
            problems.append(f"{tag} unknown personalization_context")
        known_owners = {o for _, o in seed_files.values()}
        if q.user_id not in known_owners:
            problems.append(f"{tag} user_id {q.user_id!r} owns no seed file")
        for fid, grade in q.relevance.items():
            if fid not in seed_files:
                problems.append(f"{tag} unknown relevant file_id {fid}")
                continue
            if grade not in (1, 2):
                problems.append(f"{tag} invalid grade {grade} for file {fid}")
            if seed_files[fid][1] != q.user_id:
                problems.append(f"{tag} relevant file {fid} is not accessible to {q.user_id}")
        for fid in q.forbidden_file_ids:
            if fid not in seed_files:
                problems.append(f"{tag} unknown forbidden file_id {fid}")
            elif seed_files[fid][1] == q.user_id:
                problems.append(f"{tag} forbidden file {fid} is actually owned by {q.user_id}")
            if fid in q.relevance:
                problems.append(f"{tag} file {fid} is both relevant and forbidden")
        if q.expected_outcome == "no_result" and q.relevance:
            problems.append(f"{tag} no_result query must not list relevant files")
        if q.expected_outcome != "no_result" and not q.relevance:
            problems.append(f"{tag} {q.expected_outcome} query has no relevant files")
        if q.personalization_context == "history" and q.user_id not in PERSONALIZATION_FIXTURE:
            problems.append(f"{tag} history context but no fixture for {q.user_id}")
        if q.personalization_context == "cold_start" and q.user_id in PERSONALIZATION_FIXTURE:
            problems.append(f"{tag} cold_start context but {q.user_id} has fixture history")
    for user, spec in PERSONALIZATION_FIXTURE.items():
        for fid in spec["events"]:
            if fid not in seed_files:
                problems.append(f"fixture: unknown file_id {fid}")
            elif seed_files[fid][1] != user:
                problems.append(f"fixture: file {fid} not owned by {user}")
    return problems


def validate_against_db(rows) -> List[str]:
    """`rows`: iterable of dicts with file_id/filename/owner_id from the live DB."""
    live = {r["file_id"]: (r["filename"], r["owner_id"]) for r in rows}
    problems = []
    for fid, expected in SEED_FILES.items():
        if fid not in live:
            problems.append(f"file_id {fid} ({expected[0]}) missing from database")
        elif live[fid] != expected:
            problems.append(f"file_id {fid}: expected {expected}, database has {live[fid]}")
    return problems
