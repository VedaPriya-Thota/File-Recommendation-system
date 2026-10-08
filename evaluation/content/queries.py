"""
Phase 11 real-content query set -- hand-labelled, written BEFORE the first
measurement run and frozen. Ground truth is (owner, relative path) from
corpus_spec.py only; it is resolved to file_ids at run time from the ingested
database rows, never from system output.

Grades: 2 = intended / strongest answer, 1 = acceptable. `forbidden` lists files
of ANOTHER user that match the query textually and must never appear.

Per-query expectations (used for stage attribution, never to score):
  expect_bm25 / expect_semantic : which retrieval path is meant to find the answer
  personalization               : the cases where user history is meant to matter
Success rule is the same single definition as Phase 7 (evaluation.metrics.judge_success).

Categories
  L lexical filename   C content-only lexical   S semantic (no lexical overlap)
  G goal probes (substring keyword class)        A ambiguous
  P personalization    N no relevant file        X cross-user isolation
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple

CATEGORIES = ("L_filename", "C_content", "S_semantic", "G_goal_probe", "A_ambiguous", "P_personalization", "N_no_result", "X_cross_user")


@dataclass(frozen=True)
class ContentQuery:
    qid: str
    user: str                         # "alice" | "bob"
    query: str
    category: str
    relevant: Dict[str, int]          # rel_path (of the querying user) -> grade
    outcome: str = "results"          # results | clarify_or_results | no_result
    forbidden: Tuple[Tuple[str, str], ...] = ()   # (owner, rel_path) of other users' files
    expect_bm25: bool = False
    expect_semantic: bool = False
    personalization: bool = False
    notes: str = ""


def _q(qid, user, query, relevant, outcome="results", forbidden=(), bm25=False, semantic=False, pers=False, notes=""):
    cat = {"L": "L_filename", "C": "C_content", "S": "S_semantic", "G": "G_goal_probe", "A": "A_ambiguous",
           "P": "P_personalization", "N": "N_no_result", "X": "X_cross_user", "D": "diagnostic"}[qid[0]]
    return ContentQuery(qid, user, query, cat, dict(relevant), outcome, tuple(forbidden), bm25, semantic, pers, notes)


QUERIES: Tuple[ContentQuery, ...] = (
    # --- L: the filename states the topic ---------------------------------------
    _q("L1", "alice", "quarterly budget review", {"quarterly_budget_review.txt": 2}, bm25=True),
    _q("L2", "alice", "employee handbook", {"employee_handbook.pdf": 2}, bm25=True),
    _q("L3", "alice", "server migration plan", {"server_migration_plan.md": 2}, bm25=True,
       notes="'plan' also appears in other bodies (marathon, budget) -- a mild lexical distractor"),
    _q("L4", "alice", "chess_openings.md", {"chess_openings.md": 2}, bm25=True, notes="exact filename"),
    # --- C: distinctive words only in the body -------------------------------------
    _q("C1", "alice", "sourdough levain hydration", {"kitchen_notes.txt": 2}, bm25=True),
    _q("C2", "alice", "glacier moraine sediment samples", {"field_report_2023.docx": 2}, bm25=True),
    _q("C3", "alice", "kubernetes autoscaling postmortem", {"meeting_minutes_q2.txt": 2}, bm25=True),
    _q("C4", "alice", "tetanus booster flu shot", {"vaccination_records.txt": 2}, bm25=True),
    # --- S: paraphrases with zero lexical overlap with the target ---------------------
    _q("S1", "alice", "growing vegetables outdoors", {"garden_journal.md": 2}, semantic=True),
    _q("S2", "alice", "keeping my cardiovascular system in good shape", {"heart_health_leaflet.pdf": 2}, semantic=True),
    _q("S3", "alice", "obligations when renting an apartment", {"lease_agreement_summary.docx": 2}, semantic=True),
    _q("S4", "alice", "watching galaxies and constellations at night", {"astronomy_observing_guide.txt": 2}, semantic=True),
    _q("S5", "alice", "routine for learning a musical instrument", {"piano_practice_plan.md": 2}, semantic=True),
    _q("S6", "alice", "generating electricity from sunlight at home", {"solar_install_quote.txt": 2}, semantic=True,
       notes="no lexical overlap with the target; 'sunlight' does appear in the distractor plant_biology_primer.md (a lexical trap)"),
    # --- G: lexical queries containing a word a SUBSTRING match misreads as a file-type hint ----
    _q("G1", "alice", "doctor appointment and vaccination schedule", {"vaccination_records.txt": 2}, bm25=True,
       notes="'doctor' contains the docx keyword 'doc'"),
    _q("G2", "alice", "how photosynthesis works in leaves", {"plant_biology_primer.md": 2}, bm25=True,
       notes="'photosynthesis' contains the png keyword 'photo'"),
    _q("G3", "alice", "excellent seafood restaurants in Lisbon", {"lisbon_dining_guide.pdf": 2}, bm25=True,
       notes="'excellent' contains the xlsx keyword 'excel'; flight_booking_receipt.pdf also mentions Lisbon but is not relevant"),
    _q("G4", "alice", "learned numeric representations of words", {"embedding_primer.md": 2}, bm25=True,
       notes="'representations' contains the pptx keyword 'presentation' (the Phase 10 observation)"),
    _q("G5", "alice", "landslides on the northern slope", {"erosion_survey.txt": 2}, bm25=True,
       notes="'landslides' contains the pptx keyword 'slides'"),
    # --- A: genuinely ambiguous ----------------------------------------------------------------
    _q("A1", "alice", "the notes", {"book_club_notes.md": 1, "kitchen_notes.txt": 1}, outcome="clarify_or_results", bm25=True),
    _q("A2", "alice", "the plan", {"server_migration_plan.md": 1, "marathon_training_schedule.docx": 1}, outcome="clarify_or_results", bm25=True),
    _q("A3", "alice", "something about money",
       {"quarterly_budget_review.txt": 1, "stock_market_primer.txt": 1, "tax_filing_checklist.pdf": 1},
       outcome="clarify_or_results", semantic=True),
    # --- P: several acceptable files; history says which is intended -----------------------------
    _q("P1", "alice", "dinner recipe ideas", {"thai_curry_guide.txt": 2, "recipe_pasta.md": 1}, bm25=True, pers=True,
       notes="both are dinners; the fixture's history is heavy on thai_curry_guide.txt"),
    _q("P2", "alice", "trip plans", {"hiking_trip_itinerary.txt": 2, "flight_booking_receipt.pdf": 1}, bm25=True, semantic=True, pers=True,
       notes="history is heavy on hiking_trip_itinerary.txt"),
    # --- N: nothing in the corpus is relevant ------------------------------------------------------------
    _q("N1", "alice", "quantum chromodynamics lattice gauge theory", {}, outcome="no_result"),
    _q("N2", "alice", "medieval siege warfare catapults and trebuchets", {}, outcome="no_result"),
    _q("N3", "alice", "cryptocurrency mining hardware profitability", {}, outcome="no_result",
       notes="finance documents exist (budget, stocks, tax) but none concern cryptocurrency"),
    # --- X: isolation -------------------------------------------------------------------------------------------
    _q("X1", "alice", "merger acquisition negotiation valuation terms", {}, outcome="no_result",
       forbidden=[("bob", "merger_negotiation_terms.txt")], bm25=True),
    _q("X2", "alice", "tomato seedlings raised beds compost", {"garden_journal.md": 2},
       forbidden=[("bob", "bob_garden_diary.txt")], bm25=True, notes="bob's diary matches this almost verbatim"),
    _q("X3", "bob", "growing vegetables outdoors", {"bob_garden_diary.txt": 2},
       forbidden=[("alice", "garden_journal.md")], semantic=True),
    _q("X4", "bob", "quarterly budget review", {"quarterly_budget_review.txt": 2},
       forbidden=[("alice", "quarterly_budget_review.txt")], bm25=True, notes="same filename, different owner and content"),
)

# Phase 12 DIAGNOSTICS -- pre-registered before being run, reported SEPARATELY and never part of the frozen 31
# queries or of any overall metric. D1 exists only to verify that the existing personalization path (get_user_activity ->
# Rule A) CAN be reached on real content: both dinner documents contain "dinner" and "basil" in the body only (no
# filename/summary match), so metadata_search finds nothing and the BM25 pool should be close to flat.
DIAGNOSTIC_QUERIES: Tuple[ContentQuery, ...] = (
    _q("D1", "alice", "dinner with basil", {"thai_curry_guide.txt": 2, "recipe_pasta.md": 1}, bm25=True, pers=True,
       notes="history is heavy on thai_curry_guide.txt; diagnostic of reachability, not a score"),
)

# Explicit history for the personalization cases (alice only; bob stays cold-start).
# rel_path -> [(action, seconds, days_ago)]
TELEMETRY = {
    "alice": {
        "profile": {"preferred_file_types": ["txt"], "frequent_topics": {"cooking": 0.5, "travel": 0.4}},
        "events": {
            "thai_curry_guide.txt": [("open", 300, 1), ("open", 240, 2), ("open", 200, 3), ("open", 180, 4), ("open", 160, 6),
                                     ("edit", 500, 1), ("edit", 420, 2), ("edit", 380, 5)],
            "hiking_trip_itinerary.txt": [("open", 260, 1), ("open", 220, 3), ("open", 150, 5),
                                          ("edit", 400, 2), ("edit", 350, 4)],
            "recipe_pasta.md": [("view", 60, 40)],
            "flight_booking_receipt.pdf": [("view", 45, 30)],
        },
    }
}


def validate(corpus) -> list:
    """Static validation against corpus_spec (no system involved). Returns problems."""
    problems, seen = [], set()
    for q in QUERIES:
        tag = f"[{q.qid}]"
        if q.qid in seen:
            problems.append(f"{tag} duplicate qid")
        seen.add(q.qid)
        if q.category not in CATEGORIES:
            problems.append(f"{tag} bad category")
        if q.user not in corpus:
            problems.append(f"{tag} unknown user")
            continue
        for rel in q.relevant:
            if rel not in corpus[q.user]:
                problems.append(f"{tag} relevant file {rel!r} is not in {q.user}'s corpus")
        for owner, rel in q.forbidden:
            if owner == q.user or rel not in corpus.get(owner, {}):
                problems.append(f"{tag} forbidden ({owner},{rel}) invalid")
        if q.outcome == "no_result" and q.relevant:
            problems.append(f"{tag} no_result with relevant files")
        if q.outcome != "no_result" and not q.relevant:
            problems.append(f"{tag} {q.outcome} without relevant files")
    for q in DIAGNOSTIC_QUERIES:
        if any(rel not in corpus[q.user] for rel in q.relevant):
            problems.append(f"[{q.qid}] diagnostic relevant file missing")
    pairs = [(q.user, " ".join(q.query.lower().split())) for q in (*QUERIES, *DIAGNOSTIC_QUERIES)]
    if len(pairs) != len(set(pairs)):
        problems.append("duplicate (user, query)")
    return problems
