"""
Phase 11 -- goal-understanding file-type inference is whole-token, not substring.

Demonstrated harm (Phase 11 real-content evaluation, probes G1-G5): a substring hit such as
"representations" -> `presentation` -> file_type=pptx put a hard filter on bm25/semantic that removed every
relevant file, so the agent returned NOTHING for G4/G5/G2 and a wrong file for G1.
"""
import pytest

from app.goal import FILE_TYPE_KEYWORDS, understand_goal


def ftype(q):
    return understand_goal(q).filters["file_type"]


@pytest.mark.parametrize("query,spurious", [
    ("learned numeric representations of words", "pptx"),      # presentation
    ("landslides on the northern slope", "pptx"),              # slides
    ("doctor appointment and vaccination schedule", "docx"),   # doc
    ("how photosynthesis works in leaves", "png"),             # photo
    ("excellent seafood restaurants in Lisbon", "xlsx"),       # excel
    ("poetic imagery in the novel", "png"),                    # imagery contains `image`
])
def test_keywords_inside_longer_words_no_longer_set_a_file_type(query, spurious):
    assert ftype(query) is None


def test_the_old_substring_rule_really_was_wrong_on_these_probes():
    """Guards the claim above: each probe contained a keyword as a bare substring."""
    for query in ("learned numeric representations of words", "landslides on the northern slope",
                  "doctor appointment and vaccination schedule", "how photosynthesis works in leaves",
                  "excellent seafood restaurants in Lisbon"):
        assert any(kw in query for kws in FILE_TYPE_KEYWORDS.values() for kw in kws), query


@pytest.mark.parametrize("query,expected", [
    ("find my spreadsheet", "xlsx"), ("find my presentation", "pptx"), ("find that document", "docx"),
    ("find that pdf", "pdf"), ("find that photo", "png"),
    ("documents about planning", "docx"),          # plural of a keyword
    ("my slides", "pptx"), ("the team offsite photos", "png"), ("two spreadsheets", "xlsx"), ("old pdfs", "pdf"),
    ("the slide deck about AI architecture", "pptx"), ("budget sheet", "xlsx"), ("write-up of the trip", "docx"),
    ("a word doc", "docx"), ("Excel file", "xlsx"), ("PowerPoint", "pptx"),                         # case-insensitive
])
def test_genuine_file_type_words_still_set_the_type(query, expected):
    assert ftype(query) == expected


@pytest.mark.parametrize("query,expected", [
    ("Q3_Budget_Review.xlsx", "xlsx"), ("Budget_Slides_Draft.pptx", "pptx"), ("Recipe_Collection.docx", "docx"),
    ("file named Meeting_Notes_January", "docx"),   # '_' separates tokens, exactly as before
    ("notes_from_march", "docx"), ("the-slides-v2", "pptx"),
])
def test_filenames_keep_their_behaviour_underscores_dots_and_hyphens_are_separators(query, expected):
    assert ftype(query) == expected


def test_the_first_matching_type_in_dict_order_still_wins():
    assert ftype("slides and a spreadsheet") == "pptx"   # pptx is listed before xlsx; unchanged precedence


def test_goal_keeps_its_other_fields_untouched():
    g = understand_goal("learned numeric representations of words")
    assert g.filters["file_type"] is None and g.filters["date_range"] is None
    assert g.topic_tokens == ["learned", "numeric", "representations", "words"]
