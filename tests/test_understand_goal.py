from app.goal import understand_goal


def test_specific_query_low_ambiguity():
    g = understand_goal("Find the slide deck about AI architecture I edited last week.")
    assert g.intent == "file_discovery"
    assert g.filters["file_type"] == "pptx"
    assert g.filters["date_range"] is not None
    assert "ai" in g.topic_tokens
    assert "architecture" in g.topic_tokens
    assert g.ambiguity_score < 0.5, "specific query should have low ambiguity"


def test_ambiguous_query_high_ambiguity():
    g = understand_goal("The budget thing I was looking at.")
    assert g.filters["file_type"] is None
    assert g.filters["date_range"] is None
    assert g.filters["filename_candidate"] is None
    assert "budget" in g.topic_tokens
    assert g.ambiguity_score > 0.5, "ambiguous query should have high ambiguity"


def test_specific_query_less_ambiguous_than_ambiguous_query():
    specific = understand_goal("Find the slide deck about AI architecture I edited last week.")
    ambiguous = understand_goal("The budget thing I was looking at.")
    assert specific.ambiguity_score < ambiguous.ambiguity_score


def test_filename_extension_detected():
    g = understand_goal("Can you find Q3_Budget_Review.xlsx for me?")
    assert g.filters["filename_candidate"] == "Q3_Budget_Review.xlsx"


def test_named_filename_hint_detected():
    g = understand_goal("Find the file called Vendor Budget Estimate.")
    assert g.filters["filename_candidate"] is not None
    assert "vendor budget estimate" in g.filters["filename_candidate"].lower()


def test_file_type_keyword_mapping():
    assert understand_goal("find my spreadsheet").filters["file_type"] == "xlsx"
    assert understand_goal("find my presentation").filters["file_type"] == "pptx"
    assert understand_goal("find that document").filters["file_type"] == "docx"
    assert understand_goal("find that pdf").filters["file_type"] == "pdf"
    assert understand_goal("find that photo").filters["file_type"] == "png"


def test_topic_tokens_exclude_stopwords_and_file_type_words():
    g = understand_goal("Find the slide deck about AI architecture I edited last week.")
    for stop in ["find", "the", "about", "i", "edited", "last", "week", "slide", "deck"]:
        assert stop not in g.topic_tokens
