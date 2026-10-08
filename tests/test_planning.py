from app.goal import understand_goal
from app.planning import args_signature, plan_next_action


def test_initial_action_uses_all_available_signals():
    goal = understand_goal("Find the slide deck about AI architecture I edited last week.").to_dict()
    action = plan_next_action(goal, committed_actions=[], failed_strategies=[], iteration=1)
    assert action["tool"] == "metadata_search"
    assert action["args"]["file_type"] == "pptx"
    assert "date_from" in action["args"]
    assert action["args"]["topic_tokens"] == ["ai", "architecture"]


def test_ambiguous_goal_produces_topic_only_action():
    goal = understand_goal("The budget thing I was looking at.").to_dict()
    action = plan_next_action(goal, committed_actions=[], failed_strategies=[], iteration=1)
    assert action["args"] == {"topic_tokens": ["budget"]}


def test_zero_result_replan_relaxes_file_type_first():
    goal = understand_goal("Find the spreadsheet about AI architecture I edited last week.").to_dict()
    first = plan_next_action(goal, committed_actions=[], failed_strategies=[], iteration=1)
    assert first["args"]["file_type"] == "xlsx"

    second = plan_next_action(
        goal,
        committed_actions=[first],
        failed_strategies=[args_signature(first["tool"], first["args"])],
        iteration=2,
        observation_hint="zero_results",
    )
    assert second is not None
    assert "file_type" not in second["args"]
    assert second["args"]["topic_tokens"] == ["ai", "architecture"]
    assert "date_from" in second["args"]  # date_range kept -- only file_type was dropped


def test_never_repeats_identical_failed_strategy():
    goal = understand_goal("Find the slide deck about AI architecture I edited last week.").to_dict()
    first = plan_next_action(goal, committed_actions=[], failed_strategies=[], iteration=1)
    sig = args_signature(first["tool"], first["args"])

    second = plan_next_action(
        goal,
        committed_actions=[first],
        failed_strategies=[sig],
        iteration=2,
        observation_hint="zero_results",
    )
    assert second is not None
    assert args_signature(second["tool"], second["args"]) != sig


def test_too_many_observation_never_broadens():
    goal = understand_goal("Find the slide deck about AI architecture I edited last week.").to_dict()
    first = plan_next_action(goal, committed_actions=[], failed_strategies=[], iteration=1)
    action = plan_next_action(
        goal,
        committed_actions=[first],
        failed_strategies=[],
        iteration=2,
        observation_hint="too_many",
    )
    assert action is None, "no narrower strategy exists in Phase 1; must not broaden on 'too_many'"


def test_exhausted_metadata_strategies_escalate_to_bm25(): 
    """Phase 2A: once metadata_search's structural strategies are exhausted
    with zero results, the deterministic fallback is bm25_search -- not
    None. This is the observation-driven metadata -> bm25 transition."""
    goal = understand_goal("The budget thing I was looking at.").to_dict()
    first = plan_next_action(goal, committed_actions=[], failed_strategies=[], iteration=1)
    sig = args_signature(first["tool"], first["args"])
    action = plan_next_action(
        goal,
        committed_actions=[first],
        failed_strategies=[sig],
        iteration=2,
        observation_hint="zero_results",
    )
    assert action is not None
    assert action["tool"] == "bm25_search"
    assert action["args"]["query"] == "budget"


def test_exhausted_metadata_and_bm25_escalate_to_semantic():
    """Phase 2B: once metadata_search AND bm25_search have both been tried
    with zero results, the deterministic fallback is semantic_search --
    not None. This is the observation-driven bm25 -> semantic transition."""
    goal = understand_goal("The budget thing I was looking at.").to_dict()
    first = plan_next_action(goal, committed_actions=[], failed_strategies=[], iteration=1)
    sig1 = args_signature(first["tool"], first["args"])
    second = plan_next_action(
        goal,
        committed_actions=[first],
        failed_strategies=[sig1],
        iteration=2,
        observation_hint="zero_results",
    )
    assert second["tool"] == "bm25_search"
    sig2 = args_signature(second["tool"], second["args"])

    third = plan_next_action(
        goal,
        committed_actions=[first, second],
        failed_strategies=[sig1, sig2],
        iteration=3,
        observation_hint="zero_results",
    )
    assert third is not None
    assert third["tool"] == "semantic_search"
    assert third["args"]["query"] == "The budget thing I was looking at."  # raw_query, not concept_text


def test_exhausted_strategies_including_semantic_return_none():
    """Once metadata_search, bm25_search, AND semantic_search have all been
    tried and all returned zero results, there is genuinely nothing left."""
    goal = understand_goal("The budget thing I was looking at.").to_dict()
    first = plan_next_action(goal, committed_actions=[], failed_strategies=[], iteration=1)
    sig1 = args_signature(first["tool"], first["args"])
    second = plan_next_action(
        goal,
        committed_actions=[first],
        failed_strategies=[sig1],
        iteration=2,
        observation_hint="zero_results",
    )
    sig2 = args_signature(second["tool"], second["args"])
    third = plan_next_action(
        goal,
        committed_actions=[first, second],
        failed_strategies=[sig1, sig2],
        iteration=3,
        observation_hint="zero_results",
    )
    assert third["tool"] == "semantic_search"
    sig3 = args_signature(third["tool"], third["args"])

    action = plan_next_action(
        goal,
        committed_actions=[first, second, third],
        failed_strategies=[sig1, sig2, sig3],
        iteration=4,
        observation_hint="zero_results",
    )
    assert action is None, "metadata_search, bm25_search, and semantic_search all exhausted; nothing left to try"


def test_args_signature_is_stable_and_order_independent_of_dict_construction():
    sig1 = args_signature("metadata_search", {"file_type": "pptx", "topic_tokens": ["a", "b"]})
    sig2 = args_signature("metadata_search", {"topic_tokens": ["a", "b"], "file_type": "pptx"})
    assert sig1 == sig2
