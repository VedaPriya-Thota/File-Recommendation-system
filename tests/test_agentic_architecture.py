"""
Tests that the multi-agent architecture really behaves like one: the orchestrator picks agents per task, agents use only their own tools,
tool results flow between agents as structured data, verification can stop or redirect the work, failures are retried a bounded number of
times, results are cached, independent tasks run in parallel, and the event stream is a faithful record of what actually executed.
"""
import time

import pytest

from app import folders as F
from app.agents import tools as agent_tools
from app.agents.core import AgentFailure, Board, EventBus, Metrics, ToolBox
from app.assistant import conversation as convmod
from app.assistant.engine import run_assistant
from evaluation.agentic import harness as H

OWNER = H.OWNER


@pytest.fixture(scope="module")
def corpus():
    root, folder = H.setup()
    yield root
    H.teardown(folder)


def ask(q, conv=None):
    ev = []
    res = run_assistant(OWNER, q, ev.append, conversation_id=conv or convmod.new_id())
    res["_events"] = ev
    return res


# ----------------------------------------------------------------------------- the seven task types
@pytest.mark.parametrize("sc", H.SCENARIOS, ids=[s.id for s in H.SCENARIOS])
def test_scenario(corpus, sc):
    res = H.run_one(sc)
    checks = H.judge(sc, res)
    failed = {k: d for k, (ok, d) in checks.items() if not ok}
    assert not failed, failed


# ----------------------------------------------------------------------------- dynamic agent selection and handoffs
def test_orchestrator_skips_agents_it_does_not_need(corpus):
    rec = ask("Find files about the quarterly budget")
    assert "Analysis Agent" not in rec["agents"]["agents"]                    # a recommendation needs no analysis
    summ = ask("Summarize final.html")
    assert "Analysis Agent" in summ["agents"]["agents"]
    plan_events = [e for e in summ["_events"] if e["kind"] == "plan" and e.get("plan")]
    assert len(plan_events) >= 2 and plan_events[0]["plan"][0]["agent"] == "discovery"          # stage 1 plans discovery only
    assert {t["agent"] for t in plan_events[-1]["plan"]} >= {"discovery", "retrieval", "analysis", "verification", "response"}   # stage 2 grows from what was found


def test_follow_up_reuses_context_and_skips_a_new_search(corpus):
    first = ask("Summarize pricing_2024.txt")
    cid = first["conversation_id"]
    second = ask("What does it say about support?", cid)
    assert second["followup"] is True and second["status"] == "ok" and "live chat" in second["answer"].lower()
    tool_names = [e["tool"] for e in second["_events"] if e.get("tool")]
    assert "search_files" not in tool_names                                    # discovery reused the previous answer's file
    assert any(e["agent"] == "File Discovery Agent" and "previous answer" in (e["detail"] + e["title"]) for e in second["_events"])


def test_replans_when_a_calculation_request_has_no_matching_spreadsheet(corpus):
    res = ask("How much does the standard plan cost per month?")
    assert any("Re-planning" in e["title"] for e in res["_events"])
    assert res["conflicts"], "the two pricing files disagree and both must be reported"
    assert "20 dollars" in res["answer"] and "25 dollars" in res["answer"]


def test_verification_sends_retrieval_back_out_once_then_stops(corpus):
    res = ask("What is the CEO's salary?")
    assert res["agents"]["rounds"] == 2                                       # exactly one extra round: bounded
    titles = [e["title"] for e in res["_events"]]
    assert any("Verification asked for more evidence" in t for t in titles)
    assert res["status"] == "insufficient" and "couldn't find" in res["answer"]
    assert res["agents"]["tasks_run"] <= 16


# ----------------------------------------------------------------------------- tools, permissions, provenance
def test_tool_events_are_real_and_agents_use_only_their_own_tools(corpus):
    res = ask("Analyze my machine learning project and explain how the model is trained.")
    key = {"File Discovery Agent": "discovery", "Retrieval Agent": "retrieval", "Analysis Agent": "analysis", "Verification Agent": "verification", "Response Agent": "response"}
    tool_events = [e for e in res["_events"] if e["kind"] == "tool" and e.get("tool") in agent_tools.REGISTRY]
    assert tool_events
    for e in tool_events:
        assert key[e["agent"]] in agent_tools.REGISTRY[e["tool"]].agents, (e["agent"], e["tool"])
    assert res["agents"]["tool_calls"] >= len(tool_events) - 2
    assert res["agents"]["parallel_batches"] >= 1                              # per-file reads ran as parallel tasks
    assert any(e["kind"] == "handoff" and "parallel" in e["title"] for e in res["_events"])


def test_a_tool_is_refused_to_an_agent_that_does_not_own_it(corpus):
    bus = EventBus("t", lambda e: None)
    b = Board(owner_id=OWNER, question="q", plan_q=None, settings={}, bus=bus)
    tb = ToolBox(agent_tools.REGISTRY, b)
    r = tb.call("retrieval", "analyze_table", file_id=1, question="total")
    assert not r.ok and "not permitted" in r.error
    assert any(e["status"] == "error" and "not permitted" in e["title"] for e in bus.events)


def test_sources_keep_provenance_through_the_whole_workflow(corpus):
    res = ask("How many dollars is the Q3 marketing budget?")
    ref = res["references"][0]
    assert ref["filename"] == "quarterly_budget_q3.txt" and ref["abs_path"].endswith("quarterly_budget_q3.txt")
    ex = ref["excerpts"][0]
    assert ex["verified"] is True and "30,000 dollars" in ex["text"] and ex["location"]
    assert "[1" in res["answer"]


# ----------------------------------------------------------------------------- failures, retries, caching
def test_a_transient_tool_failure_is_retried_once_and_the_run_succeeds(corpus, monkeypatch):
    real = agent_tools.REGISTRY["summarize_document"].fn
    state = {"n": 0}

    def flaky(b, **kw):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("temporary failure")
        return real(b, **kw)
    monkeypatch.setitem(agent_tools.REGISTRY, "summarize_document", agent_tools.REGISTRY["summarize_document"].__class__("summarize_document", "x", flaky, {"analysis"}))
    res = ask("Summarize pricing_2023.txt")
    assert res["status"] == "ok" and res["agents"]["retries"] >= 1 and state["n"] >= 2
    assert any(e["kind"] == "retry" for e in res["_events"])


def test_a_permanently_failing_tool_ends_in_an_honest_answer_not_a_crash(corpus, monkeypatch):
    def broken(b, **kw):
        raise RuntimeError("disk on fire")
    monkeypatch.setitem(agent_tools.REGISTRY, "summarize_document", agent_tools.REGISTRY["summarize_document"].__class__("summarize_document", "x", broken, {"analysis"}))
    res = ask("Summarize pricing_2023.txt")
    assert res["status"] in ("insufficient", "error")
    assert "disk on fire" not in res["answer"]                                  # internal error text is not shown as an answer
    assert any(e["status"] == "error" for e in res["_events"])
    assert res["agents"]["tool_calls"] < 20                                     # bounded: no retry storm


def test_reusable_file_processing_is_cached(corpus):
    ask("Summarize final.html")
    again = ask("Summarize final.html")
    cached = [e for e in again["_events"] if e["kind"] == "tool" and e.get("cached")]
    assert cached and again["agents"]["cache_hits"] >= 1


# ----------------------------------------------------------------------------- verification stops fabricated conclusions
def _fake_llm(monkeypatch, text):
    import app.agents.response_agent as ra
    import app.assistant.qa as qa

    class Fake:
        def complete(self, s, u):
            Fake.prompt = u
            return text
    for mod in (qa, ra):
        monkeypatch.setattr(mod, "llm_status", lambda: {"active": True, "provider": "fake", "model": "x", "external": False, "reason": ""})
        monkeypatch.setattr(mod, "get_default_provider", lambda: Fake())
    return Fake


def test_an_invented_model_answer_is_rejected_by_the_verification_agent(corpus, monkeypatch):
    fake = _fake_llm(monkeypatch, "The Q3 marketing budget is 90,000 dollars [1]. Travel is unlimited this quarter [1].")
    res = ask("How many dollars is the Q3 marketing budget?")
    assert "90,000" not in res["answer"] and "unlimited" not in res["answer"]
    assert "30,000" in res["answer"]                                             # the grounded extractive answer replaced it
    assert res["agents"]["llm_calls"] == 1                                       # exactly one model call, not one per agent
    assert any("rejected" in e["title"].lower() and e["agent"] == "Verification Agent" for e in res["_events"])
    assert "<document" in fake.prompt                                            # sources are delimited as untrusted data


def test_a_grounded_model_answer_is_kept(corpus, monkeypatch):
    _fake_llm(monkeypatch, "The Q3 marketing budget is 30,000 dollars, and travel is capped at 5,000 dollars [1].")
    res = ask("How many dollars is the Q3 marketing budget?")
    assert res["llm"]["used"] is True and "5,000" in res["answer"] and res["agents"]["llm_calls"] == 1


def test_no_model_call_is_made_when_there_is_nothing_to_phrase(corpus, monkeypatch):
    _fake_llm(monkeypatch, "invented")
    for q in ("What is the CEO's salary?", "Which month has the highest sales?", "Find files about the quarterly budget"):
        assert ask(q)["agents"]["llm_calls"] == 0


# ----------------------------------------------------------------------------- answers read like answers
def test_html_summary_explains_the_page_instead_of_dumping_tags(corpus):
    res = ask("Summarize final.html")
    a = res["answer"]
    assert "<" not in a and "{" not in a
    assert a.lower().startswith("**final.html** is a web page") and "interpretation" in a   # inference is labelled as inference


def test_ml_workflow_answer_is_ordered_and_sourced(corpus):
    res = ask("Analyze my machine learning project and explain how the model is trained.")
    a = res["answer"]
    assert a.index("main()") < a.index("build_model") < a.index("train(")             # entry point first, then what it calls
    assert "epochs = 20" in a and "learning_rate = 0.001" in a                        # hyper-parameters come from the config file
    assert "not from running" in a or "not on running" in a
    assert {r["filename"] for r in res["references"]} >= {"train.py", "model.py", "data.py", "config.yaml"}
