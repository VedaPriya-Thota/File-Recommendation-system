"""Phase 8 -- the before/after comparison script is pure JSON post-processing."""
import json

from evaluation import compare


def _res(variant, status, returned, tools, success=True):
    return {"variant": variant, "query_id": "Q1", "status": status, "returned_file_ids": returned,
            "tool_sequence": tools, "tool_calls": len(tools), "success": success}


def _doc(mrr, clar, results, passed):
    summ = {"overall": {}, "behavior": {}}
    for v in compare.CORE:
        summ["overall"][v] = {k: mrr for k in compare.RETRIEVAL}
        summ["behavior"][v] = {k: clar for k in compare.BEHAVIOR}
    return {"summary": summ, "results": results, "scenarios": [{"name": "s1", "passed": passed}]}


def test_render_reports_metric_deltas_changed_queries_and_scenarios():
    before = _doc(0.5, 0.2, [_res("C_agentic_personalized", "clarify", [], ["metadata_search"], False),
                             _res("A_single_shot_metadata", "finish", [1], ["metadata_search"])], False)
    after = _doc(0.8, 0.0, [_res("C_agentic_personalized", "finish", [1], ["metadata_search"], True),
                            _res("A_single_shot_metadata", "finish", [1], ["metadata_search"])], True)
    text = compare.render(before, after)
    assert "| C | MRR | 0.500 | 0.800 |" in text
    assert "clarify [] (1 tools) | finish [1] (1 tools) | False -> True" in text  # the changed query
    assert text.count("| Q1 |") == 1  # the unchanged variant-A query is not listed
    assert "- s1: FAIL -> PASS" in text


def test_committed_phase7_baseline_and_current_results_are_comparable():
    import os
    out = compare.OUT
    with open(os.path.join(out, "phase7_baseline_results.json")) as fb, open(os.path.join(out, "benchmark_results.json")) as fa:
        b, a = json.load(fb), json.load(fa)
    assert b["metadata"]["dataset_size"] == a["metadata"]["dataset_size"]
    assert [r["query_id"] for r in b["results"]] == [r["query_id"] for r in a["results"]]  # same dataset, same order
    assert "Per-query behaviour changes" in compare.render(b, a)
