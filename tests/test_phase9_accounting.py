"""Phase 9 -- the 5C accounting script is pure post-processing of benchmark records."""
import json
import os

from evaluation import compare, stage5c


def _rec(qid, status, attempts=0, applied=False, scores=(0.6, 0.5), clarified=False, variant="D_full_system"):
    return {"variant": variant, "query_id": qid, "clarified": clarified, "top3_final_scores": list(scores),
            "llm_attempts_by_stage": {"comparison": attempts},
            "llm_stage_status": {"comparison": status, "comparison_applied": applied}}


def test_account_separates_eligible_ran_suppressed_changed_and_rejected():
    recs = [
        _rec("ran_changed", "selected", attempts=1, applied=True),
        _rec("ran_agreed", "selected", attempts=1),
        _rec("rejected", "low_confidence", attempts=1),
        _rec("suppressed", "skipped_personalization_resolved"),
        _rec("clear_winner", "not_needed", scores=(0.9, 0.4)),               # gate closed -> not eligible
        _rec("clarified", "not_needed", clarified=True),                     # clarified -> not eligible
        _rec("outage", "unavailable", attempts=1, variant="D_llm_unavailable"),
    ]
    t = stage5c.account(recs)
    d = t["D_full_system"]
    assert d["eligible"] == ["ran_changed", "ran_agreed", "rejected", "suppressed"]
    assert d["ran"] == ["ran_changed", "ran_agreed", "rejected"]
    assert d["suppressed"] == ["suppressed"] and d["changed"] == ["ran_changed"] and d["rejected"] == ["rejected"]
    assert t["D_llm_unavailable"]["rejected"] == ["outage"] and t["D_llm_unavailable"]["ran"] == ["outage"]
    assert "D_cap3_diagnostic" not in t  # variants with no records are omitted


def test_committed_results_show_suppression_only_on_llm_variants_and_never_without_a_ran_or_gate():
    with open(os.path.join(stage5c.OUT, "benchmark_results.json")) as f:
        results = json.load(f)["results"]
    t = stage5c.account(results)
    for v, row in t.items():
        assert set(row["suppressed"]) <= set(row["eligible"])           # suppression only where 5C was otherwise eligible
        assert not set(row["suppressed"]) & set(row["ran"])             # a suppressed run never called the provider
        assert set(row["changed"]) <= set(row["ran"])                   # only a run that happened can change the order
    assert all(r["llm_attempts_by_stage"]["comparison"] == 0 for r in results
               if r["llm_stage_status"]["comparison"] == "skipped_personalization_resolved")
    assert not any(r["variant"] in ("A_single_shot_metadata", "B_agentic_deterministic", "C_agentic_personalized")
                   and r["llm_calls"] for r in results)                  # deterministic variants never bill an LLM call


def test_compare_accepts_a_custom_output_name(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "OUT", str(tmp_path))
    src = os.path.join(os.path.dirname(os.path.abspath(compare.__file__)), "output", "benchmark_results.json")
    assert compare.main([src, src, "custom.md"]) == 0
    assert (tmp_path / "custom.md").exists()
