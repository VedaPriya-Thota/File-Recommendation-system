"""
python -m evaluation.stage5c [RESULTS.json ...]

Phase 5C accounting over benchmark result files -- answers "does 5C still have a
legitimate role?" without touching production code. Pure post-processing of the
per-run records (no app imports beyond the shared near-tie definition).

Per LLM-capable variant:
  eligible     near-tie gate open (>= 2 of the top-N within DOMINANT_SCORE_GAP), result not clarified
  ran          the 5C provider was actually called
  suppressed   eligible but skipped because the pool was already resolved by the deterministic
               personalization path (comparison_status == skipped_personalization_resolved)
  changed      5C was applied and changed the order
  rejected     5C ran but its answer was not applied (low confidence / invalid / unknown candidate / outage)
"""
import json
import os
import sys

from evaluation.scenarios import _near_tie

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
LLM_VARIANTS = ("D_full_system", "D_llm_unavailable", "D_llm_invalid_output", "D_no_explanation", "D_cap3_diagnostic")
REJECTED = {"low_confidence", "invalid_output", "unknown_candidate", "unavailable", "provider_error"}


def account(results: list) -> dict:
    table = {}
    for v in LLM_VARIANTS:
        rows = [r for r in results if r["variant"] == v]
        if not rows:
            continue
        t = {"runs": len(rows), "eligible": [], "ran": [], "suppressed": [], "changed": [], "rejected": []}
        for r in rows:
            st = r["llm_stage_status"]["comparison"]
            suppressed = st == "skipped_personalization_resolved"
            ran = r["llm_attempts_by_stage"]["comparison"] > 0
            eligible = (_near_tie(r["top3_final_scores"]) and not r["clarified"]) or suppressed
            for key, flag in (("eligible", eligible), ("ran", ran), ("suppressed", suppressed),
                              ("changed", bool(r["llm_stage_status"]["comparison_applied"])),
                              ("rejected", ran and st in REJECTED)):
                if flag:
                    t[key].append(r["query_id"])
        table[v] = t
    return table


def render(label: str, table: dict) -> str:
    lines = [f"## {label}", "", "| variant | runs | 5C eligible | 5C ran | suppressed (personalization-resolved) | order changed | rejected |",
             "|---|---|---|---|---|---|---|"]
    for v, t in table.items():
        lines.append(f"| {v} | {t['runs']} | {len(t['eligible'])} | {len(t['ran'])} | {len(t['suppressed'])} | {len(t['changed'])} | {len(t['rejected'])} |")
    lines += ["", "Query IDs (D_full_system): " + "; ".join(
        f"{k}={table['D_full_system'][k] or '-'}" for k in ("eligible", "ran", "suppressed", "changed", "rejected")
    ) if "D_full_system" in table else "", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    paths = argv or [os.path.join(OUT, "phase8_baseline_results.json"), os.path.join(OUT, "benchmark_results.json")]
    text = "# Phase 5C accounting\n\n"
    for p in paths:
        with open(p) as f:
            text += render(os.path.basename(p), account(json.load(f)["results"])) + "\n"
    out = os.path.join(OUT, "phase9_5c_accounting.md")
    with open(out, "w") as f:
        f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
